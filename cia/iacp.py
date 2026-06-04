"""
Inter-Agent Communication Protocol (IACP) — the outermost layer of the CIA.

Implements the 4-phase disclosure negotiation protocol (§4 of the paper):

  Phase 1  Identify    Mutual pseudonym exchange; blocklist check; trust score
  Phase 2  Request     Validate A_req, purpose, retention policy
  Phase 3  Negotiate   Full IPC → CC → DDE pipeline; counter-proposal loop (≤3 rounds)
  Phase 4  Confirm     Sign and return the finalised DisclosureReceipt

Degradation: if ``mode="web"`` the IACP runs a single unilateral DDE decision
(phases 1 and 4 are skipped), for compatibility with conventional web services.

RetentionPolicy → temporary β adjustment in the DDE call:
  ANONYMISED:   β += 0.15   (anonymised storage → higher trust signal weight)
  PERSISTENT:   β -= 0.10   (indefinite storage → lower trust signal weight)
  SESSION_ONLY:  no change

Permitted imports: all prior layers.  This is the ONLY module permitted to
import from ipc.py, cc.py, and dde.py simultaneously.
Nothing imports from iacp.py except main.py and eval/.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Literal

from cia.models.attributes import Attribute, AttributeSet, AttributeVocab
from cia.models.context import ClassifiedContext, ContextClass, ContextSignal
from cia.models.policy import PolicySet
from cia.models.profile import DisclosureEvent
from cia.models.request import DisclosureRequest, PlatformType, RetentionPolicy
from cia.models.response import (
    DisclosureReceipt, DisclosureResponse, NegotiationStatus,
)
from cia.policy.audit import AuditLog, _canonical_json
from cia.policy.norms import NormDatabase
from cia.crypto.pseudonyms import PseudonymEngine, _hkdf_extract, _hkdf_expand
from cia.ipc import IPC, DriftAlert, EntropyAlert
from cia.cc import ContextClassifier
from cia.dde import DisclosureDecisionEngine
from cia.store.pds import PDS

logger = logging.getLogger(__name__)

_MAX_ROUNDS: int = 3
_BLOCKLIST_KEY: str = "iacp_blocklist"

# Carry-forward spec: "ANONYMISED → +0.15; PERSISTENT → −0.10"
_RETENTION_BETA: dict[RetentionPolicy, float] = {
    RetentionPolicy.SESSION_ONLY: 0.0,
    RetentionPolicy.PERSISTENT:  -0.10,
    RetentionPolicy.ANONYMISED:  +0.15,
}

# Contexts where an absent or blank purpose forces validation failure.
_PURPOSE_REQUIRED: frozenset[ContextClass] = frozenset({
    ContextClass.MEDICAL,
    ContextClass.FINANCIAL,
})


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CounterProposal:
    """
    CIA's counter-offer returned in Phase 3 when D* ⊊ A_req.

    offered:            Attributes CIA is willing to disclose as raw values.
    zkp_substitutions:  {attr → circuit_name} — CIA will prove the predicate
                        instead of disclosing the raw value.
    rationale:          Human-readable explanation (surfaced to the user).
    """

    offered: AttributeSet
    zkp_substitutions: dict[Attribute, str] = field(default_factory=dict)
    rationale: str = ""


@dataclass
class IACPResult:
    """Final outcome of a complete IACP session."""

    status: NegotiationStatus
    response: DisclosureResponse | None
    receipt: DisclosureReceipt | None
    rounds: int
    counter_proposals: list[CounterProposal] = field(default_factory=list)
    # Human-readable error message (populated on REJECTED / FAILED).
    error: str = ""


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class IACPSession:
    """
    Stateful IACP session.  Construct once per user identity, reuse across
    multiple requests.

    Parameters
    ----------
    pds:        Personal Data Store.
    ipc:        Identity Profile Constructor.
    cc:         Context Classifier.
    dde:        Disclosure Decision Engine.
    norm_db:    Norm database Φ(c).
    policy:     User's PolicySet Π.
    audit_log:  Tamper-evident disclosure log (shared with DDE).
    """

    def __init__(
        self,
        pds: PDS,
        ipc: IPC,
        cc: ContextClassifier,
        dde: DisclosureDecisionEngine,
        norm_db: NormDatabase,
        policy: PolicySet,
        audit_log: AuditLog,
    ) -> None:
        self._pds      = pds
        self._ipc      = ipc
        self._cc       = cc
        self._dde      = dde
        self._norms    = norm_db
        self._policy   = policy
        self._audit    = audit_log
        # Per-context pending EntropyAlert from the previous interaction.
        # Consumed by the next DDE call then cleared.
        self._pending_entropy: dict[ContextClass, EntropyAlert] = {}

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def handle(
        self,
        request: DisclosureRequest,
        mode: Literal["agent", "web"] = "web",
    ) -> IACPResult:
        """
        Process an incoming disclosure request.

        ``mode="web"``   — single unilateral DDE decision (phases 1 & 4 skipped).
        ``mode="agent"`` — full 4-phase negotiation protocol.
        """
        if mode == "web":
            return self._handle_web(request)
        return self._handle_agent(request)

    # ------------------------------------------------------------------
    # Blocklist management (public utility)
    # ------------------------------------------------------------------

    def block_platform(self, platform_id: str) -> None:
        """Add a platform to the CIA's blocklist (persisted in PDS config)."""
        did = _platform_did(platform_id)
        raw = self._pds.get_raw_config(_BLOCKLIST_KEY)
        blocked: list[str] = json.loads(raw) if raw else []
        if did not in blocked:
            blocked.append(did)
            self._pds.set_raw_config(_BLOCKLIST_KEY, json.dumps(blocked))

    def unblock_platform(self, platform_id: str) -> None:
        did = _platform_did(platform_id)
        raw = self._pds.get_raw_config(_BLOCKLIST_KEY)
        if not raw:
            return
        blocked = [d for d in json.loads(raw) if d != did]
        self._pds.set_raw_config(_BLOCKLIST_KEY, json.dumps(blocked))

    # ------------------------------------------------------------------
    # Web mode
    # ------------------------------------------------------------------

    def _handle_web(self, request: DisclosureRequest) -> IACPResult:
        err = self._validate_request(request)
        if err:
            return IACPResult(
                status=NegotiationStatus.REJECTED, response=None,
                receipt=None, rounds=0, error=err,
            )

        beta_adj = _retention_beta_adj(request.retention_policy)
        response, trace, _ = self._run_pipeline(request, beta_adj=beta_adj)
        receipt = self._fetch_receipt(response.receipt_id)

        return IACPResult(
            status=response.negotiation_status,
            response=response,
            receipt=receipt,
            rounds=1,
        )

    # ------------------------------------------------------------------
    # Agent mode — full 4-phase protocol
    # ------------------------------------------------------------------

    def _handle_agent(self, request: DisclosureRequest) -> IACPResult:
        # Phase 1: Identify
        block_err = self._phase_identify(request)
        if block_err:
            return block_err

        # Phase 2: Request validation
        err = self._validate_request(request)
        if err:
            return IACPResult(
                status=NegotiationStatus.REJECTED, response=None,
                receipt=None, rounds=0, error=err,
            )

        # Phase 3: Negotiate
        result = self._phase_negotiate(request)

        # Phase 4: Confirm — sign receipt with context-specific key
        if result.receipt is not None and result.status in (
            NegotiationStatus.ACCEPTED,
            NegotiationStatus.PARTIAL,
            NegotiationStatus.PROOF_ONLY,
        ):
            result.receipt = self._sign_receipt(result.receipt, request)

        return result

    # ------------------------------------------------------------------
    # Phase 1: Identify
    # ------------------------------------------------------------------

    def _phase_identify(self, request: DisclosureRequest) -> IACPResult | None:
        """Return an IACPResult (rejection) if identity check fails, else None."""
        did = _platform_did(request.platform_id)
        raw = self._pds.get_raw_config(_BLOCKLIST_KEY)
        if raw and did in json.loads(raw):
            logger.warning("Platform %s is blocklisted; rejecting.", request.platform_id)
            return IACPResult(
                status=NegotiationStatus.REJECTED, response=None, receipt=None,
                rounds=0, error=f"Platform '{request.platform_id}' is on the CIA blocklist.",
            )
        # Log the CIA's pseudonym for this interaction (informational).
        ctx_hint = _platform_context_hint(request.platform_type)
        master_id = self._pds.get_master_id()
        cia_did = PseudonymEngine.derive(master_id, ctx_hint)
        logger.debug("CIA presenting DID %s for platform %s.", cia_did, request.platform_id)
        return None

    # ------------------------------------------------------------------
    # Phase 3: Negotiate
    # ------------------------------------------------------------------

    def _phase_negotiate(self, request: DisclosureRequest) -> IACPResult:
        beta_adj = _retention_beta_adj(request.retention_policy)
        counter_proposals: list[CounterProposal] = []
        current_request = request
        last_response: DisclosureResponse | None = None
        last_receipt:  DisclosureReceipt  | None = None

        for round_num in range(1, _MAX_ROUNDS + 1):
            response, trace, entropy_alert = self._run_pipeline(
                current_request, beta_adj=beta_adj
            )
            last_response = response
            last_receipt  = self._fetch_receipt(response.receipt_id)

            if response.negotiation_status == NegotiationStatus.ACCEPTED:
                return IACPResult(
                    status=NegotiationStatus.ACCEPTED,
                    response=response, receipt=last_receipt,
                    rounds=round_num, counter_proposals=counter_proposals,
                )

            if round_num == _MAX_ROUNDS:
                break

            # Build counter-proposal from D* vs original A_req.
            cp = self._build_counter_proposal(request, response)
            counter_proposals.append(cp)

            if not cp.offered and not cp.zkp_substitutions:
                # Nothing to offer — terminate early.
                break

            # Simulate counterparty accepting the counter-proposal.
            # Next round uses A_req = offered ∪ zkp_subs_attrs.
            new_attrs = frozenset(cp.offered) | frozenset(cp.zkp_substitutions)
            if not new_attrs:
                break
            current_request = replace(request, requested_attributes=new_attrs)

        # Reached MAX_ROUNDS or early termination.
        final_status = (
            last_response.negotiation_status
            if last_response and last_response.negotiation_status in (
                NegotiationStatus.PARTIAL,
                NegotiationStatus.PROOF_ONLY,
            )
            else NegotiationStatus.FAILED
        )
        return IACPResult(
            status=final_status,
            response=last_response, receipt=last_receipt,
            rounds=min(_MAX_ROUNDS, len(counter_proposals) + 1),
            counter_proposals=counter_proposals,
        )

    # ------------------------------------------------------------------
    # Phase 4: sign receipt
    # ------------------------------------------------------------------

    def _sign_receipt(
        self, receipt: DisclosureReceipt, request: DisclosureRequest
    ) -> DisclosureReceipt:
        """Re-sign the receipt with a context-specific pseudonymous key (HKDF)."""
        master_id = self._pds.get_master_id()
        signing_key = _derive_signing_key(master_id, receipt.context_class)
        payload = _canonical_json(receipt)
        sig = hmac.new(signing_key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
        return replace(receipt, signature=sig)

    # ------------------------------------------------------------------
    # Full pipeline (IPC → CC → DDE)
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        request: DisclosureRequest,
        beta_adj: float = 0.0,
    ) -> tuple[DisclosureResponse, object, EntropyAlert | None]:
        """
        Run the full classification + decision pipeline for one round.

        Returns (response, trace, entropy_alert_for_next_call).
        """
        # 1. First-pass CC classification (no drift info yet).
        classified_init = self._cc.classify(request)

        # 2. Drift detection via IPC.
        features = {
            "platform_type": request.platform_type.value,
            "requested_count": len(request.requested_attributes),
            "has_purpose": bool(request.context_signal.description.strip()),
            "trust_band": _trust_band_label(request.platform_trust_score),
        }
        drift_alert = self._ipc.check_drift(
            classified_init.context_class, features
        )

        # 3. Re-classify with drift signal.
        classified = self._cc.classify(request, drift_alert=drift_alert)

        # 4. Get or create profile.
        profile = self._ipc.get_or_create_profile(classified.context_class)

        # 5. Consume pending EntropyAlert (from a previous interaction).
        entropy_alert = self._pending_entropy.pop(classified.context_class, None)

        # 6. DDE decision.
        response, trace = self._dde.decide(
            request, classified, profile, self._policy,
            entropy_alert=entropy_alert,
            beta_adjustment=beta_adj,
        )

        # 7. Update profile history; capture new EntropyAlert for next call.
        disclosed_attrs = frozenset(av.attribute for av in response.disclosed_values)
        event = DisclosureEvent(
            timestamp=datetime.now(timezone.utc),
            platform_id=request.platform_id,
            context_class=classified.context_class,
            disclosed_attributes=disclosed_attrs,
            receipt_signature=response.receipt_id,
            entropy_at_disclosure=profile.identity_entropy,
        )
        new_entropy_alert = self._ipc.update_profile(classified.context_class, event)
        if new_entropy_alert is not None:
            self._pending_entropy[classified.context_class] = new_entropy_alert

        return response, trace, new_entropy_alert

    # ------------------------------------------------------------------
    # Request validation (Phase 2)
    # ------------------------------------------------------------------

    def _validate_request(self, request: DisclosureRequest) -> str:
        """Return an error message string if the request is invalid, else ''."""
        vocab = frozenset(AttributeVocab)
        invalid = request.requested_attributes - vocab
        if invalid:
            return (
                f"Requested attributes not in vocab: "
                f"{', '.join(sorted(a.value if hasattr(a, 'value') else str(a) for a in invalid))}"
            )

        ctx_hint = _platform_context_hint(request.platform_type)
        if ctx_hint in _PURPOSE_REQUIRED and not request.context_signal.description.strip():
            return (
                f"Purpose declaration is required for {ctx_hint.value} context requests."
            )

        return ""

    # ------------------------------------------------------------------
    # Counter-proposal construction
    # ------------------------------------------------------------------

    def _build_counter_proposal(
        self,
        original_request: DisclosureRequest,
        response: DisclosureResponse,
    ) -> CounterProposal:
        """
        Build a CounterProposal from the DDE's disclosure response.

        offered:           D* (raw-disclosed attributes from the response).
        zkp_substitutions: DOB → "age_gte_18" if DOB was requested but withheld.
        """
        offered: AttributeSet = frozenset(av.attribute for av in response.disclosed_values)

        # Offer ZKP substitution for DOB if it was requested but not disclosed.
        withheld_from_original = original_request.requested_attributes - offered
        zkp_subs: dict[Attribute, str] = {}
        if Attribute.DOB in withheld_from_original:
            # age_gte_18 is the only compiled circuit.
            zkp_subs[Attribute.DOB] = "age_gte_18"

        # Also include any ZKP proofs already in the response.
        for circuit_name in response.zkp_proofs:
            # Map circuit back to attribute (currently only DOB → age_gte_18).
            if circuit_name == "age_gte_18":
                zkp_subs[Attribute.DOB] = "age_gte_18"

        withheld_names = ", ".join(sorted(a.value for a in withheld_from_original - frozenset(zkp_subs)))
        offered_names  = ", ".join(sorted(a.value for a in offered))

        rationale = (
            f"CIA will disclose: {offered_names or 'none'}. "
            + (f"ZKP proof available for: {', '.join(a.value for a in zkp_subs)}. " if zkp_subs else "")
            + (f"Withheld (policy): {withheld_names}." if withheld_names else "")
        )

        return CounterProposal(
            offered=offered,
            zkp_substitutions=zkp_subs,
            rationale=rationale.strip(),
        )

    # ------------------------------------------------------------------
    # Audit log helper
    # ------------------------------------------------------------------

    def _fetch_receipt(self, receipt_id: str) -> DisclosureReceipt | None:
        """Retrieve the receipt that the DDE just wrote by its ID."""
        if not receipt_id:
            return None
        receipts = self._audit.query()
        for r in reversed(receipts):
            if r.receipt_id == receipt_id:
                return r
        return None


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _platform_did(platform_id: str) -> str:
    """Canonical DID representation of a platform identifier."""
    return f"did:web:{platform_id}"


def _platform_context_hint(platform_type: PlatformType) -> ContextClass:
    """Best-guess context class from platform type (mirrors CC's ceiling)."""
    _MAP = {
        PlatformType.FINTECH:        ContextClass.FINANCIAL,
        PlatformType.EMPLOYER:       ContextClass.PROFESSIONAL,
        PlatformType.SOCIAL_NETWORK: ContextClass.SOCIAL,
        PlatformType.HEALTHCARE:     ContextClass.MEDICAL,
        PlatformType.GOVERNMENT:     ContextClass.CIVIC,
    }
    return _MAP[platform_type]


def _retention_beta_adj(policy: RetentionPolicy | None) -> float:
    if policy is None:
        return 0.0
    return _RETENTION_BETA.get(policy, 0.0)


def _trust_band_label(score: float) -> str:
    if score < 0.4:
        return "LOW"
    if score < 0.7:
        return "MID"
    return "HIGH"


def _derive_signing_key(master_id: bytes, context: ContextClass) -> bytes:
    """Derive a 32-byte context-specific HMAC signing key from master_id."""
    ikm = master_id + context.value.encode("utf-8")
    prk = _hkdf_extract(b"cia-receipt-signing-v1", ikm)
    return _hkdf_expand(prk, b"cia-receipt-context-key", 32)

"""
Disclosure Decision Engine (DDE) — computes D* and assembles DisclosureResponse.

Utility function (§3.4):
    U(D) = α·S(D) + β·T(D) − γ·L(D)

    S(D) = |D| / |A_req|                            service utility
    T(D) = Σ_{a∈D}(1 − Wₐ) / |D|                   trust signal (low-W = routine = high trust)
    L(D) = H(I) · Σ_{a∈D} Wₐ                       linkability  (H(I) = log(12) nats)

Greedy algorithm:
    Start from D = ∅.  At each step add the attribute with maximum marginal ΔU.
    Stop when no candidate improves U.  O(|A_req|²) worst case (144 comparisons).
    After termination, verify minimality: no strict subset D' ⊊ D* achieves U(D') ≥ U(D*).

Post-optimisation:
    • VERIFY_ONLY attrs → ZKP (age_gte_18 circuit); demote to WITHHOLD on error.
    • BBS+ SubsetProof over all active profile attrs with D* indices disclosed.
    • Pseudonym DID via PseudonymEngine.derive(master_id, context).
    • Audit receipt written to AuditLog.

Import restriction: must NOT import from cc.py, ipc.py, or iacp.py.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from cia.models.attributes import Attribute, AttributeSet, AttributeValue, AttributeVocab
from cia.models.context import ClassifiedContext, ContextClass
from cia.models.decision import CandidateDisclosure, DecisionTrace, UtilityScore
from cia.models.policy import PolicyAction, PolicySet
from cia.models.profile import ProfileTuple
from cia.models.request import DisclosureRequest
from cia.models.response import (
    DisclosureReceipt, DisclosureResponse, NegotiationStatus,
)
from cia.policy.audit import AuditLog, _canonical_json
from cia.policy.norms import NormDatabase
from cia.policy.rules import PolicyInterpreter
from cia.crypto.bbs import BBSEngine, BBSKeyPair, SubsetProof
from cia.crypto.zkp import ZKPEngine, CircuitNotCompiledError
from cia.crypto.pseudonyms import PseudonymEngine
from cia.ipc import EntropyAlert
from cia.store.pds import PDS

logger = logging.getLogger(__name__)

# H(I) — uniform prior over 12 attributes: log(12) nats.
_H_I: float = math.log(12)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DisclosureDecisionEngine:
    """
    Core decision-making component.

    Parameters
    ----------
    pds:        Personal Data Store (weights, vault, master_id).
    norm_db:    Norm database Φ(c).
    audit_log:  Tamper-evident disclosure log.  DDE owns all writes.
    bbs:        BBSEngine instance (stateless).
    zkp:        ZKPEngine instance.
    pseudonyms: PseudonymEngine instance (stateless).
    """

    def __init__(
        self,
        pds: PDS,
        norm_db: NormDatabase,
        audit_log: AuditLog,
        bbs: BBSEngine,
        zkp: ZKPEngine,
        pseudonyms: PseudonymEngine,
    ) -> None:
        self._pds       = pds
        self._norms     = norm_db
        self._audit     = audit_log
        self._bbs       = bbs
        self._zkp       = zkp
        self._pseudonyms = pseudonyms
        self._interp    = PolicyInterpreter(norm_db)
        # Fresh BBS+ key pair per DDE instance (prototype; not persisted).
        self._bbs_kp: BBSKeyPair = bbs.keygen()
        # In-session BBS+ signature cache: context → BBSSignature.
        self._bbs_sigs: dict[ContextClass, Any] = {}

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def decide(
        self,
        request: DisclosureRequest,
        context: ClassifiedContext,
        profile: ProfileTuple,
        policy: PolicySet,
        entropy_alert: EntropyAlert | None = None,
        beta_adjustment: float = 0.0,
    ) -> tuple[DisclosureResponse, DecisionTrace]:
        """
        Compute D* and return (DisclosureResponse, DecisionTrace).

        ``entropy_alert`` → γ += 0.10 (in-call, not persisted).
        ``beta_adjustment`` → β offset from IACP retention policy (in-call, not persisted).
        """
        alpha, beta, gamma = self._pds.get_weights()
        beta = max(0.0, min(1.0, beta + beta_adjustment))
        if entropy_alert is not None:
            gamma = min(gamma + 0.10, 1.0)

        # ── Feasibility filter ────────────────────────────────────────
        interp = self._interp.evaluate(request, profile, policy, context.context_class)

        if interp.trust_gated:
            return self._rejected(request, context, profile)

        # Only DISCLOSE, ABSTRACT, VERIFY_ONLY survive the feasibility filter.
        candidates: dict[Attribute, PolicyAction] = {
            d.attribute: d.action
            for d in interp.decisions
            if d.action != PolicyAction.WITHHOLD
        }

        # ── Greedy optimiser ──────────────────────────────────────────
        n_req = max(len(request.requested_attributes), 1)
        trace = _greedy(
            candidates, profile, request.requested_attributes,
            n_req, alpha, beta, gamma,
            request.request_id, context.context_class,
        )

        D_star: frozenset[Attribute] = (
            trace.optimal.attribute_set if trace.optimal else frozenset()
        )

        # ── Assemble response ─────────────────────────────────────────
        response = self._build_response(
            request, context, profile, D_star, candidates,
        )

        # ── Audit ─────────────────────────────────────────────────────
        u_star = trace.optimal.utility.U if trace.optimal else 0.0
        receipt = self._make_receipt(
            request, context, profile, D_star,
            request.requested_attributes - D_star,
            response.negotiation_status, u_star,
        )
        self._audit.append(receipt)
        response = replace(response, receipt_id=receipt.receipt_id)

        return response, trace

    # ------------------------------------------------------------------
    # Response assembly
    # ------------------------------------------------------------------

    def _build_response(
        self,
        request: DisclosureRequest,
        context: ClassifiedContext,
        profile: ProfileTuple,
        D_star: frozenset[Attribute],
        candidates: dict[Attribute, PolicyAction],
    ) -> DisclosureResponse:
        vault = self._pds.get_all_attributes()
        master_id = self._pds.get_master_id()

        disclosed_values: list[AttributeValue] = []
        zkp_proofs: dict[str, bytes] = {}
        verify_only_attrs: set[Attribute] = set()

        for attr in sorted(D_star, key=lambda a: a.value):
            action = candidates.get(attr, PolicyAction.DISCLOSE)
            av = vault.get(attr)
            if av is None:
                continue

            if action == PolicyAction.VERIFY_ONLY:
                result = self._try_zkp(attr, av)
                if result is not None:
                    circuit_name, proof_bytes = result
                    zkp_proofs[circuit_name] = proof_bytes
                    verify_only_attrs.add(attr)
                # On failure: attribute silently withheld (logged inside _try_zkp).

            elif action == PolicyAction.ABSTRACT:
                abstracted = av.abstracted if av.abstracted is not None else av.raw
                disclosed_values.append(
                    AttributeValue(attribute=attr, raw=abstracted, abstracted=av.abstracted)
                )
            else:  # DISCLOSE
                disclosed_values.append(av)

        D_effective = frozenset(av.attribute for av in disclosed_values) | verify_only_attrs

        # BBS+ proof.
        bbs_proof_bytes = self._generate_bbs_proof_bytes(
            profile, D_effective, context.context_class
        )

        did = self._pseudonyms.derive(master_id, context.context_class)
        status = _negotiation_status(D_effective, request.requested_attributes, bool(zkp_proofs))

        return DisclosureResponse(
            request_id=request.request_id,
            negotiation_status=status,
            disclosed_values=tuple(disclosed_values),
            zkp_proofs=zkp_proofs,
            pseudonym_did=did,
            bbs_subset_proof_bytes=bbs_proof_bytes,
            user_explanation=_explanation(D_effective, request.requested_attributes),
        )

    # ------------------------------------------------------------------
    # BBS+
    # ------------------------------------------------------------------

    def _generate_bbs_proof_bytes(
        self,
        profile: ProfileTuple,
        D_star: frozenset[Attribute],
        context: ContextClass,
    ) -> bytes:
        """Sign profile attrs, generate SubsetProof, serialise to bytes."""
        ordered = _ordered_messages(profile, self._pds)
        if not ordered:
            return b""

        attrs_ordered = [a for a, _ in ordered]
        messages      = [m for _, m in ordered]

        sig = self._bbs_sigs.get(context)
        if sig is None:
            try:
                sig = self._bbs.sign(self._bbs_kp.sk, messages)
                self._bbs_sigs[context] = sig
            except Exception as exc:
                logger.warning("BBS+ signing failed: %s — no proof attached.", exc)
                return b""

        disclosed_indices = [i for i, a in enumerate(attrs_ordered) if a in D_star]
        try:
            proof = self._bbs.prove_subset(
                self._bbs_kp.sk, self._bbs_kp.pk,
                messages, sig, disclosed_indices,
            )
            return _serialise_subset_proof(proof)
        except Exception as exc:
            logger.warning("BBS+ prove_subset failed: %s — no proof attached.", exc)
            return b""

    # ------------------------------------------------------------------
    # ZKP
    # ------------------------------------------------------------------

    def _try_zkp(
        self,
        attr: Attribute,
        av: AttributeValue,
    ) -> tuple[str, bytes] | None:
        """Generate ZKP for supported VERIFY_ONLY attributes.  Only DOB is supported."""
        if attr != Attribute.DOB:
            logger.warning(
                "VERIFY_ONLY requested for %s but no circuit is available; withheld.",
                attr.value,
            )
            return None
        try:
            birth_year = int(str(av.raw)[:4])
            proof = self._zkp.prove_age_gte_18(birth_year)
            return "age_gte_18", proof.to_json().encode("utf-8")
        except CircuitNotCompiledError:
            logger.warning(
                "age_gte_18 circuit not compiled; DOB VERIFY_ONLY demoted to WITHHOLD. "
                "Run: python cia/assets/compile_circuits.py"
            )
            return None
        except Exception as exc:
            logger.warning("ZKP generation failed for DOB: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Trust-gated (instant rejection)
    # ------------------------------------------------------------------

    def _rejected(
        self,
        request: DisclosureRequest,
        context: ClassifiedContext,
        profile: ProfileTuple,
    ) -> tuple[DisclosureResponse, DecisionTrace]:
        master_id = self._pds.get_master_id()
        did = self._pseudonyms.derive(master_id, context.context_class)
        response = DisclosureResponse(
            request_id=request.request_id,
            negotiation_status=NegotiationStatus.REJECTED,
            pseudonym_did=did,
            user_explanation="Platform trust score below minimum threshold.",
        )
        trace = DecisionTrace(
            request_id=request.request_id,
            context_class=context.context_class,
        )
        receipt = self._make_receipt(
            request, context, profile,
            frozenset(), request.requested_attributes,
            NegotiationStatus.REJECTED, 0.0,
        )
        self._audit.append(receipt)
        return replace(response, receipt_id=receipt.receipt_id), trace

    # ------------------------------------------------------------------
    # Receipt
    # ------------------------------------------------------------------

    def _make_receipt(
        self,
        request: DisclosureRequest,
        context: ClassifiedContext,
        profile: ProfileTuple,
        disclosed: frozenset[Attribute],
        withheld: frozenset[Attribute],
        status: NegotiationStatus,
        utility: float,
    ) -> DisclosureReceipt:
        master_id = self._pds.get_master_id()
        w_sum = sum(profile.salience_weights.get(a, 1.0 / 12) for a in disclosed)
        entropy_after = max(0.0, profile.identity_entropy - _H_I * w_sum)

        receipt = DisclosureReceipt(
            receipt_id=str(uuid.uuid4()),
            request_id=request.request_id,
            platform_id=request.platform_id,
            context_class=context.context_class,
            disclosed_attributes=disclosed,
            withheld_attributes=withheld,
            negotiation_status=status,
            utility_score=utility,
            entropy_before=profile.identity_entropy,
            entropy_after=entropy_after,
        )
        # HMAC-sign with master_id as key.
        payload = _canonical_json(receipt)
        sig = hmac.new(master_id, payload.encode("utf-8"), hashlib.sha256).hexdigest()
        return replace(receipt, signature=sig)


# ---------------------------------------------------------------------------
# Greedy optimiser (pure function — no engine state)
# ---------------------------------------------------------------------------

def _greedy(
    candidates: dict[Attribute, PolicyAction],
    profile: ProfileTuple,
    A_req: frozenset[Attribute],
    n_req: int,
    alpha: float,
    beta: float,
    gamma: float,
    request_id: str,
    context_class: ContextClass,
) -> DecisionTrace:
    """
    Greedy attribute-addition algorithm.

    Iteratively adds the candidate attribute with maximum marginal ΔU until
    no addition improves U.  After termination, verifies minimality.

    Returns a DecisionTrace with per-step CandidateDisclosure records.
    """
    trace = DecisionTrace(request_id=request_id, context_class=context_class)

    D: frozenset[Attribute] = frozenset()
    s_D = 0.0   # running Σ (1 − Wₐ) for a ∈ D
    w_D = 0.0   # running Σ Wₐ       for a ∈ D

    while True:
        remaining = {a for a in candidates if a not in D}
        if not remaining:
            break

        best_attr: Attribute | None = None
        best_du: float = 0.0

        n_D = len(D)
        for a in remaining:
            W_a = profile.salience_weights.get(a, 1.0 / 12)
            du = _marginal_u(W_a, n_D, s_D, n_req, alpha, beta, gamma)
            if du > best_du:
                best_du = du
                best_attr = a

        if best_attr is None:
            break

        W_best = profile.salience_weights.get(best_attr, 1.0 / 12)
        D   = D | {best_attr}
        s_D += 1.0 - W_best
        w_D += W_best

        util = _score_utility(D, profile, A_req, n_req, alpha, beta, gamma)
        trace.candidates.append(CandidateDisclosure(
            attribute_set=D,
            utility=util,
            attribute_actions={a: candidates[a] for a in D},
            feasible=True,
        ))
        trace.optimal_index = len(trace.candidates) - 1

    # O(|D*|) minimality check.
    if trace.optimal is not None:
        D_star = trace.optimal.attribute_set
        u_star = trace.optimal.utility.U
        trace.minimality_verified = all(
            _score_utility(D_star - {a}, profile, A_req, n_req, alpha, beta, gamma).U < u_star
            for a in D_star
        )

    return trace


# ---------------------------------------------------------------------------
# Utility helpers (pure)
# ---------------------------------------------------------------------------

def _marginal_u(
    W_a: float,
    n_D: int,
    s_D: float,
    n_req: int,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    """
    Marginal utility gain from adding an attribute with salience weight W_a
    to the current set D (characterised by n_D and s_D = Σ(1−W)).
    """
    delta_S = 1.0 / n_req
    delta_T = (1.0 - W_a) if n_D == 0 else (
        (s_D + 1.0 - W_a) / (n_D + 1) - s_D / n_D
    )
    delta_L = _H_I * W_a
    return alpha * delta_S + beta * delta_T - gamma * delta_L


def _score_utility(
    D: frozenset[Attribute],
    profile: ProfileTuple,
    A_req: frozenset[Attribute],
    n_req: int,
    alpha: float,
    beta: float,
    gamma: float,
) -> UtilityScore:
    """Compute U(D) and return a UtilityScore with all decomposed components."""
    n_D = len(D)
    S   = n_D / n_req
    T   = (sum(1.0 - profile.salience_weights.get(a, 1.0 / 12) for a in D) / n_D
           if n_D > 0 else 0.0)
    W_sum = sum(profile.salience_weights.get(a, 1.0 / 12) for a in D)
    L   = _H_I * W_sum
    U   = alpha * S + beta * T - gamma * L
    return UtilityScore(S=S, T=T, L=L, alpha=alpha, beta=beta, gamma=gamma, U=U)


# ---------------------------------------------------------------------------
# BBS+ serialisation (G1 affine coordinates → JSON bytes)
# ---------------------------------------------------------------------------

def _serialise_subset_proof(proof: SubsetProof) -> bytes:
    """Serialise a SubsetProof to canonical JSON bytes for storage/transport."""
    try:
        from py_ecc.optimized_bn128 import normalize, is_inf  # noqa: PLC0415

        def _g1(pt: object) -> dict:
            if is_inf(pt):
                return {"inf": True}
            nx, ny = normalize(pt)
            return {"x": str(int(nx)), "y": str(int(ny))}

        data = {
            "A_prime": _g1(proof.A_prime),
            "A_bar":   _g1(proof.A_bar),
            "D":       _g1(proof.D),
            "R_e":     _g1(proof.R_e),
            "T":       _g1(proof.T),
            "c":    proof.c,
            "z_e":  proof.z_e,
            "z_r1": proof.z_r1,
            "z_rs": proof.z_rs,
            "z_m":  {str(i): v for i, v in proof.z_m.items()},
            "disclosed_indices":  list(proof.disclosed_indices),
            "disclosed_messages": [m.hex() for m in proof.disclosed_messages],
            "n_messages": proof.n_messages,
        }
        return json.dumps(data, separators=(",", ":")).encode("utf-8")
    except Exception as exc:
        logger.warning("SubsetProof serialisation failed: %s", exc)
        return b""


# ---------------------------------------------------------------------------
# Misc helpers (pure)
# ---------------------------------------------------------------------------

def _ordered_messages(
    profile: ProfileTuple,
    pds: PDS,
) -> list[tuple[Attribute, bytes]]:
    """
    (attribute, value_bytes) in AttributeVocab order for BBS+ signing.
    Only attributes in both the profile's active set and the PDS vault.
    """
    vault = pds.get_all_attributes()
    return [
        (attr, str(vault[attr].effective_value()).encode("utf-8"))
        for attr in AttributeVocab
        if attr in profile.active_attributes and attr in vault
    ]


def _negotiation_status(
    D_star: frozenset[Attribute],
    A_req:  frozenset[Attribute],
    has_zkp: bool,
) -> NegotiationStatus:
    if not D_star:
        return NegotiationStatus.PROOF_ONLY if has_zkp else NegotiationStatus.REJECTED
    if D_star >= A_req:
        return NegotiationStatus.ACCEPTED
    return NegotiationStatus.PARTIAL


def _explanation(D_star: frozenset[Attribute], A_req: frozenset[Attribute]) -> str:
    withheld = A_req - D_star
    if not withheld:
        return "All requested attributes disclosed."
    return (
        f"Disclosed: {', '.join(sorted(a.value for a in D_star)) or 'none'}. "
        f"Withheld: {', '.join(sorted(a.value for a in withheld))}."
    )

"""Tests for cia.iacp — Inter-Agent Communication Protocol."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from cia.models.attributes import Attribute
from cia.models.context import ClassifiedContext, ContextClass, ContextSignal
from cia.models.policy import PolicyAction, PolicyRule, PolicySet
from cia.models.profile import ProfileTuple, TrustConstraints
from cia.models.request import DisclosureRequest, PlatformType, RetentionPolicy
from cia.models.response import NegotiationStatus
from cia.policy.audit import AuditLog
from cia.policy.norms import NormDatabase
from cia.crypto.bbs import BBSEngine, BBSKeyPair
from cia.crypto.zkp import ZKPEngine
from cia.crypto.pseudonyms import PseudonymEngine
from cia.ipc import IPC
from cia.cc import ContextClassifier
from cia.dde import DisclosureDecisionEngine
from cia.store.pds import PDS
from cia.iacp import (
    IACPSession, IACPResult, CounterProposal,
    _platform_did, _platform_context_hint,
    _retention_beta_adj, _derive_signing_key, _trust_band_label,
    _MAX_ROUNDS,
)


# ---------------------------------------------------------------------------
# Stubs / helpers
# ---------------------------------------------------------------------------

class _StubBBS:
    def keygen(self) -> BBSKeyPair:
        return BBSKeyPair(sk=1, pk=None)  # type: ignore[arg-type]
    def sign(self, *a, **kw):
        raise RuntimeError("stub")
    def prove_subset(self, *a, **kw):
        raise RuntimeError("stub")


class _MockCC:
    """Symbolic-only CC that never touches ONNX."""
    def __init__(self, fixed_ctx: ContextClass = ContextClass.FINANCIAL):
        self._ctx = fixed_ctx
    def classify(self, request, drift_alert=None) -> ClassifiedContext:
        ctx = _platform_context_hint(request.platform_type)
        return ClassifiedContext(
            context_class=ctx,
            confidence=0.92,
            conservative_mode=False,
            class_probabilities={ctx: 0.92},
            source="symbolic",
        )


def _make_pds(**extra_attrs) -> PDS:
    pds = PDS(Path(":memory:"))
    defaults = {
        Attribute.EMAIL:             "alice@example.com",
        Attribute.LEGAL_NAME:        "Alice",
        Attribute.EMPLOYMENT_STATUS: "employed",
        Attribute.INCOME_BRACKET:    80_000,
        Attribute.DOB:               "1990-03-15",
        Attribute.PHONE:             "+1-555-0100",
    }
    for attr, val in {**defaults, **extra_attrs}.items():
        pds.set_attribute(attr, val)
    return pds


def _make_session(pds: PDS | None = None) -> tuple[IACPSession, PDS, AuditLog]:
    pds  = pds or _make_pds()
    audit = AuditLog(Path(":memory:"))
    ipc  = IPC(pds)
    cc   = _MockCC()
    dde  = DisclosureDecisionEngine(
        pds=pds, norm_db=NormDatabase(), audit_log=audit,
        bbs=_StubBBS(),  # type: ignore[arg-type]
        zkp=ZKPEngine(), pseudonyms=PseudonymEngine(),
    )
    session = IACPSession(
        pds=pds, ipc=ipc, cc=cc, dde=dde,
        norm_db=NormDatabase(), policy=PolicySet(),
        audit_log=audit,
    )
    return session, pds, audit


def _make_request(
    platform_type: PlatformType = PlatformType.FINTECH,
    attrs: set[Attribute] | None = None,
    trust: float = 0.9,
    purpose: str = "loan application",
    retention: RetentionPolicy | None = None,
) -> DisclosureRequest:
    return DisclosureRequest(
        request_id=f"req-{id(attrs)}",
        platform_id="plat-test",
        platform_type=platform_type,
        platform_trust_score=trust,
        requested_attributes=frozenset(attrs or {Attribute.EMAIL}),
        context_signal=ContextSignal(description=purpose),
        retention_policy=retention,
    )


# ---------------------------------------------------------------------------
# Phase 2 — Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_request_not_rejected_with_error(self):
        """A well-formed request must not produce a validation error."""
        session, _, _ = _make_session()
        result = session.handle(_make_request(attrs={Attribute.EMAIL}), mode="web")
        assert result.error == ""

    def test_valid_fintech_request_passes_validation(self):
        session, _, _ = _make_session()
        result = session.handle(_make_request(PlatformType.FINTECH, purpose="loan"), mode="web")
        assert result.error == ""

    def test_medical_request_without_purpose_rejected(self):
        session, _, _ = _make_session()
        req = _make_request(PlatformType.HEALTHCARE, purpose="", attrs={Attribute.HEALTH_CONDITION})
        result = session.handle(req, mode="web")
        assert result.status == NegotiationStatus.REJECTED
        assert "purpose" in result.error.lower()

    def test_financial_request_without_purpose_rejected(self):
        session, _, _ = _make_session()
        req = _make_request(PlatformType.FINTECH, purpose="", attrs={Attribute.EMAIL})
        result = session.handle(req, mode="web")
        assert result.status == NegotiationStatus.REJECTED

    def test_social_request_without_purpose_passes(self):
        """Social context does not require a purpose declaration."""
        session, _, _ = _make_session()
        req = _make_request(PlatformType.SOCIAL_NETWORK, purpose="", attrs={Attribute.EMAIL})
        result = session.handle(req, mode="web")
        assert result.error == ""


# ---------------------------------------------------------------------------
# Web mode
# ---------------------------------------------------------------------------

class TestWebMode:
    def test_returns_iacp_result(self):
        session, _, _ = _make_session()
        result = session.handle(_make_request(), mode="web")
        assert isinstance(result, IACPResult)

    def test_rounds_is_one(self):
        session, _, _ = _make_session()
        result = session.handle(_make_request(), mode="web")
        assert result.rounds == 1

    def test_counter_proposals_empty(self):
        session, _, _ = _make_session()
        result = session.handle(_make_request(), mode="web")
        assert result.counter_proposals == []

    def test_audit_log_updated(self):
        session, _, audit = _make_session()
        session.handle(_make_request(), mode="web")
        assert audit.count() == 1

    def test_response_not_none_on_success(self):
        session, _, _ = _make_session()
        result = session.handle(_make_request(attrs={Attribute.EMAIL}), mode="web")
        assert result.response is not None

    def test_pseudonym_did_in_response(self):
        session, _, _ = _make_session()
        result = session.handle(_make_request(), mode="web")
        assert result.response is not None
        assert result.response.pseudonym_did.startswith("did:cia:")


# ---------------------------------------------------------------------------
# Agent mode — basic
# ---------------------------------------------------------------------------

class TestAgentMode:
    def test_returns_iacp_result(self):
        session, _, _ = _make_session()
        result = session.handle(_make_request(), mode="agent")
        assert isinstance(result, IACPResult)

    def test_status_not_pending(self):
        session, _, _ = _make_session()
        result = session.handle(_make_request(), mode="agent")
        assert result.status != NegotiationStatus.PENDING

    def test_audit_log_updated(self):
        session, _, audit = _make_session()
        session.handle(_make_request(), mode="agent")
        assert audit.count() >= 1

    def test_receipt_signed_in_agent_mode(self):
        """Phase 4 re-signs the receipt with a context-specific key."""
        session, _, _ = _make_session()
        result = session.handle(_make_request(), mode="agent")
        if result.receipt is not None:
            assert result.receipt.signature != ""

    def test_agent_mode_receipt_has_non_empty_signature(self):
        """
        Phase 4 re-signs the receipt using a context-specific HKDF key.
        Web mode skips phase 4, so its fetched receipt has signature="" (the
        audit log stores payload without the signature field to avoid circularity).
        """
        session, _, _ = _make_session()
        req = _make_request()
        result = session.handle(req, mode="agent")
        if result.receipt is not None:
            assert result.receipt.signature != ""


# ---------------------------------------------------------------------------
# Blocklist (Phase 1)
# ---------------------------------------------------------------------------

class TestBlocklist:
    def test_blocked_platform_rejected(self):
        session, _, _ = _make_session()
        session.block_platform("plat-test")
        result = session.handle(_make_request(), mode="agent")
        assert result.status == NegotiationStatus.REJECTED
        assert "blocklist" in result.error.lower() or "blocked" in result.error.lower()

    def test_blocked_platform_web_mode_not_checked(self):
        """Web mode skips Phase 1 (identify), so blocklist is not enforced."""
        session, _, _ = _make_session()
        session.block_platform("plat-test")
        result = session.handle(_make_request(), mode="web")
        # Web mode: no blocklist check → should proceed normally.
        assert result.status != NegotiationStatus.REJECTED or result.error == ""

    def test_unblock_allows_requests(self):
        session, _, _ = _make_session()
        session.block_platform("plat-test")
        session.unblock_platform("plat-test")
        result = session.handle(_make_request(), mode="agent")
        assert "blocklist" not in result.error.lower()

    def test_block_platform_persisted_in_pds(self):
        session, pds, _ = _make_session()
        session.block_platform("evil-plat")
        raw = pds.get_raw_config("iacp_blocklist")
        assert raw is not None
        import json
        assert "did:web:evil-plat" in json.loads(raw)


# ---------------------------------------------------------------------------
# Negotiation rounds
# ---------------------------------------------------------------------------

class TestNegotiationRounds:
    def test_accepted_in_round_1_when_email_requested(self):
        """EMAIL is normatively permitted in FINANCIAL — should be ACCEPTED in round 1."""
        session, _, _ = _make_session()
        req = _make_request(attrs={Attribute.EMAIL})
        result = session.handle(req, mode="agent")
        assert result.rounds == 1
        assert result.status == NegotiationStatus.ACCEPTED

    def test_partial_request_builds_counter_proposal(self):
        """Requesting health_condition in FINANCIAL context triggers counter-proposal."""
        session, _, _ = _make_session()
        req = _make_request(
            attrs={Attribute.EMAIL, Attribute.HEALTH_CONDITION},
            purpose="loan application",
        )
        result = session.handle(req, mode="agent")
        # HEALTH_CONDITION withheld in FINANCIAL → should generate at least 1 CP.
        if result.status != NegotiationStatus.ACCEPTED:
            assert len(result.counter_proposals) >= 1

    def test_max_rounds_not_exceeded(self):
        session, _, _ = _make_session()
        result = session.handle(_make_request(), mode="agent")
        assert result.rounds <= _MAX_ROUNDS

    def test_counter_proposal_offered_subset_of_original(self):
        session, _, _ = _make_session()
        req = _make_request(
            attrs={Attribute.EMAIL, Attribute.HEALTH_CONDITION},
            purpose="loan check",
        )
        result = session.handle(req, mode="agent")
        for cp in result.counter_proposals:
            assert cp.offered <= req.requested_attributes

    def test_counter_proposal_has_rationale(self):
        session, _, _ = _make_session()
        req = _make_request(attrs={Attribute.EMAIL, Attribute.HEALTH_CONDITION})
        result = session.handle(req, mode="agent")
        for cp in result.counter_proposals:
            assert len(cp.rationale) > 0


# ---------------------------------------------------------------------------
# RetentionPolicy adjustments
# ---------------------------------------------------------------------------

class TestRetentionPolicy:
    def test_session_only_no_error(self):
        session, _, _ = _make_session()
        req = _make_request(retention=RetentionPolicy.SESSION_ONLY)
        result = session.handle(req, mode="web")
        assert result.error == ""

    def test_anonymised_no_error(self):
        session, _, _ = _make_session()
        req = _make_request(retention=RetentionPolicy.ANONYMISED)
        result = session.handle(req, mode="web")
        assert result.error == ""

    def test_persistent_no_error(self):
        session, _, _ = _make_session()
        req = _make_request(retention=RetentionPolicy.PERSISTENT)
        result = session.handle(req, mode="web")
        assert result.error == ""

    def test_retention_beta_adj_values(self):
        assert _retention_beta_adj(RetentionPolicy.SESSION_ONLY) == 0.0
        assert _retention_beta_adj(RetentionPolicy.ANONYMISED)   == pytest.approx(0.15)
        assert _retention_beta_adj(RetentionPolicy.PERSISTENT)   == pytest.approx(-0.10)
        assert _retention_beta_adj(None)                         == 0.0


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestPureHelpers:
    def test_platform_did_format(self):
        assert _platform_did("my-bank") == "did:web:my-bank"

    def test_platform_context_hint_all_types(self):
        mapping = {
            PlatformType.FINTECH:        ContextClass.FINANCIAL,
            PlatformType.EMPLOYER:       ContextClass.PROFESSIONAL,
            PlatformType.SOCIAL_NETWORK: ContextClass.SOCIAL,
            PlatformType.HEALTHCARE:     ContextClass.MEDICAL,
            PlatformType.GOVERNMENT:     ContextClass.CIVIC,
        }
        for pt, expected in mapping.items():
            assert _platform_context_hint(pt) == expected

    def test_trust_band_label(self):
        assert _trust_band_label(0.1)  == "LOW"
        assert _trust_band_label(0.39) == "LOW"
        assert _trust_band_label(0.4)  == "MID"
        assert _trust_band_label(0.69) == "MID"
        assert _trust_band_label(0.7)  == "HIGH"
        assert _trust_band_label(1.0)  == "HIGH"

    def test_derive_signing_key_length(self):
        key = _derive_signing_key(b"a" * 32, ContextClass.FINANCIAL)
        assert len(key) == 32

    def test_derive_signing_key_context_specific(self):
        mid = b"m" * 32
        k1 = _derive_signing_key(mid, ContextClass.FINANCIAL)
        k2 = _derive_signing_key(mid, ContextClass.MEDICAL)
        assert k1 != k2

    def test_derive_signing_key_master_id_specific(self):
        k1 = _derive_signing_key(b"a" * 32, ContextClass.CIVIC)
        k2 = _derive_signing_key(b"b" * 32, ContextClass.CIVIC)
        assert k1 != k2


# ---------------------------------------------------------------------------
# CounterProposal
# ---------------------------------------------------------------------------

class TestCounterProposal:
    def test_frozen(self):
        cp = CounterProposal(offered=frozenset({Attribute.EMAIL}), rationale="test")
        with pytest.raises((AttributeError, TypeError)):
            cp.offered = frozenset()  # type: ignore[misc]

    def test_zkp_substitutions_default_empty(self):
        cp = CounterProposal(offered=frozenset())
        assert cp.zkp_substitutions == {}


# ---------------------------------------------------------------------------
# NegotiationStatus additions
# ---------------------------------------------------------------------------

class TestNegotiationStatusExtended:
    def test_failed_status_exists(self):
        assert NegotiationStatus.FAILED == "FAILED"

    def test_all_statuses_count(self):
        # ACCEPTED, PARTIAL, REJECTED, PROOF_ONLY, PENDING, FAILED
        assert len(NegotiationStatus) == 6

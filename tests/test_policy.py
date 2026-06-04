"""Tests for cia.policy — norms, rules, audit."""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cia.models.attributes import Attribute
from cia.models.context import ContextClass
from cia.models.policy import PolicyAction, PolicyRule, PolicySet
from cia.models.profile import ProfileTuple, TrustConstraints
from cia.models.request import DisclosureRequest, PlatformType
from cia.models.context import ContextSignal
from cia.models.response import DisclosureReceipt, NegotiationStatus
from cia.policy.norms import NormDatabase, NormEntry, RecipientRole
from cia.policy.rules import PolicyInterpreter, InterpretationResult
from cia.policy.audit import AuditLog, _canonical_json, _compute_hmac


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile(
    attrs=None,
    min_trust=0.0,
    hard_blacklist=None,
    proof_required=None,
) -> ProfileTuple:
    if attrs is None:
        attrs = frozenset(Attribute)
    tc = TrustConstraints(
        min_trust_score=min_trust,
        hard_blacklist=frozenset(hard_blacklist or []),
        proof_required=frozenset(proof_required or []),
    )
    weights = {a: 1 / len(attrs) for a in attrs}
    return ProfileTuple(
        profile_id="test_profile",
        active_attributes=attrs,
        salience_weights=weights,
        trust_constraints=tc,
    )


def _make_request(
    platform_type=PlatformType.FINTECH,
    trust_score=0.9,
    requested=None,
) -> DisclosureRequest:
    if requested is None:
        requested = frozenset({Attribute.EMAIL, Attribute.INCOME_BRACKET})
    return DisclosureRequest(
        request_id="req-1",
        platform_id="plat-1",
        platform_type=platform_type,
        platform_trust_score=trust_score,
        requested_attributes=requested,
        context_signal=ContextSignal(description="test"),
    )


def _make_receipt(
    receipt_id="rec-1",
    request_id="req-1",
    platform_id="plat-1",
    context=ContextClass.FINANCIAL,
    disclosed=None,
    withheld=None,
    status=NegotiationStatus.ACCEPTED,
) -> DisclosureReceipt:
    return DisclosureReceipt(
        receipt_id=receipt_id,
        request_id=request_id,
        platform_id=platform_id,
        context_class=context,
        disclosed_attributes=frozenset(disclosed or [Attribute.EMAIL]),
        withheld_attributes=frozenset(withheld or []),
        negotiation_status=status,
        utility_score=0.7,
        entropy_before=3.0,
        entropy_after=2.5,
        timestamp=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# NormDatabase
# ---------------------------------------------------------------------------

class TestNormDatabase:
    def test_seed_has_entries(self):
        db = NormDatabase()
        assert sum(1 for _ in db.all_entries()) > 0

    def test_financial_email_permitted_for_institution(self):
        db = NormDatabase()
        assert db.permitted(
            ContextClass.FINANCIAL, Attribute.EMAIL,
            RecipientRole.FINANCIAL_INSTITUTION
        )

    def test_health_condition_not_permitted_in_social(self):
        db = NormDatabase()
        assert not db.permitted(
            ContextClass.SOCIAL, Attribute.HEALTH_CONDITION,
            RecipientRole.SOCIAL_PEER
        )

    def test_income_bracket_not_permitted_for_employer(self):
        db = NormDatabase()
        assert not db.permitted(
            ContextClass.PROFESSIONAL, Attribute.INCOME_BRACKET,
            RecipientRole.EMPLOYER
        )

    def test_gaming_history_permitted_in_social(self):
        db = NormDatabase()
        assert db.permitted(
            ContextClass.SOCIAL, Attribute.GAMING_HISTORY,
            RecipientRole.SOCIAL_PEER
        )

    def test_dob_proof_required_in_financial(self):
        db = NormDatabase()
        assert db.proof_required(
            ContextClass.FINANCIAL, Attribute.DOB,
            RecipientRole.FINANCIAL_INSTITUTION
        )

    def test_email_proof_not_required_in_financial(self):
        db = NormDatabase()
        assert not db.proof_required(
            ContextClass.FINANCIAL, Attribute.EMAIL,
            RecipientRole.FINANCIAL_INSTITUTION
        )

    def test_permitted_attributes_financial(self):
        db = NormDatabase()
        attrs = db.permitted_attributes(
            ContextClass.FINANCIAL, RecipientRole.FINANCIAL_INSTITUTION
        )
        assert Attribute.EMAIL in attrs
        assert Attribute.INCOME_BRACKET in attrs
        assert Attribute.HEALTH_CONDITION not in attrs

    def test_unknown_role_always_blocked(self):
        db = NormDatabase()
        for attr in Attribute:
            for ctx in ContextClass:
                assert not db.permitted(ctx, attr, RecipientRole.UNKNOWN)

    def test_roundtrip_json(self, tmp_path):
        db = NormDatabase()
        p = tmp_path / "norm_db.json"
        db.to_json(p)
        db2 = NormDatabase.from_json(p)
        original = list(db.all_entries())
        restored = list(db2.all_entries())
        assert len(original) == len(restored)
        assert set(original) == set(restored)

    def test_all_5_contexts_covered(self):
        db = NormDatabase()
        covered = {e.context for e in db.all_entries()}
        assert covered == set(ContextClass)

    def test_entries_for_context(self):
        db = NormDatabase()
        entries = db.entries_for_context(ContextClass.MEDICAL)
        assert all(e.context == ContextClass.MEDICAL for e in entries)
        assert len(entries) > 0


# ---------------------------------------------------------------------------
# PolicyInterpreter
# ---------------------------------------------------------------------------

class TestPolicyInterpreter:
    def setup_method(self):
        self.interp = PolicyInterpreter()

    def test_norm_disclose_email_financial(self):
        req = _make_request(
            platform_type=PlatformType.FINTECH,
            requested=frozenset({Attribute.EMAIL}),
        )
        result = self.interp.evaluate(
            req, _make_profile(), PolicySet(), ContextClass.FINANCIAL
        )
        assert result.action_for(Attribute.EMAIL) == PolicyAction.DISCLOSE
        assert result.decisions[0].source == "norm"

    def test_norm_withhold_health_in_financial(self):
        req = _make_request(
            platform_type=PlatformType.FINTECH,
            requested=frozenset({Attribute.HEALTH_CONDITION}),
        )
        result = self.interp.evaluate(
            req, _make_profile(), PolicySet(), ContextClass.FINANCIAL
        )
        assert result.action_for(Attribute.HEALTH_CONDITION) == PolicyAction.WITHHOLD

    def test_hard_blacklist_overrides_norm(self):
        req = _make_request(
            platform_type=PlatformType.FINTECH,
            requested=frozenset({Attribute.EMAIL}),
        )
        profile = _make_profile(hard_blacklist=[Attribute.EMAIL])
        result = self.interp.evaluate(
            req, profile, PolicySet(), ContextClass.FINANCIAL
        )
        assert result.action_for(Attribute.EMAIL) == PolicyAction.WITHHOLD
        assert result.decisions[0].source == "hard_blacklist"

    def test_proof_required_overrides_norm_disclose(self):
        req = _make_request(
            platform_type=PlatformType.FINTECH,
            requested=frozenset({Attribute.EMAIL}),
        )
        profile = _make_profile(proof_required=[Attribute.EMAIL])
        result = self.interp.evaluate(
            req, profile, PolicySet(), ContextClass.FINANCIAL
        )
        assert result.action_for(Attribute.EMAIL) == PolicyAction.VERIFY_ONLY
        assert result.decisions[0].source == "proof_required"

    def test_policy_rule_overrides_norm(self):
        req = _make_request(
            platform_type=PlatformType.FINTECH,
            requested=frozenset({Attribute.EMAIL}),
        )
        ps = PolicySet()
        ps.add_rule(PolicyRule(
            attributes=frozenset({Attribute.EMAIL}),
            contexts=frozenset({ContextClass.FINANCIAL}),
            action=PolicyAction.ABSTRACT,
            priority=5,
        ))
        result = self.interp.evaluate(
            req, _make_profile(), ps, ContextClass.FINANCIAL
        )
        assert result.action_for(Attribute.EMAIL) == PolicyAction.ABSTRACT
        assert result.decisions[0].source == "policy_rule"

    def test_trust_gate_rejects_all(self):
        req = _make_request(trust_score=0.3)
        profile = _make_profile(min_trust=0.8)
        result = self.interp.evaluate(
            req, profile, PolicySet(), ContextClass.FINANCIAL
        )
        assert result.trust_gated is True
        assert result.decisions == []

    def test_norm_dob_verify_only_financial(self):
        req = _make_request(
            platform_type=PlatformType.FINTECH,
            requested=frozenset({Attribute.DOB}),
        )
        result = self.interp.evaluate(
            req, _make_profile(), PolicySet(), ContextClass.FINANCIAL
        )
        assert result.action_for(Attribute.DOB) == PolicyAction.VERIFY_ONLY

    def test_attribute_set_helpers(self):
        req = _make_request(
            platform_type=PlatformType.FINTECH,
            requested=frozenset({Attribute.EMAIL, Attribute.HEALTH_CONDITION}),
        )
        result = self.interp.evaluate(
            req, _make_profile(), PolicySet(), ContextClass.FINANCIAL
        )
        assert Attribute.EMAIL in result.disclose_set
        assert Attribute.HEALTH_CONDITION in result.withhold_set

    def test_default_withhold_for_unknown_attribute_context(self):
        # GAMING_HISTORY has no norm in FINANCIAL context.
        req = _make_request(
            platform_type=PlatformType.FINTECH,
            requested=frozenset({Attribute.GAMING_HISTORY}),
        )
        result = self.interp.evaluate(
            req, _make_profile(), PolicySet(), ContextClass.FINANCIAL
        )
        action = result.action_for(Attribute.GAMING_HISTORY)
        assert action == PolicyAction.WITHHOLD
        assert result.decisions[0].source == "default"


# ---------------------------------------------------------------------------
# AuditLog
# ---------------------------------------------------------------------------

class TestAuditLog:
    def test_append_and_count(self):
        log = AuditLog(Path(":memory:"))
        log.append(_make_receipt())
        assert log.count() == 1

    def test_chain_valid_after_append(self):
        log = AuditLog(Path(":memory:"))
        for i in range(5):
            log.append(_make_receipt(receipt_id=f"rec-{i}", request_id=f"req-{i}"))
        assert log.verify_chain() is True

    def test_query_by_context(self):
        log = AuditLog(Path(":memory:"))
        log.append(_make_receipt(receipt_id="r1", context=ContextClass.FINANCIAL))
        log.append(_make_receipt(receipt_id="r2", context=ContextClass.MEDICAL))
        results = log.query(context=ContextClass.FINANCIAL)
        assert len(results) == 1
        assert results[0].context_class == ContextClass.FINANCIAL

    def test_query_by_platform(self):
        log = AuditLog(Path(":memory:"))
        log.append(_make_receipt(receipt_id="r1", platform_id="plat-A"))
        log.append(_make_receipt(receipt_id="r2", platform_id="plat-B"))
        results = log.query(platform_id="plat-A")
        assert len(results) == 1
        assert results[0].platform_id == "plat-A"

    def test_query_by_since(self):
        from dataclasses import replace
        log = AuditLog(Path(":memory:"))
        early = replace(
            _make_receipt(receipt_id="r1"),
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        log.append(early)
        log.append(_make_receipt(receipt_id="r2"))  # now()
        since = datetime(2025, 6, 1, tzinfo=timezone.utc)
        results = log.query(since=since)
        assert all(r.receipt_id != "r1" for r in results)

    def test_tamper_detection(self):
        log = AuditLog(Path(":memory:"))
        log.append(_make_receipt(receipt_id="r1"))
        log.append(_make_receipt(receipt_id="r2", request_id="req-2"))
        # Corrupt the first row's payload directly.
        log._conn.execute(
            "UPDATE disclosure_log SET payload = 'TAMPERED' WHERE id = 1"
        )
        log._conn.commit()
        assert log.verify_chain() is False

    def test_canonical_json_is_deterministic(self):
        r = _make_receipt()
        j1 = _canonical_json(r)
        j2 = _canonical_json(r)
        assert j1 == j2

    def test_canonical_json_sorts_attributes(self):
        r = _make_receipt(disclosed=[Attribute.PHONE, Attribute.EMAIL])
        data = json.loads(_canonical_json(r))
        assert data["disclosed_attributes"] == sorted(data["disclosed_attributes"])

    def test_roundtrip_receipt(self):
        log = AuditLog(Path(":memory:"))
        original = _make_receipt(
            receipt_id="r1",
            disclosed=[Attribute.EMAIL, Attribute.PHONE],
            withheld=[Attribute.INCOME_BRACKET],
            status=NegotiationStatus.PARTIAL,
        )
        log.append(original)
        results = log.query()
        assert len(results) == 1
        r = results[0]
        assert r.receipt_id == "r1"
        assert r.negotiation_status == NegotiationStatus.PARTIAL
        assert Attribute.EMAIL in r.disclosed_attributes
        assert Attribute.INCOME_BRACKET in r.withheld_attributes

    def test_empty_chain_is_valid(self):
        log = AuditLog(Path(":memory:"))
        assert log.verify_chain() is True

    def test_persistent_db(self, tmp_path):
        db_path = tmp_path / "audit.db"
        log1 = AuditLog(db_path)
        log1.append(_make_receipt(receipt_id="r1"))
        log1.close()

        log2 = AuditLog(db_path)
        assert log2.count() == 1
        assert log2.verify_chain() is True

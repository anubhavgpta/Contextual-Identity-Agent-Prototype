"""Tests for cia.models — structural correctness, no logic yet."""

import math
from datetime import datetime

import pytest

from cia.models import (
    Attribute,
    AttributeVocab,
    AttributeValue,
    ClassifiedContext,
    ContextClass,
    ContextSignal,
    ProfileTuple,
    TrustConstraints,
    PolicyAction,
    PolicyRule,
    PolicySet,
    PlatformType,
    RequestTier,
    DisclosureRequest,
    NegotiationStatus,
    DisclosureReceipt,
    DisclosureResponse,
    UtilityScore,
    CandidateDisclosure,
    DecisionTrace,
)


# ---------------------------------------------------------------------------
# Attribute vocabulary
# ---------------------------------------------------------------------------

def test_attribute_vocab_has_12_members():
    assert len(AttributeVocab) == 12


def test_attribute_vocab_contains_all_expected():
    expected = {
        "legal_name", "dob", "email", "phone", "geolocation",
        "employment_status", "income_bracket", "health_condition",
        "gaming_history", "social_graph", "device_fingerprint",
        "behavioural_metadata",
    }
    assert {a.value for a in AttributeVocab} == expected


def test_attribute_value_effective_value_prefers_abstracted():
    av = AttributeValue(
        attribute=Attribute.DOB, raw="1990-03-15", abstracted="1990s"
    )
    assert av.effective_value() == "1990s"


def test_attribute_value_effective_value_falls_back_to_raw():
    av = AttributeValue(attribute=Attribute.EMAIL, raw="a@b.com")
    assert av.effective_value() == "a@b.com"


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

def test_context_class_has_5_members():
    assert len(ContextClass) == 5


def test_classified_context_conservative_flag():
    cc = ClassifiedContext(
        context_class=ContextClass.FINANCIAL,
        confidence=0.72,
        conservative_mode=True,
        class_probabilities={ContextClass.FINANCIAL: 0.72, ContextClass.CIVIC: 0.28},
    )
    assert cc.conservative_mode is True
    assert cc.confidence < 0.80


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

def test_profile_tuple_creation():
    attrs = frozenset({Attribute.EMAIL, Attribute.LEGAL_NAME})
    weights = {Attribute.EMAIL: 0.6, Attribute.LEGAL_NAME: 0.4}
    tc = TrustConstraints(min_trust_score=0.5)
    profile = ProfileTuple(
        profile_id="p1",
        active_attributes=attrs,
        salience_weights=weights,
        trust_constraints=tc,
    )
    assert profile.profile_id == "p1"
    assert Attribute.EMAIL in profile.active_attributes
    assert profile.disclosure_history == ()
    assert profile.drift_flagged is False


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

def test_policy_rule_wildcard_matches_any():
    rule = PolicyRule(
        attributes=frozenset(),
        contexts=frozenset(),
        action=PolicyAction.WITHHOLD,
        priority=200,
    )
    assert rule.matches(Attribute.PHONE, ContextClass.SOCIAL)
    assert rule.matches(Attribute.INCOME_BRACKET, ContextClass.MEDICAL)


def test_policy_rule_specific_match():
    rule = PolicyRule(
        attributes=frozenset({Attribute.HEALTH_CONDITION}),
        contexts=frozenset({ContextClass.MEDICAL}),
        action=PolicyAction.VERIFY_ONLY,
        priority=10,
    )
    assert rule.matches(Attribute.HEALTH_CONDITION, ContextClass.MEDICAL)
    assert not rule.matches(Attribute.HEALTH_CONDITION, ContextClass.SOCIAL)
    assert not rule.matches(Attribute.EMAIL, ContextClass.MEDICAL)


def test_policy_set_resolution_priority_order():
    ps = PolicySet()
    # Low-priority wildcard: withhold everything.
    ps.add_rule(PolicyRule(
        attributes=frozenset(), contexts=frozenset(),
        action=PolicyAction.WITHHOLD, priority=100,
    ))
    # High-priority specific rule: disclose email in PROFESSIONAL.
    ps.add_rule(PolicyRule(
        attributes=frozenset({Attribute.EMAIL}),
        contexts=frozenset({ContextClass.PROFESSIONAL}),
        action=PolicyAction.DISCLOSE, priority=10,
    ))
    assert ps.resolve(Attribute.EMAIL, ContextClass.PROFESSIONAL) == PolicyAction.DISCLOSE
    assert ps.resolve(Attribute.PHONE, ContextClass.PROFESSIONAL) == PolicyAction.WITHHOLD
    assert ps.resolve(Attribute.EMAIL, ContextClass.SOCIAL) == PolicyAction.WITHHOLD


def test_policy_set_returns_none_when_no_rule():
    ps = PolicySet()
    assert ps.resolve(Attribute.EMAIL, ContextClass.SOCIAL) is None


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

def test_disclosure_request_immutable():
    req = DisclosureRequest(
        request_id="r1",
        platform_id="plat_fintech",
        platform_type=PlatformType.FINTECH,
        platform_trust_score=0.85,
        requested_attributes=frozenset({Attribute.EMAIL, Attribute.INCOME_BRACKET}),
        context_signal=ContextSignal(description="loan application"),
    )
    with pytest.raises((AttributeError, TypeError)):
        req.platform_id = "mutate"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

def test_negotiation_status_has_6_values():
    # ACCEPTED, PARTIAL, REJECTED, PROOF_ONLY, PENDING, FAILED
    assert len(NegotiationStatus) == 6


def test_disclosure_receipt_fields():
    receipt = DisclosureReceipt(
        receipt_id="rec1",
        request_id="r1",
        platform_id="plat_fintech",
        context_class=ContextClass.FINANCIAL,
        disclosed_attributes=frozenset({Attribute.EMAIL}),
        withheld_attributes=frozenset({Attribute.INCOME_BRACKET}),
        negotiation_status=NegotiationStatus.PARTIAL,
        utility_score=0.63,
        entropy_before=3.2,
        entropy_after=2.9,
    )
    assert receipt.entropy_before > receipt.entropy_after


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

def test_utility_score_formula():
    u = UtilityScore(S=0.8, T=0.9, L=1.2, alpha=0.3, beta=0.3, gamma=0.6, U=0.0)
    expected = 0.3 * 0.8 + 0.3 * 0.9 - 0.6 * 1.2
    # U stored separately; we verify formula manually.
    assert math.isclose(u.alpha * u.S + u.beta * u.T - u.gamma * u.L, expected)


def test_decision_trace_optimal_property():
    attrs = frozenset({Attribute.EMAIL})
    u = UtilityScore(S=1.0, T=0.9, L=0.5, alpha=0.3, beta=0.3, gamma=0.6, U=0.27)
    cand = CandidateDisclosure(attribute_set=attrs, utility=u)
    trace = DecisionTrace(request_id="r1", context_class=ContextClass.PROFESSIONAL)
    assert trace.optimal is None
    trace.candidates.append(cand)
    trace.optimal_index = 0
    assert trace.optimal is cand


def test_decision_trace_out_of_bounds_optimal():
    trace = DecisionTrace(request_id="r1", context_class=ContextClass.CIVIC)
    trace.optimal_index = 99
    assert trace.optimal is None

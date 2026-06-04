"""Tests for cia.eval.metrics — pure metric computation."""

from __future__ import annotations

import math

import pytest

from cia.models.attributes import Attribute, AttributeVocab
from cia.models.context import ContextClass
from cia.models.request import RequestTier
from cia.models.response import NegotiationStatus
from cia.eval.metrics import (
    BaselineComparison,
    ClassificationResult,
    SimSession,
    TraceResult,
    _sessions_from_traces,
    _sessions_from_traces_with_disclosed,
    classification_accuracy,
    compare_baselines,
    cross_platform_linkability,
    decision_latency_stats,
    disclosure_minimisation_rate,
    reidentification_risk,
)

_ALL = frozenset(AttributeVocab)   # 12 attrs
_H_I = math.log(12)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tr(
    platform: str = "financial_portal",
    tier: RequestTier = RequestTier.BENIGN,
    requested: set[Attribute] | None = None,
    disclosed: set[Attribute] | None = None,
    status: NegotiationStatus = NegotiationStatus.PARTIAL,
    ctx: ContextClass = ContextClass.FINANCIAL,
    conservative: bool = False,
) -> TraceResult:
    req = frozenset(requested or {Attribute.EMAIL, Attribute.PHONE})
    dis = frozenset(disclosed if disclosed is not None else {Attribute.EMAIL})
    return TraceResult(
        platform=platform, tier=tier,
        requested=req, disclosed=dis,
        latency_ms=5.0, status=status,
        context_classified=ctx,
        conservative_mode_triggered=conservative,
    )


# ---------------------------------------------------------------------------
# disclosure_minimisation_rate
# ---------------------------------------------------------------------------

class TestDMR:
    def test_empty(self):
        assert disclosure_minimisation_rate([], []) == 0.0

    def test_full_disclosure(self):
        attrs = frozenset({Attribute.EMAIL, Attribute.PHONE})
        dmr = disclosure_minimisation_rate([attrs], [attrs])
        assert dmr == pytest.approx(0.0)

    def test_full_withholding(self):
        requested = frozenset({Attribute.EMAIL, Attribute.PHONE})
        dmr = disclosure_minimisation_rate([frozenset()], [requested])
        assert dmr == pytest.approx(1.0)

    def test_partial(self):
        requested = frozenset({Attribute.EMAIL, Attribute.PHONE, Attribute.LEGAL_NAME})
        disclosed = frozenset({Attribute.EMAIL})
        dmr = disclosure_minimisation_rate([disclosed], [requested])
        assert dmr == pytest.approx(1.0 - 1 / 3)

    def test_mean_over_traces(self):
        req = frozenset({Attribute.EMAIL, Attribute.PHONE})
        # trace 1: disclose 1/2 → DMR 0.5; trace 2: disclose 0/2 → DMR 1.0
        dmr = disclosure_minimisation_rate(
            [frozenset({Attribute.EMAIL}), frozenset()],
            [req, req],
        )
        assert dmr == pytest.approx(0.75)

    def test_skips_empty_request(self):
        # Empty A_req traces are skipped; only the non-empty trace counts.
        dmr = disclosure_minimisation_rate(
            [frozenset(), frozenset()],
            [frozenset(), frozenset({Attribute.EMAIL})],
        )
        assert dmr == pytest.approx(1.0)

    def test_oauth_baseline_is_zero(self):
        req = frozenset({Attribute.EMAIL, Attribute.PHONE})
        dmr = disclosure_minimisation_rate([req], [req])
        assert dmr == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# reidentification_risk
# ---------------------------------------------------------------------------

class TestReidRisk:
    def test_empty(self):
        assert reidentification_risk([]) == 0.0

    def test_disclose_nothing(self):
        risk = reidentification_risk([frozenset()])
        assert risk == pytest.approx(0.0)

    def test_disclose_all(self):
        # H(I|D*) = log(12 - 12) → 0, so L = H(I) - 0 = log(12)
        risk = reidentification_risk([_ALL])
        assert risk == pytest.approx(_H_I)

    def test_disclose_one(self):
        # H(I|D*) = log(11), L = log(12) - log(11)
        risk = reidentification_risk([frozenset({Attribute.EMAIL})])
        assert risk == pytest.approx(math.log(12) - math.log(11))

    def test_mean_over_traces(self):
        r1 = math.log(12) - math.log(11)  # disclose 1
        r2 = math.log(12) - math.log(10)  # disclose 2
        risk = reidentification_risk([
            frozenset({Attribute.EMAIL}),
            frozenset({Attribute.EMAIL, Attribute.PHONE}),
        ])
        assert risk == pytest.approx((r1 + r2) / 2)


# ---------------------------------------------------------------------------
# cross_platform_linkability
# ---------------------------------------------------------------------------

class TestLinkability:
    def _session(self, platform: str, tier: RequestTier, disclosed_sets: list[set[Attribute]]) -> SimSession:
        traces = [
            _tr(platform=platform, tier=tier, disclosed=ds)
            for ds in disclosed_sets
        ]
        return SimSession(platform=platform, tier=tier, traces=traces)

    def test_single_session_returns_zero(self):
        s = self._session("A", RequestTier.BENIGN, [{Attribute.EMAIL, Attribute.PHONE}])
        assert cross_platform_linkability([s]) == 0.0

    def test_no_intersection_returns_zero(self):
        s1 = self._session("A", RequestTier.BENIGN, [{Attribute.EMAIL}])
        s2 = self._session("B", RequestTier.BENIGN, [{Attribute.PHONE}])
        assert cross_platform_linkability([s1, s2]) == 0.0

    def test_one_common_attr_not_linkable(self):
        s1 = self._session("A", RequestTier.BENIGN, [{Attribute.EMAIL}])
        s2 = self._session("B", RequestTier.BENIGN, [{Attribute.EMAIL}])
        # intersection = 1 < 2
        assert cross_platform_linkability([s1, s2]) == 0.0

    def test_two_common_attrs_linkable(self):
        s1 = self._session("A", RequestTier.BENIGN, [{Attribute.EMAIL, Attribute.PHONE}])
        s2 = self._session("B", RequestTier.BENIGN, [{Attribute.EMAIL, Attribute.PHONE}])
        assert cross_platform_linkability([s1, s2]) == pytest.approx(1.0)

    def test_different_tiers_not_paired(self):
        s1 = self._session("A", RequestTier.BENIGN,    [{Attribute.EMAIL, Attribute.PHONE}])
        s2 = self._session("B", RequestTier.PROBING,   [{Attribute.EMAIL, Attribute.PHONE}])
        # Different tiers → no pairing → no denominator → 0
        assert cross_platform_linkability([s1, s2]) == 0.0

    def test_same_platform_not_paired(self):
        s1 = self._session("A", RequestTier.BENIGN, [{Attribute.EMAIL, Attribute.PHONE}])
        s2 = self._session("A", RequestTier.BENIGN, [{Attribute.EMAIL, Attribute.PHONE}])
        assert cross_platform_linkability([s1, s2]) == 0.0

    def test_partial_linkability(self):
        # Two interactions: first linkable (2 common), second not (0 common).
        s1 = self._session("A", RequestTier.BENIGN, [
            {Attribute.EMAIL, Attribute.PHONE},
            {Attribute.EMAIL},
        ])
        s2 = self._session("B", RequestTier.BENIGN, [
            {Attribute.EMAIL, Attribute.PHONE},
            {Attribute.LEGAL_NAME},
        ])
        # 1/2 interactions linkable
        assert cross_platform_linkability([s1, s2]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# classification_accuracy
# ---------------------------------------------------------------------------

class TestClassificationAccuracy:
    def test_empty(self):
        assert classification_accuracy([]) == 0.0

    def test_all_correct(self):
        results = [
            ClassificationResult(ContextClass.FINANCIAL, ContextClass.FINANCIAL),
            ClassificationResult(ContextClass.SOCIAL, ContextClass.SOCIAL),
        ]
        assert classification_accuracy(results) == pytest.approx(1.0)

    def test_all_wrong(self):
        results = [
            ClassificationResult(ContextClass.FINANCIAL, ContextClass.SOCIAL),
        ]
        assert classification_accuracy(results) == pytest.approx(0.0)

    def test_partial(self):
        results = [
            ClassificationResult(ContextClass.FINANCIAL, ContextClass.FINANCIAL),
            ClassificationResult(ContextClass.SOCIAL, ContextClass.MEDICAL),
        ]
        assert classification_accuracy(results) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# decision_latency_stats
# ---------------------------------------------------------------------------

class TestLatencyStats:
    def test_empty(self):
        stats = decision_latency_stats([])
        assert stats == {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}

    def test_single(self):
        stats = decision_latency_stats([10.0])
        assert stats["mean"] == pytest.approx(10.0)
        assert stats["p50"]  == pytest.approx(10.0)
        assert stats["p95"]  == pytest.approx(10.0)
        assert stats["p99"]  == pytest.approx(10.0)

    def test_uniform(self):
        stats = decision_latency_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats["mean"] == pytest.approx(3.0)
        assert stats["p50"]  == pytest.approx(3.0)

    def test_percentiles_ordered(self):
        import random as _rand
        _rand.seed(0)
        lats = [_rand.uniform(1, 100) for _ in range(1000)]
        stats = decision_latency_stats(lats)
        assert stats["p50"] <= stats["p95"] <= stats["p99"]
        assert stats["p99"] <= max(lats) + 1e-9


# ---------------------------------------------------------------------------
# compare_baselines
# ---------------------------------------------------------------------------

class TestCompareBaselines:
    def _make_traces(self) -> list[TraceResult]:
        # financial_portal: discloses 1 of 2 → DMR 0.5
        # social_platform: discloses 0 of 2 → DMR 1.0
        return [
            _tr(
                platform="financial_portal",
                tier=RequestTier.BENIGN,
                requested={Attribute.EMAIL, Attribute.PHONE},
                disclosed={Attribute.EMAIL},
                ctx=ContextClass.FINANCIAL,
            ),
            _tr(
                platform="social_platform",
                tier=RequestTier.BENIGN,
                requested={Attribute.EMAIL, Attribute.SOCIAL_GRAPH},
                disclosed=set(),
                ctx=ContextClass.SOCIAL,
            ),
        ]

    def test_returns_baseline_comparison(self):
        traces = self._make_traces()
        result = compare_baselines(traces, scenario="test")
        assert isinstance(result, BaselineComparison)

    def test_oauth_dmr_is_zero(self):
        traces = self._make_traces()
        result = compare_baselines(traces)
        assert result.oauth_dmr == pytest.approx(0.0)

    def test_unmediated_dmr_is_zero(self):
        traces = self._make_traces()
        result = compare_baselines(traces)
        assert result.unmediated_dmr == pytest.approx(0.0)

    def test_cia_dmr_above_zero(self):
        traces = self._make_traces()
        result = compare_baselines(traces)
        assert result.cia_dmr > 0.0

    def test_cia_reid_risk_le_oauth(self):
        traces = self._make_traces()
        result = compare_baselines(traces)
        assert result.cia_reid_risk <= result.oauth_reid_risk + 1e-9

    def test_ssi_dmr_between_oauth_and_cia(self):
        # SSI should be somewhere between OAuth (0) and full withholding.
        traces = self._make_traces()
        result = compare_baselines(traces)
        assert 0.0 <= result.ssi_dmr

    def test_scenario_label(self):
        traces = self._make_traces()
        result = compare_baselines(traces, scenario="unit_test")
        assert result.scenario == "unit_test"


# ---------------------------------------------------------------------------
# _sessions_from_traces helpers
# ---------------------------------------------------------------------------

class TestSessionHelpers:
    def test_groups_by_platform_and_tier(self):
        traces = [
            _tr(platform="A", tier=RequestTier.BENIGN),
            _tr(platform="A", tier=RequestTier.BENIGN),
            _tr(platform="B", tier=RequestTier.BENIGN),
            _tr(platform="A", tier=RequestTier.PROBING),
        ]
        sessions = _sessions_from_traces(traces)
        assert len(sessions) == 3
        ab = next(s for s in sessions if s.platform == "A" and s.tier == RequestTier.BENIGN)
        assert len(ab.traces) == 2

    def test_replaces_disclosed(self):
        traces = [_tr(disclosed={Attribute.EMAIL})]
        new_d  = [frozenset({Attribute.PHONE})]
        sessions = _sessions_from_traces_with_disclosed(traces, new_d)
        assert sessions[0].traces[0].disclosed == frozenset({Attribute.PHONE})

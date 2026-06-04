"""Tests for cia.eval.simulator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from cia.models.attributes import Attribute
from cia.models.context import ClassifiedContext, ContextClass, ContextSignal
from cia.models.request import DisclosureRequest, PlatformType, RequestTier
from cia.models.response import NegotiationStatus
from cia.eval.metrics import SimSession, TraceResult
from cia.eval.simulator import (
    SimPlatformConfig,
    SimulationReport,
    _PLATFORMS,
    _SharedTrackingCC,
    _StubBBS,
    _make_iacp_session,
    _make_request,
    _populate_pds,
    _run_one_session,
    run_simulation,
)
from cia.iacp import _platform_context_hint
from cia.policy.norms import NormDatabase
from cia.store.pds import PDS


# ---------------------------------------------------------------------------
# Minimal mock CC (no ONNX needed)
# ---------------------------------------------------------------------------

class _MockCC:
    def classify(
        self, request: DisclosureRequest, drift_alert: Any = None
    ) -> ClassifiedContext:
        ctx = _platform_context_hint(request.platform_type)
        return ClassifiedContext(
            context_class=ctx,
            confidence=0.92,
            conservative_mode=False,
            class_probabilities={ctx: 0.92},
            source="symbolic",
        )


# ---------------------------------------------------------------------------
# Platform config invariants
# ---------------------------------------------------------------------------

class TestPlatformConfigs:
    def test_five_platforms(self):
        assert len(_PLATFORMS) == 5

    def test_unique_names(self):
        names = [p.name for p in _PLATFORMS]
        assert len(names) == len(set(names))

    def test_trust_scores_in_range(self):
        for p in _PLATFORMS:
            assert 0.0 < p.trust_score <= 1.0

    def test_twenty_purpose_phrases_each(self):
        for p in _PLATFORMS:
            assert len(p.purpose_vocab) == 20, f"{p.name} has {len(p.purpose_vocab)} phrases"

    def test_base_attrs_non_empty(self):
        for p in _PLATFORMS:
            assert len(p.base_attrs) >= 4, f"{p.name} has too few base attrs"

    def test_probing_extra_non_empty(self):
        for p in _PLATFORMS:
            assert len(p.probing_extra) >= 2

    def test_probing_extra_disjoint_from_base(self):
        for p in _PLATFORMS:
            overlap = p.base_attrs & p.probing_extra
            assert not overlap, f"{p.name}: probing_extra overlaps base_attrs: {overlap}"

    def test_known_platform_names(self):
        names = {p.name for p in _PLATFORMS}
        assert names == {
            "financial_portal", "professional_network",
            "social_platform", "healthcare_portal", "gaming_platform",
        }

    @pytest.mark.parametrize("name,expected_ctx", [
        ("financial_portal",    ContextClass.FINANCIAL),
        ("professional_network", ContextClass.PROFESSIONAL),
        ("social_platform",     ContextClass.SOCIAL),
        ("healthcare_portal",   ContextClass.MEDICAL),
        ("gaming_platform",     ContextClass.SOCIAL),
    ])
    def test_expected_context(self, name: str, expected_ctx: ContextClass):
        cfg = next(p for p in _PLATFORMS if p.name == name)
        assert cfg.expected_context == expected_ctx

    @pytest.mark.parametrize("name,expected_trust", [
        ("financial_portal",    0.85),
        ("professional_network", 0.70),
        ("social_platform",     0.45),
        ("healthcare_portal",   0.90),
        ("gaming_platform",     0.55),
    ])
    def test_trust_scores(self, name: str, expected_trust: float):
        cfg = next(p for p in _PLATFORMS if p.name == name)
        assert cfg.trust_score == pytest.approx(expected_trust)


# ---------------------------------------------------------------------------
# _StubBBS
# ---------------------------------------------------------------------------

class TestStubBBS:
    def test_keygen_returns_keypair(self):
        from cia.crypto.bbs import BBSKeyPair
        kp = _StubBBS().keygen()
        assert isinstance(kp, BBSKeyPair)

    def test_sign_raises(self):
        with pytest.raises(RuntimeError):
            _StubBBS().sign()

    def test_prove_raises(self):
        with pytest.raises(RuntimeError):
            _StubBBS().prove_subset()


# ---------------------------------------------------------------------------
# _SharedTrackingCC
# ---------------------------------------------------------------------------

class TestSharedTrackingCC:
    def _make_req(self, rid: str) -> DisclosureRequest:
        return DisclosureRequest(
            request_id=rid,
            platform_id="test.example.com",
            platform_type=PlatformType.FINTECH,
            platform_trust_score=0.8,
            requested_attributes=frozenset({Attribute.EMAIL}),
            context_signal=ContextSignal(description="test"),
        )

    def test_records_result(self):
        cc = _SharedTrackingCC(_MockCC())
        req = self._make_req("req-1")
        cc.classify(req)
        assert cc.get("req-1") is not None
        assert cc.get("req-1").context_class == ContextClass.FINANCIAL

    def test_second_call_overwrites(self):
        from unittest.mock import MagicMock
        mock_cc = MagicMock()
        r1 = ClassifiedContext(ContextClass.FINANCIAL, 0.9, False, {}, "symbolic")
        r2 = ClassifiedContext(ContextClass.SOCIAL,    0.6, True,  {}, "symbolic")
        mock_cc.classify.side_effect = [r1, r2]
        cc = _SharedTrackingCC(mock_cc)
        req = self._make_req("req-2")
        cc.classify(req)
        cc.classify(req)
        assert cc.get("req-2").context_class == ContextClass.SOCIAL

    def test_unknown_request_id_returns_none(self):
        cc = _SharedTrackingCC(_MockCC())
        assert cc.get("nonexistent") is None


# ---------------------------------------------------------------------------
# _populate_pds
# ---------------------------------------------------------------------------

class TestPopulatePDS:
    def test_all_twelve_attrs_present(self):
        pds = PDS(Path(":memory:"))
        _populate_pds(pds)
        for attr in Attribute:
            val = pds.get_attribute(attr)
            assert val is not None, f"Missing attribute: {attr}"


# ---------------------------------------------------------------------------
# _make_request
# ---------------------------------------------------------------------------

class TestMakeRequest:
    def _cfg(self) -> SimPlatformConfig:
        return next(p for p in _PLATFORMS if p.name == "financial_portal")

    def test_benign_uses_base_attrs(self):
        cfg = self._cfg()
        req = _make_request(cfg, RequestTier.BENIGN, "loan", 0)
        assert req.requested_attributes == cfg.base_attrs

    def test_probing_adds_extra_attrs(self):
        cfg = self._cfg()
        req = _make_request(cfg, RequestTier.PROBING, "service", 0)
        assert cfg.probing_extra.issubset(req.requested_attributes)
        assert cfg.base_attrs.issubset(req.requested_attributes)

    def test_colluding_uses_base_attrs(self):
        cfg = self._cfg()
        req = _make_request(cfg, RequestTier.COLLUDING, "login", 0)
        assert req.requested_attributes == cfg.base_attrs

    def test_request_id_unique_per_interaction(self):
        cfg = self._cfg()
        ids = {_make_request(cfg, RequestTier.BENIGN, "loan", i).request_id for i in range(10)}
        assert len(ids) == 10

    def test_platform_type_matches_config(self):
        cfg = self._cfg()
        req = _make_request(cfg, RequestTier.BENIGN, "loan", 0)
        assert req.platform_type == cfg.platform_type

    def test_purpose_set(self):
        cfg = self._cfg()
        req = _make_request(cfg, RequestTier.BENIGN, "mortgage review", 0)
        assert req.context_signal.description == "mortgage review"


# ---------------------------------------------------------------------------
# _make_iacp_session
# ---------------------------------------------------------------------------

class TestMakeIACPSession:
    def test_returns_session_pds_audit(self):
        from cia.iacp import IACPSession
        cc = _SharedTrackingCC(_MockCC())
        norm_db = NormDatabase()
        session, pds, audit = _make_iacp_session(cc, norm_db, _StubBBS())
        assert isinstance(session, IACPSession)

    def test_pds_has_attributes(self):
        cc = _SharedTrackingCC(_MockCC())
        norm_db = NormDatabase()
        _, pds, _ = _make_iacp_session(cc, norm_db, _StubBBS())
        assert pds.get_attribute(Attribute.EMAIL) is not None


# ---------------------------------------------------------------------------
# _run_one_session
# ---------------------------------------------------------------------------

class TestRunOneSession:
    def _setup(self):
        import random
        random.seed(42)
        cc = _SharedTrackingCC(_MockCC())
        norm_db = NormDatabase()
        platform_cfg = next(p for p in _PLATFORMS if p.name == "financial_portal")
        session, _, _ = _make_iacp_session(cc, norm_db, _StubBBS())
        rng = random.Random(42)
        return platform_cfg, session, cc, rng

    def test_returns_expected_counts(self):
        platform_cfg, session, cc, rng = self._setup()
        traces, cls_results = _run_one_session(
            platform_cfg, RequestTier.BENIGN,
            session, cc, n_interactions=5, rng=rng,
        )
        assert len(traces) == 5
        assert len(cls_results) == 5

    def test_trace_fields_populated(self):
        platform_cfg, session, cc, rng = self._setup()
        traces, _ = _run_one_session(
            platform_cfg, RequestTier.BENIGN,
            session, cc, n_interactions=2, rng=rng,
        )
        for tr in traces:
            assert tr.platform == platform_cfg.name
            assert tr.tier == RequestTier.BENIGN
            assert tr.latency_ms >= 0.0
            assert isinstance(tr.disclosed, frozenset)
            assert isinstance(tr.status, NegotiationStatus)

    def test_disclosed_subset_of_requested(self):
        platform_cfg, session, cc, rng = self._setup()
        traces, _ = _run_one_session(
            platform_cfg, RequestTier.BENIGN,
            session, cc, n_interactions=3, rng=rng,
        )
        for tr in traces:
            assert tr.disclosed <= tr.requested

    def test_probing_tier_uses_extra_attrs(self):
        import random
        random.seed(42)
        cc = _SharedTrackingCC(_MockCC())
        norm_db = NormDatabase()
        platform_cfg = next(p for p in _PLATFORMS if p.name == "financial_portal")
        session, _, _ = _make_iacp_session(cc, norm_db, _StubBBS())
        rng = random.Random(42)
        traces, _ = _run_one_session(
            platform_cfg, RequestTier.PROBING,
            session, cc, n_interactions=3, rng=rng,
        )
        for tr in traces:
            assert platform_cfg.probing_extra.issubset(tr.requested)


# ---------------------------------------------------------------------------
# run_simulation (fast: n_interactions=3)
# ---------------------------------------------------------------------------

class TestRunSimulation:
    @pytest.fixture(scope="class")
    def report(self) -> SimulationReport:
        return run_simulation(n_interactions=3, cc_override=_MockCC())

    def test_returns_simulation_report(self, report: SimulationReport):
        assert isinstance(report, SimulationReport)

    def test_total_traces(self, report: SimulationReport):
        # 5 platforms × 3 tiers × 3 interactions = 45
        assert report.total_traces == 45

    def test_n_interactions(self, report: SimulationReport):
        assert report.n_interactions == 3

    def test_dmr_by_platform_keys(self, report: SimulationReport):
        expected = {
            "financial_portal", "professional_network",
            "social_platform", "healthcare_portal", "gaming_platform",
        }
        assert set(report.dmr_by_platform.keys()) == expected

    def test_dmr_values_in_range(self, report: SimulationReport):
        for v in report.dmr_by_platform.values():
            assert 0.0 <= v <= 1.0
        assert 0.0 <= report.overall_dmr <= 1.0

    def test_linkability_by_tier_keys(self, report: SimulationReport):
        assert set(report.linkability_by_tier.keys()) == {"BENIGN", "PROBING", "COLLUDING"}

    def test_linkability_values_in_range(self, report: SimulationReport):
        for v in report.linkability_by_tier.values():
            assert 0.0 <= v <= 1.0

    def test_classification_accuracy_keys(self, report: SimulationReport):
        assert set(report.classification_accuracy_by_tier.keys()) == {
            "BENIGN", "PROBING", "COLLUDING"
        }

    def test_classification_accuracy_in_range(self, report: SimulationReport):
        for v in report.classification_accuracy_by_tier.values():
            assert 0.0 <= v <= 1.0

    def test_negotiation_success_rate_in_range(self, report: SimulationReport):
        assert 0.0 <= report.negotiation_success_rate <= 1.0

    def test_latency_stats_keys(self, report: SimulationReport):
        assert set(report.latency_stats.keys()) == {"mean", "p50", "p95", "p99"}

    def test_latency_positive(self, report: SimulationReport):
        assert report.latency_stats["mean"] > 0.0

    def test_baseline_dict_present(self, report: SimulationReport):
        b = report.baseline
        assert "cia_dmr" in b
        assert "oauth_dmr" in b
        assert "ssi_dmr" in b

    def test_to_dict_json_serialisable(self, report: SimulationReport):
        import json
        d = report.to_dict()
        # Should not raise
        json.dumps(d)

    def test_reid_risk_reduction_pct_computed(self, report: SimulationReport):
        # May be negative in edge cases with tiny n_interactions, but must be a float.
        assert isinstance(report.reid_risk_reduction_pct, float)

    def test_deterministic_across_runs(self):
        r1 = run_simulation(n_interactions=2, cc_override=_MockCC())
        r2 = run_simulation(n_interactions=2, cc_override=_MockCC())
        assert r1.overall_dmr == pytest.approx(r2.overall_dmr)
        assert r1.negotiation_success_rate == pytest.approx(r2.negotiation_success_rate)

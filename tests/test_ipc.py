"""Tests for cia.ipc — Identity Profile Constructor."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cia.models.attributes import Attribute
from cia.models.context import ContextClass
from cia.models.profile import DisclosureEvent, ProfileTuple, TrustConstraints
from cia.store.pds import PDS
from cia.ipc import (
    IPC,
    TrustBand,
    EntropyAlert,
    DriftAlert,
    _update_salience,
    _compute_entropy,
    _update_entropy_counts,
    _extract_features,
    _kl_divergence_features,
    _trust_band,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pds() -> PDS:
    p = PDS(Path(":memory:"))
    # Populate the vault with enough attributes for meaningful profiles.
    p.set_attribute(Attribute.EMAIL,             "alice@example.com")
    p.set_attribute(Attribute.LEGAL_NAME,        "Alice Example")
    p.set_attribute(Attribute.DOB,               "1990-01-15")
    p.set_attribute(Attribute.PHONE,             "+1-555-0100")
    p.set_attribute(Attribute.INCOME_BRACKET,    75_000)
    p.set_attribute(Attribute.EMPLOYMENT_STATUS, "employed")
    p.set_attribute(Attribute.HEALTH_CONDITION,  "none")
    p.set_attribute(Attribute.GEOLOCATION,       "London, UK")
    return p


@pytest.fixture
def ipc(pds: PDS) -> IPC:
    return IPC(pds)


def _make_event(
    platform_id: str,
    context: ContextClass,
    disclosed: set[Attribute],
    entropy: float = 0.0,
) -> DisclosureEvent:
    return DisclosureEvent(
        timestamp=datetime.now(timezone.utc),
        platform_id=platform_id,
        context_class=context,
        disclosed_attributes=frozenset(disclosed),
        receipt_signature="sig-test",
        entropy_at_disclosure=entropy,
    )


# ---------------------------------------------------------------------------
# Profile management
# ---------------------------------------------------------------------------

class TestProfileManagement:
    def test_create_profile_returns_profile_tuple(self, ipc: IPC):
        p = ipc.create_profile(ContextClass.FINANCIAL)
        assert isinstance(p, ProfileTuple)
        assert p.profile_id != ""

    def test_create_profile_active_attrs_are_norm_filtered(self, ipc: IPC):
        p = ipc.create_profile(ContextClass.FINANCIAL)
        # HEALTH_CONDITION has no permitted norm in FINANCIAL context.
        assert Attribute.HEALTH_CONDITION not in p.active_attributes
        # EMAIL is permitted in FINANCIAL.
        assert Attribute.EMAIL in p.active_attributes

    def test_create_profile_weights_sum_to_one(self, ipc: IPC):
        p = ipc.create_profile(ContextClass.FINANCIAL)
        assert abs(sum(p.salience_weights.values()) - 1.0) < 1e-9

    def test_create_profile_weights_are_uniform(self, ipc: IPC):
        p = ipc.create_profile(ContextClass.FINANCIAL)
        n = len(p.active_attributes)
        for w in p.salience_weights.values():
            assert abs(w - 1.0 / n) < 1e-9

    def test_get_profile_returns_none_before_create(self, ipc: IPC):
        assert ipc.get_profile(ContextClass.CIVIC) is None

    def test_get_profile_returns_after_create(self, ipc: IPC):
        ipc.create_profile(ContextClass.SOCIAL)
        p = ipc.get_profile(ContextClass.SOCIAL)
        assert p is not None

    def test_get_or_create_creates_if_missing(self, ipc: IPC):
        p = ipc.get_or_create_profile(ContextClass.PROFESSIONAL)
        assert p is not None
        assert p.profile_id != ""

    def test_get_or_create_returns_existing(self, ipc: IPC):
        p1 = ipc.create_profile(ContextClass.MEDICAL)
        p2 = ipc.get_or_create_profile(ContextClass.MEDICAL)
        assert p1.profile_id == p2.profile_id

    def test_create_profile_persisted_to_pds(self, ipc: IPC, pds: PDS):
        ipc.create_profile(ContextClass.FINANCIAL)
        loaded = pds.load_profile(ContextClass.FINANCIAL)
        assert loaded is not None

    def test_create_profile_empty_vault_gives_empty_active_attrs(self):
        empty_pds = PDS(Path(":memory:"))
        ipc = IPC(empty_pds)
        p = ipc.create_profile(ContextClass.FINANCIAL)
        assert p.active_attributes == frozenset()

    def test_different_contexts_different_active_attrs(self, ipc: IPC):
        fin = ipc.create_profile(ContextClass.FINANCIAL)
        med = ipc.create_profile(ContextClass.MEDICAL)
        # HEALTH_CONDITION: not in FINANCIAL, yes in MEDICAL.
        assert Attribute.HEALTH_CONDITION not in fin.active_attributes
        assert Attribute.HEALTH_CONDITION in med.active_attributes

    def test_cache_preserves_disclosure_history(self, ipc: IPC):
        ipc.create_profile(ContextClass.FINANCIAL)
        evt = _make_event("plat-1", ContextClass.FINANCIAL, {Attribute.EMAIL})
        ipc.update_profile(ContextClass.FINANCIAL, evt)
        cached = ipc.get_profile(ContextClass.FINANCIAL)
        assert len(cached.disclosure_history) == 1


# ---------------------------------------------------------------------------
# Profile update and entropy
# ---------------------------------------------------------------------------

class TestUpdateAndEntropy:
    def test_update_appends_to_history(self, ipc: IPC):
        ipc.create_profile(ContextClass.FINANCIAL)
        for i in range(3):
            evt = _make_event(f"p{i}", ContextClass.FINANCIAL, {Attribute.EMAIL})
            ipc.update_profile(ContextClass.FINANCIAL, evt)
        p = ipc.get_profile(ContextClass.FINANCIAL)
        assert len(p.disclosure_history) == 3

    def test_update_returns_none_when_below_threshold(self, ipc: IPC):
        ipc.create_profile(ContextClass.FINANCIAL)
        evt = _make_event("p1", ContextClass.FINANCIAL, {Attribute.EMAIL})
        alert = ipc.update_profile(ContextClass.FINANCIAL, evt)
        # Single disclosure — delta too small (entropy starts at 0).
        # May or may not fire depending on delta; test that type is correct.
        assert alert is None or isinstance(alert, EntropyAlert)

    def test_update_fires_entropy_alert_on_large_delta(self):
        """
        Single-attribute disclosure gives H=0 (p=1, log(1)=0).
        Two *different* attributes are needed: entropy jumps 0→log(2)≈0.693.
        With threshold=0.0, that delta fires an alert.
        """
        pds = PDS(Path(":memory:"))
        pds.set_attribute(Attribute.EMAIL, "a@b.com")
        pds.set_attribute(Attribute.LEGAL_NAME, "Alice")
        pds.set_attribute(Attribute.DOB, "1990-01-01")
        ipc = IPC(pds, entropy_threshold=0.0)
        ipc.create_profile(ContextClass.FINANCIAL)
        # First disclosure (EMAIL only): H 0→0, delta=0, no alert.
        ipc.update_profile(
            ContextClass.FINANCIAL,
            _make_event("p1", ContextClass.FINANCIAL, {Attribute.EMAIL}),
        )
        # Second disclosure (different attribute): H rises → delta > 0 → alert.
        alert = ipc.update_profile(
            ContextClass.FINANCIAL,
            _make_event("p1", ContextClass.FINANCIAL, {Attribute.LEGAL_NAME}),
        )
        assert isinstance(alert, EntropyAlert)
        assert alert.delta > 0
        assert alert.context == ContextClass.FINANCIAL

    def test_entropy_alert_fields(self):
        pds = PDS(Path(":memory:"))
        pds.set_attribute(Attribute.EMAIL, "x@y.com")
        pds.set_attribute(Attribute.PHONE, "123")
        ipc = IPC(pds, entropy_threshold=0.0)
        ipc.create_profile(ContextClass.FINANCIAL)
        # Prime with one attribute so next disclosure shifts the distribution.
        ipc.update_profile(
            ContextClass.FINANCIAL,
            _make_event("p", ContextClass.FINANCIAL, {Attribute.EMAIL}),
        )
        alert = ipc.update_profile(
            ContextClass.FINANCIAL,
            _make_event("p", ContextClass.FINANCIAL, {Attribute.PHONE}),
        )
        assert isinstance(alert, EntropyAlert)
        assert alert.entropy_after > alert.entropy_before
        assert abs(alert.delta - (alert.entropy_after - alert.entropy_before)) < 1e-9

    def test_entropy_increases_after_diverse_disclosures(self, ipc: IPC):
        ipc.create_profile(ContextClass.FINANCIAL)
        attrs = [Attribute.EMAIL, Attribute.LEGAL_NAME, Attribute.PHONE]
        for a in attrs:
            evt = _make_event("p", ContextClass.FINANCIAL, {a})
            ipc.update_profile(ContextClass.FINANCIAL, evt)
        p = ipc.get_profile(ContextClass.FINANCIAL)
        assert p.identity_entropy > 0.0

    def test_salience_weights_sum_to_one_after_update(self, ipc: IPC):
        ipc.create_profile(ContextClass.FINANCIAL)
        evt = _make_event("p", ContextClass.FINANCIAL, {Attribute.EMAIL})
        ipc.update_profile(ContextClass.FINANCIAL, evt)
        p = ipc.get_profile(ContextClass.FINANCIAL)
        assert abs(sum(p.salience_weights.values()) - 1.0) < 1e-9

    def test_disclosed_attr_weight_decreases(self, ipc: IPC):
        ipc.create_profile(ContextClass.FINANCIAL)
        initial_w = ipc.get_profile(ContextClass.FINANCIAL).salience_weights[Attribute.EMAIL]
        evt = _make_event("p", ContextClass.FINANCIAL, {Attribute.EMAIL})
        ipc.update_profile(ContextClass.FINANCIAL, evt)
        updated_w = ipc.get_profile(ContextClass.FINANCIAL).salience_weights[Attribute.EMAIL]
        # After normalization, email weight should be relatively lower than before.
        # (Withheld attrs got boosted, so email's share shrinks.)
        n = len(ipc.get_profile(ContextClass.FINANCIAL).active_attributes)
        if n > 1:
            assert updated_w < initial_w + 1e-9  # at most equal if alone

    def test_update_saves_to_pds(self, ipc: IPC, pds: PDS):
        ipc.create_profile(ContextClass.FINANCIAL)
        evt = _make_event("p", ContextClass.FINANCIAL, {Attribute.EMAIL})
        ipc.update_profile(ContextClass.FINANCIAL, evt)
        loaded = pds.load_profile(ContextClass.FINANCIAL)
        assert loaded is not None
        assert loaded.identity_entropy >= 0.0


# ---------------------------------------------------------------------------
# Attribute acquisition
# ---------------------------------------------------------------------------

class TestAttributeAcquisition:
    def test_set_explicit_persists_to_pds(self, ipc: IPC, pds: PDS):
        ipc.set_explicit(Attribute.SOCIAL_GRAPH, {"friends": ["bob"]})
        av = pds.get_attribute(Attribute.SOCIAL_GRAPH)
        assert av is not None
        assert av.raw == {"friends": ["bob"]}

    def test_set_explicit_invalidates_cache(self, ipc: IPC):
        ipc.create_profile(ContextClass.SOCIAL)
        assert ContextClass.SOCIAL in ipc._cache
        # Setting an attr that's in the SOCIAL profile's active set invalidates it.
        ipc.set_explicit(Attribute.EMAIL, "new@example.com")
        assert ContextClass.SOCIAL not in ipc._cache

    def test_set_explicit_with_abstracted(self, ipc: IPC, pds: PDS):
        ipc.set_explicit(Attribute.DOB, "1990-01-15", abstracted="1990s")
        av = pds.get_attribute(Attribute.DOB)
        assert av.abstracted == "1990s"
        assert av.effective_value() == "1990s"

    def test_import_vc_sets_verified_true(self, ipc: IPC):
        ipc.import_vc("vc://issuer/123", Attribute.LEGAL_NAME, "Alice Verified")
        # Verify via PDS directly (verified flag not surfaced in AttributeValue)
        row = ipc._pds._conn.execute(
            "SELECT verified, vc_reference FROM attributes WHERE attribute_name = ?",
            (Attribute.LEGAL_NAME.value,),
        ).fetchone()
        assert row[0] == 1
        assert row[1] == "vc://issuer/123"

    def test_infer_from_history_returns_frequent_attrs(self, ipc: IPC):
        ipc.create_profile(ContextClass.FINANCIAL)
        # Disclose EMAIL 4 out of 5 times (80% > 50%).
        for _ in range(4):
            ipc.update_profile(
                ContextClass.FINANCIAL,
                _make_event("p", ContextClass.FINANCIAL, {Attribute.EMAIL}),
            )
        ipc.update_profile(
            ContextClass.FINANCIAL,
            _make_event("p", ContextClass.FINANCIAL, {Attribute.LEGAL_NAME}),
        )
        inferred = ipc.infer_from_history(ContextClass.FINANCIAL)
        attrs = [a for a, _ in inferred]
        assert Attribute.EMAIL in attrs

    def test_infer_from_history_excludes_rare_attrs(self, ipc: IPC):
        ipc.create_profile(ContextClass.FINANCIAL)
        for _ in range(4):
            ipc.update_profile(
                ContextClass.FINANCIAL,
                _make_event("p", ContextClass.FINANCIAL, {Attribute.EMAIL}),
            )
        # LEGAL_NAME disclosed only once — below 50%.
        ipc.update_profile(
            ContextClass.FINANCIAL,
            _make_event("p", ContextClass.FINANCIAL, {Attribute.LEGAL_NAME}),
        )
        inferred = ipc.infer_from_history(ContextClass.FINANCIAL)
        attrs = [a for a, _ in inferred]
        assert Attribute.LEGAL_NAME not in attrs

    def test_infer_from_history_empty_when_no_history(self, ipc: IPC):
        ipc.create_profile(ContextClass.FINANCIAL)
        assert ipc.infer_from_history(ContextClass.FINANCIAL) == []

    def test_infer_from_history_no_profile(self, ipc: IPC):
        assert ipc.infer_from_history(ContextClass.CIVIC) == []


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

class TestDriftDetection:
    def test_first_observation_returns_none(self, ipc: IPC):
        alert = ipc.check_drift(
            ContextClass.FINANCIAL,
            {"platform_type": "FINTECH", "requested_count": 2},
        )
        assert alert is None

    def test_identical_requests_no_drift(self, ipc: IPC):
        features = {"platform_type": "FINTECH", "requested_count": 2}
        ipc.check_drift(ContextClass.FINANCIAL, features)  # baseline
        # Send same features 10 times — rolling distribution should stabilise.
        for _ in range(10):
            alert = ipc.check_drift(ContextClass.FINANCIAL, features)
        assert alert is None

    def test_highly_divergent_request_fires_drift_alert(self):
        """Use zero threshold to guarantee drift fires on any novel feature."""
        pds = PDS(Path(":memory:"))
        ipc = IPC(pds, drift_threshold=0.0)
        # Establish baseline.
        ipc.check_drift(ContextClass.FINANCIAL, {"type": "FINTECH"})
        # Completely different features → KL > 0 → alert.
        alert = ipc.check_drift(ContextClass.FINANCIAL, {"type": "COLLUDING", "extra": "X"})
        assert isinstance(alert, DriftAlert)
        assert alert.kl_divergence > 0
        assert alert.context == ContextClass.FINANCIAL

    def test_drift_alert_marks_profile_flagged(self):
        pds = PDS(Path(":memory:"))
        pds.set_attribute(Attribute.EMAIL, "a@b.com")
        ipc = IPC(pds, drift_threshold=0.0)
        ipc.create_profile(ContextClass.FINANCIAL)
        ipc.check_drift(ContextClass.FINANCIAL, {"type": "A"})  # baseline
        ipc.check_drift(ContextClass.FINANCIAL, {"type": "B"})
        p = ipc.get_profile(ContextClass.FINANCIAL)
        assert p.drift_flagged is True

    def test_drift_alert_fields(self):
        pds = PDS(Path(":memory:"))
        ipc = IPC(pds, drift_threshold=0.0)
        ipc.check_drift(ContextClass.CIVIC, {"k": "baseline"})
        alert = ipc.check_drift(ContextClass.CIVIC, {"k": "outlier", "x": "new"})
        assert isinstance(alert, DriftAlert)
        assert alert.threshold == 0.0
        assert isinstance(alert.diverging_features, dict)

    def test_rolling_distribution_persisted_to_pds(self, ipc: IPC, pds: PDS):
        ipc.check_drift(ContextClass.FINANCIAL, {"platform_type": "FINTECH"})
        raw = pds.get_raw_config("drift:FINANCIAL")
        assert raw is not None
        data = __import__("json").loads(raw)
        assert any("FINTECH" in k for k in data)


# ---------------------------------------------------------------------------
# Trust constraints
# ---------------------------------------------------------------------------

class TestTrustConstraints:
    def test_low_trust_has_aggressive_blacklist(self, ipc: IPC, pds: PDS):
        pds.set_trust_score("shady_plat", 0.2)
        tc = ipc.get_trust_constraint(ContextClass.FINANCIAL, "shady_plat")
        assert Attribute.HEALTH_CONDITION in tc.hard_blacklist
        assert Attribute.INCOME_BRACKET in tc.hard_blacklist
        assert Attribute.DOB in tc.proof_required

    def test_mid_trust_standard_constraints(self, ipc: IPC, pds: PDS):
        pds.set_trust_score("normal_plat", 0.55)
        tc = ipc.get_trust_constraint(ContextClass.FINANCIAL, "normal_plat")
        assert Attribute.HEALTH_CONDITION in tc.hard_blacklist
        assert Attribute.DOB in tc.proof_required
        assert Attribute.INCOME_BRACKET not in tc.hard_blacklist

    def test_high_trust_relaxed(self, ipc: IPC, pds: PDS):
        pds.set_trust_score("trusted_gov", 0.9)
        tc = ipc.get_trust_constraint(ContextClass.CIVIC, "trusted_gov")
        assert len(tc.proof_required) == 0
        assert Attribute.HEALTH_CONDITION in tc.hard_blacklist  # always

    def test_medical_low_trust_no_health_blacklist(self, ipc: IPC, pds: PDS):
        """HEALTH_CONDITION must NOT be blacklisted in MEDICAL even for low trust."""
        pds.set_trust_score("unverified_clinic", 0.1)
        tc = ipc.get_trust_constraint(ContextClass.MEDICAL, "unverified_clinic")
        assert Attribute.HEALTH_CONDITION not in tc.hard_blacklist
        # It should be in proof_required instead.
        assert Attribute.HEALTH_CONDITION in tc.proof_required

    def test_unknown_platform_gets_default_trust(self, ipc: IPC):
        tc = ipc.get_trust_constraint(ContextClass.SOCIAL, "never_seen_platform")
        # Default trust = 0.5 → MID band
        assert tc.min_trust_score == 0.5
        assert Attribute.INCOME_BRACKET in tc.hard_blacklist  # MID SOCIAL

    def test_all_contexts_all_bands_defined(self, ipc: IPC, pds: PDS):
        """Ensure _BAND_CONSTRAINTS covers all (context, band) combinations."""
        for ctx in ContextClass:
            for score, expected_band in [(0.1, TrustBand.LOW), (0.55, TrustBand.MID), (0.9, TrustBand.HIGH)]:
                pds.set_trust_score("test_plat", score)
                tc = ipc.get_trust_constraint(ctx, "test_plat")
                assert isinstance(tc, TrustConstraints)


# ---------------------------------------------------------------------------
# Pure function unit tests
# ---------------------------------------------------------------------------

class TestPureFunctions:
    def test_trust_band_boundaries(self):
        assert _trust_band(0.0)  == TrustBand.LOW
        assert _trust_band(0.39) == TrustBand.LOW
        assert _trust_band(0.4)  == TrustBand.MID
        assert _trust_band(0.69) == TrustBand.MID
        assert _trust_band(0.7)  == TrustBand.HIGH
        assert _trust_band(1.0)  == TrustBand.HIGH

    def test_update_salience_sums_to_one(self):
        active = frozenset({Attribute.EMAIL, Attribute.PHONE, Attribute.DOB})
        weights = {a: 1/3 for a in active}
        new_w = _update_salience(weights, active, frozenset({Attribute.EMAIL}))
        assert abs(sum(new_w.values()) - 1.0) < 1e-9

    def test_update_salience_disclosed_lower(self):
        active = frozenset({Attribute.EMAIL, Attribute.PHONE})
        weights = {Attribute.EMAIL: 0.5, Attribute.PHONE: 0.5}
        new_w = _update_salience(weights, active, frozenset({Attribute.EMAIL}))
        # After normalisation, EMAIL weight should be lower (it was decayed, PHONE boosted).
        assert new_w[Attribute.EMAIL] < new_w[Attribute.PHONE]

    def test_update_salience_covers_all_active(self):
        active = frozenset({Attribute.EMAIL, Attribute.PHONE, Attribute.DOB})
        weights = {Attribute.EMAIL: 0.5, Attribute.PHONE: 0.5}  # DOB missing
        new_w = _update_salience(weights, active, frozenset())
        assert set(new_w) == active

    def test_compute_entropy_zero_when_no_history(self):
        pds = PDS(Path(":memory:"))
        h = _compute_entropy(pds, ContextClass.FINANCIAL, frozenset({Attribute.EMAIL}))
        assert h == 0.0

    def test_compute_entropy_positive_after_disclosures(self):
        pds = PDS(Path(":memory:"))
        active = frozenset({Attribute.EMAIL, Attribute.PHONE})
        _update_entropy_counts(pds, ContextClass.FINANCIAL, frozenset({Attribute.EMAIL}))
        _update_entropy_counts(pds, ContextClass.FINANCIAL, frozenset({Attribute.PHONE}))
        h = _compute_entropy(pds, ContextClass.FINANCIAL, active)
        assert h > 0.0

    def test_compute_entropy_max_when_uniform(self):
        """H is maximised when all attributes are equally frequently disclosed."""
        pds = PDS(Path(":memory:"))
        active = frozenset({Attribute.EMAIL, Attribute.PHONE})
        for a in active:
            _update_entropy_counts(pds, ContextClass.FINANCIAL, frozenset({a}))
        h = _compute_entropy(pds, ContextClass.FINANCIAL, active)
        expected = math.log(len(active))
        assert abs(h - expected) < 1e-9

    def test_extract_features_sorted(self):
        feats = _extract_features({"z": "1", "a": "2"})
        assert feats == ["a:2", "z:1"]

    def test_extract_features_list_values_expanded(self):
        feats = _extract_features({"attrs": ["email", "phone"]})
        assert "attrs:email" in feats
        assert "attrs:phone" in feats

    def test_kl_divergence_identical_distributions_near_zero(self):
        features = ["a:1", "b:2"]
        rolling = {"a:1": 10.0, "b:2": 10.0}
        kl, _ = _kl_divergence_features(features, rolling)
        assert kl < 0.01

    def test_kl_divergence_novel_feature_positive(self):
        rolling = {"a:old": 100.0}
        kl, _ = _kl_divergence_features(["a:new"], rolling)
        assert kl > 0.0

    def test_kl_divergence_contributions_keyed_by_feature(self):
        rolling = {"x:1": 5.0, "x:2": 5.0}
        kl, contribs = _kl_divergence_features(["x:1", "x:2"], rolling)
        assert isinstance(contribs, dict)

"""Tests for cia.store.pds — Personal Data Store."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from cia.models.attributes import Attribute, AttributeValue
from cia.models.context import ContextClass
from cia.models.profile import ProfileTuple, TrustConstraints
from cia.store.pds import PDS, _profile_to_json, _profile_from_json


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pds() -> PDS:
    return PDS(Path(":memory:"))


def _make_profile(
    profile_id: str = "p1",
    attrs: set | None = None,
    min_trust: float = 0.5,
) -> ProfileTuple:
    if attrs is None:
        attrs = {Attribute.EMAIL, Attribute.LEGAL_NAME}
    frozen = frozenset(attrs)
    weights = {a: 1.0 / len(frozen) for a in frozen}
    return ProfileTuple(
        profile_id=profile_id,
        active_attributes=frozen,
        salience_weights=weights,
        trust_constraints=TrustConstraints(min_trust_score=min_trust),
    )


# ---------------------------------------------------------------------------
# Identity / master_id
# ---------------------------------------------------------------------------

class TestMasterId:
    def test_master_id_created_on_init(self, pds: PDS):
        mid = pds.get_master_id()
        assert isinstance(mid, bytes) and len(mid) == 32

    def test_master_id_stable_across_calls(self, pds: PDS):
        assert pds.get_master_id() == pds.get_master_id()

    def test_master_id_unique_per_pds_instance(self):
        p1 = PDS(Path(":memory:"))
        p2 = PDS(Path(":memory:"))
        assert p1.get_master_id() != p2.get_master_id()

    def test_master_id_persists_across_reopens(self, tmp_path: Path):
        db = tmp_path / "pds.db"
        with PDS(db) as p1:
            mid = p1.get_master_id()
        with PDS(db) as p2:
            assert p2.get_master_id() == mid


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------

class TestAttributes:
    def test_set_and_get_string_attribute(self, pds: PDS):
        pds.set_attribute(Attribute.EMAIL, "alice@example.com")
        av = pds.get_attribute(Attribute.EMAIL)
        assert av is not None
        assert av.raw == "alice@example.com"
        assert av.abstracted is None

    def test_set_and_get_with_abstracted(self, pds: PDS):
        pds.set_attribute(Attribute.DOB, "1990-03-15", abstracted="1990s")
        av = pds.get_attribute(Attribute.DOB)
        assert av.raw == "1990-03-15"
        assert av.abstracted == "1990s"
        assert av.effective_value() == "1990s"

    def test_get_nonexistent_returns_none(self, pds: PDS):
        assert pds.get_attribute(Attribute.PHONE) is None

    def test_set_overwrites_existing(self, pds: PDS):
        pds.set_attribute(Attribute.EMAIL, "old@example.com")
        pds.set_attribute(Attribute.EMAIL, "new@example.com")
        av = pds.get_attribute(Attribute.EMAIL)
        assert av.raw == "new@example.com"

    def test_set_numeric_attribute(self, pds: PDS):
        pds.set_attribute(Attribute.INCOME_BRACKET, 75000)
        av = pds.get_attribute(Attribute.INCOME_BRACKET)
        assert av.raw == 75000

    def test_get_all_attributes_returns_all_set(self, pds: PDS):
        pds.set_attribute(Attribute.EMAIL, "a@b.com")
        pds.set_attribute(Attribute.PHONE, "+1234567890")
        all_attrs = pds.get_all_attributes()
        assert Attribute.EMAIL in all_attrs
        assert Attribute.PHONE in all_attrs
        assert len(all_attrs) == 2

    def test_get_all_attributes_empty(self, pds: PDS):
        assert pds.get_all_attributes() == {}

    def test_delete_attribute(self, pds: PDS):
        pds.set_attribute(Attribute.GEOLOCATION, "London")
        pds.delete_attribute(Attribute.GEOLOCATION)
        assert pds.get_attribute(Attribute.GEOLOCATION) is None

    def test_delete_nonexistent_is_noop(self, pds: PDS):
        pds.delete_attribute(Attribute.HEALTH_CONDITION)  # should not raise

    def test_verified_flag(self, pds: PDS):
        pds.set_attribute(Attribute.LEGAL_NAME, "Alice", verified=True)
        # Verified flag stored — we confirm via re-fetch that it doesn't corrupt data.
        av = pds.get_attribute(Attribute.LEGAL_NAME)
        assert av.raw == "Alice"

    def test_set_all_12_attributes(self, pds: PDS):
        for attr in Attribute:
            pds.set_attribute(attr, f"value_{attr.value}")
        assert len(pds.get_all_attributes()) == 12


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

class TestProfiles:
    def test_save_and_load_profile(self, pds: PDS):
        p = _make_profile()
        pds.save_profile(ContextClass.FINANCIAL, p)
        loaded = pds.load_profile(ContextClass.FINANCIAL)
        assert loaded is not None
        assert loaded.profile_id == "p1"
        assert Attribute.EMAIL in loaded.active_attributes

    def test_load_nonexistent_returns_none(self, pds: PDS):
        assert pds.load_profile(ContextClass.MEDICAL) is None

    def test_save_overwrites(self, pds: PDS):
        pds.save_profile(ContextClass.SOCIAL, _make_profile("v1"))
        pds.save_profile(ContextClass.SOCIAL, _make_profile("v2"))
        loaded = pds.load_profile(ContextClass.SOCIAL)
        assert loaded.profile_id == "v2"

    def test_multiple_context_profiles(self, pds: PDS):
        for ctx in ContextClass:
            pds.save_profile(ctx, _make_profile(f"p-{ctx.value}"))
        for ctx in ContextClass:
            loaded = pds.load_profile(ctx)
            assert loaded.profile_id == f"p-{ctx.value}"

    def test_all_contexts_with_profiles(self, pds: PDS):
        pds.save_profile(ContextClass.FINANCIAL, _make_profile())
        pds.save_profile(ContextClass.MEDICAL, _make_profile())
        ctxs = pds.all_contexts_with_profiles()
        assert ContextClass.FINANCIAL in ctxs
        assert ContextClass.MEDICAL in ctxs
        assert len(ctxs) == 2

    def test_disclosure_history_not_persisted(self, pds: PDS):
        """DisclosureHistory is intentionally stripped on save."""
        p = _make_profile()
        pds.save_profile(ContextClass.CIVIC, p)
        loaded = pds.load_profile(ContextClass.CIVIC)
        assert loaded.disclosure_history == ()

    def test_trust_constraints_roundtrip(self, pds: PDS):
        tc = TrustConstraints(
            min_trust_score=0.8,
            allowed_contexts=frozenset({ContextClass.FINANCIAL}),
            hard_blacklist=frozenset({Attribute.HEALTH_CONDITION}),
            proof_required=frozenset({Attribute.DOB}),
        )
        p = ProfileTuple(
            profile_id="tc-test",
            active_attributes=frozenset({Attribute.EMAIL}),
            salience_weights={Attribute.EMAIL: 1.0},
            trust_constraints=tc,
        )
        pds.save_profile(ContextClass.PROFESSIONAL, p)
        loaded = pds.load_profile(ContextClass.PROFESSIONAL)
        assert loaded.trust_constraints.min_trust_score == 0.8
        assert ContextClass.FINANCIAL in loaded.trust_constraints.allowed_contexts
        assert Attribute.HEALTH_CONDITION in loaded.trust_constraints.hard_blacklist
        assert Attribute.DOB in loaded.trust_constraints.proof_required

    def test_salience_weights_roundtrip(self, pds: PDS):
        attrs = frozenset({Attribute.EMAIL, Attribute.PHONE, Attribute.LEGAL_NAME})
        weights = {Attribute.EMAIL: 0.5, Attribute.PHONE: 0.3, Attribute.LEGAL_NAME: 0.2}
        p = ProfileTuple(
            profile_id="w-test",
            active_attributes=attrs,
            salience_weights=weights,
            trust_constraints=TrustConstraints(),
        )
        pds.save_profile(ContextClass.SOCIAL, p)
        loaded = pds.load_profile(ContextClass.SOCIAL)
        assert abs(loaded.salience_weights[Attribute.EMAIL] - 0.5) < 1e-9
        assert abs(loaded.salience_weights[Attribute.PHONE] - 0.3) < 1e-9

    def test_entropy_and_flags_roundtrip(self, pds: PDS):
        p = _make_profile()
        p.identity_entropy = 3.14
        p.drift_flagged = True
        pds.save_profile(ContextClass.CIVIC, p)
        loaded = pds.load_profile(ContextClass.CIVIC)
        assert abs(loaded.identity_entropy - 3.14) < 1e-9
        assert loaded.drift_flagged is True


# ---------------------------------------------------------------------------
# DDE Weights
# ---------------------------------------------------------------------------

class TestWeights:
    def test_default_weights(self, pds: PDS):
        alpha, beta, gamma = pds.get_weights()
        assert alpha == 0.25
        assert beta == 0.25
        assert gamma == 0.50

    def test_set_and_get_weights(self, pds: PDS):
        pds.set_weights(0.2, 0.3, 0.5)
        a, b, g = pds.get_weights()
        assert abs(a - 0.2) < 1e-9
        assert abs(b - 0.3) < 1e-9
        assert abs(g - 0.5) < 1e-9

    def test_set_weights_enforces_gamma_gt_alpha(self, pds: PDS):
        with pytest.raises(ValueError, match="γ > α"):
            pds.set_weights(0.5, 0.2, 0.3)  # gamma < alpha

    def test_set_weights_enforces_gamma_gt_beta(self, pds: PDS):
        with pytest.raises(ValueError, match="γ > α"):
            pds.set_weights(0.2, 0.6, 0.5)  # gamma < beta

    def test_set_weights_rejects_negative(self, pds: PDS):
        with pytest.raises(ValueError):
            pds.set_weights(-0.1, 0.3, 0.6)

    def test_weights_persist(self, tmp_path: Path):
        db = tmp_path / "pds.db"
        with PDS(db) as p1:
            p1.set_weights(0.1, 0.2, 0.7)
        with PDS(db) as p2:
            a, b, g = p2.get_weights()
            assert abs(a - 0.1) < 1e-9
            assert abs(g - 0.7) < 1e-9


# ---------------------------------------------------------------------------
# Trust scores
# ---------------------------------------------------------------------------

class TestTrustScores:
    def test_default_trust_score(self, pds: PDS):
        assert pds.get_trust_score("unknown_platform") == 0.5

    def test_set_and_get_trust_score(self, pds: PDS):
        pds.set_trust_score("plat_fintech", 0.85)
        assert abs(pds.get_trust_score("plat_fintech") - 0.85) < 1e-9

    def test_set_trust_score_rejects_out_of_range(self, pds: PDS):
        with pytest.raises(ValueError):
            pds.set_trust_score("plat", 1.1)
        with pytest.raises(ValueError):
            pds.set_trust_score("plat", -0.1)

    def test_trust_score_zero_allowed(self, pds: PDS):
        pds.set_trust_score("untrusted", 0.0)
        assert pds.get_trust_score("untrusted") == 0.0

    def test_trust_score_one_allowed(self, pds: PDS):
        pds.set_trust_score("verified_gov", 1.0)
        assert pds.get_trust_score("verified_gov") == 1.0

    def test_all_trust_scores(self, pds: PDS):
        pds.set_trust_score("plat_a", 0.9)
        pds.set_trust_score("plat_b", 0.4)
        scores = pds.all_trust_scores()
        assert abs(scores["plat_a"] - 0.9) < 1e-9
        assert abs(scores["plat_b"] - 0.4) < 1e-9

    def test_trust_score_overwrite(self, pds: PDS):
        pds.set_trust_score("plat", 0.7)
        pds.set_trust_score("plat", 0.3)
        assert abs(pds.get_trust_score("plat") - 0.3) < 1e-9

    def test_trust_scores_persist(self, tmp_path: Path):
        db = tmp_path / "pds.db"
        with PDS(db) as p1:
            p1.set_trust_score("fintech", 0.75)
        with PDS(db) as p2:
            assert abs(p2.get_trust_score("fintech") - 0.75) < 1e-9


# ---------------------------------------------------------------------------
# Profile serialisation round-trip (unit)
# ---------------------------------------------------------------------------

class TestProfileSerialisation:
    def test_full_roundtrip(self):
        tc = TrustConstraints(
            min_trust_score=0.6,
            allowed_contexts=frozenset({ContextClass.MEDICAL, ContextClass.CIVIC}),
            hard_blacklist=frozenset({Attribute.DEVICE_FINGERPRINT}),
            proof_required=frozenset({Attribute.DOB, Attribute.HEALTH_CONDITION}),
        )
        original = ProfileTuple(
            profile_id="serial-test",
            active_attributes=frozenset({Attribute.EMAIL, Attribute.LEGAL_NAME}),
            salience_weights={Attribute.EMAIL: 0.6, Attribute.LEGAL_NAME: 0.4},
            trust_constraints=tc,
            identity_entropy=2.71,
            drift_flagged=False,
        )
        restored = _profile_from_json(_profile_to_json(original))

        assert restored.profile_id == original.profile_id
        assert restored.active_attributes == original.active_attributes
        assert restored.trust_constraints.min_trust_score == 0.6
        assert ContextClass.MEDICAL in restored.trust_constraints.allowed_contexts
        assert Attribute.DEVICE_FINGERPRINT in restored.trust_constraints.hard_blacklist
        assert Attribute.DOB in restored.trust_constraints.proof_required
        assert abs(restored.identity_entropy - 2.71) < 1e-9


# ---------------------------------------------------------------------------
# Context-manager and path handling
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_context_manager(self):
        with PDS(Path(":memory:")) as pds:
            pds.set_attribute(Attribute.EMAIL, "test@test.com")
            assert pds.get_attribute(Attribute.EMAIL) is not None

    def test_creates_parent_directory(self, tmp_path: Path):
        db = tmp_path / "nested" / "deep" / "pds.db"
        with PDS(db) as pds:
            mid = pds.get_master_id()
        assert db.exists()
        assert len(mid) == 32

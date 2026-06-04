"""
Tests for cia.cc — Context Classifier.

All unit tests run with download_model=False (symbolic-only mode) so they
require no network access and no ONNX model file.

Neural-layer tests are skipped automatically when cc_model.onnx is absent.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cia.models.context import ClassifiedContext, ContextClass, ContextSignal
from cia.models.request import DisclosureRequest, PlatformType
from cia.models.attributes import Attribute
from cia.ipc import DriftAlert
from cia.cc import (
    ContextClassifier,
    ModelNotAvailableError,
    _build_text,
    _PLATFORM_CEILING,
    _PRIVACY_ORDER,
    _PRIVACY_RANK,
    _CTX_ORDER,
    _NO_PURPOSE_CONFIDENCE_CAP,
    _PURPOSE_EXEMPT,
    _DRIFT_PENALTY,
)
from datetime import datetime, timezone

_ASSETS_DIR = Path(__file__).parent.parent / "cia" / "assets"
_MODEL_PRESENT = (_ASSETS_DIR / "cc_model.onnx").exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _req(
    platform_type: PlatformType = PlatformType.FINTECH,
    purpose: str = "loan application",
    attrs: set[Attribute] | None = None,
    trust: float = 0.85,
) -> DisclosureRequest:
    return DisclosureRequest(
        request_id="r1",
        platform_id="plat-1",
        platform_type=platform_type,
        platform_trust_score=trust,
        requested_attributes=frozenset(attrs or {Attribute.EMAIL, Attribute.INCOME_BRACKET}),
        context_signal=ContextSignal(description=purpose),
    )


_EMPTY_ASSETS = Path(__file__).parent / "_cc_empty_assets"
_EMPTY_ASSETS.mkdir(exist_ok=True)


def _sym_cc(tau: float = 0.80) -> ContextClassifier:
    """Symbolic-only classifier — uses an empty assets dir so ONNX is never found."""
    return ContextClassifier(_EMPTY_ASSETS, tau=tau, download_model=False)


# ---------------------------------------------------------------------------
# Symbolic layer
# ---------------------------------------------------------------------------

class TestSymbolicLayer:
    def test_fintech_ceiling_is_financial(self):
        cc = _sym_cc()
        ctx, conf = cc._symbolic(_req(PlatformType.FINTECH))
        assert ctx == ContextClass.FINANCIAL
        assert conf == pytest.approx(0.90)

    def test_employer_ceiling_is_professional(self):
        cc = _sym_cc()
        ctx, _ = cc._symbolic(_req(PlatformType.EMPLOYER))
        assert ctx == ContextClass.PROFESSIONAL

    def test_social_network_ceiling_is_social(self):
        cc = _sym_cc()
        ctx, _ = cc._symbolic(_req(PlatformType.SOCIAL_NETWORK))
        assert ctx == ContextClass.SOCIAL

    def test_healthcare_ceiling_is_medical(self):
        cc = _sym_cc()
        ctx, _ = cc._symbolic(_req(PlatformType.HEALTHCARE))
        assert ctx == ContextClass.MEDICAL

    def test_government_ceiling_is_civic(self):
        cc = _sym_cc()
        ctx, _ = cc._symbolic(_req(PlatformType.GOVERNMENT))
        assert ctx == ContextClass.CIVIC

    def test_all_platforms_have_ceiling(self):
        cc = _sym_cc()
        for pt in PlatformType:
            ctx, conf = cc._symbolic(_req(pt))
            assert isinstance(ctx, ContextClass)
            assert 0.0 <= conf <= 1.0

    def test_empty_purpose_non_exempt_platform_clamped(self):
        cc = _sym_cc()
        _, conf = cc._symbolic(_req(PlatformType.EMPLOYER, purpose=""))
        assert conf <= _NO_PURPOSE_CONFIDENCE_CAP

    def test_empty_purpose_fintech_not_clamped(self):
        """FINTECH is exempt: no purpose → still high confidence."""
        cc = _sym_cc()
        _, conf = cc._symbolic(_req(PlatformType.FINTECH, purpose=""))
        assert conf > _NO_PURPOSE_CONFIDENCE_CAP

    def test_empty_purpose_healthcare_not_clamped(self):
        cc = _sym_cc()
        _, conf = cc._symbolic(_req(PlatformType.HEALTHCARE, purpose=""))
        assert conf > _NO_PURPOSE_CONFIDENCE_CAP

    def test_empty_purpose_government_clamped(self):
        cc = _sym_cc()
        _, conf = cc._symbolic(_req(PlatformType.GOVERNMENT, purpose=""))
        assert conf <= _NO_PURPOSE_CONFIDENCE_CAP

    def test_empty_purpose_social_clamped(self):
        cc = _sym_cc()
        _, conf = cc._symbolic(_req(PlatformType.SOCIAL_NETWORK, purpose=""))
        assert conf <= _NO_PURPOSE_CONFIDENCE_CAP


# ---------------------------------------------------------------------------
# classify() — symbolic-only mode
# ---------------------------------------------------------------------------

class TestClassifySymbolicOnly:
    def test_returns_classified_context(self):
        cc = _sym_cc()
        result = cc.classify(_req())
        assert isinstance(result, ClassifiedContext)

    def test_classify_fintech_returns_financial(self):
        cc = _sym_cc()
        result = cc.classify(_req(PlatformType.FINTECH, purpose="credit check"))
        # With purpose present and high symbolic confidence, no conservative mode.
        assert result.context_class == ContextClass.FINANCIAL

    def test_class_probabilities_sum_to_one(self):
        cc = _sym_cc()
        result = cc.classify(_req())
        total = sum(result.class_probabilities.values())
        assert abs(total - 1.0) < 1e-6

    def test_class_probabilities_all_non_negative(self):
        cc = _sym_cc()
        result = cc.classify(_req())
        for v in result.class_probabilities.values():
            assert v >= 0.0

    def test_source_is_symbolic_when_no_onnx(self):
        cc = _sym_cc()
        assert cc.classify(_req()).source == "symbolic"

    def test_conservative_mode_when_confidence_below_tau(self):
        """Empty purpose on EMPLOYER clamps sym_conf ≤ 0.65 < τ=0.80."""
        cc = _sym_cc(tau=0.80)
        result = cc.classify(_req(PlatformType.EMPLOYER, purpose=""))
        assert result.conservative_mode is True

    def test_not_conservative_when_confidence_above_tau(self):
        cc = _sym_cc(tau=0.80)
        result = cc.classify(_req(PlatformType.FINTECH, purpose="loan application"))
        assert result.conservative_mode is False
        assert result.confidence >= 0.80

    def test_conservative_mode_selects_strictest_candidate(self):
        """
        In conservative mode the classifier picks the most privacy-preserving
        class among those with non-zero probability.

        In symbolic-only mode residual probability is spread over ALL classes
        (including MEDICAL), so the most strict candidate with non-zero mass
        is MEDICAL (rank 0), not FINANCIAL.
        """
        cc = _sym_cc(tau=0.95)   # tau so high that sym_conf (0.90) < tau
        result = cc.classify(_req(PlatformType.FINTECH, purpose="x"))
        assert result.conservative_mode is True
        # Residual prob is non-zero for MEDICAL → MEDICAL is strictest candidate.
        assert result.context_class == ContextClass.MEDICAL

    def test_conservative_mode_confidence_below_tau(self):
        """Conservative mode clamps returned confidence below τ."""
        cc = _sym_cc(tau=0.95)
        result = cc.classify(_req(PlatformType.FINTECH, purpose="x"))
        assert result.conservative_mode is True
        assert result.confidence < cc._tau

    def test_conservative_mode_picks_most_strict_from_spread(self):
        """
        When residual probability is spread over all classes (symbolic-only),
        conservative mode should select the most strict class that has mass.
        In symbolic-only with MEDICAL ceiling, MEDICAL gets the highest mass.
        """
        cc = _sym_cc(tau=0.95)  # force conservative
        result = cc.classify(_req(PlatformType.HEALTHCARE, purpose="x"))
        assert result.conservative_mode is True
        # MEDICAL is most strict and has highest mass → selected.
        assert result.context_class == ContextClass.MEDICAL


# ---------------------------------------------------------------------------
# Drift penalty
# ---------------------------------------------------------------------------

class TestDriftPenalty:
    def test_drift_alert_reduces_confidence(self):
        cc = _sym_cc()
        no_drift = cc.classify(_req(PlatformType.FINTECH, purpose="loan"))
        alert = DriftAlert(
            context=ContextClass.FINANCIAL,
            kl_divergence=1.2,
            diverging_features={"platform_type:FINTECH": 0.8},
            threshold=0.5,
        )
        with_drift = cc.classify(_req(PlatformType.FINTECH, purpose="loan"), drift_alert=alert)
        # Confidence must be lower or conservative mode triggered.
        assert with_drift.confidence < no_drift.confidence or with_drift.conservative_mode

    def test_drift_alert_wrong_context_has_no_effect(self):
        """A drift alert for SOCIAL should not affect a FINANCIAL classification."""
        cc = _sym_cc(tau=0.80)
        baseline = cc.classify(_req(PlatformType.FINTECH, purpose="loan"))
        alert = DriftAlert(
            context=ContextClass.SOCIAL,    # different context
            kl_divergence=2.0,
            diverging_features={"type:social": 1.5},
            threshold=0.5,
        )
        result = cc.classify(_req(PlatformType.FINTECH, purpose="loan"), drift_alert=alert)
        assert result.conservative_mode == baseline.conservative_mode

    def test_drift_can_push_into_conservative_mode(self):
        """
        Drift reduces FINANCIAL prob by 0.10 then renormalises.

        In symbolic-only mode the residual spread means post-drift
        FINANCIAL ≈ (0.90 - 0.10) / 0.90 ≈ 0.889, so τ must exceed
        0.889 to trigger conservative mode.  We use τ = 0.91.
        """
        cc = _sym_cc(tau=0.91)
        alert = DriftAlert(
            context=ContextClass.FINANCIAL,
            kl_divergence=1.0,
            diverging_features={},
            threshold=0.5,
        )
        result = cc.classify(_req(PlatformType.FINTECH, purpose="loan"), drift_alert=alert)
        assert result.conservative_mode is True

    def test_probabilities_still_sum_to_one_after_drift(self):
        cc = _sym_cc()
        alert = DriftAlert(
            context=ContextClass.FINANCIAL,
            kl_divergence=0.8,
            diverging_features={},
            threshold=0.5,
        )
        result = cc.classify(_req(PlatformType.FINTECH, purpose="loan"), drift_alert=alert)
        total = sum(result.class_probabilities.values())
        assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Privacy order / ranked constants
# ---------------------------------------------------------------------------

class TestPrivacyOrder:
    def test_medical_is_most_strict(self):
        assert _PRIVACY_RANK[ContextClass.MEDICAL] == 0

    def test_social_is_least_strict(self):
        assert _PRIVACY_RANK[ContextClass.SOCIAL] == len(ContextClass) - 1

    def test_all_contexts_in_rank(self):
        assert set(_PRIVACY_RANK) == set(ContextClass)

    def test_privacy_order_matches_rank(self):
        for i, ctx in enumerate(_PRIVACY_ORDER):
            assert _PRIVACY_RANK[ctx] == i

    def test_ctx_order_covers_all_contexts(self):
        assert set(_CTX_ORDER) == set(ContextClass)

    def test_platform_ceiling_covers_all_platforms(self):
        assert set(_PLATFORM_CEILING) == set(PlatformType)


# ---------------------------------------------------------------------------
# build_text helper
# ---------------------------------------------------------------------------

class TestBuildText:
    def test_includes_purpose(self):
        r = _req(purpose="credit check for loan")
        t = _build_text(r)
        assert "credit check for loan" in t

    def test_includes_attribute_names(self):
        r = _req(attrs={Attribute.EMAIL, Attribute.INCOME_BRACKET})
        t = _build_text(r)
        assert "email" in t
        assert "income bracket" in t

    def test_fallback_when_empty_purpose_and_no_attrs(self):
        r = DisclosureRequest(
            request_id="r",
            platform_id="p",
            platform_type=PlatformType.FINTECH,
            platform_trust_score=0.8,
            requested_attributes=frozenset(),
            context_signal=ContextSignal(description=""),
        )
        t = _build_text(r)
        assert t  # non-empty
        assert "fintech" in t.lower()

    def test_attrs_are_space_separated(self):
        r = _req(attrs={Attribute.LEGAL_NAME})
        t = _build_text(r)
        assert "legal name" in t


# ---------------------------------------------------------------------------
# ModelNotAvailableError
# ---------------------------------------------------------------------------

class TestModelNotAvailableError:
    def test_download_failure_raises_typed_error(self, tmp_path: Path):
        """Patch urllib so the download fails; expect ModelNotAvailableError."""
        with patch("cia.cc.urllib.request.urlretrieve", side_effect=OSError("timeout")):
            with pytest.raises(ModelNotAvailableError):
                ContextClassifier(tmp_path, download_model=True)

    def test_is_runtime_error_subclass(self):
        assert issubclass(ModelNotAvailableError, RuntimeError)


# ---------------------------------------------------------------------------
# Neural layer — skipped when ONNX model not present
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _MODEL_PRESENT, reason="cc_model.onnx not downloaded")
class TestNeuralLayer:
    @pytest.fixture(scope="class")
    def cc(self):
        return ContextClassifier(_ASSETS_DIR, tau=0.80, download_model=False)

    def test_neural_session_loaded(self, cc: ContextClassifier):
        assert cc._session is not None

    def test_head_shape(self, cc: ContextClassifier):
        assert cc._head is not None
        assert cc._head.shape == (5, 384)

    def test_encode_returns_unit_vector(self, cc: ContextClassifier):
        emb = cc._encode("financial loan application")
        assert emb.shape == (384,)
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-4

    def test_neural_probs_sum_to_one(self, cc: ContextClassifier):
        probs = cc._neural(_req(PlatformType.FINTECH), ContextClass.FINANCIAL)
        assert abs(sum(probs.values()) - 1.0) < 1e-5

    def test_neural_ceiling_constraint_medical(self, cc: ContextClassifier):
        """Neural cannot assign prob to classes more strict than the ceiling."""
        probs = cc._neural(_req(PlatformType.SOCIAL_NETWORK), ContextClass.SOCIAL)
        # SOCIAL is least strict (rank 4). Nothing is above it.
        for ctx, p in probs.items():
            if _PRIVACY_RANK[ctx] < _PRIVACY_RANK[ContextClass.SOCIAL]:
                assert p == pytest.approx(0.0), f"{ctx} should be zeroed"

    def test_neural_ceiling_constraint_professional(self, cc: ContextClassifier):
        probs = cc._neural(_req(PlatformType.EMPLOYER), ContextClass.PROFESSIONAL)
        # MEDICAL and FINANCIAL (rank 0, 1) should be zeroed.
        assert probs[ContextClass.MEDICAL]    == pytest.approx(0.0)
        assert probs[ContextClass.FINANCIAL]  == pytest.approx(0.0)

    def test_classify_source_is_combined(self, cc: ContextClassifier):
        result = cc.classify(_req(PlatformType.FINTECH, purpose="credit check"))
        assert result.source == "combined"

    def test_classify_financial_request_tends_toward_financial(self, cc: ContextClassifier):
        result = cc.classify(_req(
            PlatformType.FINTECH,
            purpose="loan application income verification",
            attrs={Attribute.INCOME_BRACKET, Attribute.EMPLOYMENT_STATUS},
        ))
        # Should pick FINANCIAL with high confidence (if head is well-calibrated).
        assert result.context_class == ContextClass.FINANCIAL

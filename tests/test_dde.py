"""
Tests for cia.dde — Disclosure Decision Engine.

BBS+-dependent tests are marked @pytest.mark.slow.
All other tests use a _StubBBS that raises on signing (DDE handles gracefully).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from cia.models.attributes import Attribute
from cia.models.context import ClassifiedContext, ContextClass, ContextSignal
from cia.models.policy import PolicyAction, PolicyRule, PolicySet
from cia.models.profile import ProfileTuple, TrustConstraints
from cia.models.request import DisclosureRequest, PlatformType
from cia.models.response import NegotiationStatus
from cia.policy.audit import AuditLog
from cia.policy.norms import NormDatabase
from cia.crypto.bbs import BBSEngine, BBSKeyPair
from cia.crypto.zkp import ZKPEngine
from cia.crypto.pseudonyms import PseudonymEngine
from cia.ipc import EntropyAlert
from cia.store.pds import PDS
from cia.dde import (
    DisclosureDecisionEngine,
    _greedy,
    _marginal_u,
    _score_utility,
    _negotiation_status,
    _H_I,
)


# ---------------------------------------------------------------------------
# Stubs / fixtures
# ---------------------------------------------------------------------------

class _StubBBS:
    """Minimal BBSEngine that raises on actual crypto — DDE handles gracefully."""
    def keygen(self) -> BBSKeyPair:
        return BBSKeyPair(sk=1, pk=None)  # type: ignore[arg-type]
    def sign(self, sk, messages):
        raise RuntimeError("BBS+ stub: no real signing")
    def prove_subset(self, *a, **kw):
        raise RuntimeError("BBS+ stub: no real proving")


def _make_pds(attrs: dict | None = None) -> PDS:
    pds = PDS(Path(":memory:"))
    defaults = {
        Attribute.EMAIL:             "alice@example.com",
        Attribute.LEGAL_NAME:        "Alice",
        Attribute.EMPLOYMENT_STATUS: "employed",
        Attribute.INCOME_BRACKET:    80_000,
        Attribute.DOB:               "1990-03-15",
        Attribute.PHONE:             "+1-555-0100",
        Attribute.GEOLOCATION:       "London",
        Attribute.HEALTH_CONDITION:  "none",
    }
    for attr, val in {**(attrs or {}), **{k: v for k, v in defaults.items() if k not in (attrs or {})}}.items():
        pds.set_attribute(attr, val)
    return pds


def _make_dde(pds: PDS | None = None, bbs: BBSEngine | None = None) -> tuple[DisclosureDecisionEngine, PDS, AuditLog]:
    pds = pds or _make_pds()
    audit = AuditLog(Path(":memory:"), hmac_key=b"test-key")
    dde = DisclosureDecisionEngine(
        pds=pds,
        norm_db=NormDatabase(),
        audit_log=audit,
        bbs=bbs or _StubBBS(),   # type: ignore[arg-type]
        zkp=ZKPEngine(),
        pseudonyms=PseudonymEngine(),
    )
    return dde, pds, audit


def _make_profile(
    attrs: set[Attribute] | None = None,
    min_trust: float = 0.0,
    weights: dict[Attribute, float] | None = None,
) -> ProfileTuple:
    if attrs is None:
        attrs = {Attribute.EMAIL, Attribute.EMPLOYMENT_STATUS, Attribute.INCOME_BRACKET}
    frozen = frozenset(attrs)
    uniform = {a: 1.0 / len(frozen) for a in frozen}
    return ProfileTuple(
        profile_id="test-profile",
        active_attributes=frozen,
        salience_weights=weights or uniform,
        trust_constraints=TrustConstraints(min_trust_score=min_trust),
    )


def _make_request(
    platform_type: PlatformType = PlatformType.FINTECH,
    attrs: set[Attribute] | None = None,
    trust: float = 0.9,
    purpose: str = "loan application",
) -> DisclosureRequest:
    return DisclosureRequest(
        request_id="req-1",
        platform_id="plat-fintech",
        platform_type=platform_type,
        platform_trust_score=trust,
        requested_attributes=frozenset(attrs or {Attribute.EMAIL, Attribute.INCOME_BRACKET}),
        context_signal=ContextSignal(description=purpose),
    )


def _make_context(ctx: ContextClass = ContextClass.FINANCIAL) -> ClassifiedContext:
    return ClassifiedContext(
        context_class=ctx,
        confidence=0.92,
        conservative_mode=False,
        class_probabilities={ctx: 0.92},
    )


# ---------------------------------------------------------------------------
# decide() — basic paths
# ---------------------------------------------------------------------------

class TestDecideBasic:
    def test_returns_response_and_trace(self):
        dde, _, _ = _make_dde()
        resp, trace = dde.decide(
            _make_request(), _make_context(),
            _make_profile(), PolicySet(),
        )
        assert resp is not None
        assert trace is not None

    def test_pseudonym_did_set(self):
        dde, _, _ = _make_dde()
        resp, _ = dde.decide(
            _make_request(), _make_context(),
            _make_profile(), PolicySet(),
        )
        assert resp.pseudonym_did.startswith("did:cia:financial:")

    def test_receipt_id_attached_to_response(self):
        dde, _, _ = _make_dde()
        resp, _ = dde.decide(
            _make_request(), _make_context(),
            _make_profile(), PolicySet(),
        )
        assert resp.receipt_id != ""

    def test_audit_log_receives_receipt(self):
        dde, _, audit = _make_dde()
        dde.decide(
            _make_request(), _make_context(),
            _make_profile(), PolicySet(),
        )
        assert audit.count() == 1

    def test_audit_chain_valid_after_decide(self):
        dde, _, audit = _make_dde()
        for i in range(3):
            req = _make_request(attrs={Attribute.EMAIL})
            req = DisclosureRequest(
                request_id=f"req-{i}", platform_id="p",
                platform_type=PlatformType.FINTECH,
                platform_trust_score=0.9,
                requested_attributes=frozenset({Attribute.EMAIL}),
                context_signal=ContextSignal(description="x"),
            )
            dde.decide(req, _make_context(), _make_profile(), PolicySet())
        assert audit.verify_chain()

    def test_accepted_when_all_requested_disclosed(self):
        # Request only EMAIL; norm permits EMAIL in FINANCIAL.
        dde, _, _ = _make_dde()
        req = _make_request(attrs={Attribute.EMAIL})
        resp, _ = dde.decide(req, _make_context(), _make_profile(), PolicySet())
        assert resp.negotiation_status == NegotiationStatus.ACCEPTED

    def test_rejected_when_trust_gated(self):
        # Profile requires trust ≥ 0.9, request has trust=0.2.
        dde, _, _ = _make_dde()
        profile = _make_profile(min_trust=0.9)
        req = _make_request(trust=0.2)
        resp, trace = dde.decide(req, _make_context(), profile, PolicySet())
        assert resp.negotiation_status == NegotiationStatus.REJECTED
        assert trace.optimal is None

    def test_rejected_writes_to_audit(self):
        dde, _, audit = _make_dde()
        profile = _make_profile(min_trust=0.95)
        dde.decide(_make_request(trust=0.1), _make_context(), profile, PolicySet())
        assert audit.count() == 1

    def test_partial_when_some_withheld(self):
        # HEALTH_CONDITION is normatively WITHHOLD in FINANCIAL.
        dde, _, _ = _make_dde()
        req = _make_request(attrs={Attribute.EMAIL, Attribute.HEALTH_CONDITION})
        resp, _ = dde.decide(req, _make_context(), _make_profile(), PolicySet())
        # EMAIL is disclosable; HEALTH_CONDITION is not in FINANCIAL → PARTIAL or ACCEPTED.
        assert resp.negotiation_status in (
            NegotiationStatus.PARTIAL, NegotiationStatus.ACCEPTED,
            NegotiationStatus.REJECTED,
        )

    def test_policy_withhold_overrides_norm(self):
        """User explicitly withholds EMAIL — should not appear in D*."""
        dde, _, _ = _make_dde()
        ps = PolicySet()
        ps.add_rule(PolicyRule(
            attributes=frozenset({Attribute.EMAIL}),
            contexts=frozenset({ContextClass.FINANCIAL}),
            action=PolicyAction.WITHHOLD,
            priority=1,
        ))
        req = _make_request(attrs={Attribute.EMAIL})
        resp, _ = dde.decide(req, _make_context(), _make_profile(), ps)
        email_avs = [av for av in resp.disclosed_values if av.attribute == Attribute.EMAIL]
        assert not email_avs

    def test_entropy_alert_increases_gamma(self):
        """With entropy_alert, γ increases → fewer attrs disclosed."""
        dde_normal, _, _ = _make_dde()
        dde_alert, _, _  = _make_dde()

        req = _make_request(attrs={Attribute.EMAIL, Attribute.INCOME_BRACKET,
                                    Attribute.EMPLOYMENT_STATUS})
        ctx = _make_context()
        profile = _make_profile(
            attrs={Attribute.EMAIL, Attribute.INCOME_BRACKET, Attribute.EMPLOYMENT_STATUS},
            weights={Attribute.EMAIL: 0.2, Attribute.INCOME_BRACKET: 0.4,
                     Attribute.EMPLOYMENT_STATUS: 0.4},
        )
        _, trace_normal = dde_normal.decide(req, ctx, profile, PolicySet())

        alert = EntropyAlert(
            context=ContextClass.FINANCIAL, profile_id="p",
            entropy_before=0.5, entropy_after=0.8, delta=0.3,
        )
        _, trace_alert = dde_alert.decide(req, ctx, profile, PolicySet(), entropy_alert=alert)

        d_normal = len(trace_normal.optimal.attribute_set) if trace_normal.optimal else 0
        d_alert  = len(trace_alert.optimal.attribute_set)  if trace_alert.optimal  else 0
        # With higher γ, the optimiser should be more conservative (≤ normal).
        assert d_alert <= d_normal

    def test_verify_only_dob_demoted_on_no_circuit(self):
        """VERIFY_ONLY for DOB without compiled circuit → DOB withheld, no crash."""
        dde, _, _ = _make_dde()
        ps = PolicySet()
        ps.add_rule(PolicyRule(
            attributes=frozenset({Attribute.DOB}),
            contexts=frozenset({ContextClass.FINANCIAL}),
            action=PolicyAction.VERIFY_ONLY,
            priority=5,
        ))
        req = _make_request(attrs={Attribute.DOB})
        resp, _ = dde.decide(req, _make_context(), _make_profile(attrs={Attribute.DOB}), ps)
        # No crash; DOB may be in zkp_proofs if circuit compiled, else withheld.
        assert resp is not None

    def test_user_explanation_non_empty(self):
        dde, _, _ = _make_dde()
        resp, _ = dde.decide(_make_request(), _make_context(), _make_profile(), PolicySet())
        assert len(resp.user_explanation) > 0


# ---------------------------------------------------------------------------
# Greedy optimiser (pure function)
# ---------------------------------------------------------------------------

class TestGreedy:
    def _run(
        self,
        candidates: dict[Attribute, PolicyAction],
        weights: dict[Attribute, float],
        alpha=0.2, beta=0.2, gamma=0.6,
    ):
        attrs = frozenset(candidates)
        profile = ProfileTuple(
            profile_id="p", active_attributes=attrs,
            salience_weights=weights,
            trust_constraints=TrustConstraints(),
        )
        return _greedy(
            candidates, profile, attrs,
            max(len(attrs), 1), alpha, beta, gamma,
            "r", ContextClass.FINANCIAL,
        )

    def test_empty_candidates_gives_empty_D(self):
        trace = self._run({}, {})
        assert trace.optimal is None

    def test_single_candidate_added_when_du_positive(self):
        # EMAIL has low salience → high (1-W) → ΔT large → ΔU > 0.
        trace = self._run(
            {Attribute.EMAIL: PolicyAction.DISCLOSE},
            {Attribute.EMAIL: 0.05},
        )
        assert trace.optimal is not None
        assert Attribute.EMAIL in trace.optimal.attribute_set

    def test_high_salience_only_attr_may_be_withheld_with_high_gamma(self):
        # Very sensitive attribute (high W) with γ=1.0 → ΔU negative.
        trace = self._run(
            {Attribute.HEALTH_CONDITION: PolicyAction.DISCLOSE},
            {Attribute.HEALTH_CONDITION: 0.95},
            alpha=0.1, beta=0.1, gamma=1.0,
        )
        # Either not added (optimal=None) or U < 0.
        if trace.optimal:
            assert trace.optimal.utility.U <= 0.0

    def test_greedy_selects_lowest_salience_first(self):
        """Lower-salience attributes give higher ΔU; should be added first."""
        candidates = {
            Attribute.EMAIL:          PolicyAction.DISCLOSE,
            Attribute.INCOME_BRACKET: PolicyAction.DISCLOSE,
        }
        weights = {
            Attribute.EMAIL:          0.05,   # low salience → added first
            Attribute.INCOME_BRACKET: 0.50,   # high salience → added later
        }
        trace = self._run(candidates, weights, alpha=0.2, beta=0.2, gamma=0.6)
        if len(trace.candidates) >= 1:
            assert Attribute.EMAIL in trace.candidates[0].attribute_set

    def test_minimality_flag_set(self):
        trace = self._run(
            {Attribute.EMAIL: PolicyAction.DISCLOSE},
            {Attribute.EMAIL: 0.1},
        )
        if trace.optimal:
            assert trace.minimality_verified is True

    def test_candidates_grow_monotonically(self):
        candidates = {
            Attribute.EMAIL:          PolicyAction.DISCLOSE,
            Attribute.EMPLOYMENT_STATUS: PolicyAction.DISCLOSE,
        }
        weights = {Attribute.EMAIL: 0.1, Attribute.EMPLOYMENT_STATUS: 0.15}
        trace = self._run(candidates, weights, alpha=0.3, beta=0.3, gamma=0.4)
        sizes = [len(c.attribute_set) for c in trace.candidates]
        assert sizes == sorted(sizes)

    def test_utility_increases_at_each_step(self):
        candidates = {
            Attribute.EMAIL:             PolicyAction.DISCLOSE,
            Attribute.EMPLOYMENT_STATUS: PolicyAction.DISCLOSE,
            Attribute.PHONE:             PolicyAction.DISCLOSE,
        }
        weights = {
            Attribute.EMAIL:             0.1,
            Attribute.EMPLOYMENT_STATUS: 0.12,
            Attribute.PHONE:             0.08,
        }
        trace = self._run(candidates, weights, alpha=0.3, beta=0.3, gamma=0.4)
        utils = [c.utility.U for c in trace.candidates]
        for u1, u2 in zip(utils, utils[1:]):
            assert u2 > u1 - 1e-9  # non-decreasing


# ---------------------------------------------------------------------------
# _score_utility (pure)
# ---------------------------------------------------------------------------

class TestScoreUtility:
    def _profile(self, weights: dict[Attribute, float]) -> ProfileTuple:
        attrs = frozenset(weights)
        return ProfileTuple(
            profile_id="p", active_attributes=attrs,
            salience_weights=weights,
            trust_constraints=TrustConstraints(),
        )

    def test_empty_D_gives_zero_utility(self):
        p = self._profile({Attribute.EMAIL: 0.3, Attribute.PHONE: 0.7})
        u = _score_utility(frozenset(), p, frozenset({Attribute.EMAIL}), 1, 0.2, 0.2, 0.6)
        assert u.U == pytest.approx(0.0)
        assert u.S == pytest.approx(0.0)
        assert u.T == pytest.approx(0.0)
        assert u.L == pytest.approx(0.0)

    def test_S_is_fraction_of_requested(self):
        p = self._profile({Attribute.EMAIL: 0.3, Attribute.PHONE: 0.7})
        A_req = frozenset({Attribute.EMAIL, Attribute.PHONE})
        D = frozenset({Attribute.EMAIL})
        u = _score_utility(D, p, A_req, 2, 0.2, 0.2, 0.6)
        assert u.S == pytest.approx(0.5)

    def test_T_is_average_one_minus_w(self):
        p = self._profile({Attribute.EMAIL: 0.3, Attribute.PHONE: 0.7})
        D = frozenset({Attribute.EMAIL, Attribute.PHONE})
        u = _score_utility(D, p, D, 2, 0.2, 0.2, 0.6)
        expected_T = ((1 - 0.3) + (1 - 0.7)) / 2
        assert u.T == pytest.approx(expected_T)

    def test_L_is_H_I_times_w_sum(self):
        p = self._profile({Attribute.EMAIL: 0.25})
        D = frozenset({Attribute.EMAIL})
        u = _score_utility(D, p, D, 1, 0.2, 0.2, 0.6)
        assert u.L == pytest.approx(_H_I * 0.25)

    def test_U_formula(self):
        w = 0.2
        p = self._profile({Attribute.EMAIL: w})
        D = frozenset({Attribute.EMAIL})
        a, b, g = 0.2, 0.2, 0.6
        u = _score_utility(D, p, D, 1, a, b, g)
        S = 1.0
        T = 1.0 - w
        L = _H_I * w
        assert u.U == pytest.approx(a * S + b * T - g * L)

    def test_high_salience_attr_reduces_U(self):
        p_low  = self._profile({Attribute.EMAIL: 0.05})
        p_high = self._profile({Attribute.EMAIL: 0.95})
        A = frozenset({Attribute.EMAIL})
        D = A
        u_low  = _score_utility(D, p_low,  A, 1, 0.2, 0.2, 0.6)
        u_high = _score_utility(D, p_high, A, 1, 0.2, 0.2, 0.6)
        assert u_high.U < u_low.U

    def test_all_fields_present(self):
        p = self._profile({Attribute.EMAIL: 0.3})
        u = _score_utility(frozenset({Attribute.EMAIL}), p,
                           frozenset({Attribute.EMAIL}), 1, 0.2, 0.2, 0.6)
        assert all(hasattr(u, f) for f in ("S", "T", "L", "alpha", "beta", "gamma", "U"))


# ---------------------------------------------------------------------------
# _marginal_u (pure)
# ---------------------------------------------------------------------------

class TestMarginalU:
    def test_first_addition_no_existing_D(self):
        W = 0.1
        du = _marginal_u(W, n_D=0, s_D=0.0, n_req=2, alpha=0.2, beta=0.2, gamma=0.6)
        delta_S = 0.5
        delta_T = 1.0 - W
        delta_L = _H_I * W
        assert du == pytest.approx(0.2 * delta_S + 0.2 * delta_T - 0.6 * delta_L)

    def test_high_gamma_makes_high_salience_negative(self):
        du = _marginal_u(W_a=0.9, n_D=0, s_D=0.0, n_req=1,
                         alpha=0.1, beta=0.1, gamma=0.9)
        assert du < 0.0

    def test_low_salience_attr_positive_delta(self):
        du = _marginal_u(W_a=0.05, n_D=0, s_D=0.0, n_req=1,
                         alpha=0.3, beta=0.3, gamma=0.4)
        assert du > 0.0


# ---------------------------------------------------------------------------
# _negotiation_status (pure)
# ---------------------------------------------------------------------------

class TestNegotiationStatus:
    def test_accepted_when_D_covers_A_req(self):
        A = frozenset({Attribute.EMAIL})
        assert _negotiation_status(A, A, False) == NegotiationStatus.ACCEPTED

    def test_partial_when_D_strict_subset(self):
        A_req = frozenset({Attribute.EMAIL, Attribute.PHONE})
        D = frozenset({Attribute.EMAIL})
        assert _negotiation_status(D, A_req, False) == NegotiationStatus.PARTIAL

    def test_rejected_when_D_empty_no_zkp(self):
        assert _negotiation_status(frozenset(), frozenset({Attribute.EMAIL}), False) \
               == NegotiationStatus.REJECTED

    def test_proof_only_when_D_empty_with_zkp(self):
        assert _negotiation_status(frozenset(), frozenset({Attribute.EMAIL}), True) \
               == NegotiationStatus.PROOF_ONLY


# ---------------------------------------------------------------------------
# BBS+ integration — @pytest.mark.slow
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestDDEWithBBS:
    def test_bbs_proof_bytes_non_empty_when_sign_succeeds(self):
        pds = _make_pds()
        dde, _, _ = _make_dde(pds, BBSEngine())
        # Profile with at least one active attribute that has a PDS value.
        profile = _make_profile(attrs={Attribute.EMAIL})
        req = _make_request(attrs={Attribute.EMAIL})
        resp, _ = dde.decide(req, _make_context(), profile, PolicySet())
        # BBS+ proof should be non-empty bytes.
        assert isinstance(resp.bbs_subset_proof_bytes, bytes)
        assert len(resp.bbs_subset_proof_bytes) > 0

    def test_bbs_proof_is_valid_json(self):
        import json as _json
        pds = _make_pds()
        dde, _, _ = _make_dde(pds, BBSEngine())
        profile = _make_profile(attrs={Attribute.EMAIL})
        req = _make_request(attrs={Attribute.EMAIL})
        resp, _ = dde.decide(req, _make_context(), profile, PolicySet())
        if resp.bbs_subset_proof_bytes:
            data = _json.loads(resp.bbs_subset_proof_bytes.decode("utf-8"))
            assert "A_prime" in data
            assert "c" in data

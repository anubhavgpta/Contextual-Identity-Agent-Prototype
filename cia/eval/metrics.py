"""
Evaluation metrics for the CIA prototype (Section 9 of the paper).

Pure computation — no I/O.  CIA imports limited to models/ and policy/norms
(norms is data-only; no I/O side effects).  All inputs are pre-computed by
eval/simulator.py.

Formal definitions (§9):
  DMR       = mean(1 − |D*| / |A_req|) over traces
  L(D*)     = H(I) − H(I|D*)  in nats,  H(I) = log(12),
                H(I|D*) = log(12 − |D*|) if 12−|D*|>0 else 0
  Linkability = fraction of (interaction, platform-pair) tuples
                where |D*_a ∩ D*_b| ≥ 2
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from itertools import combinations
from typing import Sequence

from cia.models.attributes import Attribute, AttributeSet, AttributeVocab
from cia.models.context import ContextClass
from cia.models.request import RequestTier
from cia.models.response import NegotiationStatus


# ---------------------------------------------------------------------------
# Shared data types — imported by simulator.py
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TraceResult:
    """Record of a single IACP interaction in the simulation."""

    platform: str
    tier: RequestTier
    requested: AttributeSet
    disclosed: AttributeSet
    latency_ms: float
    status: NegotiationStatus
    context_classified: ContextClass
    conservative_mode_triggered: bool


@dataclass(frozen=True)
class ClassificationResult:
    """Ground-truth vs CC-classified context for one interaction."""

    expected_context: ContextClass
    actual_context: ContextClass


@dataclass
class SimSession:
    """All traces from one (platform, tier) combination (typically 500 items)."""

    platform: str
    tier: RequestTier
    traces: list[TraceResult] = field(default_factory=list)


@dataclass(frozen=True)
class BaselineComparison:
    scenario: str
    # Disclosure minimisation rate (higher = more private)
    cia_dmr: float
    oauth_dmr: float        # always 0.0
    ssi_dmr: float          # modelled from norm DB
    unmediated_dmr: float   # always 0.0
    # Re-identification risk in nats (lower = more private)
    cia_reid_risk: float
    oauth_reid_risk: float
    ssi_reid_risk: float
    unmediated_reid_risk: float
    # Cross-platform linkability fraction (lower = more private)
    cia_linkability: float
    oauth_linkability: float  # ≈ 1.0
    ssi_linkability: float


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALL_ATTRS: AttributeSet = frozenset(AttributeVocab)
_N_ATTRS: int = len(_ALL_ATTRS)          # 12
_H_I: float = math.log(_N_ATTRS)        # log(12) nats


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def disclosure_minimisation_rate(
    disclosed: Sequence[AttributeSet],
    requested: Sequence[AttributeSet],
) -> float:
    """
    Mean over traces of (1 − |D*| / |A_req|).
    Traces with |A_req| = 0 are skipped.
    """
    if not disclosed:
        return 0.0
    rates = [
        1.0 - len(d) / len(r)
        for d, r in zip(disclosed, requested)
        if r
    ]
    return statistics.mean(rates) if rates else 0.0


def reidentification_risk(
    disclosed: Sequence[AttributeSet],
    all_attributes: AttributeSet = _ALL_ATTRS,
) -> float:
    """
    Mean L(D*) = H(I) − H(I|D*) in nats.
      H(I)     = log(|all_attributes|)
      H(I|D*)  = log(|all_attributes| − |D*|)  when > 0,  else 0
    """
    if not disclosed:
        return 0.0
    n = len(all_attributes)
    if n == 0:
        return 0.0
    h_i = math.log(n)
    risks = []
    for d in disclosed:
        remaining = n - len(d)
        h_cond = math.log(remaining) if remaining > 0 else 0.0
        risks.append(h_i - h_cond)
    return statistics.mean(risks)


def cross_platform_linkability(sessions: Sequence[SimSession]) -> float:
    """
    Linkage oracle: for each pair of same-tier sessions with different platforms,
    check every aligned interaction index i for attribute intersection ≥ 2.
    Returns fraction of (i, pair) tuples that are linkable.
    """
    if len(sessions) < 2:
        return 0.0

    # Pair sessions: same tier, different platforms.
    pairs = [
        (a, b)
        for a, b in combinations(sessions, 2)
        if a.tier == b.tier and a.platform != b.platform
    ]

    total = linked = 0
    for s_a, s_b in pairs:
        n = min(len(s_a.traces), len(s_b.traces))
        for i in range(n):
            intersection = s_a.traces[i].disclosed & s_b.traces[i].disclosed
            total += 1
            if len(intersection) >= 2:
                linked += 1

    return linked / total if total > 0 else 0.0


def classification_accuracy(results: Sequence[ClassificationResult]) -> float:
    """Fraction where actual_context == expected_context."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.actual_context == r.expected_context) / len(results)


def decision_latency_stats(latencies_ms: Sequence[float]) -> dict[str, float]:
    """Returns mean, p50, p95, p99 over latency measurements (milliseconds)."""
    if not latencies_ms:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    s = sorted(latencies_ms)
    n = len(s)

    def _pct(p: float) -> float:
        idx = p / 100.0 * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        return s[lo] + (s[hi] - s[lo]) * (idx - lo)

    return {
        "mean": statistics.mean(s),
        "p50":  _pct(50),
        "p95":  _pct(95),
        "p99":  _pct(99),
    }


def compare_baselines(
    cia_traces: Sequence[TraceResult],
    scenario: str = "",
) -> BaselineComparison:
    """
    Compute DMR, re-id risk, and linkability for CIA vs baselines.

    OAuth / Unmediated: disclose all of A_req (DMR = 0, max re-id risk).
    SSI: disclose A_req ∩ norm_permitted(platform_context, recipient_role).
    """
    from cia.policy.norms import NormDatabase, RecipientRole

    # Platform name → (ContextClass, RecipientRole) for SSI computation.
    _P2CTX: dict[str, tuple[ContextClass, RecipientRole]] = {
        "financial_portal":    (ContextClass.FINANCIAL,    RecipientRole.FINANCIAL_INSTITUTION),
        "professional_network": (ContextClass.PROFESSIONAL, RecipientRole.EMPLOYER),
        "social_platform":     (ContextClass.SOCIAL,       RecipientRole.SOCIAL_PEER),
        "healthcare_portal":   (ContextClass.MEDICAL,      RecipientRole.HEALTHCARE_PROVIDER),
        "gaming_platform":     (ContextClass.SOCIAL,       RecipientRole.SOCIAL_PEER),
    }

    norm_db = NormDatabase()
    requested_sets = [t.requested for t in cia_traces]
    cia_disclosed  = [t.disclosed for t in cia_traces]

    # SSI: disclose norm-permitted subset of A_req.
    ssi_disclosed: list[AttributeSet] = []
    for t in cia_traces:
        ctx_role = _P2CTX.get(t.platform)
        if ctx_role is None:
            # Unknown platform: fall back to classified context + UNKNOWN role.
            ctx, role = t.context_classified, RecipientRole.UNKNOWN
        else:
            ctx, role = ctx_role
        permitted = norm_db.permitted_attributes(ctx, role)
        ssi_disclosed.append(t.requested & permitted)

    # OAuth / Unmediated disclose everything requested.
    oauth_disclosed = requested_sets

    cia_dmr  = disclosure_minimisation_rate(cia_disclosed,  requested_sets)
    ssi_dmr  = disclosure_minimisation_rate(ssi_disclosed,  requested_sets)
    oauth_dmr = 0.0

    cia_reid   = reidentification_risk(cia_disclosed)
    ssi_reid   = reidentification_risk(ssi_disclosed)
    oauth_reid = reidentification_risk(oauth_disclosed)

    cia_sessions   = _sessions_from_traces(cia_traces)
    ssi_sessions   = _sessions_from_traces_with_disclosed(cia_traces, ssi_disclosed)
    oauth_sessions = _sessions_from_traces_with_disclosed(cia_traces, oauth_disclosed)

    cia_link   = cross_platform_linkability(cia_sessions)
    ssi_link   = cross_platform_linkability(ssi_sessions)
    oauth_link = cross_platform_linkability(oauth_sessions)

    return BaselineComparison(
        scenario=scenario,
        cia_dmr=cia_dmr,
        oauth_dmr=oauth_dmr,
        ssi_dmr=ssi_dmr,
        unmediated_dmr=0.0,
        cia_reid_risk=cia_reid,
        oauth_reid_risk=oauth_reid,
        ssi_reid_risk=ssi_reid,
        unmediated_reid_risk=oauth_reid,
        cia_linkability=cia_link,
        oauth_linkability=oauth_link,
        ssi_linkability=ssi_link,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sessions_from_traces(traces: Sequence[TraceResult]) -> list[SimSession]:
    """Group traces by (platform, tier) preserving insertion order."""
    buckets: dict[tuple[str, RequestTier], SimSession] = {}
    for t in traces:
        key = (t.platform, t.tier)
        if key not in buckets:
            buckets[key] = SimSession(platform=t.platform, tier=t.tier)
        buckets[key].traces.append(t)
    return list(buckets.values())


def _sessions_from_traces_with_disclosed(
    traces: Sequence[TraceResult],
    new_disclosed: Sequence[AttributeSet],
) -> list[SimSession]:
    """Like _sessions_from_traces but replaces each trace's disclosed set."""
    modified = [
        TraceResult(
            platform=t.platform,
            tier=t.tier,
            requested=t.requested,
            disclosed=nd,
            latency_ms=t.latency_ms,
            status=t.status,
            context_classified=t.context_classified,
            conservative_mode_triggered=t.conservative_mode_triggered,
        )
        for t, nd in zip(traces, new_disclosed)
    ]
    return _sessions_from_traces(modified)

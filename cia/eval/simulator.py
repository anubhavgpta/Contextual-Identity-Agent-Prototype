"""
eval/simulator.py — orchestrates 7,500 synthetic IACP traces (§9.1).

5 platform types × 3 scenario tiers × 500 interactions = 7,500 traces.
All randomness seeded via random.seed(42) for reproducibility.

Usage:
    from cia.eval.simulator import run_simulation
    report = run_simulation()                      # stub BBS+, 500 interactions
    report = run_simulation(use_real_crypto=True)  # paper final run
    report = run_simulation(n_interactions=10)     # fast test run
"""

from __future__ import annotations

import dataclasses
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from cia.models.attributes import Attribute, AttributeSet
from cia.models.context import ClassifiedContext, ContextClass, ContextSignal
from cia.models.policy import PolicySet
from cia.models.request import DisclosureRequest, PlatformType, RequestTier
from cia.models.response import NegotiationStatus
from cia.policy.audit import AuditLog
from cia.policy.norms import NormDatabase
from cia.crypto.bbs import BBSEngine, BBSKeyPair
from cia.crypto.zkp import ZKPEngine
from cia.crypto.pseudonyms import PseudonymEngine
from cia.ipc import IPC
from cia.dde import DisclosureDecisionEngine
from cia.store.pds import PDS
from cia.iacp import IACPSession

from cia.eval.metrics import (
    BaselineComparison,
    ClassificationResult,
    SimSession,
    TraceResult,
    classification_accuracy,
    compare_baselines,
    cross_platform_linkability,
    decision_latency_stats,
    disclosure_minimisation_rate,
    reidentification_risk,
    _sessions_from_traces,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stub BBS+ (default — avoids expensive pairings in simulation)
# ---------------------------------------------------------------------------

class _StubBBS:
    """BBS+ stub: keygen succeeds, sign/prove raise RuntimeError (DDE catches)."""

    def keygen(self) -> BBSKeyPair:
        return BBSKeyPair(sk=1, pk=None)  # type: ignore[arg-type]

    def sign(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("_StubBBS: BBS+ disabled")

    def prove_subset(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("_StubBBS: BBS+ disabled")


# ---------------------------------------------------------------------------
# Tracking CC wrapper — records conservative_mode per request_id
# ---------------------------------------------------------------------------

class _CCProtocol(Protocol):
    def classify(
        self, request: DisclosureRequest, drift_alert: Any = None
    ) -> ClassifiedContext: ...


class _SharedTrackingCC:
    """
    Wraps any CC instance and records the last ClassifiedContext per request_id.
    Since IACP calls classify() twice per pipeline round (pre-drift, post-drift),
    the second call (operative classification) overwrites the first.
    Single-threaded simulation — no concurrency concern.
    """

    def __init__(self, cc: _CCProtocol) -> None:
        self._cc = cc
        self._records: dict[str, ClassifiedContext] = {}

    def classify(
        self, request: DisclosureRequest, drift_alert: Any = None
    ) -> ClassifiedContext:
        result = self._cc.classify(request, drift_alert=drift_alert)
        self._records[request.request_id] = result
        return result

    def get(self, request_id: str) -> ClassifiedContext | None:
        return self._records.get(request_id)


# ---------------------------------------------------------------------------
# Platform configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimPlatformConfig:
    name: str
    platform_type: PlatformType
    base_attrs: frozenset[Attribute]
    probing_extra: frozenset[Attribute]  # contextually inappropriate additions
    trust_score: float
    expected_context: ContextClass       # ground truth for classification accuracy
    purpose_vocab: tuple[str, ...]       # 20 legitimate purpose phrases


_PLATFORMS: tuple[SimPlatformConfig, ...] = (
    SimPlatformConfig(
        name="financial_portal",
        platform_type=PlatformType.FINTECH,
        base_attrs=frozenset({
            Attribute.LEGAL_NAME, Attribute.DOB, Attribute.EMAIL,
            Attribute.PHONE, Attribute.INCOME_BRACKET, Attribute.EMPLOYMENT_STATUS,
        }),
        probing_extra=frozenset({
            Attribute.HEALTH_CONDITION, Attribute.GAMING_HISTORY, Attribute.SOCIAL_GRAPH,
        }),
        trust_score=0.85,
        expected_context=ContextClass.FINANCIAL,
        purpose_vocab=(
            "loan application review",
            "credit assessment",
            "account opening",
            "mortgage pre-approval",
            "investment risk profiling",
            "credit limit increase",
            "identity verification for banking",
            "KYC compliance check",
            "fraud prevention screening",
            "insurance premium calculation",
            "debt consolidation request",
            "refinancing evaluation",
            "savings account setup",
            "credit card application",
            "financial advisory onboarding",
            "tax document verification",
            "wire transfer authorization",
            "payment method registration",
            "creditworthiness review",
            "bank account verification",
        ),
    ),
    SimPlatformConfig(
        name="professional_network",
        platform_type=PlatformType.EMPLOYER,
        base_attrs=frozenset({
            Attribute.LEGAL_NAME, Attribute.EMAIL,
            Attribute.EMPLOYMENT_STATUS, Attribute.SOCIAL_GRAPH,
        }),
        probing_extra=frozenset({
            Attribute.HEALTH_CONDITION, Attribute.INCOME_BRACKET,
            Attribute.DEVICE_FINGERPRINT,
        }),
        trust_score=0.70,
        expected_context=ContextClass.PROFESSIONAL,
        purpose_vocab=(
            "employment background check",
            "professional profile creation",
            "job application screening",
            "recruiter outreach",
            "skills verification",
            "employment history confirmation",
            "professional networking signup",
            "reference check",
            "talent acquisition",
            "career platform onboarding",
            "work experience validation",
            "certification verification",
            "professional endorsement",
            "business contact exchange",
            "headhunter inquiry",
            "contractor vetting",
            "freelancer registration",
            "professional portfolio review",
            "employment eligibility check",
            "workforce management setup",
        ),
    ),
    SimPlatformConfig(
        name="social_platform",
        platform_type=PlatformType.SOCIAL_NETWORK,
        base_attrs=frozenset({
            Attribute.LEGAL_NAME, Attribute.EMAIL, Attribute.GEOLOCATION,
            Attribute.SOCIAL_GRAPH, Attribute.BEHAVIOURAL_METADATA,
            Attribute.DEVICE_FINGERPRINT,
        }),
        probing_extra=frozenset({
            Attribute.HEALTH_CONDITION, Attribute.INCOME_BRACKET, Attribute.PHONE,
        }),
        trust_score=0.45,
        expected_context=ContextClass.SOCIAL,
        purpose_vocab=(
            "social account registration",
            "friend connection request",
            "profile creation",
            "event RSVP",
            "community membership",
            "content sharing setup",
            "location check-in services",
            "social login integration",
            "photo sharing setup",
            "group membership application",
            "recommendation personalisation",
            "social matching",
            "user discovery",
            "notification preferences",
            "privacy settings configuration",
            "account recovery",
            "two-factor authentication setup",
            "public profile creation",
            "newsfeed customisation",
            "social analytics setup",
        ),
    ),
    SimPlatformConfig(
        name="healthcare_portal",
        platform_type=PlatformType.HEALTHCARE,
        base_attrs=frozenset({
            Attribute.LEGAL_NAME, Attribute.DOB,
            Attribute.HEALTH_CONDITION, Attribute.GEOLOCATION,
        }),
        probing_extra=frozenset({
            Attribute.EMPLOYMENT_STATUS, Attribute.SOCIAL_GRAPH,
            Attribute.DEVICE_FINGERPRINT,
        }),
        trust_score=0.90,
        expected_context=ContextClass.MEDICAL,
        purpose_vocab=(
            "patient registration for primary care",
            "appointment booking",
            "medical history intake",
            "prescription management",
            "referral processing",
            "health insurance verification",
            "telehealth consultation setup",
            "emergency contact registration",
            "vaccination record access",
            "lab results access",
            "medication adherence tracking",
            "chronic condition management",
            "surgical pre-assessment",
            "mental health intake",
            "allergy documentation",
            "specialist referral",
            "health screening questionnaire",
            "care plan setup",
            "hospital admission process",
            "genetic counselling intake",
        ),
    ),
    SimPlatformConfig(
        name="gaming_platform",
        platform_type=PlatformType.SOCIAL_NETWORK,  # closest ceiling match
        base_attrs=frozenset({
            Attribute.EMAIL, Attribute.GAMING_HISTORY,
            Attribute.DEVICE_FINGERPRINT, Attribute.BEHAVIOURAL_METADATA,
        }),
        probing_extra=frozenset({
            Attribute.INCOME_BRACKET, Attribute.HEALTH_CONDITION,
            Attribute.GEOLOCATION,
        }),
        trust_score=0.55,
        expected_context=ContextClass.SOCIAL,
        purpose_vocab=(
            "game account creation",
            "leaderboard registration",
            "in-game purchase setup",
            "multiplayer matchmaking",
            "achievement tracking",
            "game progress sync",
            "player statistics collection",
            "competitive ranking setup",
            "friend list creation",
            "game recommendation personalisation",
            "session analytics",
            "anti-cheat verification",
            "tournament registration",
            "beta access signup",
            "loyalty program enrollment",
            "gaming community membership",
            "streaming integration",
            "controller profile setup",
            "game library sync",
            "cross-platform account linking",
        ),
    ),
)

# Vague purpose phrases used in PROBING tier (appended to standard vocab).
_PROBING_PURPOSE_ADDENDUM: tuple[str, ...] = (
    "service optimisation",
    "personalisation",
    "analytics",
    "platform improvement",
    "enhanced user experience",
)


# ---------------------------------------------------------------------------
# SimulationReport
# ---------------------------------------------------------------------------

@dataclass
class SimulationReport:
    """
    Aggregated results from one simulation run.  JSON-serialisable via to_dict().
    Table 3 of the paper is derived from this report.
    """

    total_traces: int
    n_interactions: int

    # Per-platform DMR (keyed by platform name)
    dmr_by_platform: dict[str, float]
    overall_dmr: float

    # Re-id risk reduction vs OAuth baseline, as a percentage
    reid_risk_reduction_pct: float

    # Linkability fraction by tier name
    linkability_by_tier: dict[str, float]

    # CC classification accuracy by tier name
    classification_accuracy_by_tier: dict[str, float]

    # Decision latency stats (ms) — overall
    latency_stats: dict[str, float]

    # Negotiation success rate (ACCEPTED | PARTIAL | PROOF_ONLY)
    negotiation_success_rate: float

    # Full baseline comparison
    baseline: dict[str, Any]

    # Per-platform conservative-mode fraction (all tiers combined)
    conservative_mode_by_platform: dict[str, float] = field(default_factory=dict)

    # Per-platform p95 decision latency (ms)
    latency_p95_by_platform: dict[str, float] = field(default_factory=dict)

    # DMR broken down by tier
    dmr_by_tier: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

def _populate_pds(pds: PDS) -> None:
    """Seed PDS with plausible values for all 12 attributes."""
    pds.set_attribute(Attribute.LEGAL_NAME,         "Alice Nakamura")
    pds.set_attribute(Attribute.DOB,                "1992-07-14")
    pds.set_attribute(Attribute.EMAIL,              "alice@example.com")
    pds.set_attribute(Attribute.PHONE,              "+1-555-0142")
    pds.set_attribute(Attribute.GEOLOCATION,        "37.7749,-122.4194")
    pds.set_attribute(Attribute.EMPLOYMENT_STATUS,  "full-time")
    pds.set_attribute(Attribute.INCOME_BRACKET,     "75000-100000")
    pds.set_attribute(Attribute.HEALTH_CONDITION,   "none")
    pds.set_attribute(Attribute.GAMING_HISTORY,     "casual")
    pds.set_attribute(Attribute.SOCIAL_GRAPH,       "500-connections")
    pds.set_attribute(Attribute.DEVICE_FINGERPRINT, "linux-x64-firefox")
    pds.set_attribute(Attribute.BEHAVIOURAL_METADATA, "standard")


def _make_iacp_session(
    shared_cc: _SharedTrackingCC,
    norm_db: NormDatabase,
    bbs: BBSEngine,
) -> tuple[IACPSession, PDS, AuditLog]:
    pds   = PDS(Path(":memory:"))
    _populate_pds(pds)
    audit = AuditLog(Path(":memory:"))
    ipc   = IPC(pds, norm_db)
    dde   = DisclosureDecisionEngine(
        pds=pds, norm_db=norm_db, audit_log=audit,
        bbs=bbs, zkp=ZKPEngine(), pseudonyms=PseudonymEngine(),
    )
    session = IACPSession(
        pds=pds, ipc=ipc, cc=shared_cc,  # type: ignore[arg-type]
        dde=dde, norm_db=norm_db,
        policy=PolicySet(), audit_log=audit,
    )
    return session, pds, audit


# ---------------------------------------------------------------------------
# Request builder
# ---------------------------------------------------------------------------

def _make_request(
    platform_cfg: SimPlatformConfig,
    tier: RequestTier,
    purpose: str,
    interaction_idx: int,
) -> DisclosureRequest:
    if tier == RequestTier.PROBING:
        attrs: AttributeSet = platform_cfg.base_attrs | platform_cfg.probing_extra
    else:
        attrs = platform_cfg.base_attrs

    return DisclosureRequest(
        request_id=f"{platform_cfg.name}-{tier.value}-{interaction_idx}-{uuid.uuid4().hex[:8]}",
        platform_id=f"{platform_cfg.name}.example.com",
        platform_type=platform_cfg.platform_type,
        platform_trust_score=platform_cfg.trust_score,
        requested_attributes=attrs,
        context_signal=ContextSignal(description=purpose),
        _sim_tier=tier,
    )


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def _run_one_session(
    platform_cfg: SimPlatformConfig,
    tier: RequestTier,
    session: IACPSession,
    tracking_cc: _SharedTrackingCC,
    n_interactions: int,
    rng: random.Random,
) -> tuple[list[TraceResult], list[ClassificationResult]]:
    traces: list[TraceResult] = []
    cls_results: list[ClassificationResult] = []

    for i in range(n_interactions):
        if tier == RequestTier.PROBING:
            purpose = rng.choice(
                platform_cfg.purpose_vocab + _PROBING_PURPOSE_ADDENDUM
            )
        else:
            purpose = rng.choice(platform_cfg.purpose_vocab)

        req = _make_request(platform_cfg, tier, purpose, i)

        t0 = time.perf_counter()
        result = session.handle(req, mode="agent")
        latency_ms = (time.perf_counter() - t0) * 1000.0

        disclosed: AttributeSet
        if result.response is not None:
            disclosed = frozenset(
                av.attribute for av in result.response.disclosed_values
            )
        else:
            disclosed = frozenset()

        # Operative CC classification (second classify() call, post-drift).
        cc_record = tracking_cc.get(req.request_id)
        ctx_classified = (
            cc_record.context_class
            if cc_record is not None
            else platform_cfg.expected_context
        )
        conservative = cc_record.conservative_mode if cc_record is not None else False

        traces.append(TraceResult(
            platform=platform_cfg.name,
            tier=tier,
            requested=req.requested_attributes,
            disclosed=disclosed,
            latency_ms=latency_ms,
            status=result.status,
            context_classified=ctx_classified,
            conservative_mode_triggered=conservative,
        ))
        cls_results.append(ClassificationResult(
            expected_context=platform_cfg.expected_context,
            actual_context=ctx_classified,
        ))

    return traces, cls_results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_simulation(
    n_interactions: int = 500,
    use_real_crypto: bool = False,
    assets_dir: Path | None = None,
    cc_override: Any = None,
) -> SimulationReport:
    """
    Run the full evaluation simulation.

    Parameters
    ----------
    n_interactions : int
        Interactions per (platform, tier) pair.  Paper uses 500.
    use_real_crypto : bool
        If True, use actual BBS+ pairings (slow).  Default: stub BBS+.
    assets_dir : Path | None
        Override the CC assets directory.  Defaults to cia/assets/.
    cc_override : any
        Inject a custom CC (e.g. MockCC for tests).  Skips real CC init.
    """
    random.seed(42)

    # ── CC init ────────────────────────────────────────────────────────────
    if cc_override is not None:
        raw_cc = cc_override
    else:
        from cia.cc import ContextClassifier, ModelNotAvailableError
        _assets = assets_dir or Path(__file__).parent.parent / "assets"
        try:
            raw_cc = ContextClassifier(_assets, download_model=False)
        except (ModelNotAvailableError, Exception):
            # Fall back to symbolic-only mode via an empty assets dir.
            _empty = Path(__file__).parent.parent.parent / "tests" / "_cc_empty_assets"
            _empty.mkdir(parents=True, exist_ok=True)
            raw_cc = ContextClassifier(_empty, download_model=False)

    tracking_cc = _SharedTrackingCC(raw_cc)

    bbs: BBSEngine = BBSEngine() if use_real_crypto else _StubBBS()  # type: ignore[assignment]
    norm_db = NormDatabase()
    rng = random.Random(42)

    # ── Build 15 sessions (5 platforms × 3 tiers) ─────────────────────────
    sessions_map: dict[tuple[str, RequestTier], tuple[IACPSession, PDS, AuditLog]] = {}
    for platform_cfg in _PLATFORMS:
        for tier in RequestTier:
            sessions_map[(platform_cfg.name, tier)] = _make_iacp_session(
                tracking_cc, norm_db, bbs
            )

    # ── Run interactions ───────────────────────────────────────────────────
    all_traces: list[TraceResult] = []
    all_cls: list[ClassificationResult] = []
    sim_sessions: list[SimSession] = []

    for platform_cfg in _PLATFORMS:
        for tier in RequestTier:
            iacp_session, _, _ = sessions_map[(platform_cfg.name, tier)]
            logger.info(
                "Simulating %s / %s (%d interactions)...",
                platform_cfg.name, tier.value, n_interactions,
            )
            traces, cls_res = _run_one_session(
                platform_cfg, tier, iacp_session, tracking_cc,
                n_interactions, rng,
            )
            all_traces.extend(traces)
            all_cls.extend(cls_res)
            sim_sessions.append(SimSession(
                platform=platform_cfg.name,
                tier=tier,
                traces=traces,
            ))

    # ── Aggregate metrics ──────────────────────────────────────────────────
    _SUCCESS = {NegotiationStatus.ACCEPTED, NegotiationStatus.PARTIAL,
                NegotiationStatus.PROOF_ONLY}

    # Per-platform DMR (aggregated over all tiers).
    dmr_by_platform: dict[str, float] = {}
    for platform_cfg in _PLATFORMS:
        pt = [t for t in all_traces if t.platform == platform_cfg.name]
        dmr_by_platform[platform_cfg.name] = disclosure_minimisation_rate(
            [t.disclosed for t in pt],
            [t.requested for t in pt],
        )

    overall_dmr = disclosure_minimisation_rate(
        [t.disclosed for t in all_traces],
        [t.requested for t in all_traces],
    )

    # Re-id risk reduction vs OAuth.
    cia_reid = reidentification_risk([t.disclosed for t in all_traces])
    oauth_reid = reidentification_risk([t.requested for t in all_traces])
    reid_reduction_pct = (
        (oauth_reid - cia_reid) / oauth_reid * 100.0 if oauth_reid > 0 else 0.0
    )

    # Linkability by tier.
    linkability_by_tier: dict[str, float] = {}
    for tier in RequestTier:
        tier_sessions = [s for s in sim_sessions if s.tier == tier]
        linkability_by_tier[tier.value] = cross_platform_linkability(tier_sessions)

    # Classification accuracy by tier.
    cls_by_tier: dict[str, float] = {}
    for tier in RequestTier:
        tier_cls = [
            r for t, r in zip(all_traces, all_cls) if t.tier == tier
        ]
        cls_by_tier[tier.value] = classification_accuracy(tier_cls)

    latency_stats = decision_latency_stats([t.latency_ms for t in all_traces])

    n_success = sum(1 for t in all_traces if t.status in _SUCCESS)
    negotiation_success_rate = n_success / len(all_traces) if all_traces else 0.0

    baseline = compare_baselines(all_traces, scenario="full_simulation")

    # Per-platform conservative-mode fraction and p95 latency.
    conservative_mode_by_platform: dict[str, float] = {}
    latency_p95_by_platform: dict[str, float] = {}
    for platform_cfg in _PLATFORMS:
        pt = [t for t in all_traces if t.platform == platform_cfg.name]
        conservative_mode_by_platform[platform_cfg.name] = (
            sum(1 for t in pt if t.conservative_mode_triggered) / len(pt)
            if pt else 0.0
        )
        lat_stats = decision_latency_stats([t.latency_ms for t in pt])
        latency_p95_by_platform[platform_cfg.name] = lat_stats["p95"]

    # DMR by tier.
    dmr_by_tier: dict[str, float] = {}
    for tier in RequestTier:
        tt = [t for t in all_traces if t.tier == tier]
        dmr_by_tier[tier.value] = disclosure_minimisation_rate(
            [t.disclosed for t in tt],
            [t.requested for t in tt],
        )

    return SimulationReport(
        total_traces=len(all_traces),
        n_interactions=n_interactions,
        dmr_by_platform=dmr_by_platform,
        overall_dmr=overall_dmr,
        reid_risk_reduction_pct=reid_reduction_pct,
        linkability_by_tier=linkability_by_tier,
        classification_accuracy_by_tier=cls_by_tier,
        latency_stats=latency_stats,
        negotiation_success_rate=negotiation_success_rate,
        baseline=dataclasses.asdict(baseline),
        conservative_mode_by_platform=conservative_mode_by_platform,
        latency_p95_by_platform=latency_p95_by_platform,
        dmr_by_tier=dmr_by_tier,
    )

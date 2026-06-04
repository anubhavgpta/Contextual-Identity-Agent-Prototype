"""
Identity Profile Constructor (IPC) — constructs and maintains P = {pᵢ}.

The IPC sits between the PDS (raw attribute storage) and the DDE (decision
engine).  It is the *only* component that writes ProfileTuple snapshots back
to the PDS.

Responsibilities
────────────────
  Profile management   create / get / update profiles per context class
  Entropy tracking     H(Aᵢ) over empirical disclosure distribution; fires
                       EntropyAlert when delta > entropy_threshold
  Attribute acquisition
    Channel 1  set_explicit()  — direct user input
    Channel 2  infer_from_history() — patterns from Hᵢ
    Channel 3  import_vc()    — VC-verified attribute
  Drift detection      KL-divergence over rolling request features;
                       fires DriftAlert when KL > drift_threshold
  Salience weights     Wᵢ: disclosed attrs ↓, withheld attrs ↑, normalised
  Trust constraints    trust score band → (hard_blacklist, proof_required)

Import restrictions: IPC must NOT import from cc.py, dde.py, or iacp.py.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from cia.models.attributes import Attribute
from cia.models.context import ContextClass
from cia.models.profile import (
    DisclosureEvent,
    DisclosureHistory,
    ProfileTuple,
    TrustConstraints,
)
from cia.policy.norms import NormDatabase, RecipientRole
from cia.store.pds import PDS


# ---------------------------------------------------------------------------
# Alert types emitted by the IPC
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EntropyAlert:
    """
    Fired when the identity-entropy of a profile rises by more than
    ``entropy_threshold`` nats in a single update step.

    The DDE consumes this to prompt the user for confirmation before
    proceeding with the disclosure that caused the spike.
    """

    context: ContextClass
    profile_id: str
    entropy_before: float
    entropy_after: float
    delta: float  # entropy_after − entropy_before (always > 0 when alert fires)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class DriftAlert:
    """
    Fired when the incoming request's feature distribution diverges
    significantly from the rolling baseline (KL > drift_threshold).

    cc.py consumes this to trigger norm recalibration.
    """

    context: ContextClass
    kl_divergence: float
    # Feature strings with the largest per-feature KL contribution.
    diverging_features: dict[str, float]
    threshold: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Trust-band definitions
# ---------------------------------------------------------------------------

class TrustBand(str, Enum):
    LOW  = "LOW"   # trust < 0.4 — aggressive blacklist
    MID  = "MID"   # 0.4 ≤ trust < 0.7 — standard
    HIGH = "HIGH"  # trust ≥ 0.7 — relaxed proof requirements


def _trust_band(score: float) -> TrustBand:
    if score < 0.4:
        return TrustBand.LOW
    if score < 0.7:
        return TrustBand.MID
    return TrustBand.HIGH


# Attribute sets used to build band constraints below.
_ALWAYS_SENSITIVE = frozenset({
    Attribute.HEALTH_CONDITION,
    Attribute.BEHAVIOURAL_METADATA,
    Attribute.DEVICE_FINGERPRINT,
})
_FINANCIAL_SENSITIVE = _ALWAYS_SENSITIVE | frozenset({
    Attribute.SOCIAL_GRAPH,
    Attribute.GAMING_HISTORY,
    Attribute.INCOME_BRACKET,
})

# (hard_blacklist, proof_required) per (ContextClass, TrustBand)
_BAND_CONSTRAINTS: dict[ContextClass, dict[TrustBand, tuple[frozenset, frozenset]]] = {
    ContextClass.FINANCIAL: {
        TrustBand.LOW:  (_FINANCIAL_SENSITIVE,
                         frozenset({Attribute.DOB, Attribute.GEOLOCATION})),
        TrustBand.MID:  (_ALWAYS_SENSITIVE,
                         frozenset({Attribute.DOB})),
        TrustBand.HIGH: (frozenset({Attribute.HEALTH_CONDITION}),
                         frozenset()),
    },
    ContextClass.PROFESSIONAL: {
        TrustBand.LOW:  (_ALWAYS_SENSITIVE | frozenset({
                             Attribute.INCOME_BRACKET, Attribute.SOCIAL_GRAPH,
                             Attribute.GAMING_HISTORY,
                         }),
                         frozenset({Attribute.DOB, Attribute.GEOLOCATION})),
        TrustBand.MID:  (_ALWAYS_SENSITIVE | frozenset({Attribute.INCOME_BRACKET}),
                         frozenset({Attribute.DOB})),
        TrustBand.HIGH: (frozenset({Attribute.HEALTH_CONDITION}),
                         frozenset()),
    },
    ContextClass.SOCIAL: {
        TrustBand.LOW:  (_ALWAYS_SENSITIVE | frozenset({
                             Attribute.INCOME_BRACKET, Attribute.EMPLOYMENT_STATUS,
                         }),
                         frozenset({Attribute.DOB, Attribute.GEOLOCATION})),
        TrustBand.MID:  (_ALWAYS_SENSITIVE | frozenset({Attribute.INCOME_BRACKET}),
                         frozenset({Attribute.GEOLOCATION})),
        TrustBand.HIGH: (_ALWAYS_SENSITIVE,
                         frozenset()),
    },
    ContextClass.MEDICAL: {
        # health_condition is NOT blacklisted in MEDICAL — it's the purpose.
        TrustBand.LOW:  (frozenset({Attribute.BEHAVIOURAL_METADATA, Attribute.DEVICE_FINGERPRINT}),
                         frozenset({Attribute.DOB, Attribute.HEALTH_CONDITION,
                                    Attribute.GEOLOCATION})),
        TrustBand.MID:  (frozenset({Attribute.BEHAVIOURAL_METADATA, Attribute.DEVICE_FINGERPRINT}),
                         frozenset({Attribute.DOB})),
        TrustBand.HIGH: (frozenset({Attribute.BEHAVIOURAL_METADATA, Attribute.DEVICE_FINGERPRINT}),
                         frozenset()),
    },
    ContextClass.CIVIC: {
        TrustBand.LOW:  (_ALWAYS_SENSITIVE | frozenset({
                             Attribute.SOCIAL_GRAPH, Attribute.GAMING_HISTORY,
                             Attribute.INCOME_BRACKET,
                         }),
                         frozenset({Attribute.DOB, Attribute.GEOLOCATION})),
        TrustBand.MID:  (_ALWAYS_SENSITIVE,
                         frozenset({Attribute.DOB})),
        TrustBand.HIGH: (frozenset({Attribute.HEALTH_CONDITION}),
                         frozenset()),
    },
}

# Decay/boost factors for salience weight updates.
_SALIENCE_DECAY = 0.10   # applied to disclosed attributes
_SALIENCE_BOOST = 0.10   # applied to withheld attributes
_SALIENCE_MIN   = 1e-6   # floor to avoid zero weights

# KL smoothing epsilon — prevents log(0).
_KL_EPSILON = 1e-6


# ---------------------------------------------------------------------------
# IPC
# ---------------------------------------------------------------------------

class IPC:
    """
    Identity Profile Constructor.

    Parameters
    ----------
    pds:
        Opened PDS instance (caller owns lifecycle).
    norm_db:
        Norm database Φ(c); defaults to seed NormDatabase().
    entropy_threshold:
        Minimum entropy delta (nats) that triggers an EntropyAlert.
    drift_threshold:
        KL-divergence threshold that triggers a DriftAlert.
    """

    def __init__(
        self,
        pds: PDS,
        norm_db: NormDatabase | None = None,
        entropy_threshold: float = 0.15,
        drift_threshold: float = 0.5,
    ) -> None:
        self._pds = pds
        self._norms = norm_db if norm_db is not None else NormDatabase()
        self._entropy_threshold = entropy_threshold
        self._drift_threshold = drift_threshold
        # In-session profile cache — keeps full Hᵢ which PDS does not persist.
        self._cache: dict[ContextClass, ProfileTuple] = {}

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------

    def create_profile(self, context: ContextClass) -> ProfileTuple:
        """
        Initialise a fresh ProfileTuple for ``context``.

        Active attributes Aᵢ are those in the PDS vault that have at
        least one permitted norm entry for this context.
        Salience weights are uniform; TrustConstraints are default.
        """
        active = self._relevant_attrs_for_context(context)
        n = max(len(active), 1)
        weights = {a: 1.0 / n for a in active}
        profile = ProfileTuple(
            profile_id=str(uuid.uuid4()),
            active_attributes=active,
            salience_weights=weights,
            trust_constraints=TrustConstraints(),
        )
        self._cache[context] = profile
        self._pds.save_profile(context, profile)
        return profile

    def get_profile(self, context: ContextClass) -> ProfileTuple | None:
        """Return cached profile (with Hᵢ) first, then PDS snapshot."""
        if context in self._cache:
            return self._cache[context]
        profile = self._pds.load_profile(context)
        if profile is not None:
            self._cache[context] = profile
        return profile

    def get_or_create_profile(self, context: ContextClass) -> ProfileTuple:
        """Return existing profile or create a fresh one."""
        p = self.get_profile(context)
        return p if p is not None else self.create_profile(context)

    def update_profile(
        self,
        context: ContextClass,
        event: DisclosureEvent,
    ) -> EntropyAlert | None:
        """
        Append ``event`` to Hᵢ, recalculate Wᵢ, update H(I), and
        persist the updated snapshot to the PDS.

        Returns an EntropyAlert when the entropy delta exceeds
        ``entropy_threshold``; None otherwise.
        """
        profile = self.get_or_create_profile(context)
        entropy_before = profile.identity_entropy

        new_history: DisclosureHistory = profile.disclosure_history + (event,)
        new_weights = _update_salience(
            profile.salience_weights,
            profile.active_attributes,
            event.disclosed_attributes,
        )

        _update_entropy_counts(self._pds, context, event.disclosed_attributes)
        entropy_after = _compute_entropy(self._pds, context, profile.active_attributes)

        updated = ProfileTuple(
            profile_id=profile.profile_id,
            active_attributes=profile.active_attributes,
            salience_weights=new_weights,
            trust_constraints=profile.trust_constraints,
            disclosure_history=new_history,
            identity_entropy=entropy_after,
            drift_flagged=profile.drift_flagged,
            created_at=profile.created_at,
            updated_at=datetime.now(timezone.utc),
        )
        self._cache[context] = updated
        self._pds.save_profile(context, updated)

        delta = entropy_after - entropy_before
        if delta > self._entropy_threshold:
            return EntropyAlert(
                context=context,
                profile_id=profile.profile_id,
                entropy_before=entropy_before,
                entropy_after=entropy_after,
                delta=delta,
            )
        return None

    # ------------------------------------------------------------------
    # Attribute acquisition
    # ------------------------------------------------------------------

    def set_explicit(
        self,
        attr: Attribute,
        value: object,
        abstracted: object | None = None,
    ) -> None:
        """Channel 1: direct user-provided attribute; invalidates cached profiles."""
        self._pds.set_attribute(attr, value, abstracted=abstracted)
        self._invalidate_profiles_with(attr)

    def import_vc(
        self,
        vc_reference: str,
        attr: Attribute,
        value: object,
    ) -> None:
        """Channel 3: VC-verified attribute import; sets verified=True in PDS."""
        self._pds.set_attribute(attr, value, verified=True, vc_reference=vc_reference)
        self._invalidate_profiles_with(attr)

    def infer_from_history(
        self, context: ContextClass
    ) -> list[tuple[Attribute, object]]:
        """
        Channel 2: passive inference from Hᵢ.

        Returns (attribute, value) pairs for attributes that appear in
        > 50% of historical disclosure events in this context, using
        their current PDS value.  The caller can use these to auto-populate
        profiles or adjust salience.
        """
        profile = self.get_profile(context)
        if profile is None or not profile.disclosure_history:
            return []

        total = len(profile.disclosure_history)
        counts: dict[Attribute, int] = {}
        for evt in profile.disclosure_history:
            for a in evt.disclosed_attributes:
                counts[a] = counts.get(a, 0) + 1

        frequent = [a for a, c in counts.items() if c / total > 0.5]
        vault = self._pds.get_all_attributes()
        return [(a, vault[a].effective_value()) for a in frequent if a in vault]

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def check_drift(
        self,
        context: ContextClass,
        new_request_features: dict,
    ) -> DriftAlert | None:
        """
        Compute KL(new_obs ‖ rolling_dist) for the request feature distribution.

        Rolling distribution is stored in PDS config under ``drift:<context>``
        as a JSON dict of feature-string → decayed count.  Each call applies
        a 0.9 decay factor to existing counts before adding the new observation.

        Returns a DriftAlert if KL > drift_threshold; None otherwise.
        On the very first observation for a context no alert is fired —
        the observation becomes the baseline distribution.
        """
        key = f"drift:{context.value}"
        raw = self._pds.get_raw_config(key)
        rolling: dict[str, float] = json.loads(raw) if raw else {}

        new_features = _extract_features(new_request_features)
        if not new_features:
            return None

        alert: DriftAlert | None = None

        if rolling:
            kl, contributions = _kl_divergence_features(new_features, rolling)
            if kl > self._drift_threshold:
                self._flag_profile_drift(context)
                alert = DriftAlert(
                    context=context,
                    kl_divergence=kl,
                    diverging_features=contributions,
                    threshold=self._drift_threshold,
                )

        # Update rolling: decay existing + add new observation.
        updated: dict[str, float] = {k: v * 0.9 for k, v in rolling.items()}
        for f in new_features:
            updated[f] = updated.get(f, 0.0) + 1.0
        self._pds.set_raw_config(key, json.dumps(updated))

        return alert

    # ------------------------------------------------------------------
    # Trust constraints
    # ------------------------------------------------------------------

    def get_trust_constraint(
        self,
        context: ContextClass,
        platform_id: str,
    ) -> TrustConstraints:
        """
        Return TrustConstraints derived from the platform's trust score band.

        LOW  (< 0.4):   aggressive hard_blacklist, broad proof_required
        MID  (0.4–0.7): standard constraints
        HIGH (≥ 0.7):   relaxed proof requirements
        """
        score = self._pds.get_trust_score(platform_id)
        band = _trust_band(score)
        blacklist, proof_req = _BAND_CONSTRAINTS[context][band]
        return TrustConstraints(
            min_trust_score=score,
            hard_blacklist=blacklist,
            proof_required=proof_req,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _relevant_attrs_for_context(self, context: ContextClass) -> frozenset[Attribute]:
        """
        Attributes in the PDS vault with ≥ 1 permitted norm flow for
        this context (any recipient role).
        """
        vault = set(self._pds.get_all_attributes())
        return frozenset(
            a for a in vault
            if any(self._norms.permitted(context, a, role) for role in RecipientRole)
        )

    def _invalidate_profiles_with(self, attr: Attribute) -> None:
        """Drop cached profiles that include ``attr`` in their active set."""
        for ctx in [c for c, p in self._cache.items() if attr in p.active_attributes]:
            del self._cache[ctx]

    def _flag_profile_drift(self, context: ContextClass) -> None:
        profile = self.get_profile(context)
        if profile is None or profile.drift_flagged:
            return
        flagged = ProfileTuple(
            profile_id=profile.profile_id,
            active_attributes=profile.active_attributes,
            salience_weights=profile.salience_weights,
            trust_constraints=profile.trust_constraints,
            disclosure_history=profile.disclosure_history,
            identity_entropy=profile.identity_entropy,
            drift_flagged=True,
            created_at=profile.created_at,
            updated_at=datetime.now(timezone.utc),
        )
        self._cache[context] = flagged
        self._pds.save_profile(context, flagged)


# ---------------------------------------------------------------------------
# Pure functions (no IPC state dependency)
# ---------------------------------------------------------------------------

def _update_salience(
    current: dict[Attribute, float],
    active: frozenset[Attribute],
    disclosed: frozenset[Attribute],
) -> dict[Attribute, float]:
    """
    Update salience weights after a disclosure event.

    Disclosed  → weight * (1 − DECAY)   [routine disclosure = less sensitive]
    Withheld   → weight * (1 + BOOST)   [withheld = more sensitive]
    All weights normalised to sum 1.0 after update.
    """
    weights = {a: current.get(a, 1.0 / max(len(active), 1)) for a in active}
    for a in active:
        if a in disclosed:
            weights[a] = max(weights[a] * (1.0 - _SALIENCE_DECAY), _SALIENCE_MIN)
        else:
            weights[a] = weights[a] * (1.0 + _SALIENCE_BOOST)
    total = sum(weights.values()) or 1.0
    return {a: w / total for a, w in weights.items()}


def _update_entropy_counts(
    pds: PDS,
    context: ContextClass,
    disclosed: frozenset[Attribute],
) -> None:
    """Increment per-attribute disclosure counts in PDS config."""
    key = f"entropy_counts:{context.value}"
    raw = pds.get_raw_config(key)
    counts: dict[str, float] = json.loads(raw) if raw else {}
    for a in disclosed:
        counts[a.value] = counts.get(a.value, 0.0) + 1.0
    counts["_total"] = counts.get("_total", 0.0) + 1.0
    pds.set_raw_config(key, json.dumps(counts))


def _compute_entropy(
    pds: PDS,
    context: ContextClass,
    active: frozenset[Attribute],
) -> float:
    """
    Shannon entropy H(Aᵢ) in nats over the empirical disclosure distribution.

    Returns 0.0 when no disclosures have occurred yet (no information).
    """
    key = f"entropy_counts:{context.value}"
    raw = pds.get_raw_config(key)
    if raw is None:
        return 0.0
    counts: dict[str, float] = json.loads(raw)
    total = counts.get("_total", 0.0)
    if total == 0.0:
        return 0.0
    h = 0.0
    for a in active:
        c = counts.get(a.value, 0.0)
        if c > 0.0:
            p = c / total
            h -= p * math.log(p)
    return h


def _extract_features(features: dict) -> list[str]:
    """
    Flatten a feature dict into sorted ``key:value`` strings.

    List values are expanded per-element; non-string values are str()-cast.
    """
    result: list[str] = []
    for k in sorted(features):
        v = features[k]
        if isinstance(v, (list, tuple)):
            for item in sorted(str(x) for x in v):
                result.append(f"{k}:{item}")
        else:
            result.append(f"{k}:{v}")
    return result


def _kl_divergence_features(
    new_features: list[str],
    rolling: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """
    KL(new_obs ‖ rolling_dist) where new_obs is uniform over new_features.

    Laplace smoothing (ε = 1e-6) prevents log(0).
    Returns (total_kl, {feature: contribution}) for contributions > 0.01.
    """
    n_new = max(len(new_features), 1)
    new_dist = {f: 1.0 / n_new for f in new_features}

    all_keys = set(new_dist) | set(rolling)
    n_vocab = max(len(all_keys), 1)
    total_rolling = sum(rolling.values()) or 1.0

    kl = 0.0
    contributions: dict[str, float] = {}
    for f in new_features:
        p = (new_dist[f] + _KL_EPSILON) / (1.0 + _KL_EPSILON * n_vocab)
        q = (rolling.get(f, 0.0) + _KL_EPSILON) / (total_rolling + _KL_EPSILON * n_vocab)
        c = p * math.log(p / q)
        contributions[f] = c
        kl += c

    return kl, {f: v for f, v in contributions.items() if v > 0.01}

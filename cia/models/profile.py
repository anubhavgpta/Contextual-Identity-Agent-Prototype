"""Profile tuple pᵢ = (Aᵢ, Wᵢ, Tᵢ, Hᵢ) as defined in §3.1 of the paper."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping

from .attributes import Attribute, AttributeSet
from .context import ContextClass


# Wᵢ — salience weights ∈ (0, 1] per attribute, summing to 1 within a profile.
SalienceWeights = Mapping[Attribute, float]


@dataclass(frozen=True)
class TrustConstraints:
    """Tᵢ — per-profile trust thresholds and allowed context classes."""

    # Minimum platform trust score in [0, 1] required to disclose anything.
    min_trust_score: float = 0.0
    # Context classes explicitly permitted; empty set = all permitted.
    allowed_contexts: frozenset[ContextClass] = field(
        default_factory=frozenset
    )
    # Attributes that must never be disclosed regardless of other rules.
    hard_blacklist: AttributeSet = field(default_factory=frozenset)
    # Attributes that require an additional ZKP/BBS+ presentation rather than raw disclosure.
    proof_required: AttributeSet = field(default_factory=frozenset)


@dataclass(frozen=True)
class DisclosureEvent:
    """A single entry in the disclosure history Hᵢ."""

    timestamp: datetime
    platform_id: str
    context_class: ContextClass
    # Attributes disclosed (may include abstracted variants).
    disclosed_attributes: AttributeSet
    # HMAC/signature over the disclosure receipt for tamper-evidence.
    receipt_signature: str
    # Running identity-entropy value H(I) at time of disclosure.
    entropy_at_disclosure: float


# Hᵢ — ordered disclosure history (chronological, immutable entries).
DisclosureHistory = tuple[DisclosureEvent, ...]


@dataclass
class ProfileTuple:
    """
    pᵢ = (Aᵢ, Wᵢ, Tᵢ, Hᵢ)

    Mutable container; the IPC is responsible for mutation.  All writes
    must go through ipc.py — direct field assignment is intentionally
    possible here (dataclass, not frozen) because the IPC needs it.
    """

    profile_id: str
    # Aᵢ — active attributes currently held for this profile.
    active_attributes: AttributeSet
    # Wᵢ — salience weights; must cover every attribute in active_attributes.
    salience_weights: dict[Attribute, float]
    # Tᵢ — trust constraints.
    trust_constraints: TrustConstraints
    # Hᵢ — immutable history; IPC appends by replacing with a new tuple.
    disclosure_history: DisclosureHistory = field(default_factory=tuple)
    # Running estimate of H(I) — identity entropy in bits.
    identity_entropy: float = 0.0
    # Whether the IPC has flagged this profile for user review (drift).
    drift_flagged: bool = False
    # ISO-8601 creation timestamp.
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # ISO-8601 last-modified timestamp; IPC updates on every write.
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

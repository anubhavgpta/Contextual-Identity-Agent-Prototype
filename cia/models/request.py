"""Incoming disclosure request types — what a platform sends to the CIA."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Mapping

from .attributes import AttributeSet
from .context import ContextClass, ContextSignal


class PlatformType(str, Enum):
    """The 5 synthetic platform types used in the simulator (§5.1)."""

    FINTECH = "FINTECH"
    EMPLOYER = "EMPLOYER"
    SOCIAL_NETWORK = "SOCIAL_NETWORK"
    HEALTHCARE = "HEALTHCARE"
    GOVERNMENT = "GOVERNMENT"


class RetentionPolicy(str, Enum):
    """
    How the relying party intends to store disclosed data.
    Affects the DDE trust signal T(D) via a temporary β adjustment (IACP only).
    """

    SESSION_ONLY = "SESSION_ONLY"   # data deleted after session — β unchanged
    PERSISTENT   = "PERSISTENT"     # data stored indefinitely — β −0.10
    ANONYMISED   = "ANONYMISED"     # data anonymised before storage — β +0.15


class RequestTier(str, Enum):
    """
    Threat tier used by the simulator (§5.2).

    BENIGN     — legitimate minimal-scope request.
    PROBING    — over-asks, testing CIA boundaries.
    COLLUDING  — coordinates with other platforms to enable re-identification.
    """

    BENIGN = "BENIGN"
    PROBING = "PROBING"
    COLLUDING = "COLLUDING"


@dataclass(frozen=True)
class DisclosureRequest:
    """
    A single disclosure request arriving via the IACP identify→request phase.
    Corresponds to the 'request' message in §4.1.
    """

    request_id: str
    # Platform sending the request.
    platform_id: str
    platform_type: PlatformType
    # Trust score in [0, 1] derived from platform's TLS cert + reputation DB.
    platform_trust_score: float
    # Attributes the platform is asking for.
    requested_attributes: AttributeSet
    # Contextual signal provided by the platform for classification.
    context_signal: ContextSignal
    # Pre-classified context, if the CC has already run.
    declared_context: ContextClass | None = None
    # ISO-8601 request timestamp.
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Arbitrary platform-supplied metadata (treated as untrusted).
    platform_metadata: Mapping[str, str] = field(default_factory=dict)
    # How the relying party will retain the disclosed data.
    retention_policy: RetentionPolicy | None = None
    # Simulator-only: ground-truth tier (never visible to CIA logic).
    _sim_tier: RequestTier | None = None

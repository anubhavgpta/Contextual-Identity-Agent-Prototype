"""Attribute vocabulary and value types — the 12-attribute fixed vocab from the paper."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, FrozenSet


class Attribute(str, Enum):
    """The 12 fixed attributes in the paper's vocabulary (§3.1)."""

    LEGAL_NAME = "legal_name"
    DOB = "dob"
    EMAIL = "email"
    PHONE = "phone"
    GEOLOCATION = "geolocation"
    EMPLOYMENT_STATUS = "employment_status"
    INCOME_BRACKET = "income_bracket"
    HEALTH_CONDITION = "health_condition"
    GAMING_HISTORY = "gaming_history"
    SOCIAL_GRAPH = "social_graph"
    DEVICE_FINGERPRINT = "device_fingerprint"
    BEHAVIOURAL_METADATA = "behavioural_metadata"


# Ordered tuple used wherever deterministic iteration matters.
AttributeVocab: tuple[Attribute, ...] = tuple(Attribute)


@dataclass(frozen=True)
class AttributeValue:
    """A single attribute paired with its raw value and an optional abstracted form."""

    attribute: Attribute
    # Raw value as stored in the PDS.
    raw: Any
    # Abstracted/generalised value (e.g. age range instead of exact DOB).
    # None means no abstraction has been applied.
    abstracted: Any | None = None

    def effective_value(self) -> Any:
        """Return abstracted if available, otherwise raw."""
        return self.abstracted if self.abstracted is not None else self.raw


# A disclosure set: frozenset so it can be hashed and used as a dict key.
AttributeSet = FrozenSet[Attribute]

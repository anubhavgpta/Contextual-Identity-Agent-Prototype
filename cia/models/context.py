"""Context classification types — ContextClass enum and classified-context output."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class ContextClass(str, Enum):
    """The 5 context classes recognised by the CC (§3.2)."""

    FINANCIAL = "FINANCIAL"
    PROFESSIONAL = "PROFESSIONAL"
    SOCIAL = "SOCIAL"
    MEDICAL = "MEDICAL"
    CIVIC = "CIVIC"


@dataclass(frozen=True)
class ContextSignal:
    """Raw signal fed to the Context Classifier."""

    # Free-text description of the platform or interaction.
    description: str
    # Structured metadata from the requesting platform (arbitrary k/v).
    metadata: Mapping[str, str] = field(default_factory=dict)
    # Optional explicit hint from the platform (not trusted, used as a prior).
    declared_class: ContextClass | None = None


@dataclass(frozen=True)
class ClassifiedContext:
    """Output of the Context Classifier for a single signal."""

    context_class: ContextClass
    # Posterior confidence in [0, 1]. Below τ=0.80 → conservative mode.
    confidence: float
    # True when confidence < τ and conservative mode is active.
    conservative_mode: bool
    # Per-class probability distribution (sums to 1.0).
    class_probabilities: Mapping[ContextClass, float] = field(default_factory=dict)
    # Which layer produced the classification: "symbolic" | "neural" | "combined".
    source: str = "combined"

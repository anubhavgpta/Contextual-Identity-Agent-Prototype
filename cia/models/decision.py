"""DDE internal decision types — utility scoring and trace structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .attributes import Attribute, AttributeSet
from .context import ContextClass
from .policy import PolicyAction


@dataclass(frozen=True)
class UtilityScore:
    """
    Decomposed U(D) = α·S(D) + β·T(D) − γ·L(D)  (§3.4).

    S(D) — service utility: fraction of requested attributes satisfied.
    T(D) — trust utility: platform trust score weighted by disclosure depth.
    L(D) — linkability cost: H(I) − H(I|D), the identity-entropy drop.
    U     — final scalar utility.
    """

    S: float          # service utility ∈ [0, 1]
    T: float          # trust utility ∈ [0, 1]
    L: float          # linkability cost ≥ 0 (in bits)
    alpha: float      # weight on S
    beta: float       # weight on T
    gamma: float      # weight on L  (γ > α, β enforced by DDE)
    U: float          # α·S + β·T − γ·L


@dataclass(frozen=True)
class CandidateDisclosure:
    """
    A single candidate set D evaluated during the greedy search.
    The DDE builds a sequence of these, adding one attribute per step.
    """

    attribute_set: AttributeSet
    utility: UtilityScore
    # Per-attribute action the DDE intends to apply (matches policy resolution).
    attribute_actions: dict[Attribute, PolicyAction] = field(default_factory=dict)
    # True when this candidate satisfies both Π and Φ(c).
    feasible: bool = True


@dataclass
class DecisionTrace:
    """
    Full audit trail of the DDE's greedy optimisation for one request.
    Written to the disclosure log alongside the receipt.
    """

    request_id: str
    context_class: ContextClass
    # Ordered sequence of candidate sets evaluated (greedy steps).
    candidates: list[CandidateDisclosure] = field(default_factory=list)
    # Index into candidates of the optimal D*.
    optimal_index: int = -1
    # Whether the minimality condition holds: ∄ D'⊊D* with U(D')≥U(D*).
    minimality_verified: bool = False
    # Conservative mode was active during this decision.
    conservative_mode: bool = False

    @property
    def optimal(self) -> CandidateDisclosure | None:
        if 0 <= self.optimal_index < len(self.candidates):
            return self.candidates[self.optimal_index]
        return None

"""User policy DSL types — Π as defined in §3.3 of the paper."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from .attributes import Attribute, AttributeSet
from .context import ContextClass


class PolicyAction(str, Enum):
    """The four actions available in the policy DSL (§3.3)."""

    DISCLOSE = "DISCLOSE"          # release raw attribute value
    WITHHOLD = "WITHHOLD"          # never release in this context
    ABSTRACT = "ABSTRACT"          # release generalised/bucketed value
    VERIFY_ONLY = "VERIFY_ONLY"    # release ZKP/BBS+ proof, not raw value


@dataclass(frozen=True)
class PolicyRule:
    """
    A single rule in Π.

    Rules are evaluated in priority order (lower number = higher priority).
    The first matching rule wins.  If no rule matches, the DDE falls back
    to the norm database Φ(c).
    """

    # Attributes this rule covers; empty frozenset = wildcard (all attributes).
    attributes: AttributeSet
    # Contexts this rule covers; empty frozenset = wildcard (all contexts).
    contexts: frozenset[ContextClass]
    action: PolicyAction
    priority: int = 100
    # Optional free-text justification stored in audit log.
    label: str = ""

    def matches(self, attribute: Attribute, context: ContextClass) -> bool:
        attr_match = not self.attributes or attribute in self.attributes
        ctx_match = not self.contexts or context in self.contexts
        return attr_match and ctx_match


@dataclass
class PolicySet:
    """Π — the complete user policy, ordered by priority."""

    rules: list[PolicyRule] = field(default_factory=list)

    def add_rule(self, rule: PolicyRule) -> None:
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority)

    def resolve(
        self, attribute: Attribute, context: ContextClass
    ) -> PolicyAction | None:
        """Return the action for (attribute, context), or None if no rule matches."""
        for rule in self.rules:
            if rule.matches(attribute, context):
                return rule.action
        return None

    def applicable_rules(
        self, context: ContextClass
    ) -> Sequence[PolicyRule]:
        """All rules that could fire in the given context."""
        return [r for r in self.rules if not r.contexts or context in r.contexts]

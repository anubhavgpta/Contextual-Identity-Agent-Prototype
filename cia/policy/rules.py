"""
PolicyInterpreter — Π evaluation over an incoming DisclosureRequest.

Combines the user's PolicySet (from models/policy.py) with the norm
database Φ(c) to produce a final PolicyAction for every attribute in
the requested set.  The DDE calls this module; it never touches storage.

Resolution order (highest priority first):
  1. TrustConstraints.hard_blacklist  → always WITHHOLD
  2. TrustConstraints.proof_required  → always VERIFY_ONLY
  3. Platform trust score < min_trust → always WITHHOLD (entire request)
  4. Matching PolicyRule (by priority) → rule's action
  5. Norm database Φ(c)               → DISCLOSE / VERIFY_ONLY / WITHHOLD
  6. Default                          → WITHHOLD  (closed-world)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cia.models.attributes import Attribute, AttributeSet
from cia.models.context import ContextClass
from cia.models.policy import PolicyAction, PolicySet
from cia.models.profile import ProfileTuple
from cia.models.request import DisclosureRequest
from .norms import NormDatabase, RecipientRole


# Mapping from PlatformType string to RecipientRole — bridging the two enums.
# PlatformType lives in models (what platforms are); RecipientRole lives in
# norms (what role they play in a contextual flow).  The mapping is stable
# for the 5 synthetic platform types used in the paper.
from cia.models.request import PlatformType

_PLATFORM_TO_ROLE: dict[PlatformType, RecipientRole] = {
    PlatformType.FINTECH:         RecipientRole.FINANCIAL_INSTITUTION,
    PlatformType.EMPLOYER:        RecipientRole.EMPLOYER,
    PlatformType.SOCIAL_NETWORK:  RecipientRole.SOCIAL_PEER,
    PlatformType.HEALTHCARE:      RecipientRole.HEALTHCARE_PROVIDER,
    PlatformType.GOVERNMENT:      RecipientRole.GOVERNMENT_AGENCY,
}


@dataclass(frozen=True)
class AttributeDecision:
    """Per-attribute outcome from the interpreter."""

    attribute: Attribute
    action: PolicyAction
    # Which layer produced this decision (for audit transparency).
    source: str   # "hard_blacklist" | "proof_required" | "trust_gate" |
                  # "policy_rule" | "norm" | "default"
    # The norm rationale string if source == "norm", else "".
    rationale: str = ""


@dataclass
class InterpretationResult:
    """Full output of PolicyInterpreter.evaluate()."""

    request_id: str
    context_class: ContextClass
    # Trust gate fired — entire request rejected before per-attribute eval.
    trust_gated: bool = False
    # Per-attribute decisions; empty when trust_gated is True.
    decisions: list[AttributeDecision] = field(default_factory=list)

    def action_for(self, attribute: Attribute) -> PolicyAction | None:
        for d in self.decisions:
            if d.attribute == attribute:
                return d.action
        return None

    def attributes_with_action(self, action: PolicyAction) -> AttributeSet:
        return frozenset(d.attribute for d in self.decisions if d.action == action)

    @property
    def disclose_set(self) -> AttributeSet:
        return self.attributes_with_action(PolicyAction.DISCLOSE)

    @property
    def withhold_set(self) -> AttributeSet:
        return self.attributes_with_action(PolicyAction.WITHHOLD)

    @property
    def abstract_set(self) -> AttributeSet:
        return self.attributes_with_action(PolicyAction.ABSTRACT)

    @property
    def verify_only_set(self) -> AttributeSet:
        return self.attributes_with_action(PolicyAction.VERIFY_ONLY)


class PolicyInterpreter:
    """
    Stateless evaluator; construct once per session and call evaluate()
    for each incoming request.
    """

    def __init__(
        self,
        norm_db: NormDatabase | None = None,
    ) -> None:
        self._norms = norm_db if norm_db is not None else NormDatabase()

    def evaluate(
        self,
        request: DisclosureRequest,
        profile: ProfileTuple,
        policy_set: PolicySet,
        context_class: ContextClass,
    ) -> InterpretationResult:
        """
        Produce an AttributeDecision for every attribute in
        request.requested_attributes.

        Does NOT mutate any argument.
        """
        result = InterpretationResult(
            request_id=request.request_id,
            context_class=context_class,
        )

        # ── Trust gate (step 3) ────────────────────────────────────────────
        if request.platform_trust_score < profile.trust_constraints.min_trust_score:
            result.trust_gated = True
            return result

        recipient_role = _PLATFORM_TO_ROLE.get(
            request.platform_type, RecipientRole.UNKNOWN
        )

        for attr in request.requested_attributes:
            decision = self._decide(
                attr, context_class, recipient_role, profile, policy_set
            )
            result.decisions.append(decision)

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _decide(
        self,
        attribute: Attribute,
        context: ContextClass,
        recipient_role: RecipientRole,
        profile: ProfileTuple,
        policy_set: PolicySet,
    ) -> AttributeDecision:
        tc = profile.trust_constraints

        # Step 1 — hard blacklist
        if attribute in tc.hard_blacklist:
            return AttributeDecision(attribute, PolicyAction.WITHHOLD, "hard_blacklist")

        # Step 2 — proof-required set
        if attribute in tc.proof_required:
            return AttributeDecision(attribute, PolicyAction.VERIFY_ONLY, "proof_required")

        # Step 4 — user policy rule
        action = policy_set.resolve(attribute, context)
        if action is not None:
            return AttributeDecision(attribute, action, "policy_rule")

        # Step 5 — norm database
        if self._norms.permitted(context, attribute, recipient_role):
            norm_action = (
                PolicyAction.VERIFY_ONLY
                if self._norms.proof_required(context, attribute, recipient_role)
                else PolicyAction.DISCLOSE
            )
            # Collect rationale for audit log.
            rationale = next(
                (e.rationale for e in self._norms.entries_for_context(context)
                 if e.attribute == attribute and e.recipient_role == recipient_role),
                "",
            )
            return AttributeDecision(attribute, norm_action, "norm", rationale)

        # Step 6 — closed-world default
        return AttributeDecision(attribute, PolicyAction.WITHHOLD, "default")

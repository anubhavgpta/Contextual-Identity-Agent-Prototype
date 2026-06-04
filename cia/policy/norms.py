"""
Φ(c) — seed norm database per context class.

Encodes contextual integrity constraints after Nissenbaum (2004):
each norm is a permitted information flow ⟨attribute, sender_role, recipient_role⟩
within a given context class.  A flow NOT listed is impermissible by default
(closed-world assumption).

The database is queryable at runtime and can be round-tripped through
assets/norm_db.json for inspection / offline editing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Iterator, Sequence

from cia.models.attributes import Attribute
from cia.models.context import ContextClass


class RecipientRole(str, Enum):
    """
    Roles that a requesting platform may occupy.

    Used in norm matching: a platform declares (or is inferred to have)
    a role, and only flows to that role are permitted by Φ(c).
    """

    FINANCIAL_INSTITUTION = "FINANCIAL_INSTITUTION"
    EMPLOYER = "EMPLOYER"
    SOCIAL_PEER = "SOCIAL_PEER"
    HEALTHCARE_PROVIDER = "HEALTHCARE_PROVIDER"
    GOVERNMENT_AGENCY = "GOVERNMENT_AGENCY"
    # Generic / unclassified platform — most restrictive treatment.
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class NormEntry:
    """
    A single permitted flow ⟨context, attribute, sender_role, recipient_role⟩.

    sender_role is always "USER" in this system (the subject discloses);
    it is kept for schema completeness and future multi-party extensions.
    """

    context: ContextClass
    attribute: Attribute
    # Role of the entity that may legitimately receive this attribute in
    # this context.  "*" encodes any role (used sparingly).
    recipient_role: RecipientRole
    # Whether the norm mandates a proof instead of a raw value.
    proof_required: bool = False
    # Human-readable justification (regulatory basis, social convention, etc.).
    rationale: str = ""


# ---------------------------------------------------------------------------
# Seed norm table
# Encoding principles:
#   FINANCIAL  — lenders/insurers need income, employment, not health/social
#   PROFESSIONAL — employers see employment status, not health/gaming/income
#   SOCIAL     — peers see name, social graph; nothing sensitive
#   MEDICAL    — providers need DOB, health condition (proof ok); strict otherwise
#   CIVIC      — government sees legal identity, address; proof-only for age
# ---------------------------------------------------------------------------

_SEED_NORMS: list[NormEntry] = [
    # ── FINANCIAL ──────────────────────────────────────────────────────────
    NormEntry(ContextClass.FINANCIAL, Attribute.LEGAL_NAME,
              RecipientRole.FINANCIAL_INSTITUTION,
              rationale="KYC obligation"),
    NormEntry(ContextClass.FINANCIAL, Attribute.DOB,
              RecipientRole.FINANCIAL_INSTITUTION,
              proof_required=True,
              rationale="Age/identity verification; proof preferred"),
    NormEntry(ContextClass.FINANCIAL, Attribute.EMAIL,
              RecipientRole.FINANCIAL_INSTITUTION,
              rationale="Account communication"),
    NormEntry(ContextClass.FINANCIAL, Attribute.PHONE,
              RecipientRole.FINANCIAL_INSTITUTION,
              rationale="Two-factor authentication"),
    NormEntry(ContextClass.FINANCIAL, Attribute.EMPLOYMENT_STATUS,
              RecipientRole.FINANCIAL_INSTITUTION,
              rationale="Creditworthiness assessment"),
    NormEntry(ContextClass.FINANCIAL, Attribute.INCOME_BRACKET,
              RecipientRole.FINANCIAL_INSTITUTION,
              rationale="Loan eligibility"),
    # Geolocation: permitted only for fraud detection (abstracted to region).
    NormEntry(ContextClass.FINANCIAL, Attribute.GEOLOCATION,
              RecipientRole.FINANCIAL_INSTITUTION,
              rationale="Fraud / AML compliance; region-level only"),

    # ── PROFESSIONAL ───────────────────────────────────────────────────────
    NormEntry(ContextClass.PROFESSIONAL, Attribute.LEGAL_NAME,
              RecipientRole.EMPLOYER,
              rationale="Employment record"),
    NormEntry(ContextClass.PROFESSIONAL, Attribute.EMAIL,
              RecipientRole.EMPLOYER,
              rationale="Workplace communication"),
    NormEntry(ContextClass.PROFESSIONAL, Attribute.PHONE,
              RecipientRole.EMPLOYER,
              rationale="Workplace communication"),
    NormEntry(ContextClass.PROFESSIONAL, Attribute.EMPLOYMENT_STATUS,
              RecipientRole.EMPLOYER,
              rationale="Role / contract type"),
    # DOB only as proof for background-check age verification.
    NormEntry(ContextClass.PROFESSIONAL, Attribute.DOB,
              RecipientRole.EMPLOYER,
              proof_required=True,
              rationale="Right-to-work age check; proof only"),

    # ── SOCIAL ─────────────────────────────────────────────────────────────
    NormEntry(ContextClass.SOCIAL, Attribute.LEGAL_NAME,
              RecipientRole.SOCIAL_PEER,
              rationale="Identity on social platforms"),
    NormEntry(ContextClass.SOCIAL, Attribute.EMAIL,
              RecipientRole.SOCIAL_PEER,
              rationale="Optional contact sharing"),
    NormEntry(ContextClass.SOCIAL, Attribute.SOCIAL_GRAPH,
              RecipientRole.SOCIAL_PEER,
              rationale="Friend/connection lists"),
    NormEntry(ContextClass.SOCIAL, Attribute.GAMING_HISTORY,
              RecipientRole.SOCIAL_PEER,
              rationale="Gaming profile / achievements"),
    # Geolocation at coarse granularity (city-level) for social discovery.
    NormEntry(ContextClass.SOCIAL, Attribute.GEOLOCATION,
              RecipientRole.SOCIAL_PEER,
              rationale="Location sharing; coarse only"),

    # ── MEDICAL ────────────────────────────────────────────────────────────
    NormEntry(ContextClass.MEDICAL, Attribute.LEGAL_NAME,
              RecipientRole.HEALTHCARE_PROVIDER,
              rationale="Patient identity"),
    NormEntry(ContextClass.MEDICAL, Attribute.DOB,
              RecipientRole.HEALTHCARE_PROVIDER,
              rationale="Patient age / date-of-birth for dosing"),
    NormEntry(ContextClass.MEDICAL, Attribute.HEALTH_CONDITION,
              RecipientRole.HEALTHCARE_PROVIDER,
              rationale="Clinical necessity"),
    NormEntry(ContextClass.MEDICAL, Attribute.PHONE,
              RecipientRole.HEALTHCARE_PROVIDER,
              rationale="Appointment / prescription contact"),
    NormEntry(ContextClass.MEDICAL, Attribute.EMAIL,
              RecipientRole.HEALTHCARE_PROVIDER,
              rationale="Appointment / prescription contact"),

    # ── CIVIC ──────────────────────────────────────────────────────────────
    NormEntry(ContextClass.CIVIC, Attribute.LEGAL_NAME,
              RecipientRole.GOVERNMENT_AGENCY,
              rationale="Statutory identity obligation"),
    NormEntry(ContextClass.CIVIC, Attribute.DOB,
              RecipientRole.GOVERNMENT_AGENCY,
              proof_required=True,
              rationale="Age eligibility (voting, benefits); ZKP preferred"),
    NormEntry(ContextClass.CIVIC, Attribute.GEOLOCATION,
              RecipientRole.GOVERNMENT_AGENCY,
              rationale="Electoral roll / residency verification"),
    NormEntry(ContextClass.CIVIC, Attribute.EMAIL,
              RecipientRole.GOVERNMENT_AGENCY,
              rationale="Government service notifications"),
    NormEntry(ContextClass.CIVIC, Attribute.EMPLOYMENT_STATUS,
              RecipientRole.GOVERNMENT_AGENCY,
              rationale="Benefits / tax eligibility"),
    NormEntry(ContextClass.CIVIC, Attribute.INCOME_BRACKET,
              RecipientRole.GOVERNMENT_AGENCY,
              rationale="Tax / means-tested benefits"),
]


class NormDatabase:
    """
    In-memory Φ(c) implementation.

    Lookup is O(n) over the seed table; n ≤ ~30 so indexing is not needed.
    The closed-world default means any unlisted flow is impermissible.
    """

    def __init__(self, entries: Sequence[NormEntry] | None = None) -> None:
        self._entries: list[NormEntry] = list(entries if entries is not None else _SEED_NORMS)

    # ------------------------------------------------------------------
    # Primary query interface
    # ------------------------------------------------------------------

    def permitted(
        self,
        context: ContextClass,
        attribute: Attribute,
        recipient_role: RecipientRole,
    ) -> bool:
        """Return True if the flow is normatively permitted."""
        return any(self._matches(e, context, attribute, recipient_role) for e in self._entries)

    def proof_required(
        self,
        context: ContextClass,
        attribute: Attribute,
        recipient_role: RecipientRole,
    ) -> bool:
        """Return True if a norm requires proof rather than raw disclosure."""
        for e in self._entries:
            if self._matches(e, context, attribute, recipient_role):
                return e.proof_required
        return False

    def permitted_attributes(
        self,
        context: ContextClass,
        recipient_role: RecipientRole,
    ) -> frozenset[Attribute]:
        """All attributes normatively permitted for this (context, role) pair."""
        return frozenset(
            e.attribute for e in self._entries
            if self._matches(e, context, e.attribute, recipient_role)
        )

    def entries_for_context(self, context: ContextClass) -> list[NormEntry]:
        return [e for e in self._entries if e.context == context]

    def all_entries(self) -> Iterator[NormEntry]:
        yield from self._entries

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self, path: Path) -> None:
        data = [
            {
                "context": e.context.value,
                "attribute": e.attribute.value,
                "recipient_role": e.recipient_role.value,
                "proof_required": e.proof_required,
                "rationale": e.rationale,
            }
            for e in self._entries
        ]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "NormDatabase":
        raw = json.loads(path.read_text(encoding="utf-8"))
        entries = [
            NormEntry(
                context=ContextClass(r["context"]),
                attribute=Attribute(r["attribute"]),
                recipient_role=RecipientRole(r["recipient_role"]),
                proof_required=r.get("proof_required", False),
                rationale=r.get("rationale", ""),
            )
            for r in raw
        ]
        return cls(entries)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _matches(
        entry: NormEntry,
        context: ContextClass,
        attribute: Attribute,
        recipient_role: RecipientRole,
    ) -> bool:
        return (
            entry.context == context
            and entry.attribute == attribute
            and entry.recipient_role == recipient_role
        )

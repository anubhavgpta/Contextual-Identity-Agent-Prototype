"""Outgoing disclosure response and receipt types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Mapping, Sequence

from .attributes import Attribute, AttributeValue, AttributeSet
from .context import ContextClass


class NegotiationStatus(str, Enum):
    """
    Terminal status of the 4-phase IACP negotiation.

    ACCEPTED         — CIA disclosed D* after full negotiation.
    PARTIAL          — CIA disclosed a strict subset of what was requested.
    REJECTED         — CIA disclosed nothing (policy/trust violation).
    PROOF_ONLY       — CIA provided ZKP/BBS+ proofs, no raw attributes.
    PENDING          — Negotiation still in progress (transient, not stored).
    """

    ACCEPTED = "ACCEPTED"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"
    PROOF_ONLY = "PROOF_ONLY"
    PENDING = "PENDING"
    FAILED  = "FAILED"    # 3-round cap exhausted with no agreement (IACP only)


@dataclass(frozen=True)
class DisclosureReceipt:
    """
    Tamper-evident audit record written to the disclosure log after every
    completed negotiation.  Stored as a signed JSON blob in SQLite (audit.py).
    """

    receipt_id: str
    request_id: str
    platform_id: str
    context_class: ContextClass
    # D* — the optimal disclosure set chosen by the DDE.
    disclosed_attributes: AttributeSet
    # Attributes requested but withheld.
    withheld_attributes: AttributeSet
    negotiation_status: NegotiationStatus
    # U(D*) as computed by the DDE.
    utility_score: float
    # H(I) before this disclosure.
    entropy_before: float
    # H(I|D*) after — the residual uncertainty.
    entropy_after: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # HMAC-SHA256 over canonical JSON of all other fields.
    signature: str = ""


@dataclass(frozen=True)
class DisclosureResponse:
    """
    What the CIA sends back to the requesting platform in the confirm phase.
    Contains only what the platform should see — no internal scoring.
    """

    request_id: str
    negotiation_status: NegotiationStatus
    # Disclosed attribute values (empty on REJECTED or PROOF_ONLY).
    disclosed_values: tuple[AttributeValue, ...] = field(default_factory=tuple)
    # BBS+ selective-disclosure proofs keyed by attribute (PROOF_ONLY / VERIFY_ONLY).
    bbs_proofs: Mapping[Attribute, bytes] = field(default_factory=dict)
    # zk-SNARK proofs keyed by predicate label (e.g. "age_gte_18").
    zkp_proofs: Mapping[str, bytes] = field(default_factory=dict)
    # Pairwise pseudonymous DID for this (user, platform, context) triple.
    pseudonym_did: str = ""
    # Human-readable explanation surfaced to the user (not sent to platform).
    user_explanation: str = ""
    # Serialised BBS+ SubsetProof covering all disclosed attributes as one blob.
    # Keyed by Attribute for forward-compat with per-attribute proofs; the DDE
    # stores the single SubsetProof bytes under the first (sorted) disclosed attr.
    bbs_subset_proof_bytes: bytes = b""
    # Receipt ID for cross-referencing the audit log.
    receipt_id: str = ""

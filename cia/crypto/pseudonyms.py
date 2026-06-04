"""
Pairwise pseudonymous DIDs via HKDF-SHA256 (RFC 5869).

Each (master_id, context_class) pair yields a stable, unlinkable DID:
    DID = "did:cia:<context>:<32-byte hex>"

Two platforms in the same context class receive the same pseudonym for a
given user — this is intentional per the paper (§2.3): pseudonymity is
context-scoped, not platform-scoped, so intra-context correlation is
permitted while cross-context linkage is prevented.

A colluding pair from different context classes cannot correlate their
pseudonyms because HKDF output is indistinguishable from random under
the PRF security of HMAC-SHA256.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Final

from cia.models.context import ContextClass


_SALT: Final[bytes] = b"cia-pseudonym-v1-salt"


class PseudonymEngine:
    """
    Stateless engine; derive() is deterministic for fixed (master_id, context).
    The master_id is a 32-byte secret stored in the PDS — pass it in on
    construction or per-call.
    """

    # ------------------------------------------------------------------
    # Master-ID generation (call once per user, store result in PDS)
    # ------------------------------------------------------------------

    @staticmethod
    def generate_master_id() -> bytes:
        """Return a cryptographically random 32-byte master identity secret."""
        return secrets.token_bytes(32)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @staticmethod
    def derive(master_id: bytes, context: ContextClass) -> str:
        """
        Derive the context-scoped pseudonymous DID for this user.

        Parameters
        ----------
        master_id:
            The user's 32-byte master secret (from PDS).
        context:
            The context class for which the pseudonym is needed.

        Returns
        -------
        str
            A DID string of the form ``did:cia:<context>:<hex>``.
        """
        if len(master_id) < 16:
            raise ValueError("master_id must be at least 16 bytes")

        prk = _hkdf_extract(_SALT, master_id)
        info = f"cia-did-{context.value}".encode("utf-8")
        okm = _hkdf_expand(prk, info, 32)
        return f"did:cia:{context.value.lower()}:{okm.hex()}"

    @staticmethod
    def verify(did: str, master_id: bytes, context: ContextClass) -> bool:
        """
        Confirm that ``did`` was derived from ``(master_id, context)``.

        Uses a constant-time comparison to prevent timing oracle attacks.
        """
        expected = PseudonymEngine.derive(master_id, context)
        return hmac.compare_digest(did, expected)

    @staticmethod
    def context_from_did(did: str) -> ContextClass | None:
        """
        Parse the context class encoded in the DID string.
        Returns None if the DID is not a valid CIA DID.
        """
        parts = did.split(":")
        if len(parts) != 4 or parts[0] != "did" or parts[1] != "cia":
            return None
        try:
            return ContextClass(parts[2].upper())
        except ValueError:
            return None


# ---------------------------------------------------------------------------
# HKDF-SHA256 (RFC 5869) — implemented directly to avoid adding a dep
# ---------------------------------------------------------------------------

def _hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    """HKDF-Extract: PRK = HMAC-SHA256(salt, IKM)."""
    return hmac.new(salt, ikm, hashlib.sha256).digest()


def _hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    """HKDF-Expand: produce ``length`` bytes of key material."""
    n = (length + 31) // 32  # SHA-256 hash length = 32
    okm = b""
    t = b""
    for i in range(1, n + 1):
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        okm += t
    return okm[:length]

"""
BBS+ selective disclosure over the BN128 pairing curve (§4.2).

Uses py_ecc's optimized_bn128 implementation.

Scheme overview (Boneh–Boyen–Shacham variant with PoK extension):
─────────────────────────────────────────────────────────────────
KeyGen:
    sk ← Zr*          private signing key
    pk = sk · G2      public verification key

Sign(sk, messages=[m₁,…,mₙ]):
    e, s ← Zr*
    B = G1 + s·h₀ + Σᵢ mᵢ·hᵢ      (message commitment)
    A = B · (e + sk)⁻¹              (signature point)
    return (A, e, s)

Verify(pk, messages, sig=(A,e,s)):
    Recompute B; check: e(A, pk + e·G2) == e(B, G2)

ProveSubset(sk, messages, sig, disclosed_indices D):
    r1 ← Zr*
    A' = r1 · A
    Ā  = sk · A'       (= r1·A·sk; verifiable via pairing without revealing sk)
    D  = r1 · B        (randomised base)

    Schnorr for e (statement: D − Ā = e · A'):
        t_e ← Zr;  R_e = t_e · A'
        c   = H(A', Ā, D, R_e, T, G_pub, disclosed_data)
        z_e = (t_e + c·e) mod r

    Schnorr for D decomposition (statement: D = r1·G_pub + (r1s)·h₀ + Σᵢ∈U (r1·mᵢ)·hᵢ):
        G_pub = G1 + Σᵢ∈D mᵢ·hᵢ     (public — computable by verifier)
        t_r1, t_rs, {t_mᵢ} ← Zr
        T  = t_r1·G_pub + t_rs·h₀ + Σᵢ∈U t_mᵢ·hᵢ
        z_r1 = (t_r1 + c·r1) mod r
        z_rs = (t_rs + c·(r1·s)) mod r
        z_mᵢ = (t_mᵢ + c·(r1·mᵢ)) mod r  for i ∈ U

VerifySubset(pk, disclosed_msgs, disclosed_indices, proof):
    1. Check e(G2, Ā) == e(pk, A')                     [Ā = sk·A']
    2. Recompute G_pub; check z_e·A' == R_e + c·(D − Ā) [e binds D to (A',Ā)]
    3. Check z_r1·G_pub + z_rs·h₀ + Σ z_mᵢ·hᵢ == T + c·D
    4. Verify Fiat–Shamir: c == H(A', Ā, D, R_e, T, G_pub, disclosed_data)
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field
from typing import Sequence

try:
    from py_ecc.optimized_bn128 import (
        G1, G2, Z1,
        add, multiply, neg, pairing,
        curve_order, normalize, is_inf,
    )
    _PY_ECC_AVAILABLE = True
except ImportError:
    _PY_ECC_AVAILABLE = False


def _require_py_ecc() -> None:
    if not _PY_ECC_AVAILABLE:
        raise ImportError("py_ecc is required for BBS+: pip install py_ecc")


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# A G1/G2 point as returned by py_ecc (Jacobian 3-tuple internally; we
# normalize to affine 2-tuple only for serialisation / hashing).
G1Point = tuple
G2Point = tuple


@dataclass(frozen=True)
class BBSKeyPair:
    """BBS+ key pair. sk is an integer in Zr*; pk is a G2 point."""

    sk: int
    pk: G2Point


@dataclass(frozen=True)
class BBSSignature:
    """BBS+ signature over a list of messages."""

    A: G1Point
    e: int
    s: int
    # Number of messages this signature covers (needed to re-derive generators).
    n_messages: int


@dataclass(frozen=True)
class SubsetProof:
    """
    Non-interactive PoK of a BBS+ signature with selective disclosure.

    The verifier only learns the disclosed_messages; hidden messages are
    computationally indistinguishable from noise.
    """

    A_prime: G1Point          # r1 · A
    A_bar: G1Point            # sk · A_prime  (= r1·A·sk)
    D: G1Point                # r1 · B        (randomised base)
    R_e: G1Point              # Schnorr nonce commitment for e
    T: G1Point                # Schnorr nonce commitment for D decomposition
    c: int                    # Fiat–Shamir challenge
    z_e: int                  # Schnorr response for e
    z_r1: int                 # Schnorr response for r1
    z_rs: int                 # Schnorr response for r1·s
    z_m: dict[int, int]       # Schnorr responses for hidden messages {index: response}
    disclosed_indices: tuple[int, ...]
    disclosed_messages: tuple[bytes, ...]
    n_messages: int


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BBSEngine:
    """
    Stateless BBS+ engine.  All methods are deterministic given their inputs
    (randomness is generated internally).

    Generator derivation: hᵢ = (sha256("cia-bbs-gen-{i}") mod r) · G1
    This is a simplified hash-to-curve suitable for a research prototype.
    """

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    @staticmethod
    def keygen() -> BBSKeyPair:
        _require_py_ecc()
        sk = _random_scalar()
        pk = multiply(G2, sk)
        return BBSKeyPair(sk=sk, pk=pk)

    # ------------------------------------------------------------------
    # Sign
    # ------------------------------------------------------------------

    @staticmethod
    def sign(sk: int, messages: list[bytes]) -> BBSSignature:
        """
        Sign a list of messages.  Message order is significant — verify
        and prove_subset must use the same ordering.
        """
        _require_py_ecc()
        n = len(messages)
        e = _random_scalar()
        s = _random_scalar()
        B = _compute_B(s, messages)
        e_plus_sk_inv = pow(e + sk, curve_order - 2, curve_order)
        A = multiply(B, e_plus_sk_inv)
        return BBSSignature(A=A, e=e, s=s, n_messages=n)

    # ------------------------------------------------------------------
    # Verify (full message set)
    # ------------------------------------------------------------------

    @staticmethod
    def verify(pk: G2Point, messages: list[bytes], sig: BBSSignature) -> bool:
        """
        Verify a BBS+ signature over the full message list.
        Returns False on any mismatch; never raises.
        """
        _require_py_ecc()
        try:
            B = _compute_B(sig.s, messages)
            lhs = pairing(add(pk, multiply(G2, sig.e)), sig.A)
            rhs = pairing(G2, B)
            return lhs == rhs
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Selective-disclosure proof
    # ------------------------------------------------------------------

    @staticmethod
    def prove_subset(
        sk: int,
        pk: G2Point,
        messages: list[bytes],
        sig: BBSSignature,
        disclosed_indices: Sequence[int],
    ) -> SubsetProof:
        """
        Create a non-interactive PoK revealing only the messages at
        ``disclosed_indices``.  All other messages are computationally hidden.
        """
        _require_py_ecc()
        n = len(messages)
        D_idx = sorted(disclosed_indices)
        U_idx = [i for i in range(n) if i not in set(D_idx)]

        # ── Randomise signature ─────────────────────────────────────
        r1 = _random_scalar()
        A_prime = multiply(sig.A, r1)
        A_bar = multiply(A_prime, sk)        # = r1·A·sk (pairing-verifiable)
        B = _compute_B(sig.s, messages)
        D_pt = multiply(B, r1)               # = r1·B

        # ── G_pub: public point computable by verifier ───────────────
        G_pub = _compute_G_pub(D_idx, messages)

        # ── Schnorr nonces ───────────────────────────────────────────
        t_e  = _random_scalar()
        t_r1 = _random_scalar()
        t_rs = _random_scalar()
        t_m  = {i: _random_scalar() for i in U_idx}

        R_e = multiply(A_prime, t_e)
        T_pt = add(
            add(multiply(G_pub, t_r1), multiply(_h(0), t_rs)),
            _multiexp([(i + 1, t_m[i]) for i in U_idx]),
        )

        # ── Fiat–Shamir challenge ────────────────────────────────────
        c = _fiat_shamir(
            A_prime, A_bar, D_pt, R_e, T_pt, G_pub,
            D_idx, [messages[i] for i in D_idx],
        )

        # ── Responses ───────────────────────────────────────────────
        r = curve_order
        z_e  = (t_e  + c * sig.e)                    % r
        z_r1 = (t_r1 + c * r1)                       % r
        z_rs = (t_rs + c * (r1 * sig.s % r))         % r
        z_m  = {i: (t_m[i] + c * (r1 * _msg_int(messages[i]) % r)) % r
                for i in U_idx}

        return SubsetProof(
            A_prime=A_prime,
            A_bar=A_bar,
            D=D_pt,
            R_e=R_e,
            T=T_pt,
            c=c,
            z_e=z_e,
            z_r1=z_r1,
            z_rs=z_rs,
            z_m=z_m,
            disclosed_indices=tuple(D_idx),
            disclosed_messages=tuple(messages[i] for i in D_idx),
            n_messages=n,
        )

    # ------------------------------------------------------------------
    # Verify subset proof
    # ------------------------------------------------------------------

    @staticmethod
    def verify_subset(pk: G2Point, proof: SubsetProof) -> bool:
        """
        Verify a selective-disclosure proof.  The caller learns only the
        disclosed (index, message) pairs; hidden messages are not needed.

        Returns False on any failed check; never raises.
        """
        _require_py_ecc()
        try:
            D_idx = list(proof.disclosed_indices)
            disc_msgs = list(proof.disclosed_messages)

            # ── Check 1: pairing equation (Ā = sk·A') ───────────────
            if pairing(G2, proof.A_bar) != pairing(pk, proof.A_prime):
                return False

            # ── Check 2: Schnorr for e (D − Ā = e·A') ───────────────
            # z_e·A' == R_e + c·(D − Ā)
            lhs2 = multiply(proof.A_prime, proof.z_e)
            rhs2 = add(proof.R_e, multiply(add(proof.D, neg(proof.A_bar)), proof.c))
            if not _points_eq(lhs2, rhs2):
                return False

            # ── Check 3: Schnorr for D decomposition ────────────────
            # z_r1·G_pub + z_rs·h₀ + Σ z_mᵢ·hᵢ == T + c·D
            G_pub = _compute_G_pub(D_idx, disc_msgs)
            acc = add(
                add(multiply(G_pub, proof.z_r1), multiply(_h(0), proof.z_rs)),
                _multiexp([(i + 1, proof.z_m[i]) for i in proof.z_m]),
            )
            rhs3 = add(proof.T, multiply(proof.D, proof.c))
            if not _points_eq(acc, rhs3):
                return False

            # ── Check 4: Fiat–Shamir consistency ────────────────────
            c_prime = _fiat_shamir(
                proof.A_prime, proof.A_bar, proof.D,
                proof.R_e, proof.T, G_pub,
                D_idx, disc_msgs,
            )
            return c_prime == proof.c

        except Exception:
            return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _h(index: int) -> G1Point:
    """Deterministic generator hᵢ = sha256("cia-bbs-gen-{i}") · G1."""
    digest = hashlib.sha256(f"cia-bbs-gen-{index}".encode()).digest()
    scalar = int.from_bytes(digest, "big") % curve_order or 1
    return multiply(G1, scalar)


def _msg_int(msg: bytes) -> int:
    """Map a message byte-string to a non-zero field element."""
    val = int.from_bytes(hashlib.sha256(msg).digest(), "big") % curve_order
    return val or 1


def _compute_B(s: int, messages: list[bytes]) -> G1Point:
    """B = G1 + s·h₀ + Σᵢ mᵢ·hᵢ  (1-indexed generators for messages)."""
    acc = add(G1, multiply(_h(0), s))
    for i, msg in enumerate(messages):
        acc = add(acc, multiply(_h(i + 1), _msg_int(msg)))
    return acc


def _compute_G_pub(
    disclosed_indices: list[int],
    disclosed_messages: list[bytes],
) -> G1Point:
    """G_pub = G1 + Σᵢ∈D mᵢ·hᵢ  (known to the verifier)."""
    acc = G1
    for idx, msg in zip(disclosed_indices, disclosed_messages):
        acc = add(acc, multiply(_h(idx + 1), _msg_int(msg)))
    return acc


def _multiexp(index_scalar_pairs: list[tuple[int, int]]) -> G1Point:
    """Σ scalar·hᵢ for (index, scalar) pairs; returns Z1 if empty."""
    acc: G1Point = Z1
    for idx, scalar in index_scalar_pairs:
        acc = add(acc, multiply(_h(idx), scalar))
    return acc


def _point_bytes(pt: G1Point) -> bytes:
    """Serialise a G1 point to bytes for hashing (affine coordinates)."""
    if is_inf(pt):
        return b"\x00" * 64
    nx, ny = normalize(pt)
    return int(nx).to_bytes(32, "big") + int(ny).to_bytes(32, "big")


def _points_eq(a: G1Point, b: G1Point) -> bool:
    """Constant-structure equality check on G1 points (affine)."""
    if is_inf(a) and is_inf(b):
        return True
    if is_inf(a) or is_inf(b):
        return False
    na, nb = normalize(a), normalize(b)
    return int(na[0]) == int(nb[0]) and int(na[1]) == int(nb[1])


def _fiat_shamir(
    A_prime: G1Point,
    A_bar: G1Point,
    D: G1Point,
    R_e: G1Point,
    T: G1Point,
    G_pub: G1Point,
    disclosed_indices: list[int],
    disclosed_messages: list[bytes],
) -> int:
    """Fiat–Shamir hash binding all public proof elements."""
    h = hashlib.sha256()
    for pt in (A_prime, A_bar, D, R_e, T, G_pub):
        h.update(_point_bytes(pt))
    for idx, msg in zip(disclosed_indices, disclosed_messages):
        h.update(idx.to_bytes(4, "big"))
        h.update(msg)
    return int.from_bytes(h.digest(), "big") % curve_order or 1


def _random_scalar() -> int:
    """Uniform random scalar in [1, curve_order)."""
    while True:
        val = int.from_bytes(secrets.token_bytes(32), "big") % curve_order
        if val != 0:
            return val

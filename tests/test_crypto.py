"""
Tests for cia.crypto — pseudonyms, BBS+, ZKP.

BBS+ tests are marked slow; skip with: pytest -m "not slow"
ZKP tests that require compiled circuits are skipped automatically when
the .wasm / .zkey artifacts are absent.
"""

from __future__ import annotations

import json
import secrets
from pathlib import Path

import pytest

from cia.models.context import ContextClass
from cia.crypto.pseudonyms import PseudonymEngine, _hkdf_extract, _hkdf_expand
from cia.crypto.zkp import ZKPEngine, ZKProof, CircuitNotCompiledError

py_ecc = pytest.importorskip("py_ecc", reason="py_ecc not installed")
from cia.crypto.bbs import BBSEngine, BBSKeyPair, BBSSignature, SubsetProof


# ---------------------------------------------------------------------------
# PseudonymEngine
# ---------------------------------------------------------------------------

class TestPseudonymEngine:
    def test_generate_master_id_32_bytes(self):
        mid = PseudonymEngine.generate_master_id()
        assert isinstance(mid, bytes) and len(mid) == 32

    def test_derive_returns_did_string(self):
        mid = PseudonymEngine.generate_master_id()
        did = PseudonymEngine.derive(mid, ContextClass.FINANCIAL)
        assert did.startswith("did:cia:financial:")
        assert len(did) == len("did:cia:financial:") + 64

    def test_derive_is_deterministic(self):
        mid = PseudonymEngine.generate_master_id()
        d1 = PseudonymEngine.derive(mid, ContextClass.FINANCIAL)
        d2 = PseudonymEngine.derive(mid, ContextClass.FINANCIAL)
        assert d1 == d2

    def test_different_contexts_produce_different_dids(self):
        mid = PseudonymEngine.generate_master_id()
        dids = {PseudonymEngine.derive(mid, ctx) for ctx in ContextClass}
        assert len(dids) == len(ContextClass)

    def test_different_master_ids_produce_different_dids(self):
        ctx = ContextClass.SOCIAL
        d1 = PseudonymEngine.derive(secrets.token_bytes(32), ctx)
        d2 = PseudonymEngine.derive(secrets.token_bytes(32), ctx)
        assert d1 != d2

    def test_verify_accepts_correct_pair(self):
        mid = PseudonymEngine.generate_master_id()
        did = PseudonymEngine.derive(mid, ContextClass.MEDICAL)
        assert PseudonymEngine.verify(did, mid, ContextClass.MEDICAL)

    def test_verify_rejects_wrong_context(self):
        mid = PseudonymEngine.generate_master_id()
        did = PseudonymEngine.derive(mid, ContextClass.MEDICAL)
        assert not PseudonymEngine.verify(did, mid, ContextClass.FINANCIAL)

    def test_verify_rejects_wrong_master_id(self):
        mid1 = PseudonymEngine.generate_master_id()
        mid2 = PseudonymEngine.generate_master_id()
        did = PseudonymEngine.derive(mid1, ContextClass.CIVIC)
        assert not PseudonymEngine.verify(did, mid2, ContextClass.CIVIC)

    def test_context_from_did_roundtrip(self):
        mid = PseudonymEngine.generate_master_id()
        for ctx in ContextClass:
            did = PseudonymEngine.derive(mid, ctx)
            assert PseudonymEngine.context_from_did(did) == ctx

    def test_context_from_did_invalid(self):
        assert PseudonymEngine.context_from_did("not:a:valid:did") is None
        assert PseudonymEngine.context_from_did("did:cia:unknown:abcdef") is None

    def test_short_master_id_raises(self):
        with pytest.raises(ValueError):
            PseudonymEngine.derive(b"short", ContextClass.FINANCIAL)

    def test_hkdf_expand_correct_length(self):
        prk = secrets.token_bytes(32)
        okm = _hkdf_expand(prk, b"test", 64)
        assert len(okm) == 64

    def test_same_platform_same_context_same_pseudonym(self):
        """Two platforms in the same context get the same pseudonym (§2.3)."""
        mid = PseudonymEngine.generate_master_id()
        ctx = ContextClass.PROFESSIONAL
        did_plat_a = PseudonymEngine.derive(mid, ctx)
        did_plat_b = PseudonymEngine.derive(mid, ctx)
        assert did_plat_a == did_plat_b


# ---------------------------------------------------------------------------
# BBSEngine — marked slow because py_ecc pairings take ~1-2s each
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestBBSEngine:
    def setup_method(self):
        self.engine = BBSEngine()
        self.kp = BBSEngine.keygen()

    def test_keygen_types(self):
        assert isinstance(self.kp.sk, int)
        assert self.kp.sk > 0
        assert isinstance(self.kp.pk, tuple)

    def test_sign_and_verify_single_message(self):
        msgs = [b"hello"]
        sig = BBSEngine.sign(self.kp.sk, msgs)
        assert BBSEngine.verify(self.kp.pk, msgs, sig)

    def test_sign_and_verify_multiple_messages(self):
        msgs = [b"attr1", b"attr2", b"attr3"]
        sig = BBSEngine.sign(self.kp.sk, msgs)
        assert BBSEngine.verify(self.kp.pk, msgs, sig)

    def test_verify_rejects_wrong_message(self):
        msgs = [b"original"]
        sig = BBSEngine.sign(self.kp.sk, msgs)
        assert not BBSEngine.verify(self.kp.pk, [b"tampered"], sig)

    def test_verify_rejects_wrong_key(self):
        msgs = [b"msg"]
        sig = BBSEngine.sign(self.kp.sk, msgs)
        other_kp = BBSEngine.keygen()
        assert not BBSEngine.verify(other_kp.pk, msgs, sig)

    def test_subset_proof_all_disclosed(self):
        msgs = [b"name", b"email", b"age"]
        sig = BBSEngine.sign(self.kp.sk, msgs)
        proof = BBSEngine.prove_subset(self.kp.sk, self.kp.pk, msgs, sig, [0, 1, 2])
        assert BBSEngine.verify_subset(self.kp.pk, proof)

    def test_subset_proof_partial_disclosure(self):
        msgs = [b"name", b"dob", b"salary"]
        sig = BBSEngine.sign(self.kp.sk, msgs)
        # Disclose only name; hide dob and salary.
        proof = BBSEngine.prove_subset(self.kp.sk, self.kp.pk, msgs, sig, [0])
        assert BBSEngine.verify_subset(self.kp.pk, proof)
        assert proof.disclosed_indices == (0,)
        assert proof.disclosed_messages == (b"name",)
        assert 1 in proof.z_m  # response for hidden index 1 (dob)
        assert 2 in proof.z_m  # response for hidden index 2 (salary)

    def test_subset_proof_no_disclosure(self):
        msgs = [b"secret1", b"secret2"]
        sig = BBSEngine.sign(self.kp.sk, msgs)
        proof = BBSEngine.prove_subset(self.kp.sk, self.kp.pk, msgs, sig, [])
        assert BBSEngine.verify_subset(self.kp.pk, proof)
        assert proof.disclosed_indices == ()
        assert len(proof.z_m) == 2

    def test_subset_proof_rejects_tampered_disclosed_message(self):
        msgs = [b"real_name", b"real_email"]
        sig = BBSEngine.sign(self.kp.sk, msgs)
        proof = BBSEngine.prove_subset(self.kp.sk, self.kp.pk, msgs, sig, [0])
        # Attacker swaps the disclosed message.
        tampered = SubsetProof(
            A_prime=proof.A_prime,
            A_bar=proof.A_bar,
            D=proof.D,
            R_e=proof.R_e,
            T=proof.T,
            c=proof.c,
            z_e=proof.z_e,
            z_r1=proof.z_r1,
            z_rs=proof.z_rs,
            z_m=proof.z_m,
            disclosed_indices=proof.disclosed_indices,
            disclosed_messages=(b"fake_name",),  # tampered
            n_messages=proof.n_messages,
        )
        assert not BBSEngine.verify_subset(self.kp.pk, tampered)

    def test_subset_proof_rejects_wrong_key(self):
        msgs = [b"attr"]
        sig = BBSEngine.sign(self.kp.sk, msgs)
        proof = BBSEngine.prove_subset(self.kp.sk, self.kp.pk, msgs, sig, [0])
        other_kp = BBSEngine.keygen()
        assert not BBSEngine.verify_subset(other_kp.pk, proof)

    def test_two_proofs_for_same_sig_are_unlinkable(self):
        """Each call to prove_subset produces a different randomised proof."""
        msgs = [b"m1", b"m2"]
        sig = BBSEngine.sign(self.kp.sk, msgs)
        p1 = BBSEngine.prove_subset(self.kp.sk, self.kp.pk, msgs, sig, [0])
        p2 = BBSEngine.prove_subset(self.kp.sk, self.kp.pk, msgs, sig, [0])
        # A_prime must differ (different r1 each time).
        from cia.crypto.bbs import _points_eq
        assert not _points_eq(p1.A_prime, p2.A_prime)


# ---------------------------------------------------------------------------
# ZKPEngine
# ---------------------------------------------------------------------------

_CIRCUITS_DIR = Path(__file__).parent.parent / "cia" / "assets" / "circuits"
_AGE_CIRCUIT_COMPILED = (
    (_CIRCUITS_DIR / "age_gte_18" / "age_gte_18.wasm").exists()
    and (_CIRCUITS_DIR / "age_gte_18" / "age_gte_18_final.zkey").exists()
)


class TestZKPEngine:
    def setup_method(self):
        self.engine = ZKPEngine()

    def test_missing_circuit_raises_circuit_not_compiled(self):
        with pytest.raises(CircuitNotCompiledError):
            self.engine.generate_proof(
                "nonexistent_circuit",
                private_inputs={"x": 1},
                public_inputs={"y": 2},
            )

    def test_zkproof_json_roundtrip(self):
        proof = ZKProof(
            circuit_name="age_gte_18",
            proof={"pi_a": ["1", "2"], "pi_b": [["3", "4"]], "pi_c": ["5", "6"],
                   "protocol": "groth16", "curve": "bn128"},
            public_inputs=["2006"],
        )
        restored = ZKProof.from_json(proof.to_json())
        assert restored.circuit_name == proof.circuit_name
        assert restored.proof == proof.proof
        assert restored.public_inputs == proof.public_inputs

    @pytest.mark.skipif(
        not _AGE_CIRCUIT_COMPILED,
        reason="age_gte_18 circuit not compiled — run python cia/assets/compile_circuits.py",
    )
    def test_prove_age_gte_18_valid(self):
        from datetime import date
        threshold = date.today().year - 18
        proof = self.engine.prove_age_gte_18(birth_year=threshold - 10)
        assert isinstance(proof, ZKProof)
        assert proof.circuit_name == "age_gte_18"

    @pytest.mark.skipif(
        not _AGE_CIRCUIT_COMPILED,
        reason="age_gte_18 circuit not compiled — run python cia/assets/compile_circuits.py",
    )
    def test_verify_age_gte_18_valid(self):
        from datetime import date
        threshold = date.today().year - 18
        proof = self.engine.prove_age_gte_18(birth_year=threshold - 5)
        assert self.engine.verify_age_gte_18(proof)

    @pytest.mark.skipif(
        not _AGE_CIRCUIT_COMPILED,
        reason="age_gte_18 circuit not compiled — run python cia/assets/compile_circuits.py",
    )
    def test_prove_age_gte_18_too_young_fails(self):
        """Witness that doesn't satisfy the circuit should raise."""
        from datetime import date
        threshold = date.today().year - 18
        with pytest.raises(Exception):
            # born next year — threshold_year - birth_year is negative (wraps)
            self.engine.prove_age_gte_18(birth_year=threshold + 2)

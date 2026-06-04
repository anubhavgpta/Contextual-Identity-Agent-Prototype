"""
zk-SNARK predicate proofs via snarkjs subprocess (§4.3).

Supports the VERIFY_ONLY policy action: the user proves a predicate over
a private attribute without revealing the attribute value.

Supported circuits
──────────────────
  age_gte_18   Proves birth_year satisfies age ≥ 18 without revealing it.
               Private input:  birth_year  (e.g. 1990)
               Public input:   threshold_year  (= current_year − 18)

Circuit artifacts must be pre-compiled and present in:
    assets/circuits/<circuit_name>/
        <circuit_name>.wasm          (compiled witness generator)
        <circuit_name>_final.zkey   (proving key)
        verification_key.json        (for verification)

Run ``python cia/assets/compile_circuits.py`` once to produce these files.
If the artifacts are absent, generate_proof() raises CircuitNotCompiledError.
If node / snarkjs is not available, a RuntimeError with install instructions
is raised before any subprocess call.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


# Root of the circuits directory relative to this file's package root.
_CIRCUITS_DIR = Path(__file__).parent.parent / "assets" / "circuits"

# Cached node / npx paths; set lazily.
_NODE_PATH: str | None = None
_SNARKJS_CHECKED: bool = False


class CircuitNotCompiledError(FileNotFoundError):
    """Raised when the required circuit artifacts are not present on disk."""


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ZKProof:
    """Groth16 proof bundle as returned by snarkjs."""

    circuit_name: str
    # JSON-serialisable proof object from snarkjs (pi_a, pi_b, pi_c, protocol).
    proof: dict
    # Public inputs as a list of decimal strings (snarkjs format).
    public_inputs: list[str]

    def to_json(self) -> str:
        return json.dumps({
            "circuit_name": self.circuit_name,
            "proof": self.proof,
            "public_inputs": self.public_inputs,
        }, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> "ZKProof":
        d = json.loads(raw)
        return cls(
            circuit_name=d["circuit_name"],
            proof=d["proof"],
            public_inputs=d["public_inputs"],
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ZKPEngine:
    """
    Thin wrapper around snarkjs CLI.  All I/O goes through temporary files;
    no state is retained between calls.
    """

    def __init__(self, circuits_dir: Path | None = None) -> None:
        self._circuits_dir = circuits_dir or _CIRCUITS_DIR

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------

    def prove_age_gte_18(self, birth_year: int) -> ZKProof:
        """
        Prove that the user is at least 18 years old without revealing
        their exact birth year.

        ``threshold_year`` is derived from today's date automatically.
        """
        threshold_year = date.today().year - 18
        return self.generate_proof(
            "age_gte_18",
            private_inputs={"birth_year": birth_year},
            public_inputs={"threshold_year": threshold_year},
        )

    def verify_age_gte_18(self, proof: ZKProof) -> bool:
        """Verify an age-gte-18 proof."""
        return self.verify_proof("age_gte_18", proof)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def generate_proof(
        self,
        circuit_name: str,
        private_inputs: dict,
        public_inputs: dict,
    ) -> ZKProof:
        """
        Generate a Groth16 zk-SNARK proof for ``circuit_name``.

        Parameters
        ----------
        circuit_name:
            Name of the circuit (must match a sub-directory under circuits/).
        private_inputs:
            Witness values — private to the prover; not in the proof.
        public_inputs:
            Public statement values — included in the proof's public signals.

        Raises
        ------
        CircuitNotCompiledError
            If the compiled circuit artifacts are missing.
        RuntimeError
            If node.js is not found on PATH.
        subprocess.CalledProcessError
            If snarkjs exits non-zero (e.g. witness doesn't satisfy constraints).
        """
        self._check_node()
        circuit_dir = self._require_circuit(circuit_name)

        wasm = circuit_dir / f"{circuit_name}.wasm"
        zkey = circuit_dir / f"{circuit_name}_final.zkey"

        # snarkjs g16f wants all inputs (private + public) merged.
        all_inputs = {**private_inputs, **public_inputs}

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_file = tmp_path / "input.json"
            proof_file = tmp_path / "proof.json"
            public_file = tmp_path / "public.json"

            input_file.write_text(json.dumps(all_inputs), encoding="utf-8")

            _run_snarkjs([
                "groth16", "fullprove",
                str(input_file), str(wasm), str(zkey),
                str(proof_file), str(public_file),
            ])

            proof_data = json.loads(proof_file.read_text(encoding="utf-8"))
            public_data = json.loads(public_file.read_text(encoding="utf-8"))

        return ZKProof(
            circuit_name=circuit_name,
            proof=proof_data,
            public_inputs=public_data,
        )

    def verify_proof(self, circuit_name: str, proof: ZKProof) -> bool:
        """
        Verify a Groth16 proof.

        Returns True iff snarkjs reports the proof as valid.
        Returns False on any verification failure (not on system errors).

        Raises
        ------
        CircuitNotCompiledError
            If verification_key.json is missing.
        RuntimeError
            If node.js is not found on PATH.
        """
        self._check_node()
        circuit_dir = self._require_circuit(circuit_name)
        vk = circuit_dir / "verification_key.json"
        if not vk.exists():
            raise CircuitNotCompiledError(
                f"verification_key.json not found in {circuit_dir}. "
                "Run: python cia/assets/compile_circuits.py"
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            proof_file = tmp_path / "proof.json"
            public_file = tmp_path / "public.json"

            proof_file.write_text(
                json.dumps(proof.proof), encoding="utf-8"
            )
            public_file.write_text(
                json.dumps(proof.public_inputs), encoding="utf-8"
            )

            result = _run_snarkjs(
                ["groth16", "verify", str(vk), str(public_file), str(proof_file)],
                check=False,
            )

        # snarkjs prints "OK!" to stdout on success.
        return result.returncode == 0 and "OK" in result.stdout

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_circuit(self, circuit_name: str) -> Path:
        circuit_dir = self._circuits_dir / circuit_name
        wasm = circuit_dir / f"{circuit_name}.wasm"
        zkey = circuit_dir / f"{circuit_name}_final.zkey"
        if not wasm.exists() or not zkey.exists():
            raise CircuitNotCompiledError(
                f"Compiled artifacts for circuit '{circuit_name}' not found "
                f"in {circuit_dir}.\n"
                "Run: python cia/assets/compile_circuits.py\n"
                "Requires: circom (https://docs.circom.io/getting-started/installation/) "
                "and node.js."
            )
        return circuit_dir

    @staticmethod
    def _check_node() -> None:
        global _NODE_PATH, _SNARKJS_CHECKED
        if _SNARKJS_CHECKED:
            return
        node = shutil.which("node")
        if node is None:
            raise RuntimeError(
                "node.js not found on PATH. Install it from https://nodejs.org/\n"
                "snarkjs is invoked via: npx snarkjs <command>"
            )
        _NODE_PATH = node
        _SNARKJS_CHECKED = True


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------

def _run_snarkjs(
    args: list[str],
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run snarkjs via npx with the given arguments."""
    cmd = ["npx", "--yes", "snarkjs"] + args
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=check,
    )

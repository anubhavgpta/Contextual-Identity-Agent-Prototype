"""
compile_circuits.py — one-time setup script for all CIA zk-SNARK circuits.

Run this once before using zkp.py:
    python cia/assets/compile_circuits.py

Prerequisites
─────────────
  circom  ≥ 2.0   https://docs.circom.io/getting-started/installation/
  node.js ≥ 16    https://nodejs.org/
  snarkjs         installed automatically via npx

What this script does (per circuit)
────────────────────────────────────
  1. circom <name>.circom --r1cs --wasm --sym -o <out_dir>
  2. npx snarkjs powersoftau new bn128 12 pot12_0000.ptau
  3. npx snarkjs powersoftau contribute pot12_0000.ptau pot12_0001.ptau
  4. npx snarkjs powersoftau prepare phase2 pot12_0001.ptau pot12_final.ptau
  5. npx snarkjs groth16 setup <name>.r1cs pot12_final.ptau <name>_0000.zkey
  6. npx snarkjs zkey contribute <name>_0000.zkey <name>_final.zkey
  7. npx snarkjs zkey export verificationkey <name>_final.zkey verification_key.json

Output files (placed in assets/circuits/<name>/):
    <name>.wasm            witness generator (used by generate_proof)
    <name>_final.zkey      proving key       (used by generate_proof)
    verification_key.json  verification key  (used by verify_proof)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


CIRCUITS_DIR = Path(__file__).parent / "circuits"
CIRCUITS = ["age_gte_18"]


def run(cmd: list[str], cwd: Path) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def check_prerequisites() -> None:
    missing = []
    if shutil.which("circom") is None:
        missing.append(
            "circom  — install from https://docs.circom.io/getting-started/installation/"
        )
    if shutil.which("node") is None:
        missing.append("node.js — install from https://nodejs.org/")
    if missing:
        print("Missing prerequisites:\n" + "\n".join(f"  {m}" for m in missing))
        sys.exit(1)


def compile_circuit(name: str) -> None:
    circuit_dir = CIRCUITS_DIR / name
    circom_src = circuit_dir / f"{name}.circom"
    if not circom_src.exists():
        print(f"ERROR: {circom_src} not found — skipping.")
        return

    print(f"\n=== Compiling {name} ===")

    # Step 1: compile circom -> r1cs + wasm
    run(
        ["circom", str(circom_src), "--r1cs", "--wasm", "--sym", "-o", str(circuit_dir)],
        cwd=circuit_dir,
    )

    # wasm is placed in <name>_js/<name>.wasm by circom 2.x — move it up.
    wasm_src = circuit_dir / f"{name}_js" / f"{name}.wasm"
    wasm_dst = circuit_dir / f"{name}.wasm"
    if wasm_src.exists():
        wasm_src.rename(wasm_dst)

    # Step 2-4: Powers of Tau (ceremony + phase2 prep)
    ptau0 = circuit_dir / "pot12_0000.ptau"
    ptau1 = circuit_dir / "pot12_0001.ptau"
    ptau_final = circuit_dir / "pot12_final.ptau"

    run(["npx", "--yes", "snarkjs", "powersoftau", "new", "bn128", "12", str(ptau0)],
        cwd=circuit_dir)
    run(["npx", "snarkjs", "powersoftau", "contribute",
         str(ptau0), str(ptau1), "--name=CIA compile", "-e=cia-entropy-1"],
        cwd=circuit_dir)
    run(["npx", "snarkjs", "powersoftau", "prepare", "phase2",
         str(ptau1), str(ptau_final)],
        cwd=circuit_dir)

    # Step 5-6: Groth16 setup + contribution
    r1cs = circuit_dir / f"{name}.r1cs"
    zkey0 = circuit_dir / f"{name}_0000.zkey"
    zkey_final = circuit_dir / f"{name}_final.zkey"
    vk = circuit_dir / "verification_key.json"

    run(["npx", "snarkjs", "groth16", "setup", str(r1cs), str(ptau_final), str(zkey0)],
        cwd=circuit_dir)
    run(["npx", "snarkjs", "zkey", "contribute",
         str(zkey0), str(zkey_final), "--name=CIA phase2", "-e=cia-entropy-2"],
        cwd=circuit_dir)

    # Step 7: export verification key
    run(["npx", "snarkjs", "zkey", "export", "verificationkey",
         str(zkey_final), str(vk)],
        cwd=circuit_dir)

    # Clean up intermediary files (keep only .wasm, _final.zkey, verification_key.json)
    for f in [ptau0, ptau1, ptau_final, zkey0]:
        f.unlink(missing_ok=True)
    js_dir = circuit_dir / f"{name}_js"
    if js_dir.exists():
        shutil.rmtree(js_dir)

    print(f"  ✓ {name}: wasm + zkey + verification_key written to {circuit_dir}")


if __name__ == "__main__":
    check_prerequisites()
    for circuit in CIRCUITS:
        compile_circuit(circuit)
    print("\nAll circuits compiled. zkp.py is ready to use.")

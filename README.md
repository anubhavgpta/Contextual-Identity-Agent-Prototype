# Contextual Identity Agent (CIA)

**Prototype implementation for:**
> *Autonomous Identity Disclosure via Contextual Agents in Multi-Platform Systems*
> Anubhav Gupta & Aditi Shailesh Joshi

---

## Overview

The Contextual Identity Agent (CIA) is a Python prototype of a privacy-preserving identity disclosure system. Rather than disclosing all requested attributes unconditionally (as OAuth 2.0 and unmediated flows do), or requiring explicit per-attribute user approval on every interaction (as self-sovereign identity wallets do), the CIA acts as an autonomous agent on behalf of the user: it classifies the requesting context, evaluates a utility function over candidate disclosure sets, applies contextual integrity norms, and negotiates the minimum sufficient disclosure with the relying party — all without user interruption in the common case.

The system is grounded in three formal guarantees:

1. **Disclosure minimisation** — the agent computes the optimal disclosure set D\* ⊆ A\_req that maximises a parameterised utility function U(D) subject to norm and policy constraints, rather than the entire requested set.
2. **Re-identification resistance** — salience weights and entropy tracking cause the agent to become progressively more conservative as accumulated disclosures approach a re-identification threshold.
3. **Cross-platform unlinkability** — context-scoped pseudonymous DIDs and BBS+ selective-disclosure proofs prevent colluding platforms from linking a user's identity across contexts.

### Key results (7,500 simulated IACP interactions, 500 per platform per tier)

| Metric | CIA | SSI Wallet | OAuth / Unmediated |
|---|---|---|---|
| Disclosure Minimisation Rate | **86.1%** | 35.1% | 0.0% |
| Re-ID Risk Reduction (vs OAuth) | **91.8%** | — | baseline |
| Cross-Platform Linkability | **Low** | Medium | High |
| User Actions Required | **Low** | High | Low |

Paper targets: DMR ≥ 68%, re-ID reduction ≥ 61%. Both are exceeded.

---

## Architecture

The system is structured as a strict layered pipeline. Each layer imports only from layers below it; the dependency graph is a DAG.

```
          ┌─────────────────────────────────────────┐
          │            main.py / eval/               │  CLI, simulation, metrics
          └───────────────────┬─────────────────────┘
                              │
          ┌───────────────────▼─────────────────────┐
          │                iacp.py                   │  Inter-Agent Comm. Protocol
          └──────┬──────────────┬────────────────────┘
                 │              │
       ┌─────────▼──────┐  ┌───▼──────────────────────┐
       │    cc.py        │  │         dde.py             │  Context Classifier /
       │ (Context        │  │  (Disclosure Decision      │  Decision Engine
       │  Classifier)    │  │       Engine)              │
       └─────────┬───────┘  └──┬───────────────────┬───┘
                 │             │                   │
       ┌─────────▼─────────────▼──┐   ┌────────────▼───────┐
       │          ipc.py           │   │   policy/           │  Identity Profile
       │  (Identity Profile        │   │   norms.py          │  Constructor /
       │   Constructor)            │   │   rules.py          │  Policy Engine
       └───────────┬───────────────┘   │   audit.py          │
                   │                   └─────────────────────┘
       ┌───────────▼───────────────────────────────────────┐
       │                  store/pds.py                      │  Personal Data Store
       └───────────────────────────────────────────────────┘
                   │                   │
       ┌───────────▼──────┐  ┌─────────▼───────────────────┐
       │   crypto/         │  │        models/               │  Primitives /
       │   bbs.py          │  │   attributes, context,       │  Data types
       │   zkp.py          │  │   profile, policy,           │
       │   pseudonyms.py   │  │   request, response,         │
       └───────────────────┘  │   decision                   │
                              └─────────────────────────────┘
```

### Component descriptions

| Component | Role |
|---|---|
| `models/` | Frozen dataclasses and enums for all wire types: `Attribute` (12-member vocab), `ContextClass` (5 classes), `ProfileTuple`, `DisclosureRequest`, `DisclosureResponse`, `DisclosureReceipt`, `PolicySet`, `UtilityScore`. |
| `store/pds.py` | SQLite WAL-mode Personal Data Store. Four tables: `identity` (32-byte master secret), `attributes` (12-attr vault), `profiles` (per-context `ProfileTuple` snapshots), `config` (DDE weights α, β, γ; trust scores). Enforces γ > α and γ > β at write time. Default weights: α = 0.25, β = 0.25, γ = 0.50. |
| `policy/norms.py` | Norm database Φ(c) encoding contextual integrity constraints after Nissenbaum (2004). 26 seed entries across 5 context classes and 6 recipient roles. Closed-world: any flow not listed is impermissible. |
| `policy/rules.py` | Six-step policy resolution: (1) hard blacklist → WITHHOLD, (2) proof-required → VERIFY\_ONLY, (3) trust gate → WITHHOLD, (4) user `PolicyRule` match, (5) norm database Φ(c), (6) closed-world default → WITHHOLD. |
| `policy/audit.py` | Append-only HMAC-SHA256 hash-chain disclosure log backed by SQLite WAL. Each `DisclosureReceipt` is chained to its predecessor; `verify_chain()` detects any tamper. The DDE is the sole writer. |
| `crypto/pseudonyms.py` | HKDF-SHA256 pairwise pseudonymous DID derivation. DIDs are context-scoped: the same `ContextClass` yields the same DID across platforms, while different context classes yield unlinkable DIDs. |
| `crypto/bbs.py` | Full BBS+ selective-disclosure scheme over the BN128 pairing curve (`py_ecc.optimized_bn128`). KeyGen / Sign / Verify / `ProveSubset` with Fiat-Shamir Schnorr (e-binding + D-decomposition). Produces a `SubsetProof` covering all disclosed attributes as a single unlinkable proof blob. |
| `crypto/zkp.py` | zk-SNARK predicate proofs via a `snarkjs` Groth16 subprocess wrapper. Currently includes an `age_gte_18` circuit (8-bit range check). Raises typed `CircuitNotCompiledError` if circuit artifacts are absent; the DDE degrades gracefully to WITHHOLD. |
| `ipc.py` | Identity Profile Constructor. Maintains per-context `ProfileTuple` objects (attribute set Aᵢ, salience weights Wᵢ, trust score Tᵢ, disclosure history Hᵢ). Tracks identity entropy H(Aᵢ) in nats; fires `EntropyAlert` when a single update raises entropy by > 0.15 nats. Detects distributional drift via KL-divergence over rolling request features; fires `DriftAlert` when KL > 0.5. Salience update rule: disclosed attributes → Wₐ × 0.9, withheld → Wₐ × 1.1 (normalised). |
| `cc.py` | Two-layer Context Classifier. Symbolic layer maps `PlatformType` to a `ContextClass` ceiling (security invariant: neural layer cannot classify strictly above it). Neural layer runs `all-MiniLM-L6-v2` (ONNX, 22 MB) over the purpose declaration and attribute names; applies a 5 × 384 learned linear head; softmaxes to per-class probabilities. Confidence threshold τ = 0.80: below threshold or on ensemble disagreement, selects the most privacy-preserving class in the candidate set (MEDICAL > FINANCIAL > PROFESSIONAL > CIVIC > SOCIAL). |
| `dde.py` | Disclosure Decision Engine. Solves argmax\_D U(D) via a greedy marginal utility algorithm, then verifies minimality. U(D) = α·S(D) + β·T(D) − γ·L(D) where S is service utility, T is trust signal, L is linkability cost (see §3.4). Wraps disclosed attributes in a BBS+ `SubsetProof`; substitutes ZKP proofs for `VERIFY_ONLY` attributes; derives a pseudonymous DID; writes a signed `DisclosureReceipt` to the audit log. |
| `iacp.py` | Inter-Agent Communication Protocol. Implements the four-phase negotiation: (1) Identify — pseudonym exchange and blocklist check; (2) Request — vocabulary and purpose validation; (3) Negotiate — IPC → CC → DDE pipeline with counter-proposal loop (≤ 3 rounds); (4) Confirm — HKDF-derived context-specific receipt signing. Degrades to a single-round web mode (phases 1 and 4 skipped) for compatibility with conventional HTTP services. |
| `eval/metrics.py` | Pure-computation metric library. Implements DMR, re-identification risk L(D\*), cross-platform linkability (linkage oracle), classification accuracy, decision latency percentiles, and baseline comparison. |
| `eval/simulator.py` | Simulation orchestrator. Runs 7,500 synthetic IACP interactions (5 platforms × 3 adversarial tiers × 500 interactions). All randomness seeded deterministically. Returns a `SimulationReport` with per-platform and per-tier breakdowns. |
| `main.py` | CLI entry point. Argparse interface; routes simulator progress to stdout; prints Table 3; optionally dumps full `SimulationReport` as JSON. |

---

## Formal Model

### Attribute vocabulary

The system operates over a fixed 12-attribute vocabulary V:

```
legal_name, dob, email, phone, geolocation, employment_status,
income_bracket, health_condition, gaming_history, social_graph,
device_fingerprint, behavioural_metadata
```

### Utility function (§3.4)

For a candidate disclosure set D ⊆ A\_req, the DDE maximises:

```
U(D) = α · S(D)  +  β · T(D)  −  γ · L(D)

S(D) = |D| / |A_req|                          (service utility)
T(D) = Σ_{a∈D} (1 − Wₐ) / |D|               (trust signal)
L(D) = H(I) · Σ_{a∈D} Wₐ                    (linkability cost)

H(I) = log(12) nats   (uniform prior over |V| = 12 attributes)
```

Default weights: α = 0.25, β = 0.25, γ = 0.50. The constraint γ > α and γ > β is enforced at storage time, ensuring the linkability penalty structurally dominates the utility reward. Salience weights Wₐ are initialised uniformly and updated after each disclosure event.

**Greedy algorithm**: Starting from D = ∅, iteratively add the attribute with the highest marginal ΔU. Stop when no candidate improves U. Verify that no strict subset D' ⊊ D\* achieves U(D') ≥ U(D\*) (minimality check). Worst-case O(|A\_req|²) = O(144) comparisons per decision.

### Contextual integrity policy (§3.3)

The norm database Φ(c) encodes permitted information flows ⟨context, attribute, sender\_role, recipient\_role⟩ following Nissenbaum's contextual integrity framework. The closed-world assumption applies: any flow not explicitly listed is impermissible. Policy rules Π supplied by the user override norms at configurable priority levels; the six-step resolution order ensures hard security invariants (blacklist, trust gate) cannot be overridden by user rules.

### Evaluation metrics (§9)

```
DMR       = mean_{i} (1 − |D*_i| / |A_req_i|)      (disclosure minimisation rate)
L(D*)     = H(I) − H(I|D*)                          (re-identification risk, nats)
            H(I|D*) = log(|V| − |D*|)  if |V| > |D*|,  else 0
Linkability = fraction of (interaction i, platform-pair) tuples
              where |D*_{i,a} ∩ D*_{i,b}| ≥ 2       (linkage oracle)
```

---

## Simulation Design

### Platform configurations (§9.1)

| Platform | Requested attributes | Trust score | Context ceiling |
|---|---|---|---|
| FINANCIAL\_PORTAL | legal\_name, dob, email, phone, income\_bracket, employment\_status | 0.85 | FINANCIAL |
| PROFESSIONAL\_NETWORK | legal\_name, email, employment\_status, social\_graph | 0.70 | PROFESSIONAL |
| SOCIAL\_PLATFORM | legal\_name, email, geolocation, social\_graph, behavioural\_metadata, device\_fingerprint | 0.45 | SOCIAL |
| HEALTHCARE\_PORTAL | legal\_name, dob, health\_condition, geolocation | 0.90 | MEDICAL |
| GAMING\_PLATFORM | email, gaming\_history, device\_fingerprint, behavioural\_metadata | 0.55 | SOCIAL |

### Adversarial tiers (§9.2)

| Tier | Description |
|---|---|
| **Benign** | Standard request with a legitimate purpose declaration drawn from a fixed per-platform vocabulary. |
| **Probing (A2-class)** | Adds 2–3 contextually inappropriate attributes to A\_req (e.g. `health_condition` appended to the social platform request). Purpose declaration is present but vague. Tests the policy enforcement boundary. |
| **Colluding (A3-class)** | Requests identical to Benign; results analysed post-hoc by a linkage oracle with access to all disclosed attribute sets across platforms. Tests cross-platform re-identification resistance. |

### Simulation parameters

- 500 interactions per (platform, tier) pair → **7,500 total traces**
- 15 `IACPSession` instances (5 × 3), each backed by an independent in-memory PDS and audit log
- Single shared `ContextClassifier` instance (ONNX session initialised once)
- All randomness seeded via `random.seed(42)` — runs are fully deterministic
- Default: BBS+ signing stubbed out (fast); `--real-crypto` enables actual pairing-based proofs

---

## Results

Full 500-interaction run (`python main.py --interactions 500 --quiet`):

```
Table 3: Projected Evaluation Outcomes Across Baselines
----------------------------------------------------------
System               DMR       Linkability     Re-ID Risk     User Actions
Unmediated           0.0%      High            High           Low
OAuth / OIDC         0.0%      High            High           Low
SSI Wallet           35.1%     Medium          Medium         High
CIA (this work)      86.1%     Low             Low            Low

Per-platform breakdown:
  FINANCIAL_PORTAL      DMR:  85.2%   Conservative mode: 100.0%   Latency p95: 131ms
  PROFESSIONAL_NETWORK  DMR:  83.3%   Conservative mode: 100.0%   Latency p95:  72ms
  SOCIAL_PLATFORM       DMR: 100.0%   Conservative mode:   0.0%   Latency p95:  49ms
  HEALTHCARE_PORTAL     DMR:  78.6%   Conservative mode: 100.0%   Latency p95: 113ms
  GAMING_PLATFORM       DMR:  83.3%   Conservative mode:   0.0%   Latency p95:  72ms

Scenario tier breakdown:
  Benign     DMR:  81.7%   Accuracy: 100.0%
  Probing    DMR:  94.9%   Accuracy: 100.0%
  Colluding  DMR:  81.7%   Linkability: 0.0%

Re-identification risk vs OAuth baseline: 91.8% reduction  (target: 61%)
Negotiation success rate: 66.7%
Mean decision latency: 60ms  |  p95: 109ms  |  p99: 137ms
```

**Notable observations:**

- `SOCIAL_PLATFORM` achieves 100% DMR because `device_fingerprint` and `behavioural_metadata` have no norm support in a social context, `geolocation` and `social_graph` are withheld due to salience weights, and the CIA offers an empty counter-proposal.
- `HEALTHCARE_PORTAL` has the lowest DMR (78.6%) because `legal_name` and `dob` are genuinely norm-permitted for healthcare providers, and the utility function discloses them.
- **Conservative mode** is triggered for FINANCIAL, PROFESSIONAL, and MEDICAL platforms because the MiniLM neural layer disagrees with the symbolic ceiling (ensemble disagreement rule), forcing the classifier to the most privacy-preserving interpretation.
- **Probing DMR (94.9%)** exceeds Benign (81.7%) because the inappropriate additions (e.g. `health_condition` to a financial portal) are immediately rejected by the hard blacklist and closed-world policy steps before the utility function is even evaluated.
- **Colluding linkability 0.0%** — the linkage oracle finds no cross-platform attribute intersection of size ≥ 2 across any aligned interaction pair in the colluding tier, confirming the unlinkability property.
- **Negotiation success rate 66.7%** — one third of interactions reach the 3-round cap or produce an empty counter-proposal (status `FAILED`). These correspond predominantly to the social platform and probing tier requests where no normatively permissible attributes exist.

---

## Installation

**Requirements**: Python 3.11+

```bash
git clone <repo-url>
cd contextual-identity-agents
pip install -r requirements.txt
```

The ONNX model (22 MB) is downloaded automatically on first run from HuggingFace. To pre-download:

```bash
python -c "
from cia.cc import ContextClassifier
from pathlib import Path
ContextClassifier(Path('cia/assets'), download_model=True)
"
```

**Optional — ZKP circuits** (requires `circom` ≥ 2.0 and `node` ≥ 18):

```bash
python cia/assets/compile_circuits.py
```

Without this step, `VERIFY_ONLY` attributes degrade to `WITHHOLD` and 3 tests are skipped.

---

## Usage

### CLI

```bash
# Quick run (~5 seconds)
python main.py --interactions 10 --quiet

# Standard run matching paper (1–2 minutes)
python main.py --interactions 500 --quiet

# With per-platform progress lines
python main.py --interactions 500

# Export full report as JSON
python main.py --interactions 500 --json --quiet > report.json

# Real BBS+ pairings for final paper run (slow)
python main.py --interactions 500 --real-crypto --quiet

# Custom assets directory (CI / offline)
python main.py --assets-dir /path/to/assets --quiet
```

**Exit codes**: 0 success · 1 unhandled exception · 2 missing assets

### Python API

```python
from pathlib import Path
from cia.eval.simulator import run_simulation

# Run simulation and inspect the report
report = run_simulation(n_interactions=100)

print(f"CIA DMR:        {report.overall_dmr:.1%}")
print(f"Re-ID reduction: {report.reid_risk_reduction_pct:.1f}%")
print(f"Latency p95:    {report.latency_stats['p95']:.0f}ms")
print(report.dmr_by_platform)
```

```python
# Use the IACP directly in a custom application
from pathlib import Path
from cia.store.pds import PDS
from cia.policy.audit import AuditLog
from cia.policy.norms import NormDatabase
from cia.models.policy import PolicySet
from cia.models.request import DisclosureRequest, PlatformType
from cia.models.context import ContextSignal
from cia.models.attributes import Attribute
from cia.ipc import IPC
from cia.cc import ContextClassifier
from cia.dde import DisclosureDecisionEngine
from cia.crypto.bbs import BBSEngine
from cia.crypto.zkp import ZKPEngine
from cia.crypto.pseudonyms import PseudonymEngine
from cia.iacp import IACPSession

pds   = PDS(Path("user.db"))
audit = AuditLog(Path("audit.db"))
norm_db = NormDatabase()
cc    = ContextClassifier(Path("cia/assets"))
dde   = DisclosureDecisionEngine(
    pds=pds, norm_db=norm_db, audit_log=audit,
    bbs=BBSEngine(), zkp=ZKPEngine(), pseudonyms=PseudonymEngine(),
)
session = IACPSession(
    pds=pds, ipc=IPC(pds), cc=cc, dde=dde,
    norm_db=norm_db, policy=PolicySet(), audit_log=audit,
)

request = DisclosureRequest(
    request_id="req-001",
    platform_id="bank.example.com",
    platform_type=PlatformType.FINTECH,
    platform_trust_score=0.85,
    requested_attributes=frozenset({
        Attribute.LEGAL_NAME, Attribute.DOB,
        Attribute.EMAIL, Attribute.INCOME_BRACKET,
    }),
    context_signal=ContextSignal(description="mortgage pre-approval"),
)

result = session.handle(request, mode="agent")
print(result.status)                        # NegotiationStatus.PARTIAL
print({av.attribute for av in result.response.disclosed_values})
print(result.receipt.signature)             # HMAC-signed receipt
```

---

## Test Suite

```bash
# Full suite
python -m pytest --tb=short -q

# Individual layers
python -m pytest tests/test_models.py tests/test_policy*.py -q
python -m pytest tests/test_crypto*.py -q       # 3 skipped without compiled circuits
python -m pytest tests/test_cc.py -q            # 8 skipped without ONNX model
python -m pytest tests/test_metrics.py tests/test_simulator.py -q

# BBS+ pairing tests (slow — ~30s)
python -m pytest -m slow -q
```

**377 tests, 3 skipped** (ZKP circuits — expected without `compile_circuits.py`).

Tests that require the ONNX model use `tests/_cc_empty_assets/` (an empty directory) to force symbolic-only mode, so they pass offline. Tests that require the model are marked appropriately and are skipped when `cc_model.onnx` is absent.

---

## Repository Structure

```
contextual-identity-agents/
├── cia/
│   ├── models/
│   │   ├── attributes.py    Attribute vocab (12), AttributeValue, AttributeSet
│   │   ├── context.py       ContextClass (5), ContextSignal, ClassifiedContext
│   │   ├── profile.py       ProfileTuple, TrustConstraints, DisclosureEvent
│   │   ├── policy.py        PolicyAction, PolicyRule, PolicySet
│   │   ├── request.py       DisclosureRequest, PlatformType, RequestTier, RetentionPolicy
│   │   ├── response.py      DisclosureResponse, DisclosureReceipt, NegotiationStatus
│   │   └── decision.py      UtilityScore, CandidateDisclosure, DecisionTrace
│   ├── policy/
│   │   ├── norms.py         NormDatabase, NormEntry, RecipientRole (26 seed entries)
│   │   ├── rules.py         PolicyInterpreter, 6-step resolution
│   │   └── audit.py         AuditLog, HMAC-SHA256 hash-chain
│   ├── crypto/
│   │   ├── pseudonyms.py    PseudonymEngine, HKDF-SHA256 context-scoped DIDs
│   │   ├── bbs.py           BBSEngine, BBS+ over optimized_bn128
│   │   └── zkp.py           ZKPEngine, snarkjs Groth16, CircuitNotCompiledError
│   ├── store/
│   │   └── pds.py           PDS, SQLite WAL, 4-table schema
│   ├── eval/
│   │   ├── metrics.py       6 metric functions, shared evaluation types
│   │   └── simulator.py     run_simulation(), SimulationReport
│   ├── assets/
│   │   ├── norm_db.json
│   │   ├── cc_model.onnx        (22 MB, auto-downloaded)
│   │   ├── cc_head.npy          (auto-generated at first load)
│   │   ├── compile_circuits.py
│   │   └── circuits/age_gte_18/
│   ├── ipc.py               Identity Profile Constructor
│   ├── cc.py                Context Classifier
│   ├── dde.py               Disclosure Decision Engine
│   └── iacp.py              Inter-Agent Communication Protocol
├── tests/
│   ├── test_models.py       17 tests
│   ├── test_policy*.py      49 tests
│   ├── test_crypto*.py      75 tests (3 skipped)
│   ├── test_pds.py          41 tests
│   ├── test_ipc.py
│   ├── test_cc.py           45 tests (8 skipped without model)
│   ├── test_dde.py          36 tests
│   ├── test_iacp.py         39 tests
│   ├── test_metrics.py      }
│   ├── test_simulator.py    } 89 tests
│   └── _cc_empty_assets/    empty dir for symbolic-only CC tests
├── main.py
├── pytest.ini
└── requirements.txt
```

---

## Implementation Notes

### Cryptographic choices

- **BBS+ curve**: `py_ecc.optimized_bn128` is used throughout. The standard `py_ecc.bn128` causes a recursion overflow during final exponentiation on the BN128 Ate pairing and must not be used as a drop-in replacement.
- **BBS+ key lifecycle**: A fresh keypair is generated per `DisclosureDecisionEngine` instance. Persisting G2 points across sessions requires a serialisation scheme that is deferred to a future milestone.
- **Audit log receipts**: Receipts are stored in the audit log *without* the `signature` field to avoid HMAC-of-itself circularity. Phase 4 of the IACP re-signs the fetched receipt with an HKDF-derived context-specific key.
- **Context-scoped DIDs**: The same `ContextClass` yields the same pseudonymous DID across platforms by design (§5.5). This is not a bug — it allows verifiable credential chaining within a context while preventing cross-context linkage.

### Context Classifier

- `download_model=False` still loads `cc_model.onnx` if the file is present on disk. Unit tests that require the classifier to run in symbolic-only mode must point it at `tests/_cc_empty_assets/`, not the project's `cia/assets/`.
- Conservative mode triggers on two independent conditions: (a) neural confidence below τ = 0.80, or (b) ensemble disagreement between the symbolic ceiling and the neural top-1 prediction. In the simulation, FINANCIAL/PROFESSIONAL/MEDICAL platforms trigger conservative mode on every interaction because the neural model (trained on general text) tends to classify finance-oriented purpose declarations as PROFESSIONAL.

### Utility function invariants

- γ > α and γ > β is enforced at PDS write time. A stored weight set that violates this constraint cannot be loaded; the getter raises `ValueError`. This ensures the linkability penalty structurally dominates the utility reward and cannot be accidentally removed.
- `EntropyAlert` and `DriftAlert` adjustments (γ += 0.10, context probability − 0.10) are applied in-call only and are not persisted. Each IACP interaction starts from the stored weights.
- `RetentionPolicy` effects (`ANONYMISED` → β + 0.15, `PERSISTENT` → β − 0.10) are similarly in-call only and do not modify the stored profile.

### Determinism

The 7,500-trace simulation is fully deterministic. `random.seed(42)` is called at the top of `run_simulation()` before any randomness is consumed. The `random.Random(42)` per-session RNG for purpose phrase selection is derived independently. Running the simulation twice with the same `n_interactions` produces identical `SimulationReport` values.

---

## Extension Roadmap

| Milestone | Description |
|---|---|
| **L1 — HTTP adapter** | Flask/FastAPI endpoint implementing the IACP over HTTP. `IACPSession.handle()` is already the correct boundary; a thin adapter is all that is needed. |
| **L2 — Persistent BBS+ keypair** | Proper G2 point serialisation to allow the DDE keypair to survive process restarts. |
| **L3 — Additional ZKP circuits** | `income_above_threshold`, `location_in_region`. The ZKP dispatch table in `dde.py` is designed to be extended with new circuit names. |
| **L4 — Transparency report UI** | A read-only web interface over the audit log. The HMAC hash-chain already supports tamper detection; a UI consuming `AuditLog.query()` is a natural next step. |
| **L5 — Verifiable Credential import** | `IPC.import_vc()` is a defined interface stub. Wiring it to a W3C VC parser would allow externally-issued credentials to populate the PDS attribute vault. |

---

## Citation

```bibtex
@inproceedings{gupta2026cia,
  title     = {Autonomous Identity Disclosure via Contextual Agents
               in Multi-Platform Systems},
  author    = {Gupta, Anubhav and Joshi, Aditi Shailesh},
  booktitle = {Proceedings of Prism 2026},
  year      = {2026}
}
```

---

## Licence

Academic research prototype. All rights reserved pending publication.

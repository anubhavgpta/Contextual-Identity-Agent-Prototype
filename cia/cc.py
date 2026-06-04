"""
Context Classifier (CC) — maps a DisclosureRequest to a ClassifiedContext.

Two-layer hybrid architecture (§3.2 of the paper):

  Symbolic layer (runs first, deterministic, O(1))
  ────────────────────────────────────────────────
  PlatformType → ContextClass ceiling via a fixed lookup table.
  This ceiling is a security invariant: the neural layer cannot assign
  probability to context classes that are *stricter* (more privacy-
  demanding) than the ceiling. An advertising-category platform can never
  be reclassified as MEDICAL by request content alone.

  Neural layer (ONNX all-MiniLM-L6-v2, runs second, constrained to ceiling)
  ──────────────────────────────────────────────────────────────────────────
  Tokenises the purpose declaration + attribute names, runs through
  assets/cc_model.onnx, mean-pools the last hidden state to a 384-dim
  embedding, applies a learned 5-class linear head (assets/cc_head.npy),
  softmaxes → probability distribution over ContextClass.  Probabilities
  for classes strictly above the ceiling are zeroed before renormalisation.

  Model lifecycle
  ───────────────
  • assets/cc_model.onnx absent + download_model=True  → download from HF
  • download fails                                      → ModelNotAvailableError
  • download_model=False and model absent               → symbolic-only mode
  • assets/cc_tokenizer.json absent                     → downloaded alongside model
  • assets/cc_head.npy absent                           → generated at first load
    (representative-text embeddings per ContextClass, stacked as 5×384 matrix)

  Confidence thresholding
  ───────────────────────
  τ = 0.80 (configurable). max(P) < τ → conservative_mode = True, select the
  most privacy-preserving class in the candidate set:
    MEDICAL > FINANCIAL > PROFESSIONAL > CIVIC > SOCIAL

  Conservative mode also triggers on ensemble disagreement (symbolic ceiling ≠
  neural top-1) regardless of the neural probability score.

  Drift integration
  ─────────────────
  If a DriftAlert for the same context is passed to classify(), the operative
  context's probability is reduced by 0.10 (floor 0.0) before thresholding.

CC may import from models/ and store/ (for logging only).
Must NOT import from dde.py, iacp.py, or ipc.py.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from cia.models.context import ClassifiedContext, ContextClass
from cia.models.request import DisclosureRequest, PlatformType
from cia.ipc import DriftAlert          # alert type only — no IPC logic imported

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HF_BASE = (
    "https://huggingface.co/sentence-transformers"
    "/all-MiniLM-L6-v2/resolve/main"
)
_MODEL_URL = f"{_HF_BASE}/onnx/model.onnx"
_TOKENIZER_URL = f"{_HF_BASE}/tokenizer.json"

# Maximum token length for the MiniLM encoder.
_MAX_SEQ_LEN = 128

# Privacy strictness order: index 0 = most restrictive.
# Lower rank = higher privacy demand = preferred in conservative mode.
_PRIVACY_ORDER: list[ContextClass] = [
    ContextClass.MEDICAL,
    ContextClass.FINANCIAL,
    ContextClass.PROFESSIONAL,
    ContextClass.CIVIC,
    ContextClass.SOCIAL,
]
_PRIVACY_RANK: dict[ContextClass, int] = {c: i for i, c in enumerate(_PRIVACY_ORDER)}

# Consistent ContextClass order for numpy index ↔ class mapping.
_CTX_ORDER: list[ContextClass] = list(_PRIVACY_ORDER)

# Representative texts used to build the linear head at first run.
_REPR_TEXTS: dict[ContextClass, str] = {
    ContextClass.FINANCIAL:    "financial loan credit banking income investment tax account",
    ContextClass.PROFESSIONAL: "employment job career work professional employer company role",
    ContextClass.SOCIAL:       "social network friends community gaming entertainment profile",
    ContextClass.MEDICAL:      "health medical doctor healthcare treatment diagnosis patient",
    ContextClass.CIVIC:        "government civic identity voting benefits taxes residence",
}

# Symbolic ceiling per PlatformType.
_PLATFORM_CEILING: dict[PlatformType, ContextClass] = {
    PlatformType.FINTECH:        ContextClass.FINANCIAL,
    PlatformType.EMPLOYER:       ContextClass.PROFESSIONAL,
    PlatformType.SOCIAL_NETWORK: ContextClass.SOCIAL,
    PlatformType.HEALTHCARE:     ContextClass.MEDICAL,
    PlatformType.GOVERNMENT:     ContextClass.CIVIC,
}

# Platforms exempt from the "no purpose → clamp confidence" rule.
_PURPOSE_EXEMPT: frozenset[PlatformType] = frozenset({
    PlatformType.FINTECH,
    PlatformType.HEALTHCARE,
})

# Base confidence emitted by the symbolic layer.
_SYM_BASE_CONFIDENCE: float = 0.90

# Maximum confidence when purpose declaration is absent (§ symbolic layer).
_NO_PURPOSE_CONFIDENCE_CAP: float = 0.65

# Probability reduction applied when a DriftAlert is present.
_DRIFT_PENALTY: float = 0.10


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ModelNotAvailableError(RuntimeError):
    """Raised when the ONNX model cannot be downloaded or loaded."""


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class ContextClassifier:
    """
    Hybrid symbolic + neural context classifier.

    Parameters
    ----------
    assets_dir:
        Directory containing (or to contain) cc_model.onnx,
        cc_tokenizer.json, and cc_head.npy.
    tau:
        Confidence threshold; below → conservative mode.
    download_model:
        If True (default) and cc_model.onnx is absent, attempt download.
        If False and model is absent, operate in symbolic-only mode.
    """

    def __init__(
        self,
        assets_dir: Path,
        tau: float = 0.80,
        download_model: bool = True,
    ) -> None:
        self._assets_dir = assets_dir
        self._tau = tau
        self._session: Any = None       # onnxruntime.InferenceSession or None
        self._tokenizer: Any = None     # tokenizers.Tokenizer or None
        self._head: np.ndarray | None = None   # shape [5, 384]
        self._load_model(download_model)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def classify(
        self,
        request: DisclosureRequest,
        drift_alert: DriftAlert | None = None,
    ) -> ClassifiedContext:
        """
        Classify a DisclosureRequest to a ClassifiedContext.

        Parameters
        ----------
        request:
            The incoming platform request.
        drift_alert:
            Optional DriftAlert from the IPC.  Reduces confidence of the
            operative context class by ``_DRIFT_PENALTY`` before thresholding.
        """
        # ── Layer 1: symbolic ──────────────────────────────────────────
        sym_ctx, sym_conf = self._symbolic(request)

        # ── Layer 2: neural (if available) ────────────────────────────
        if self._session is not None:
            probs = self._neural(request, sym_ctx)
            neural_top = max(probs, key=probs.get)
            ensemble_disagrees = (neural_top != sym_ctx)
            source = "combined"
        else:
            # Symbolic-only: all probability on ceiling class.
            probs = {ctx: 0.0 for ctx in ContextClass}
            probs[sym_ctx] = sym_conf
            # Spread a small residual over other permitted classes.
            residual = (1.0 - sym_conf) / max(len(ContextClass) - 1, 1)
            for ctx in ContextClass:
                if ctx != sym_ctx:
                    probs[ctx] = residual
            neural_top = sym_ctx
            ensemble_disagrees = False
            source = "symbolic"

        # ── Drift penalty ──────────────────────────────────────────────
        if drift_alert is not None and drift_alert.context == sym_ctx:
            probs[sym_ctx] = max(probs[sym_ctx] - _DRIFT_PENALTY, 0.0)
            _total = sum(probs.values()) or 1.0
            probs = {k: v / _total for k, v in probs.items()}

        # ── Select operative class ─────────────────────────────────────
        top_class = max(probs, key=probs.get)
        confidence = probs[top_class]

        # ── Conservative mode ──────────────────────────────────────────
        conservative = (confidence < self._tau) or ensemble_disagrees

        if ensemble_disagrees:
            logger.warning(
                "CC ensemble disagreement: symbolic=%s, neural=%s — "
                "entering conservative mode.",
                sym_ctx.value,
                neural_top.value,
            )

        if conservative:
            # Override to most privacy-preserving class among candidates.
            candidates = [c for c in _PRIVACY_ORDER if probs.get(c, 0.0) > 0.0]
            if candidates:
                top_class = candidates[0]
            # Clamp confidence just below τ to make conservative_mode explicit.
            confidence = min(confidence, self._tau - 1e-4)

        return ClassifiedContext(
            context_class=top_class,
            confidence=confidence,
            conservative_mode=conservative,
            class_probabilities=dict(probs),
            source=source,
        )

    # ------------------------------------------------------------------
    # Symbolic layer
    # ------------------------------------------------------------------

    def _symbolic(self, request: DisclosureRequest) -> tuple[ContextClass, float]:
        """
        Map PlatformType → (ContextClass ceiling, confidence).

        Confidence is clamped to ≤ 0.65 when the purpose declaration is
        absent or empty and the platform is not in ``_PURPOSE_EXEMPT``.
        """
        ceiling = _PLATFORM_CEILING[request.platform_type]
        confidence = _SYM_BASE_CONFIDENCE

        has_purpose = bool(request.context_signal.description.strip())
        if not has_purpose and request.platform_type not in _PURPOSE_EXEMPT:
            confidence = min(confidence, _NO_PURPOSE_CONFIDENCE_CAP)

        return ceiling, confidence

    # ------------------------------------------------------------------
    # Neural layer
    # ------------------------------------------------------------------

    def _neural(
        self,
        request: DisclosureRequest,
        ceiling: ContextClass,
    ) -> dict[ContextClass, float]:
        """
        Run the ONNX encoder + linear head and return a ceiling-constrained
        probability distribution over ContextClass.

        Classes strictly above the ceiling (higher privacy rank) are zeroed
        before renormalisation, enforcing the symbolic-layer primacy invariant.
        """
        text = _build_text(request)
        emb = self._encode(text)               # [384]
        logits = self._head @ emb              # [5]
        exp_l = np.exp(logits - logits.max())
        probs_arr = exp_l / exp_l.sum()

        probs: dict[ContextClass, float] = {
            ctx: float(probs_arr[i]) for i, ctx in enumerate(_CTX_ORDER)
        }

        # Zero out classes strictly above the symbolic ceiling.
        ceiling_rank = _PRIVACY_RANK[ceiling]
        for ctx in ContextClass:
            if _PRIVACY_RANK[ctx] < ceiling_rank:   # more strict → above ceiling
                probs[ctx] = 0.0

        total = sum(probs.values()) or 1.0
        return {k: v / total for k, v in probs.items()}

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> np.ndarray:
        """
        Tokenise, run ONNX, mean-pool with attention mask, L2-normalise.
        Returns a 1-D float32 array of shape [384].
        """
        enc = self._tokenizer.encode(text)

        input_ids      = np.array([enc.ids],            dtype=np.int64)
        attention_mask = np.array([enc.attention_mask], dtype=np.int64)
        token_type_ids = np.array([enc.type_ids],       dtype=np.int64)

        # Some ONNX exports omit token_type_ids — inspect input names.
        ort_inputs: dict[str, np.ndarray] = {}
        for inp in self._session.get_inputs():
            if inp.name == "input_ids":
                ort_inputs["input_ids"] = input_ids
            elif inp.name == "attention_mask":
                ort_inputs["attention_mask"] = attention_mask
            elif inp.name == "token_type_ids":
                ort_inputs["token_type_ids"] = token_type_ids

        outputs = self._session.run(None, ort_inputs)
        last_hidden = outputs[0].astype(np.float32)    # [1, seq_len, 384]

        # Mean pooling (attention-masked).
        mask = attention_mask[:, :, np.newaxis].astype(np.float32)  # [1, seq, 1]
        emb = (last_hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)

        # L2 normalise.
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norm, 1e-9)
        return emb[0]                                   # [384]

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_model(self, download_model: bool) -> None:
        onnx_path = self._assets_dir / "cc_model.onnx"
        tok_path  = self._assets_dir / "cc_tokenizer.json"
        head_path = self._assets_dir / "cc_head.npy"

        if not onnx_path.exists():
            if not download_model:
                logger.warning(
                    "cc_model.onnx not found in %s; CC running in symbolic-only mode.",
                    self._assets_dir,
                )
                return
            self._download(onnx_path, tok_path)

        try:
            import onnxruntime as ort   # noqa: PLC0415
            self._session    = ort.InferenceSession(str(onnx_path))
            self._tokenizer  = _load_tokenizer(tok_path)

            if not head_path.exists():
                logger.info("Generating cc_head.npy from representative embeddings …")
                self._generate_head(head_path)
            self._head = np.load(str(head_path))
            logger.info("CC neural layer loaded (model=%s).", onnx_path.name)

        except Exception as exc:
            logger.warning(
                "Neural layer failed to load (%s); falling back to symbolic-only mode.", exc
            )
            self._session   = None
            self._tokenizer = None
            self._head      = None

    def _download(self, onnx_path: Path, tok_path: Path) -> None:
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading cc_model.onnx from HuggingFace …")
        try:
            urllib.request.urlretrieve(_MODEL_URL, str(onnx_path))
            urllib.request.urlretrieve(_TOKENIZER_URL, str(tok_path))
            logger.info("Download complete.")
        except Exception as exc:
            onnx_path.unlink(missing_ok=True)
            tok_path.unlink(missing_ok=True)
            raise ModelNotAvailableError(
                f"Failed to download ONNX model: {exc}\n"
                f"Download manually:\n  {_MODEL_URL}\n  → {onnx_path}\n"
                f"  {_TOKENIZER_URL}\n  → {tok_path}"
            ) from exc

    def _generate_head(self, head_path: Path) -> None:
        """
        Build the linear head by encoding one representative text per
        ContextClass and stacking the embeddings as rows ([5 × 384]).

        This gives a deterministic, semantically coherent initialisation:
        the head assigns maximum logit to the context whose representative
        text is most similar to the input, matching the symbolic ceiling.
        """
        rows = [self._encode(_REPR_TEXTS[ctx]) for ctx in _CTX_ORDER]
        W = np.stack(rows, axis=0).astype(np.float32)   # [5, 384]
        np.save(str(head_path), W)
        self._head = W


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _build_text(request: DisclosureRequest) -> str:
    """Concatenate purpose declaration + requested attribute names."""
    purpose = request.context_signal.description.strip()
    attrs   = " ".join(
        a.value.replace("_", " ")
        for a in sorted(request.requested_attributes, key=lambda a: a.value)
    )
    text = f"{purpose} {attrs}".strip()
    # Fall back to platform type name if nothing else is present.
    return text or request.platform_type.value.lower().replace("_", " ")


def _load_tokenizer(tok_path: Path) -> Any:
    """Load a HuggingFace fast tokenizer from tokenizer.json."""
    try:
        from tokenizers import Tokenizer    # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "tokenizers package required for CC neural layer: pip install tokenizers"
        ) from exc

    if not tok_path.exists():
        raise FileNotFoundError(
            f"Tokenizer file not found: {tok_path}\n"
            f"Run: python -c \"import urllib.request; "
            f"urllib.request.urlretrieve('{_TOKENIZER_URL}', '{tok_path}')\""
        )

    tok = Tokenizer.from_file(str(tok_path))
    tok.enable_truncation(max_length=_MAX_SEQ_LEN)
    tok.enable_padding(length=_MAX_SEQ_LEN, pad_id=0, pad_type_id=0, pad_token="[PAD]")
    return tok

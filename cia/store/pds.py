"""
Personal Data Store (PDS) — the attribute vault for the CIA.

All persistent user state lives here:
  - master_id         32-byte identity secret used by PseudonymEngine
  - attributes        The 12-attribute typed vault
  - profiles          Per-context ProfileTuple snapshots (JSON-serialised)
  - config            DDE weights (α, β, γ) and per-platform trust scores

Schema
──────
  identity   (id INTEGER PK, master_id BLOB NOT NULL, created_at TEXT)
  attributes (attribute_name TEXT PK, raw_value TEXT, abstracted_value TEXT,
              verified INTEGER, vc_reference TEXT, created_at TEXT, updated_at TEXT)
  profiles   (context_class TEXT PK, payload TEXT NOT NULL, updated_at TEXT)
  config     (key TEXT PK, value TEXT NOT NULL, updated_at TEXT)

ProfileTuple serialisation
──────────────────────────
  ProfileTuple is serialised to/from JSON by this module.  The JSON schema
  mirrors the dataclass fields exactly; DisclosureHistory entries are
  omitted on save (history is append-only via ipc.py and stored separately
  in the audit log — the PDS holds only the current profile snapshot).

Default DDE weights:  α = 0.25, β = 0.25, γ = 0.50   (γ > α, β enforced)
Default trust score:  0.5 for any previously unseen platform
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from cia.models.attributes import Attribute, AttributeValue
from cia.models.context import ContextClass
from cia.models.profile import ProfileTuple, TrustConstraints


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH = Path.home() / ".cia" / "pds.db"
_DEFAULT_ALPHA: float = 0.25
_DEFAULT_BETA:  float = 0.25
_DEFAULT_GAMMA: float = 0.50
_DEFAULT_TRUST: float = 0.50

_WEIGHTS_KEY = "dde_weights"
_TRUST_PREFIX = "trust:"


class PDS:
    """
    Personal Data Store.

    Parameters
    ----------
    db_path:
        Path to the SQLite file.  Use ``Path(":memory:")`` for tests.
        Parent directory is created automatically.
    """

    def __init__(self, db_path: Path = _DEFAULT_DB_PATH) -> None:
        self._db_path = db_path
        self._conn = self._open()
        self._init_schema()
        self._ensure_master_id()
        self._ensure_weights()

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def get_master_id(self) -> bytes:
        """Return the 32-byte master identity secret, creating it if needed."""
        row = self._conn.execute(
            "SELECT master_id FROM identity LIMIT 1"
        ).fetchone()
        if row is None:  # should not happen after __init__, but be defensive
            return self._create_master_id()
        return row[0]

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------

    def set_attribute(
        self,
        attr: Attribute,
        raw: object,
        abstracted: object | None = None,
        verified: bool = False,
        vc_reference: str | None = None,
    ) -> None:
        """Insert or replace an attribute value in the vault."""
        now = _now()
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO attributes
                    (attribute_name, raw_value, abstracted_value,
                     verified, vc_reference, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(attribute_name) DO UPDATE SET
                    raw_value       = excluded.raw_value,
                    abstracted_value= excluded.abstracted_value,
                    verified        = excluded.verified,
                    vc_reference    = excluded.vc_reference,
                    updated_at      = excluded.updated_at
                """,
                (
                    attr.value,
                    _serialise(raw),
                    _serialise(abstracted) if abstracted is not None else None,
                    int(verified),
                    vc_reference,
                    now,
                    now,
                ),
            )

    def get_attribute(self, attr: Attribute) -> AttributeValue | None:
        """Return the stored AttributeValue, or None if not set."""
        row = self._conn.execute(
            "SELECT raw_value, abstracted_value FROM attributes WHERE attribute_name = ?",
            (attr.value,),
        ).fetchone()
        if row is None:
            return None
        return AttributeValue(
            attribute=attr,
            raw=_deserialise(row[0]),
            abstracted=_deserialise(row[1]) if row[1] is not None else None,
        )

    def get_all_attributes(self) -> dict[Attribute, AttributeValue]:
        """Return all attributes currently in the vault."""
        rows = self._conn.execute(
            "SELECT attribute_name, raw_value, abstracted_value FROM attributes"
        ).fetchall()
        result: dict[Attribute, AttributeValue] = {}
        for name, raw, abstracted in rows:
            attr = Attribute(name)
            result[attr] = AttributeValue(
                attribute=attr,
                raw=_deserialise(raw),
                abstracted=_deserialise(abstracted) if abstracted is not None else None,
            )
        return result

    def delete_attribute(self, attr: Attribute) -> None:
        with self._tx() as cur:
            cur.execute(
                "DELETE FROM attributes WHERE attribute_name = ?", (attr.value,)
            )

    # ------------------------------------------------------------------
    # Profiles
    # ------------------------------------------------------------------

    def save_profile(self, context: ContextClass, profile: ProfileTuple) -> None:
        """
        Persist a ProfileTuple snapshot for the given context class.
        Replaces any existing snapshot.  DisclosureHistory is NOT stored
        here — the audit log owns history (see audit.py).
        """
        payload = _profile_to_json(profile)
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO profiles (context_class, payload, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(context_class) DO UPDATE SET
                    payload    = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (context.value, payload, _now()),
            )

    def load_profile(self, context: ContextClass) -> ProfileTuple | None:
        """
        Load the ProfileTuple snapshot for a context, or None if not set.
        The returned ProfileTuple has an empty DisclosureHistory; ipc.py
        reconstructs history from the audit log when needed.
        """
        row = self._conn.execute(
            "SELECT payload FROM profiles WHERE context_class = ?",
            (context.value,),
        ).fetchone()
        if row is None:
            return None
        return _profile_from_json(row[0])

    def all_contexts_with_profiles(self) -> list[ContextClass]:
        rows = self._conn.execute(
            "SELECT context_class FROM profiles ORDER BY context_class"
        ).fetchall()
        return [ContextClass(r[0]) for r in rows]

    # ------------------------------------------------------------------
    # DDE weights (α, β, γ)
    # ------------------------------------------------------------------

    def get_weights(self) -> tuple[float, float, float]:
        """Return (α, β, γ) as stored; falls back to defaults."""
        row = self._conn.execute(
            "SELECT value FROM config WHERE key = ?", (_WEIGHTS_KEY,)
        ).fetchone()
        if row is None:
            return _DEFAULT_ALPHA, _DEFAULT_BETA, _DEFAULT_GAMMA
        d = json.loads(row[0])
        return float(d["alpha"]), float(d["beta"]), float(d["gamma"])

    def set_weights(self, alpha: float, beta: float, gamma: float) -> None:
        """
        Persist DDE weights.  Enforces γ > α and γ > β (privacy-biased)
        as required by the paper (§3.4).
        """
        if not (gamma > alpha and gamma > beta):
            raise ValueError(
                f"Weights must satisfy γ > α and γ > β (got α={alpha}, β={beta}, γ={gamma})"
            )
        if not all(w >= 0 for w in (alpha, beta, gamma)):
            raise ValueError("All weights must be non-negative")
        payload = json.dumps({"alpha": alpha, "beta": beta, "gamma": gamma})
        self._upsert_config(_WEIGHTS_KEY, payload)

    # ------------------------------------------------------------------
    # Per-platform trust scores
    # ------------------------------------------------------------------

    def get_trust_score(self, platform_id: str) -> float:
        """Return the stored trust score for a platform, or 0.5 if unknown."""
        key = _TRUST_PREFIX + platform_id
        row = self._conn.execute(
            "SELECT value FROM config WHERE key = ?", (key,)
        ).fetchone()
        return float(row[0]) if row else _DEFAULT_TRUST

    def set_trust_score(self, platform_id: str, score: float) -> None:
        """Store a trust score in [0, 1] for a platform."""
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Trust score must be in [0, 1], got {score}")
        self._upsert_config(_TRUST_PREFIX + platform_id, str(score))

    # ------------------------------------------------------------------
    # Generic config access (used by IPC for entropy/drift state)
    # ------------------------------------------------------------------

    def get_raw_config(self, key: str) -> str | None:
        """Return the raw string value for an arbitrary config key, or None."""
        row = self._conn.execute(
            "SELECT value FROM config WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def set_raw_config(self, key: str, value: str) -> None:
        """Persist an arbitrary key → string value in the config table."""
        self._upsert_config(key, value)

    def all_trust_scores(self) -> dict[str, float]:
        """Return all stored per-platform trust scores."""
        rows = self._conn.execute(
            "SELECT key, value FROM config WHERE key LIKE ?",
            (_TRUST_PREFIX + "%",),
        ).fetchall()
        return {
            r[0][len(_TRUST_PREFIX):]: float(r[1]) for r in rows
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "PDS":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open(self) -> sqlite3.Connection:
        path_str = str(self._db_path)
        if path_str != ":memory:":
            self._db_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if not self._db_path.exists():
                self._db_path.touch(mode=0o600)
            os.chmod(path_str, 0o600)
        conn = sqlite3.connect(path_str, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        with self._tx() as cur:
            cur.executescript("""
                CREATE TABLE IF NOT EXISTS identity (
                    id         INTEGER PRIMARY KEY CHECK (id = 1),
                    master_id  BLOB    NOT NULL,
                    created_at TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS attributes (
                    attribute_name   TEXT PRIMARY KEY,
                    raw_value        TEXT,
                    abstracted_value TEXT,
                    verified         INTEGER NOT NULL DEFAULT 0,
                    vc_reference     TEXT,
                    created_at       TEXT NOT NULL,
                    updated_at       TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS profiles (
                    context_class TEXT PRIMARY KEY,
                    payload       TEXT NOT NULL,
                    updated_at    TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS config (
                    key        TEXT PRIMARY KEY,
                    value      TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)

    def _ensure_master_id(self) -> None:
        existing = self._conn.execute(
            "SELECT id FROM identity LIMIT 1"
        ).fetchone()
        if existing is None:
            self._create_master_id()

    def _create_master_id(self) -> bytes:
        mid = secrets.token_bytes(32)
        with self._tx() as cur:
            cur.execute(
                "INSERT OR IGNORE INTO identity (id, master_id, created_at) VALUES (1, ?, ?)",
                (mid, _now()),
            )
        return mid

    def _ensure_weights(self) -> None:
        existing = self._conn.execute(
            "SELECT key FROM config WHERE key = ?", (_WEIGHTS_KEY,)
        ).fetchone()
        if existing is None:
            self._upsert_config(
                _WEIGHTS_KEY,
                json.dumps({
                    "alpha": _DEFAULT_ALPHA,
                    "beta":  _DEFAULT_BETA,
                    "gamma": _DEFAULT_GAMMA,
                }),
            )

    def _upsert_config(self, key: str, value: str) -> None:
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO config (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value      = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, value, _now()),
            )

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Cursor]:
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialise(value: object) -> str:
    """JSON-encode any primitive attribute value for storage."""
    return json.dumps(value)


def _deserialise(raw: str) -> object:
    return json.loads(raw)


def _profile_to_json(profile: ProfileTuple) -> str:
    """
    Serialise a ProfileTuple to a JSON string.
    DisclosureHistory is intentionally omitted — history lives in audit.py.
    """
    data: dict = {
        "profile_id": profile.profile_id,
        "active_attributes": sorted(a.value for a in profile.active_attributes),
        "salience_weights": {k.value: v for k, v in profile.salience_weights.items()},
        "trust_constraints": {
            "min_trust_score": profile.trust_constraints.min_trust_score,
            "allowed_contexts": sorted(
                c.value for c in profile.trust_constraints.allowed_contexts
            ),
            "hard_blacklist": sorted(
                a.value for a in profile.trust_constraints.hard_blacklist
            ),
            "proof_required": sorted(
                a.value for a in profile.trust_constraints.proof_required
            ),
        },
        "identity_entropy": profile.identity_entropy,
        "drift_flagged": profile.drift_flagged,
        "created_at": profile.created_at.isoformat(),
        "updated_at": profile.updated_at.isoformat(),
    }
    return json.dumps(data, separators=(",", ":"))


def _profile_from_json(payload: str) -> ProfileTuple:
    d = json.loads(payload)
    tc_raw = d["trust_constraints"]
    tc = TrustConstraints(
        min_trust_score=tc_raw["min_trust_score"],
        allowed_contexts=frozenset(ContextClass(c) for c in tc_raw["allowed_contexts"]),
        hard_blacklist=frozenset(Attribute(a) for a in tc_raw["hard_blacklist"]),
        proof_required=frozenset(Attribute(a) for a in tc_raw["proof_required"]),
    )
    return ProfileTuple(
        profile_id=d["profile_id"],
        active_attributes=frozenset(Attribute(a) for a in d["active_attributes"]),
        salience_weights={Attribute(k): v for k, v in d["salience_weights"].items()},
        trust_constraints=tc,
        disclosure_history=(),           # history not persisted here
        identity_entropy=d["identity_entropy"],
        drift_flagged=d["drift_flagged"],
        created_at=datetime.fromisoformat(d["created_at"]),
        updated_at=datetime.fromisoformat(d["updated_at"]),
    )

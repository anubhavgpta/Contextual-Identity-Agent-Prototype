"""
Tamper-evident disclosure audit log backed by SQLite WAL.

Each DisclosureReceipt is stored as a signed JSON blob.  Tamper-evidence
is achieved via a hash-chain: every row stores HMAC-SHA256 of its own
canonical JSON concatenated with the previous row's HMAC (chain_hash).
The chain starts with HMAC of a well-known genesis string.

Schema (single table):
    id           INTEGER PRIMARY KEY AUTOINCREMENT
    receipt_id   TEXT UNIQUE NOT NULL
    request_id   TEXT NOT NULL
    platform_id  TEXT NOT NULL
    context      TEXT NOT NULL
    timestamp    TEXT NOT NULL   -- ISO-8601 UTC
    payload      TEXT NOT NULL   -- canonical JSON of DisclosureReceipt
    chain_hash   TEXT NOT NULL   -- HMAC-SHA256(payload || prev_chain_hash)

Queries:
    append(receipt)                       — add a new entry
    query(since, context, platform_id)    — filtered read
    verify_chain()                        — walk every row, check HMAC chain
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

from cia.models.attributes import Attribute
from cia.models.context import ContextClass
from cia.models.response import DisclosureReceipt, NegotiationStatus


_GENESIS_HASH = "0" * 64   # sentinel for the first row's prev_chain_hash


class AuditLog:
    """
    Thread-safe (one connection per instance), WAL-mode SQLite audit log.

    Parameters
    ----------
    db_path:
        Path to the SQLite file.  Use ``Path(":memory:")`` for tests.
    hmac_key:
        Raw bytes used as the HMAC-SHA256 key.
    """

    def __init__(
        self,
        db_path: Path,
        hmac_key: bytes,
    ) -> None:
        self._db_path = db_path
        self._key = hmac_key
        self._lock = threading.Lock()
        self._conn = self._open_connection()
        self._init_schema()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, receipt: DisclosureReceipt) -> None:
        """Persist receipt and extend the HMAC chain."""
        payload = _canonical_json(receipt)

        with self._lock:
            with self._tx() as cur:
                prev_hash = self._last_chain_hash()
                chain_hash = _compute_hmac(self._key, payload + prev_hash)

                cur.execute(
                    """
                    INSERT INTO disclosure_log
                        (receipt_id, request_id, platform_id, context,
                         timestamp, payload, chain_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        receipt.receipt_id,
                        receipt.request_id,
                        receipt.platform_id,
                        receipt.context_class.value,
                        receipt.timestamp.astimezone(timezone.utc).isoformat(),
                        payload,
                        chain_hash,
                    ),
                )

    def query(
        self,
        since: datetime | None = None,
        context: ContextClass | None = None,
        platform_id: str | None = None,
    ) -> list[DisclosureReceipt]:
        """Return receipts matching all supplied filters (AND semantics)."""
        clauses: list[str] = []
        params: list[object] = []

        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since.astimezone(timezone.utc).isoformat())
        if context is not None:
            clauses.append("context = ?")
            params.append(context.value)
        if platform_id is not None:
            clauses.append("platform_id = ?")
            params.append(platform_id)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT payload FROM disclosure_log {where} ORDER BY id ASC"

        cur = self._conn.execute(sql, params)
        return [_receipt_from_json(row[0]) for row in cur.fetchall()]

    def verify_chain(self) -> bool:
        """
        Walk every row in insertion order and recompute the HMAC chain.
        Returns True iff all chain_hashes are consistent.
        """
        cur = self._conn.execute(
            "SELECT payload, chain_hash FROM disclosure_log ORDER BY id ASC"
        )
        rows = cur.fetchall()

        prev_hash = _GENESIS_HASH
        for payload, stored_hash in rows:
            expected = _compute_hmac(self._key, payload + prev_hash)
            if not hmac.compare_digest(expected, stored_hash):
                return False
            prev_hash = stored_hash
        return True

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM disclosure_log").fetchone()[0]

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open_connection(self) -> sqlite3.Connection:
        db_path_str = str(self._db_path)
        if db_path_str == ":memory:":
            conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(db_path_str, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        with self._tx() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS disclosure_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    receipt_id  TEXT UNIQUE NOT NULL,
                    request_id  TEXT NOT NULL,
                    platform_id TEXT NOT NULL,
                    context     TEXT NOT NULL,
                    timestamp   TEXT NOT NULL,
                    payload     TEXT NOT NULL,
                    chain_hash  TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON disclosure_log (timestamp)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_context
                ON disclosure_log (context)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_platform
                ON disclosure_log (platform_id)
            """)

    def _last_chain_hash(self) -> str:
        row = self._conn.execute(
            "SELECT chain_hash FROM disclosure_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else _GENESIS_HASH

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
# Helpers
# ---------------------------------------------------------------------------

def _canonical_json(receipt: DisclosureReceipt) -> str:
    """Deterministic JSON serialisation of a DisclosureReceipt."""
    data = {
        "receipt_id": receipt.receipt_id,
        "request_id": receipt.request_id,
        "platform_id": receipt.platform_id,
        "context_class": receipt.context_class.value,
        "disclosed_attributes": sorted(a.value for a in receipt.disclosed_attributes),
        "withheld_attributes": sorted(a.value for a in receipt.withheld_attributes),
        "negotiation_status": receipt.negotiation_status.value,
        "utility_score": receipt.utility_score,
        "entropy_before": receipt.entropy_before,
        "entropy_after": receipt.entropy_after,
        "timestamp": receipt.timestamp.astimezone(timezone.utc).isoformat(),
    }
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _compute_hmac(key: bytes, data: str) -> str:
    return hmac.new(key, data.encode("utf-8"), hashlib.sha256).hexdigest()


def _receipt_from_json(payload: str) -> DisclosureReceipt:
    d = json.loads(payload)
    return DisclosureReceipt(
        receipt_id=d["receipt_id"],
        request_id=d["request_id"],
        platform_id=d["platform_id"],
        context_class=ContextClass(d["context_class"]),
        disclosed_attributes=frozenset(Attribute(a) for a in d["disclosed_attributes"]),
        withheld_attributes=frozenset(Attribute(a) for a in d["withheld_attributes"]),
        negotiation_status=NegotiationStatus(d["negotiation_status"]),
        utility_score=d["utility_score"],
        entropy_before=d["entropy_before"],
        entropy_after=d["entropy_after"],
        timestamp=datetime.fromisoformat(d["timestamp"]),
    )

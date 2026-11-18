"""
Neural Observatory — SQLite Storage Backend
"""

from __future__ import annotations

import io
import json
import logging
import sqlite3
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..collectors.base import Observation
from .base import BaseStore

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    collection  TEXT    NOT NULL,
    layer_name  TEXT    NOT NULL,
    step        INTEGER NOT NULL,
    epoch       INTEGER NOT NULL,
    seq         INTEGER NOT NULL DEFAULT 0,
    run_id      TEXT    NOT NULL DEFAULT '',
    timestamp   REAL    NOT NULL,
    stats       TEXT    NOT NULL,
    metadata    TEXT    NOT NULL,
    values_blob BLOB,
    UNIQUE (collection, layer_name, step, epoch, seq, run_id)
);
CREATE INDEX IF NOT EXISTS idx_col_layer ON observations (collection, layer_name);
CREATE INDEX IF NOT EXISTS idx_step      ON observations (step);
"""


class SQLiteStore(BaseStore):
    def __init__(self, db_path: Optional[str] = None) -> None:
        self._temp_path: Optional[Path] = None
        if db_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".observatory.db", delete=False)
            tmp.close()
            db_path = tmp.name
            self._temp_path = Path(db_path)

        self._path = Path(db_path)
        self._closed = False
        # check_same_thread=False lets any thread use the store, so every
        # operation must be serialized by us instead of by sqlite3.
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._migrate_schema()
        logger.info("SQLiteStore opened at %s", self._path)

    def _migrate_schema(self) -> None:
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version >= _SCHEMA_VERSION:
            return

        columns = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(observations)").fetchall()
        }
        if not columns:
            self._conn.executescript(
                f"BEGIN;\n{_SCHEMA}\nPRAGMA user_version = {_SCHEMA_VERSION};\nCOMMIT;"
            )
            return

        # Older tables may lack seq and/or run_id. Missing columns default to
        # 0 / ''; existing values are carried over so re-running this migration
        # never discards run identities. The version stamp commits in the same
        # transaction as the rebuild, so an interrupted migration is simply
        # retried on the next open.
        seq_source = "seq" if "seq" in columns else "0"
        run_source = "run_id" if "run_id" in columns else "''"
        self._conn.executescript(f"""
            BEGIN;
            CREATE TABLE observations_new (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                collection  TEXT    NOT NULL,
                layer_name  TEXT    NOT NULL,
                step        INTEGER NOT NULL,
                epoch       INTEGER NOT NULL,
                seq         INTEGER NOT NULL DEFAULT 0,
                run_id      TEXT    NOT NULL DEFAULT '',
                timestamp   REAL    NOT NULL,
                stats       TEXT    NOT NULL,
                metadata    TEXT    NOT NULL,
                values_blob BLOB,
                UNIQUE (collection, layer_name, step, epoch, seq, run_id)
            );
            INSERT OR IGNORE INTO observations_new
                (collection, layer_name, step, epoch, seq, run_id,
                 timestamp, stats, metadata, values_blob)
            SELECT collection, layer_name, step, epoch, {seq_source}, {run_source},
                   timestamp, stats, metadata, values_blob
            FROM observations;
            DROP TABLE observations;
            ALTER TABLE observations_new RENAME TO observations;
            CREATE INDEX idx_col_layer ON observations (collection, layer_name);
            CREATE INDEX idx_step      ON observations (step);
            PRAGMA user_version = {_SCHEMA_VERSION};
            COMMIT;
            """)

    def put(
        self,
        collection: str,
        layer_name: str,
        observation: Observation,
    ) -> None:
        values_blob = self._encode_array(observation.values)
        with self._lock:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO observations
                    (collection, layer_name, step, epoch, seq, run_id, timestamp,
                     stats, metadata, values_blob)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    collection,
                    layer_name,
                    observation.step,
                    observation.epoch,
                    observation.seq,
                    observation.run_id,
                    observation.timestamp,
                    json.dumps(observation.stats),
                    json.dumps(observation.metadata),
                    values_blob,
                ),
            )

    def commit(self) -> None:
        """Commit all pending transactions to disk."""
        with self._lock:
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._conn.commit()
            self._conn.close()
            self._closed = True
            # Only databases we created ourselves are ours to delete.
            if self._temp_path is not None:
                self._temp_path.unlink(missing_ok=True)

    def get(
        self,
        collection: str,
        layer_name: str,
        limit: Optional[int] = None,
    ) -> List[Observation]:
        q = (
            "SELECT layer_name, step, epoch, seq, run_id, timestamp, stats, metadata, values_blob "
            "FROM observations WHERE collection=? AND layer_name=? ORDER BY step, seq"
        )
        params: tuple = (collection, layer_name)
        if limit is not None:
            q += " LIMIT ?"
            params = (*params, limit)

        with self._lock:
            rows = self._conn.execute(q, params).fetchall()
        return [self._row_to_obs(r) for r in rows]

    def get_collection(self, collection: str) -> Dict[str, List[Observation]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT layer_name, step, epoch, seq, run_id, timestamp, stats, metadata, values_blob "
                "FROM observations WHERE collection=? ORDER BY step, seq",
                (collection,),
            ).fetchall()

        result: Dict[str, List[Observation]] = {}
        for row in rows:
            obs = self._row_to_obs(row)
            result.setdefault(obs.layer_name, []).append(obs)
        return result

    def layer_names(self, collection: str) -> List[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT layer_name FROM observations WHERE collection=?",
                (collection,),
            ).fetchall()
        return [r[0] for r in rows]

    def clear(self, collection: Optional[str] = None) -> None:
        with self._lock:
            if collection:
                self._conn.execute(
                    "DELETE FROM observations WHERE collection=?", (collection,)
                )
            else:
                self._conn.execute("DELETE FROM observations")
            self._conn.commit()

    def __enter__(self) -> "SQLiteStore":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    @staticmethod
    def _encode_array(arr: Optional[np.ndarray]) -> Optional[bytes]:
        if arr is None:
            return None
        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        return buf.getvalue()

    @staticmethod
    def _decode_array(blob: Optional[bytes]) -> Optional[np.ndarray]:
        if blob is None:
            return None
        return np.load(io.BytesIO(blob), allow_pickle=False)

    def _row_to_obs(self, row: tuple) -> Observation:
        (
            layer_name,
            step,
            epoch,
            seq,
            run_id,
            timestamp,
            stats_j,
            meta_j,
            values_blob,
        ) = row
        return Observation(
            layer_name=layer_name,
            step=step,
            epoch=epoch,
            seq=seq,
            run_id=run_id,
            timestamp=timestamp,
            stats=json.loads(stats_j),
            metadata=json.loads(meta_j),
            values=self._decode_array(values_blob),
        )

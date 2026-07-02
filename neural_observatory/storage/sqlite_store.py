"""
Neural Observatory — SQLite Storage Backend
"""
from __future__ import annotations

import io
import json
import logging
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..collectors.base import Observation

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    collection  TEXT    NOT NULL,
    layer_name  TEXT    NOT NULL,
    step        INTEGER NOT NULL,
    epoch       INTEGER NOT NULL,
    timestamp   REAL    NOT NULL,
    stats       TEXT    NOT NULL,
    metadata    TEXT    NOT NULL,
    values_blob BLOB
);

CREATE INDEX IF NOT EXISTS idx_col_layer ON observations (collection, layer_name);
CREATE INDEX IF NOT EXISTS idx_step      ON observations (step);
"""


class SQLiteStore:
    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            # mktemp is deprecated and insecure, using NamedTemporaryFile instead
            tmp = tempfile.NamedTemporaryFile(suffix=".observatory.db", delete=False)
            tmp.close()
            db_path = tmp.name
            
        self._path = Path(db_path)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info("SQLiteStore opened at %s", self._path)

    def put(
        self,
        collection: str,
        layer_name: str,
        observation: Observation,
    ) -> None:
        values_blob = self._encode_array(observation.values)
        self._conn.execute(
            """
            INSERT INTO observations
                (collection, layer_name, step, epoch, timestamp,
                 stats, metadata, values_blob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                collection,
                layer_name,
                observation.step,
                observation.epoch,
                observation.timestamp,
                json.dumps(observation.stats),
                json.dumps(observation.metadata),
                values_blob,
            ),
        )
        self._conn.commit()

    def get(
        self,
        collection: str,
        layer_name: str,
        limit: Optional[int] = None,
    ) -> List[Observation]:
        q = (
            "SELECT layer_name, step, epoch, timestamp, stats, metadata, values_blob "
            "FROM observations WHERE collection=? AND layer_name=? ORDER BY step"
        )
        params: tuple = (collection, layer_name)
        if limit:
            q += " LIMIT ?"
            params = (*params, limit)
            
        rows = self._conn.execute(q, params).fetchall()
        return [self._row_to_obs(r) for r in rows]

    def get_collection(self, collection: str) -> Dict[str, List[Observation]]:
        rows = self._conn.execute(
            "SELECT layer_name, step, epoch, timestamp, stats, metadata, values_blob "
            "FROM observations WHERE collection=? ORDER BY step",
            (collection,),
        ).fetchall()
        
        result: Dict[str, List[Observation]] = {}
        for row in rows:
            obs = self._row_to_obs(row)
            result.setdefault(obs.layer_name, []).append(obs)
        return result

    def layer_names(self, collection: str) -> List[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT layer_name FROM observations WHERE collection=?",
            (collection,),
        ).fetchall()
        return [r[0] for r in rows]

    def clear(self, collection: Optional[str] = None) -> None:
        if collection:
            self._conn.execute(
                "DELETE FROM observations WHERE collection=?", (collection,)
            )
        else:
            self._conn.execute("DELETE FROM observations")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

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
        layer_name, step, epoch, timestamp, stats_j, meta_j, values_blob = row
        return Observation(
            layer_name=layer_name,
            step=step,
            epoch=epoch,
            timestamp=timestamp,
            stats=json.loads(stats_j),
            metadata=json.loads(meta_j),
            values=self._decode_array(values_blob),
        )
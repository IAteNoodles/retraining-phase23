"""Trimmed Neo4j client: Bolt + HTTPS Query API, explicit DB routing.

Unlike the old monolith, `execute()` REQUIRES the caller to think about which
database runs the query: pass `database=` per call, or set a fallback at
construction. There is no silent default-DB execution hiding inside metrics.
"""

from __future__ import annotations

import json
from urllib.parse import urlparse

import requests
import threading


class Neo4jClient:
    def __init__(self, url: str = "", user: str = "neo4j", password: str = "",
                 fallback_db: str = "movies", database_mapping: dict | None = None,
                 timeout: int = 10):
        if not url:
            raise ValueError("neo4j_url is required (bolt://... or https://.../db/{database}/query/v2)")
        self.database_mapping = database_mapping or {}
        self.fallback_db = fallback_db
        self.timeout = timeout
        self._cache: dict = {}
        self._lock = threading.Lock()  # guards the shared HTTP session only

        parsed = urlparse(url)
        scheme = (parsed.scheme or "").lower()
        if scheme in {"bolt", "bolt+s", "bolt+ssc", "neo4j", "neo4j+s", "neo4j+ssc"}:
            from neo4j import GraphDatabase
            self.mode = "bolt"
            self.driver = GraphDatabase.driver(url, auth=(user, password))
            self.session = None
        else:
            self.mode = "http"
            self.driver = None
            self.base = f"{parsed.scheme}://{parsed.netloc}"
            self.session = requests.Session()
            self.session.auth = (user, password)
            self.session.headers.update({"Content-Type": "application/json"})

    def resolve(self, database: str | None) -> str:
        name = str(database or self.fallback_db)
        return self.database_mapping.get(name, name)

    @staticmethod
    def _flatten(v):
        if isinstance(v, dict):
            return tuple(sorted((k, Neo4jClient._flatten(x)) for k, v in v.items()))
        if isinstance(v, list):
            return tuple(Neo4jClient._flatten(x) for x in v)
        return v

    def execute(self, cypher: str, database: str | None = None):
        """Run cypher on the resolved database. Returns list-of-tuples, [] when
        empty, None on syntax/connection error."""
        db = self.resolve(database)
        stmt = cypher.strip().rstrip(";")
        key = (stmt, db)
        if key in self._cache:
            return self._cache[key]
        try:
            if self.mode == "bolt":
                with self.driver.session(database=db) as s:
                    # Server-side kill after `timeout`: untrained policies emit
                    # cartesian-product garbage; without this the driver blocks
                    # forever and the whole run hangs with GPU at 0%.
                    # Row cap bounds transfer; jaccard/dbcost are log-scale so
                    # a 10k cap barely moves them.
                    result = []
                    for rec in s.run(stmt, timeout=self.timeout):
                        result.append(tuple(self._flatten(v) for v in rec.values()))
                        if len(result) >= 10000:
                            break
            else:
                with self._lock:  # requests.Session is not thread-safe
                    url = f"{self.base}/db/{db}/query/v2"
                    resp = self.session.post(url, json={"statement": stmt, "parameters": {}},
                                             timeout=self.timeout)
                    data = resp.json()
                if data.get("errors"):
                    result = None
                else:
                    result = [tuple(self._flatten(x) for x in row)
                              for row in data.get("data", {}).get("values", [])]
        except Exception:
            result = None
        if result is not None:
            self._cache[key] = result
        return result

    def ping(self, database: str | None = None) -> bool:
        try:
            return self.execute("RETURN 1 AS ok", database=database) is not None
        except Exception:
            return False

    def close(self):
        try:
            if self.driver is not None:
                self.driver.close()
        except Exception:
            pass
        try:
            if self.session is not None:
                self.session.close()
        except Exception:
            pass

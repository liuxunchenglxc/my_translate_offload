from __future__ import annotations

import json
import os
import time
from urllib.parse import quote

import requests


class PersonalStoreError(RuntimeError):
    pass


class PersonalStore:
    """Client compatible with the user's PersonalStorage /store/{filename} API."""

    def __init__(self, base_url: str | None = None, token: str | None = None):
        self.base_url = (base_url or os.getenv(
            "PERSONALSTORE_URL", "https://bh.lxc.nya.je:25570/trans/"
        )).rstrip("/")
        self.token = token if token is not None else os.getenv("PERSONALSTORE_TOKEN", "")
        if not self.token:
            raise PersonalStoreError("PERSONALSTORE_TOKEN is required")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "OffloadTranslate-Worker/1.0"})
        self.timeout = (15, 90)

    def _url(self, filename: str) -> str:
        if not filename or "/" in filename or "\\" in filename:
            raise ValueError("Invalid PersonalStorage filename")
        return f"{self.base_url}/store/{quote(filename, safe='')}"

    def load_text(self, filename: str, *, missing_ok: bool = False) -> str | None:
        last_error: str | None = None
        for attempt in range(5):
            try:
                response = self.session.get(
                    self._url(filename), params={"token": self.token}, timeout=self.timeout
                )
                if response.status_code == 404 and missing_ok:
                    return None
                response.raise_for_status()
                return response.content.decode("utf-8")
            except requests.RequestException as exc:
                response = getattr(exc, "response", None)
                status = response.status_code if isinstance(response, requests.Response) else None
                last_error = f"HTTP {status}" if status is not None else type(exc).__name__
                if status is not None and 400 <= status < 500 and status != 429:
                    # Never include the exception's URL: the PersonalStorage token is
                    # carried in the query string and Actions logs may be public.
                    raise PersonalStoreError(
                        f"PersonalStorage GET {filename} failed ({status})"
                    ) from None
                if attempt < 4:
                    time.sleep(min(2 ** attempt, 12))
        raise PersonalStoreError(f"PersonalStorage GET {filename} failed ({last_error})")

    def save_text(self, filename: str, content: str) -> None:
        if not isinstance(content, str):
            raise TypeError("PersonalStorage content must be text")
        last_error: str | None = None
        for attempt in range(5):
            try:
                response = self.session.post(
                    self._url(filename),
                    params={"token": self.token},
                    json={"content": content},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return
            except requests.RequestException as exc:
                response = getattr(exc, "response", None)
                status = response.status_code if isinstance(response, requests.Response) else None
                last_error = f"HTTP {status}" if status is not None else type(exc).__name__
                if status is not None and 400 <= status < 500 and status != 429:
                    raise PersonalStoreError(
                        f"PersonalStorage POST {filename} failed ({status})"
                    ) from None
                if attempt < 4:
                    time.sleep(min(2 ** attempt, 12))
        raise PersonalStoreError(f"PersonalStorage POST {filename} failed ({last_error})")

    def load_json(self, filename: str, *, missing_ok: bool = False):
        raw = self.load_text(filename, missing_ok=missing_ok)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PersonalStoreError(f"Invalid JSON in PersonalStorage file {filename}") from exc

    def save_json(self, filename: str, value) -> None:
        self.save_text(filename, json.dumps(value, ensure_ascii=False, separators=(",", ":")))

    def close(self) -> None:
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

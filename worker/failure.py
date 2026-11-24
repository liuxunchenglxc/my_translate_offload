"""Tiny stdlib-only workflow failure handler (also works if pip installation failed)."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-key", required=True)
    parser.add_argument("--attempt", required=True, type=int)
    parser.add_argument("--phase", required=True, choices=("glossary", "translation"))
    parser.add_argument("--message", default="GitHub Actions setup failed")
    args = parser.parse_args()
    base = os.getenv("PERSONALSTORE_URL", "https://bh.lxc.nya.je:25570/trans/").rstrip("/")
    token = os.getenv("PERSONALSTORE_TOKEN", "")
    if not token:
        print("PERSONALSTORE_TOKEN is absent; cannot persist workflow failure")
        return 0
    task_name = f"{args.storage_key}.task.json"
    state_name = f"{args.storage_key}.attempt.{args.attempt}.state.json"
    task_url = f"{base}/store/{urllib.parse.quote(task_name, safe='')}?token={urllib.parse.quote(token, safe='')}"
    state_url = f"{base}/store/{urllib.parse.quote(state_name, safe='')}?token={urllib.parse.quote(token, safe='')}"
    req = urllib.request.Request(task_url, headers={"User-Agent": "OffloadTranslate-Failure/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            task = json.loads(response.read().decode("utf-8"))
        if int(task.get("attempt", -1)) != args.attempt or task.get("phase") != args.phase:
            return 0
        # Preserve the detailed exception already written by worker.py.
        try:
            with urllib.request.urlopen(state_url, timeout=15) as response:
                existing_state = json.loads(response.read().decode("utf-8"))
            if int(existing_state.get("attempt", -1)) == args.attempt and existing_state.get("status") == "failed":
                return 0
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
        state = {
            "attempt": args.attempt,
            "phase": args.phase,
            "status": "failed",
            "error": args.message[:2000],
            "message": "GitHub Actions 初始化失败",
            "updatedAt": now_iso(),
            "heartbeatAt": now_iso(),
            "worker": "github-actions",
        }
        body = json.dumps({"content": json.dumps(state, ensure_ascii=False, separators=(",", ":"))}).encode("utf-8")
        post = urllib.request.Request(state_url, data=body, headers={"Content-Type": "application/json", "User-Agent": "OffloadTranslate-Failure/1.0"}, method="POST")
        with urllib.request.urlopen(post, timeout=30) as response:
            response.read()
    except Exception as exc:
        # Avoid printing request URLs, which include the PersonalStorage token.
        print(f"Unable to persist workflow failure ({type(exc).__name__})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Test-only actual-engine probes. Importing this package never starts a server."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2))
    os.replace(temporary, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def append_json(path, value):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(value, sort_keys=True) + '\n')


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wait_for(predicate, timeout, description):
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise TimeoutError(description)
        time.sleep(.01)  # Bounded readiness polling; ordering comes from receipts.

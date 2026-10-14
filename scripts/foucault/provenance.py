"""Where a result came from: enough to rerun it."""

from __future__ import annotations

import subprocess

from cf.protocol import PROTOCOL_VERSION


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def provenance(seed: int) -> dict:
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": _git("status", "--porcelain") != "",  # uncommitted changes when it ran
        "protocol_version": PROTOCOL_VERSION,
        "seed": seed,
    }

"""Fetch remaining KAP HTML bodies without rebuilding event tables."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vesta.kap import download_bodies, download_kap, is_noise_subject  # noqa: E402


def main() -> None:
    cache = ROOT / "data" / "cache"
    rows = download_kap(cache)
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
    by_idx = {int(r["disclosure_index"]): r for r in rows if r.get("disclosure_index") is not None}

    def key(idx: int):
        rec = by_idx.get(idx, {})
        return (0 if (rec.get("date") or "") >= "2024-01-01" else 1, 1 if is_noise_subject(rec.get("subject") or "") else 0, -idx)

    need = sorted(by_idx, key=key)[: cap]
    bodies = download_bodies(cache, need, max_workers=10, retry_empty=True)
    ok = sum(1 for r in bodies.values() if r.get("ok") and r.get("body_text"))
    print(f"cache {len(bodies)} rows, nonempty {ok}")


if __name__ == "__main__":
    main()

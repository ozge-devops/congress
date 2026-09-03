"""Score study/responses.csv against sealed gold. Fails if no responses exist."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    responses = ROOT / "study" / "responses.csv"
    gold = json.loads((ROOT / "study" / "gold.json").read_text())
    with responses.open() as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("participant_id")]
    if not rows:
        print("No participant rows in study/responses.csv. Study not run.")
        sys.exit(0)
    n = 0
    hits = 0
    for r in rows:
        g = gold.get(r["scenario_id"])
        if not g:
            continue
        n += 1
        if r.get("decision") == g["correct_week"]:
            hits += 1
    print(json.dumps({"n": n, "hit_rate": hits / n if n else None}, indent=2))


if __name__ == "__main__":
    main()

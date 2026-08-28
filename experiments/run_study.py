"""Interactive 12-person study runner. Writes study/responses.csv. Does not invent answers."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    scenarios = json.loads((ROOT / "study" / "scenarios.json").read_text())
    gold = json.loads((ROOT / "study" / "gold.json").read_text())
    path = ROOT / "study" / "responses.csv"
    pid = (sys.argv[1] if len(sys.argv) > 1 else input("participant_id: ").strip()) or "p_demo"
    cap = input("time cap minutes [1/3/10]: ").strip() or "3"
    cond = input("condition [raw/unimodal/vesta]: ").strip() or "vesta"
    print("\nConsent is in study/consent.md. Empty line aborts.\n")
    rows = []
    for sc in scenarios:
        print("=" * 60)
        print(sc["scenario_id"], sc["date"], sc["ticker"])
        print(sc["stimulus"])
        print("KAP:", (sc.get("kap_text") or "")[:500])
        t0 = time.time()
        dec = input("decision [up/down/abstain]: ").strip().lower()
        if not dec:
            print("aborted")
            return
        elapsed = time.time() - t0
        print("NASA-TLX 0-20, trust 1-7")
        rec = {
            "participant_id": pid,
            "scenario_id": sc["scenario_id"],
            "condition": cond,
            "time_cap_min": cap,
            "decision": dec,
            "time_s": f"{elapsed:.1f}",
        }
        for k in [
            "tlx_mental",
            "tlx_physical",
            "tlx_temporal",
            "tlx_performance",
            "tlx_effort",
            "tlx_frustration",
            "trust",
        ]:
            rec[k] = input(f"  {k}: ").strip()
        rec["notes"] = ""
        # do not show gold during the session
        _ = gold[sc["scenario_id"]]["correct_week"]
        rows.append(rec)
    fieldnames = list(rows[0].keys())
    new_file = not path.exists() or path.stat().st_size < 50
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows for {pid} to {path}. Score later with experiments/score_study.py")


if __name__ == "__main__":
    main()

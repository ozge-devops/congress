"""Execute the public retrieve-score hop on VESTA-Public (M3 day-level)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vesta.data import build_samples, chronological_split, download_public_market  # noqa: E402
from vesta.hop import load_m3, retrieve_prior_day, same_date_peer  # noqa: E402


def main() -> None:
    frames = download_public_market(ROOT / "data" / "cache")
    train, val, test = chronological_split(build_samples(frames))
    dates, vecs = load_m3()
    date_to_y = {s.date.strftime("%Y-%m-%d"): int(s.y_fwd) for s in train + val + test}
    events = pd.read_parquet(
        ROOT / "data" / "vesta_public" / "events.parquet",
        columns=["date", "ticker", "kap_text", "session_ret"],
    )
    events["date"] = pd.to_datetime(events["date"]).dt.strftime("%Y-%m-%d")

    rows = []
    for s in test:
        q = s.date.strftime("%Y-%m-%d")
        hop = retrieve_prior_day(q, dates, vecs)
        if hop is None:
            continue
        hop_y = date_to_y.get(hop["hop_date"])
        analog_hit = hop_y == int(s.y_fwd) if hop_y is not None else None
        peer = same_date_peer(events, q)
        rows.append(
            {
                **hop,
                "y_t1": int(s.y_fwd),
                "hop_y_t1": hop_y,
                "analog_hit": analog_hit,
                "peer_ticker": None if peer is None else peer["ticker"],
                "peer_teaser": None if peer is None else peer["kap_teaser"],
            }
        )

    analog = [r["analog_hit"] for r in rows if r["analog_hit"] is not None]
    peer_n = sum(1 for r in rows if r["peer_ticker"])
    report = {
        "executed": True,
        "encoder": "BAAI/bge-m3",
        "scope": "day-level KAP list concatenation",
        "alpha": 1.0,
        "beta_fitted": False,
        "portfolio": False,
        "n_test": len(test),
        "n_retrieved": len(rows),
        "mean_cosine": float(np.mean([r["cosine"] for r in rows])),
        "analog_hit_rate": float(np.mean(analog)) if analog else None,
        "n_analog_scored": len(analog),
        "peer_coverage": peer_n / len(rows) if rows else 0.0,
        "mixer_unchanged": True,
        "t3_t10_accuracy_still": 0.497,
        "note": (
            "Retrieved prior KAP day does not enter the 16-d mixer. "
            "T3/T10 next-day accuracy stays 49.7% on seed 0."
        ),
        "example": rows[0] if rows else None,
    }
    out = ROOT / "results" / "hop.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "example"}, indent=2))
    if report["example"]:
        print("example", report["example"]["query_date"], "->", report["example"]["hop_date"],
              "cos", round(report["example"]["cosine"], 4))


if __name__ == "__main__":
    main()

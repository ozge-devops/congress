"""Build the eight sealed user-study scenarios from the public test window."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    events = pd.read_parquet(ROOT / "data" / "vesta_public" / "events_10k.parquet")
    test = events[(events["split"] == "test") & (events["has_kap"])].copy()
    test["abs_ret"] = test["session_ret"].abs()
    picks = []
    used = set()
    # Diverse: index + 7 names, mixed polarity/chart, larger moves first.
    index = test[test["ticker"] == "XU100.IS"].nlargest(3, "abs_ret")
    rest = test[test["ticker"] != "XU100.IS"].sort_values("abs_ret", ascending=False)
    for _, row in pd.concat([index, rest]).iterrows():
        key = (row["ticker"], row["chart_signal"], row["kap_polarity"])
        if row["ticker"] in used and row["ticker"] != "XU100.IS":
            continue
        if key in {(p["ticker"], p.get("chart_cue"), p.get("kap_cue")) for p in picks}:
            continue
        used.add(row["ticker"])
        picks.append(
            {
                "scenario_id": f"S{len(picks)+1:02d}",
                "event_id": row["event_id"],
                "date": row["date"],
                "ticker": row["ticker"],
                "sector": row["sector"],
                "session_ret": float(row["session_ret"]),
                "usdtry_ret": float(row["usdtry_ret"]),
                "gold_ret": float(row["gold_ret"]),
                "rsi": float(row["rsi"]),
                "stimulus": (
                    f"{row['date']} {row['ticker']}. Session return {float(row['session_ret']):+.2%}. "
                    f"USD/TRY {float(row['usdtry_ret']):+.2%}, gold {float(row['gold_ret']):+.2%}. "
                    f"RSI {float(row['rsi']):.0f}."
                ),
                "kap_text": str(row["kap_text"])[:1200],
                "n_kap": int(row["n_kap"]),
                "chart_cue": row["chart_signal"],
                "kap_cue": row["kap_polarity"],
            }
        )
        if len(picks) == 8:
            break
    public = [
        {k: v for k, v in p.items() if k not in {"chart_cue", "kap_cue", "event_id"}}
        for p in picks
    ]
    gold = {}
    by_id = events.set_index("event_id")
    for p in picks:
        r = by_id.loc[p["event_id"]]
        gold[p["scenario_id"]] = {
            "event_id": p["event_id"],
            "y_direction_1d": int(r["y_direction_1d"]),
            "y_direction_5d": int(r["y_direction_5d"]),
            "next_ret_1d": float(r["next_ret_1d"]),
            "next_ret_5d": float(r["next_ret_5d"]),
            "correct_week": "up" if int(r["y_direction_5d"]) == 1 else "down",
        }
    out = ROOT / "study"
    (out / "scenarios.json").write_text(json.dumps(public, indent=2, ensure_ascii=False))
    (out / "gold.json").write_text(json.dumps(gold, indent=2))
    print(f"wrote {len(public)} scenarios")
    for p in public:
        print(p["scenario_id"], p["date"], p["ticker"])


if __name__ == "__main__":
    main()

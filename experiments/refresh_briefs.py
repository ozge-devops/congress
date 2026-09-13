"""Rebuild event `brief` from session, KAP, and text_polarity columns."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vesta.labeling import render_brief  # noqa: E402


def _bits(row: pd.Series) -> list[str]:
    out: list[str] = []
    headlines = row.get("headlines")
    if pd.notna(headlines) and str(headlines).strip():
        out.extend(part.strip() for part in str(headlines).split(" || ") if part.strip())
    kap = row.get("kap_text")
    if pd.notna(kap) and str(kap).strip():
        out.extend(part.strip() for part in str(kap).split(" || ") if part.strip())
    return out


def rebuild_brief(row: pd.Series) -> str:
    vol = row.get("vol20")
    vol_f = float(vol) if pd.notna(vol) else 0.0
    rsi = row.get("rsi")
    rsi_f = float(rsi) if pd.notna(rsi) else 50.0
    return render_brief(
        str(row["ticker"]),
        str(row["date"])[:10],
        float(row["session_ret"]),
        float(row["usdtry_ret"]),
        float(row["gold_ret"]),
        vol_f,
        rsi_f,
        str(row["text_polarity"]),
        str(row["chart_signal"]),
        _bits(row),
    )


def main() -> None:
    out = ROOT / "data" / "vesta_public"
    events = pd.read_parquet(out / "events.parquet")
    events["date"] = pd.to_datetime(events["date"]).dt.strftime("%Y-%m-%d")
    events["brief"] = events.apply(rebuild_brief, axis=1)
    events.to_parquet(out / "events.parquet", index=False)
    slim = events.drop(columns=["ohlc_open", "ohlc_high", "ohlc_low", "ohlc_close"], errors="ignore")
    slim.to_csv(out / "events.csv", index=False)

    compact_ids = None
    compact_path = out / "events_10k.parquet"
    if compact_path.exists():
        compact_ids = set(pd.read_parquet(compact_path, columns=["event_id"])["event_id"])
        compact = events[events["event_id"].isin(compact_ids)].copy()
        compact.to_parquet(compact_path, index=False)
        compact.drop(columns=["ohlc_open", "ohlc_high", "ohlc_low", "ohlc_close"], errors="ignore").to_csv(
            out / "events_10k.csv", index=False
        )

    sample_path = out / "human_annotation_sample.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        if "event_id" in sample.columns:
            br = events.set_index("event_id")["brief"]
            sample["brief"] = sample["event_id"].map(br).fillna(sample.get("brief"))
            sample.to_csv(sample_path, index=False)

    mism = int((events["brief"].str.extract(r"Brief polarity (\w+)", expand=False) != events["text_polarity"]).sum())
    print(f"rewrote {len(events)} briefs; polarity mismatches left={mism}")


if __name__ == "__main__":
    main()

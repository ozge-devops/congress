"""Build the silver-labeled public VESTA corpus on disk."""           

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vesta.corpus import (  # noqa: E402
    CONSTITUENTS,
    build_events,
    download_constituents,
    fetch_recent_news,
    stratified_annotation_sample,
    summarize,
    make_10k,
)
from vesta.data import download_public_market  # noqa: E402
from vesta.kap import download_kap, inventory  # noqa: E402


def main() -> None:
    cache = ROOT / "data" / "cache"
    out = ROOT / "data" / "vesta_public"
    out.mkdir(parents=True, exist_ok=True)

    print("Downloading index + FX + gold …")
    index_frames = download_public_market(cache)
    print("Downloading constituents …")
    stocks = download_constituents(cache)
    print(f"  got {len(stocks)} equity series")
    print("Fetching recent Yahoo headlines (not a historical news archive) …")
    news_bag = fetch_recent_news(
        ["XU100.IS", *CONSTITUENTS.keys()],
        cache / "yahoo_news.json",
    )
    n_news = sum(len(v) for v in news_bag.values())
    print(f"  cached {n_news} recent items")

    print("Downloading KAP disclosure lists (public byCriteria API) …")
    kap_rows = download_kap(cache)
    kap_inv = inventory(kap_rows)
    (out / "kap_inventory.json").write_text(json.dumps(kap_inv, indent=2, ensure_ascii=False))
    print(f"  {len(kap_rows)} KAP filings ({kap_inv['n_unique_disclosure_index']} unique indices)")

    print("Labeling events …")
    events = build_events(index_frames, stocks, news_bag, kap_rows)
    stats = summarize(events)
    print(json.dumps(stats, indent=2))

    parquet = out / "events.parquet"
    csv = out / "events.csv"
    events.to_parquet(parquet, index=False)
    # CSV without the long OHLC arrays for spreadsheet use
    slim = events.drop(columns=["ohlc_open", "ohlc_high", "ohlc_low", "ohlc_close"])
    slim.to_csv(csv, index=False)
    (out / "label_stats.json").write_text(json.dumps(stats, indent=2))

    sample = stratified_annotation_sample(events, n=250)
    sample_path = out / "human_annotation_sample.csv"
    sample.to_csv(sample_path, index=False)

    compact = make_10k(events)
    compact.to_parquet(out / "events_10k.parquet", index=False)
    compact.drop(columns=["ohlc_open", "ohlc_high", "ohlc_low", "ohlc_close"]).to_csv(
        out / "events_10k.csv", index=False
    )
    compact_stats = summarize(compact)
    (out / "label_stats_10k.json").write_text(json.dumps(compact_stats, indent=2))

    codebook = ROOT / "docs" / "LABEL_CODEBOOK.md"
    print(f"Wrote {parquet} ({parquet.stat().st_size} bytes)")
    print(f"Wrote {csv}")
    print(f"Wrote 10k slice {len(compact)} rows")
    print(f"Wrote {sample_path} ({len(sample)} rows for annotators)")
    print(f"Codebook: {codebook}")


if __name__ == "__main__":
    main()

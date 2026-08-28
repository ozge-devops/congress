"""Attach KAP bodies, KAP-only polarity, and daily KAP features. No Yahoo re-download."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vesta.kap import (  # noqa: E402
    download_bodies,
    download_kap,
    inventory,
    is_noise_subject,
    load_bodies,
)
from vesta.labeling import POLARITY_ID, kap_polarity, kap_score, text_polarity  # noqa: E402
from vesta.corpus import summarize, stratified_annotation_sample  # noqa: E402


def _split_pipe(val: str) -> list[str]:
    if not val or (isinstance(val, float) and np.isnan(val)):
        return []
    return [p.strip() for p in str(val).split(" || ") if p.strip()]


def _indices(val: str) -> list[int]:
    out = []
    if not val or (isinstance(val, float) and np.isnan(val)):
        return out
    for p in str(val).split(","):
        p = p.strip()
        if p.isdigit():
            out.append(int(p))
    return out


def daily_feature_row(filings: list[dict], bodies: dict[int, dict]) -> dict:
    texts, subjects = [], []
    n_material = 0
    has_earnings = has_dividend = has_buyback = has_rating = has_oda = 0
    for f in filings:
        sub = f.get("subject") or ""
        idx = f.get("disclosure_index")
        body = ""
        if idx is not None and int(idx) in bodies:
            body = bodies[int(idx)].get("body_text") or ""
        blob = " ".join(x for x in (sub, f.get("text") or "", body) if x)
        texts.append(blob)
        subjects.append(sub)
        sl = sub.lower()
        if not is_noise_subject(sub):
            n_material += 1
        if "finansal rapor" in sl:
            has_earnings = 1
        if "kar payı" in sl or "kâr payı" in sl or "temettü" in sl:
            has_dividend = 1
        if "geri al" in sl:
            has_buyback = 1
        if "kredi derecelendirme" in sl:
            has_rating = 1
        if "özel durum" in sl:
            has_oda = 1
    score = kap_score(texts, subjects)
    pol = kap_polarity(texts, subjects)
    n = len(filings)
    return {
        "n_kap_day": n,
        "n_material": n_material,
        "kap_score": score,
        "kap_polarity": pol,
        "has_earnings": has_earnings,
        "has_dividend": has_dividend,
        "has_buyback": has_buyback,
        "has_rating": has_rating,
        "has_oda": has_oda,
        "kap_feat": np.array(
            [
                min(n, 30) / 30.0,
                min(n_material, 15) / 15.0,
                min(max(score, 0.0), 5.0) / 5.0,
                min(max(-score, 0.0), 5.0) / 5.0,
                float(has_earnings),
                float(has_dividend),
                float(has_buyback),
                float({"bearish": -1.0, "neutral": 0.0, "bullish": 1.0}[pol]),
            ],
            dtype=np.float32,
        ),
    }


def main() -> None:
    cache = ROOT / "data" / "cache"
    out = ROOT / "data" / "vesta_public"
    events_path = out / "events.parquet"
    events = pd.read_parquet(events_path)
    kap_rows = download_kap(cache)

    need: list[int] = []
    seen = set()
    for val in events["kap_indices"].fillna(""):
        for idx in _indices(val):
            if idx not in seen:
                seen.add(idx)
                need.append(idx)
    # Prefer test-window and non-noise first.
    by_idx = {int(r["disclosure_index"]): r for r in kap_rows if r.get("disclosure_index") is not None}
    def sort_key(idx: int) -> tuple:
        rec = by_idx.get(idx, {})
        noise = 1 if is_noise_subject(rec.get("subject") or "") else 0
        recent = 0 if (rec.get("date") or "") >= "2025-01-01" else 1
        return (recent, noise, -idx)

    need.sort(key=sort_key)
    bodies = load_bodies(cache)
    fetch_cap = int(os.environ.get("VESTA_KAP_BODY_CAP", "8000"))
    missing = [i for i in need if i not in bodies][:fetch_cap]
    print(f"Unique KAP indices on events: {len(need)}; cached bodies {len(bodies)}; fetching {len(missing)}")
    if missing:
        bodies = download_bodies(cache, missing, max_workers=10)

    # Daily KAP bag (all 27 names) for the index-level mixer.
    by_date: dict[str, list[dict]] = {}
    for r in kap_rows:
        by_date.setdefault(r["date"], []).append(r)
    daily_rows = []
    daily_vec = {}
    for day, filings in sorted(by_date.items()):
        feat = daily_feature_row(filings, bodies)
        daily_vec[day] = feat
        daily_rows.append({"date": day, **{k: feat[k] for k in feat if k != "kap_feat"}})
    daily_df = pd.DataFrame(daily_rows)
    daily_df.to_csv(out / "kap_daily_features.csv", index=False)
    feat_map = {d: v["kap_feat"].tolist() for d, v in daily_vec.items()}
    (out / "kap_daily_features.json").write_text(json.dumps(feat_map))

    # Enrich each event with body snippets + KAP-only polarity.
    body_texts = []
    kap_pols = []
    kap_scores = []
    n_bodies = []
    mixed_pols = []
    for rec in events.itertuples(index=False):
        idxs = _indices(getattr(rec, "kap_indices", ""))
        blobs = []
        n_b = 0
        for idx in idxs[:8]:
            b = bodies.get(idx, {})
            t = (b.get("body_text") or "").strip()
            if t:
                blobs.append(t[:800])
                n_b += 1
        subjects = _split_pipe(getattr(rec, "kap_subjects", ""))
        teasers = _split_pipe(getattr(rec, "kap_text", ""))
        kscore = kap_score(teasers + blobs, subjects)
        kpol = kap_polarity(teasers + blobs, subjects)
        mixed = text_polarity(
            float(rec.usdtry_ret),
            float(rec.gold_ret),
            float(rec.session_ret),
            float(rec.rsi),
            kscore,
        )
        body_texts.append(" || ".join(blobs[:4]))
        kap_pols.append(kpol)
        kap_scores.append(kscore)
        n_bodies.append(n_b)
        mixed_pols.append(mixed)

    events["kap_body"] = body_texts
    events["n_kap_bodies"] = n_bodies
    events["kap_score"] = kap_scores
    events["kap_polarity"] = kap_pols
    events["kap_polarity_id"] = [POLARITY_ID[p] for p in kap_pols]
    events["text_polarity"] = mixed_pols
    events["text_polarity_id"] = [POLARITY_ID[p] for p in mixed_pols]
    day_feat = events["date"].map(lambda d: feat_map.get(d, [0.0] * 8))
    events["kap_feat"] = day_feat.map(json.dumps)

    stats = summarize(events)
    stats["with_kap_body"] = int((events["n_kap_bodies"] > 0).sum())
    stats["kap_polarity"] = {str(k): int(v) for k, v in events["kap_polarity"].value_counts().sort_index().items()}
    stats["n_bodies_cached"] = sum(1 for r in bodies.values() if r.get("ok") and r.get("body_text"))
    (out / "label_stats.json").write_text(json.dumps(stats, indent=2))

    events.to_parquet(events_path, index=False)
    slim = events.drop(columns=["ohlc_open", "ohlc_high", "ohlc_low", "ohlc_close"], errors="ignore")
    slim.to_csv(out / "events.csv", index=False)

    from vesta.corpus import make_10k

    compact = make_10k(events)
    compact.to_parquet(out / "events_10k.parquet", index=False)
    compact.drop(columns=["ohlc_open", "ohlc_high", "ohlc_low", "ohlc_close"], errors="ignore").to_csv(
        out / "events_10k.csv", index=False
    )
    cstats = summarize(compact)
    cstats["with_kap_body"] = int((compact["n_kap_bodies"] > 0).sum())
    cstats["kap_polarity"] = {str(k): int(v) for k, v in compact["kap_polarity"].value_counts().sort_index().items()}
    (out / "label_stats_10k.json").write_text(json.dumps(cstats, indent=2))

    sample_path = out / "human_annotation_sample.csv"
    skip_sample = os.environ.get("VESTA_SKIP_ANNOTATION_SAMPLE", "").strip() in {"1", "true", "yes"}
    if skip_sample and sample_path.exists():
        print("keeping existing human_annotation_sample.csv")
    else:
        sample = stratified_annotation_sample(events, n=250)
        extra_cols = [c for c in ("kap_score", "n_kap_bodies", "kap_body") if c in events.columns]
        if extra_cols:
            extra = events[["event_id", *extra_cols]].drop_duplicates("event_id")
            sample = sample.merge(extra, on="event_id", how="left")
        sample.to_csv(sample_path, index=False)

    inv = inventory(kap_rows)
    inv["n_bodies_cached"] = stats["n_bodies_cached"]
    inv["n_bodies_attempted"] = len(bodies)
    (out / "kap_inventory.json").write_text(json.dumps(inv, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2))
    print("enriched corpus written")


if __name__ == "__main__":
    main()

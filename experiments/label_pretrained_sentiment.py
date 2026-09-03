"""Rater D: pretrained multilingual sentiment on KAP teasers and bodies.

Model: nlptown/bert-base-multilingual-uncased-sentiment.
Does not overwrite annotator_id=codebook_b.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vesta.metrics import cohen_kappa, fleiss_kappa  # noqa: E402

MODEL_ID = "nlptown/bert-base-multilingual-uncased-sentiment"


def _split(val) -> list[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    return [p.strip() for p in str(val).split(" || ") if p.strip()]


def _blob(row: pd.Series) -> str:
    parts = _split(row.get("kap_text")) + _split(row.get("kap_subjects")) + _split(row.get("kap_body"))
    text = " ".join(p for p in parts if p)[:512]
    return text or "no kap"


def _stars_to_polarity(stars: int) -> str:
    if stars <= 2:
        return "bearish"
    if stars >= 4:
        return "bullish"
    return "neutral"


def score_texts(texts: list[str], batch: int = 16) -> list[str]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
    labels = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            enc = tok(chunk, padding=True, truncation=True, max_length=256, return_tensors="pt")
            logits = model(**enc).logits
            stars = (logits.argmax(-1).cpu().numpy() + 1).tolist()
            labels.extend(_stars_to_polarity(int(s)) for s in stars)
            if (i // batch) % 20 == 0:
                print(f"  sent {i}/{len(texts)}", flush=True)
    return labels


def main() -> None:
    sample_path = ROOT / "data" / "vesta_public" / "human_annotation_sample.csv"
    sample = pd.read_csv(sample_path)
    events = pd.read_parquet(ROOT / "data" / "vesta_public" / "events.parquet")
    merged = sample.merge(
        events[["event_id", "kap_subjects", "kap_text", "kap_body"]],
        on="event_id",
        how="left",
        suffixes=("", "_ev"),
    )
    if "kap_body_ev" in merged.columns:
        merged["kap_body"] = merged["kap_body"].fillna(merged["kap_body_ev"])
    texts = [_blob(r) for _, r in merged.iterrows()]
    print(f"Scoring {len(texts)} sample rows with {MODEL_ID}", flush=True)
    d_labels = score_texts(texts)
    sample = sample.copy()
    sample["rater_d_kap"] = d_labels
    sample.to_csv(sample_path, index=False)

    compact = pd.read_parquet(ROOT / "data" / "vesta_public" / "events_10k.parquet")
    print(f"Scoring 10k slice ({len(compact)})", flush=True)
    d10 = score_texts([_blob(r) for _, r in compact.iterrows()])

    report = {
        "model": MODEL_ID,
        "not_a_human": True,
        "n_sample": int(len(sample)),
        "sample": {
            "d_vs_a": cohen_kappa(sample["rater_a_kap"], sample["rater_d_kap"]),
            "d_vs_b": cohen_kappa(sample["rater_b_kap"], sample["rater_d_kap"]),
            "d_vs_c": cohen_kappa(sample["rater_c_kap"], sample["rater_d_kap"]),
            "counts_d": sample["rater_d_kap"].value_counts().to_dict(),
        },
        "n_10k": int(len(compact)),
    }
    if "kap_polarity" in compact.columns:
        report["kap_10k_d_vs_a"] = cohen_kappa(compact["kap_polarity"].tolist(), d10)
        report["counts_d_10k"] = pd.Series(d10).value_counts().to_dict()
    dest = ROOT / "results" / "pretrained_sentiment_rater.json"
    dest.write_text(json.dumps(report, indent=2, default=float))
    (ROOT / "data" / "vesta_public" / "rater_d_10k.json").write_text(json.dumps({"model": MODEL_ID, "labels": d10}))
    print(json.dumps(report, indent=2, default=float))
    print("annotator_id still codebook_b; rater_d_kap is a pretrained model.")


if __name__ == "__main__":
    main()

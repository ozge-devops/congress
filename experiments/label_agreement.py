"""Fill the 250-row sample with independent silver codebooks and report κ.

This is not a three-human gold set. Rater A is the production silver rule,
B is a subject-taxonomy codebook, C is a body-token codebook. Chart B uses
10-bar levels instead of 20-bar. Annotator columns are filled by codebook B
and flagged as such.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vesta.labeling import (  # noqa: E402
    chart_signal,
    chart_signal_b,
    kap_polarity,
    kap_polarity_b,
    kap_polarity_c,
)
from vesta.metrics import cohen_kappa, fleiss_kappa  # noqa: E402


def _split(val) -> list[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    return [p.strip() for p in str(val).split(" || ") if p.strip()]


def _ohlc(row: pd.Series) -> np.ndarray:
    o = json.loads(row["ohlc_open"])
    h = json.loads(row["ohlc_high"])
    l = json.loads(row["ohlc_low"])
    c = json.loads(row["ohlc_close"])
    return np.stack([o, h, l, c], axis=1)


def label_row(row: pd.Series) -> dict:
    subjects = _split(row.get("kap_subjects"))
    teasers = _split(row.get("kap_text"))
    bodies = _split(row.get("kap_body")) if "kap_body" in row.index else []
    a = kap_polarity(teasers + bodies, subjects)
    b = kap_polarity_b(subjects)
    c = kap_polarity_c(teasers + bodies)
    ohlc = _ohlc(row)
    ca = chart_signal(ohlc)
    cb = chart_signal_b(ohlc)
    return {
        "rater_a_kap": a,
        "rater_b_kap": b,
        "rater_c_kap": c,
        "rater_a_chart": ca,
        "rater_b_chart": cb,
    }


def main() -> None:
    events = pd.read_parquet(ROOT / "data" / "vesta_public" / "events.parquet")
    sample_path = ROOT / "data" / "vesta_public" / "human_annotation_sample.csv"
    sample = pd.read_csv(sample_path)
    merged = sample.merge(
        events[
            [
                "event_id",
                "kap_subjects",
                "kap_text",
                "kap_body",
                "ohlc_open",
                "ohlc_high",
                "ohlc_low",
                "ohlc_close",
            ]
        ],
        on="event_id",
        how="left",
        suffixes=("", "_ev"),
    )
    if "kap_body_ev" in merged.columns:
        merged["kap_body"] = merged["kap_body"].fillna(merged["kap_body_ev"])
    rows = [label_row(r) for _, r in merged.iterrows()]
    lab = pd.DataFrame(rows)
    out = sample.copy()
    out["rater_a_kap"] = lab["rater_a_kap"].to_numpy()
    out["rater_b_kap"] = lab["rater_b_kap"].to_numpy()
    out["rater_c_kap"] = lab["rater_c_kap"].to_numpy()
    out["rater_a_chart"] = lab["rater_a_chart"].to_numpy()
    out["rater_b_chart"] = lab["rater_b_chart"].to_numpy()
    # Fill the human-shaped columns with codebook B, never pretend this is a person.
    out["annotator_text_polarity"] = out["rater_b_kap"]
    out["annotator_chart_signal"] = out["rater_b_chart"]
    out["annotator_id"] = "codebook_b"
    out["annotator_notes"] = (
        "Not a human rater. Text=subject taxonomy (codebook B). "
        "Chart=10-bar VisualClaw codebook B. See docs/AGREEMENT.md."
    )
    out.to_csv(sample_path, index=False)

    # Agreement on the 250-row sample
    report = {
        "n": int(len(out)),
        "source": "independent silver codebooks, not three human annotators",
        "kap": {
            "a_vs_b": cohen_kappa(out["rater_a_kap"], out["rater_b_kap"]),
            "a_vs_c": cohen_kappa(out["rater_a_kap"], out["rater_c_kap"]),
            "b_vs_c": cohen_kappa(out["rater_b_kap"], out["rater_c_kap"]),
            "fleiss_abc": fleiss_kappa(
                [out["rater_a_kap"].tolist(), out["rater_b_kap"].tolist(), out["rater_c_kap"].tolist()]
            ),
        },
        "chart": {
            "a_vs_b": cohen_kappa(out["rater_a_chart"], out["rater_b_chart"]),
        },
        "counts": {
            "rater_a_kap": out["rater_a_kap"].value_counts().to_dict(),
            "rater_b_kap": out["rater_b_kap"].value_counts().to_dict(),
            "rater_c_kap": out["rater_c_kap"].value_counts().to_dict(),
            "rater_a_chart": out["rater_a_chart"].value_counts().to_dict(),
            "rater_b_chart": out["rater_b_chart"].value_counts().to_dict(),
        },
    }

    # Same κ on the 10k slice (more stable)
    compact = pd.read_parquet(ROOT / "data" / "vesta_public" / "events_10k.parquet")
    c_lab = pd.DataFrame([label_row(r) for _, r in compact.iterrows()])
    report["n_10k"] = int(len(c_lab))
    report["kap_10k"] = {
        "a_vs_b": cohen_kappa(c_lab["rater_a_kap"], c_lab["rater_b_kap"]),
        "a_vs_c": cohen_kappa(c_lab["rater_a_kap"], c_lab["rater_c_kap"]),
        "b_vs_c": cohen_kappa(c_lab["rater_b_kap"], c_lab["rater_c_kap"]),
        "fleiss_abc": fleiss_kappa(
            [c_lab["rater_a_kap"].tolist(), c_lab["rater_b_kap"].tolist(), c_lab["rater_c_kap"].tolist()]
        ),
    }
    report["chart_10k"] = {"a_vs_b": cohen_kappa(c_lab["rater_a_chart"], c_lab["rater_b_chart"])}

    dest = ROOT / "results" / "agreement.json"
    dest.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {sample_path} and {dest}")


if __name__ == "__main__":
    main()

"""M3 day-level retrieve-score hop.

No user portfolio. Score is cosine on cached BGE-M3 KAP-day vectors (α=1).
The β gold / USD/TRY / session terms are not fitted and are not used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
M3_NPZ = ROOT / "data" / "vesta_public" / "kap_daily_embeddings_m3.npz"


def load_m3(path: Path | None = None) -> tuple[list[str], np.ndarray]:
    packed = np.load(path or M3_NPZ, allow_pickle=True)
    dates = [str(d) for d in packed["dates"].tolist()]
    vecs = np.asarray(packed["vectors"], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.clip(norms, 1e-8, None)
    return dates, vecs


def retrieve_prior_day(query_date: str, dates: list[str], vecs: np.ndarray) -> dict | None:
    """Top prior KAP day by cosine. Candidates are strictly before query_date."""
    if query_date not in dates:
        return None
    q = dates.index(query_date)
    if q == 0:
        return None
    scores = vecs[:q] @ vecs[q]
    j = int(np.argmax(scores))
    return {
        "query_date": query_date,
        "hop_date": dates[j],
        "cosine": float(scores[j]),
        "n_candidates": q,
        "alpha": 1.0,
        "beta_fitted": False,
        "portfolio": False,
        "encoder": "BAAI/bge-m3",
    }


def same_date_peer(events: pd.DataFrame, date: str) -> dict | None:
    """Same-calendar-day XU100 constituent with KAP; prefer largest |session_ret|."""
    day = events[(events["date"] == date) & (events["ticker"] != "XU100.IS")].copy()
    if "kap_text" in day.columns:
        kap = day["kap_text"].fillna("").astype(str).str.strip()
        day = day[kap.ne("")]
    if day.empty:
        return None
    day = day.assign(_abs=day["session_ret"].abs())
    row = day.sort_values("_abs", ascending=False).iloc[0]
    teaser = str(row.get("kap_text") or "")
    teaser = teaser.split(" || ")[0][:180]
    return {
        "date": date,
        "ticker": str(row["ticker"]),
        "session_ret": float(row["session_ret"]),
        "kap_teaser": teaser,
    }

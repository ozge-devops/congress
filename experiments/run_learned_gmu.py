"""Learned Arevalo GMU (W_v, W_t, W_z) vs score-space GMU on the public split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vesta.charts import render_candles  # noqa: E402
from vesta.data import build_samples, chronological_split, download_public_market  # noqa: E402
from vesta.metrics import binary_scores, mcnemar, mean_ci  # noqa: E402
from vesta.models import (  # noqa: E402
    fit_fusion,
    fit_learned_gmu,
    fit_unimodal,
    gmu_features,
    predict_proba,
)


def stack(samples):
    text = np.stack([s.text for s in samples])
    img = np.stack([render_candles(s.ohlc, size=24).ravel() for s in samples])
    y = np.array([s.y_fwd for s in samples], dtype=int)
    return text, img, y


def main() -> dict:
    frames = download_public_market(ROOT / "data" / "cache")
    train, val, test = chronological_split(build_samples(frames))
    fit = train + val
    Xtr_t, Xtr_v, ytr = stack(fit)
    Xte_t, Xte_v, yte = stack(test)

    accs, f1s = [], []
    seed0_pred = None
    seed0_score_gmu = None
    for seed in range(5):
        model = fit_learned_gmu(Xtr_t, Xtr_v, ytr, seed=seed)
        p = model.predict_proba(Xte_t, Xte_v)
        pred = (p >= 0.5).astype(int)
        sc = binary_scores(yte, pred)
        accs.append(sc["accuracy"])
        f1s.append(sc["f1"])
        if seed == 0:
            seed0_pred = pred
            text_m = fit_unimodal("text", Xtr_t, ytr, 0)
            vis_m = fit_unimodal("vision", Xtr_v, ytr, 0)
            h_tr = gmu_features(
                np.stack([1 - predict_proba(text_m, Xtr_t), predict_proba(text_m, Xtr_t)], 1),
                np.stack([1 - predict_proba(vis_m, Xtr_v), predict_proba(vis_m, Xtr_v)], 1),
            )
            h_te = gmu_features(
                np.stack([1 - predict_proba(text_m, Xte_t), predict_proba(text_m, Xte_t)], 1),
                np.stack([1 - predict_proba(vis_m, Xte_v), predict_proba(vis_m, Xte_v)], 1),
            )
            score_m = fit_fusion("gmu", h_tr, ytr, 0)
            seed0_score_gmu = (predict_proba(score_m, h_te) >= 0.5).astype(int)

    report = {
        "task": "next-day BIST100 direction",
        "gmu_kind": "learned_Arevalo_Wv_Wt_Wz_on_raw_16d_text_and_24x24_vision",
        "n_test": int(len(yte)),
        "acc": mean_ci(accs),
        "f1": mean_ci(f1s),
        "seed0": {
            "accuracy": float(binary_scores(yte, seed0_pred)["accuracy"]),
            "f1": float(binary_scores(yte, seed0_pred)["f1"]),
            "vs_score_gmu": mcnemar(yte, seed0_pred, seed0_score_gmu),
        },
        "score_gmu_seed0": {
            "accuracy": float(binary_scores(yte, seed0_score_gmu)["accuracy"]),
            "f1": float(binary_scores(yte, seed0_score_gmu)["f1"]),
        },
    }
    out = ROOT / "results" / "learned_gmu.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()

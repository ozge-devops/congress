"""Second chronological holdout: 2023, trained on days strictly before 2023-01-01."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vesta.charts import render_candles  # noqa: E402
from vesta.data import build_samples, download_public_market  # noqa: E402
from vesta.metrics import binary_scores, mean_ci, sharpe  # noqa: E402
from vesta.models import (  # noqa: E402
    fit_fusion,
    fit_unimodal,
    gated_features,
    predict_proba,
)

SEEDS = [0, 1, 2, 3, 4]
COST = 5.0 / 10000.0
CUT = pd.Timestamp("2023-01-01")
END = pd.Timestamp("2024-01-01")


def _stack(samples):
    tab = np.stack([s.tabular for s in samples])
    txt = np.stack([s.text for s in samples])
    img = np.stack([render_candles(s.ohlc, size=24).ravel() for s in samples])
    y = np.array([s.y_fwd for s in samples], dtype=int)
    rets = np.array([s.next_ret for s in samples], dtype=float)
    return tab, txt, img, y, rets


def _backtest(pred, rets):
    pos = np.where(pred == 1, 1.0, -1.0)
    net = pos * rets - COST
    equity = np.cumprod(1.0 + net)
    bh = np.cumprod(1.0 + rets)
    return {
        "hit_rate": float((pos * rets > 0).mean()),
        "sharpe": sharpe(net),
        "sharpe_bh": sharpe(rets),
        "total_net": float(equity[-1] - 1.0),
        "total_bh": float(bh[-1] - 1.0),
    }


def main() -> None:
    samples = build_samples(download_public_market(ROOT / "data" / "cache"))
    fit = [s for s in samples if s.date < CUT]
    test = [s for s in samples if CUT <= s.date < END]
    Xtr_tab, Xtr_txt, Xtr_img, ytr, _ = _stack(fit)
    Xte_tab, Xte_txt, Xte_img, yte, rets = _stack(test)
    accs: dict[str, list[float]] = {k: [] for k in ("majority", "text", "vision", "tabular", "mean", "gated")}
    f1s: dict[str, list[float]] = {k: [] for k in accs}
    seed0 = {}
    for seed in SEEDS:
        text_m = fit_unimodal("text", Xtr_txt, ytr, seed)
        vis_m = fit_unimodal("vision", Xtr_img, ytr, seed)
        tab_m = fit_unimodal("tabular", Xtr_tab, ytr, seed)
        p_txt_tr = predict_proba(text_m, Xtr_txt)
        p_vis_tr = predict_proba(vis_m, Xtr_img)
        h_t_tr = np.stack([1 - p_txt_tr, p_txt_tr], axis=1)
        h_v_tr = np.stack([1 - p_vis_tr, p_vis_tr], axis=1)
        mean_m = fit_fusion("mean", (h_t_tr + h_v_tr) / 2.0, ytr, seed)
        gated_m = fit_fusion("gated", gated_features(h_t_tr, h_v_tr), ytr, seed)
        p_txt = predict_proba(text_m, Xte_txt)
        p_vis = predict_proba(vis_m, Xte_img)
        p_tab = predict_proba(tab_m, Xte_tab)
        h_t = np.stack([1 - p_txt, p_txt], axis=1)
        h_v = np.stack([1 - p_vis, p_vis], axis=1)
        preds = {
            "majority": np.full(len(yte), int(ytr.mean() >= 0.5)),
            "text": (p_txt >= 0.5).astype(int),
            "vision": (p_vis >= 0.5).astype(int),
            "tabular": (p_tab >= 0.5).astype(int),
            "mean": (predict_proba(mean_m, (h_t + h_v) / 2.0) >= 0.5).astype(int),
            "gated": (predict_proba(gated_m, gated_features(h_t, h_v)) >= 0.5).astype(int),
        }
        for k, pred in preds.items():
            sc = binary_scores(yte, pred)
            accs[k].append(sc["accuracy"])
            f1s[k].append(sc["f1"])
        if seed == 0:
            seed0 = {k: _backtest(preds[k], rets) for k in ("gated", "vision", "tabular")}
            seed0["buy_hold"] = _backtest(np.ones_like(yte), rets)
    report = {
        "window": "2023",
        "train_end": "2022-12-30",
        "test_start": str(test[0].date.date()),
        "test_end": str(test[-1].date.date()),
        "n_fit": len(fit),
        "n_test": len(test),
        "prior_up": float(yte.mean()),
        "forward": {k: {"acc": mean_ci(accs[k]), "f1": mean_ci(f1s[k])} for k in accs},
        "overlay_seed0": seed0,
        "note": "Trained on days strictly before 2023-01-01. Headline table remains the 2025-2026 bull window.",
    }
    out = ROOT / "results" / "second_window.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "n_test": report["n_test"],
        "prior_up": round(100 * report["prior_up"], 1),
        "acc": {k: round(100 * v["acc"]["mean"], 1) for k, v in report["forward"].items()},
        "bh_sharpe": round(seed0["buy_hold"]["sharpe_bh"], 2),
        "bh_ret": round(100 * seed0["buy_hold"]["total_bh"], 1),
    }, indent=2))


if __name__ == "__main__":
    main()

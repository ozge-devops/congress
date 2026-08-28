"""Reproduce the public BIST100 benchmark reported in the paper."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vesta.charts import render_candles  # noqa: E402
from vesta.data import build_samples, chronological_split, download_public_market  # noqa: E402
from vesta.metrics import binary_scores, information_noise, mcnemar, mean_ci, sharpe  # noqa: E402
from vesta.models import (  # noqa: E402
    closed_form_vol_rule,
    fit_fusion,
    fit_unimodal,
    gated_features,
    gmu_features,
    mult_features,
    predict_proba,
    tfn_features,
)


SEEDS = [0, 1, 2, 3, 4]
COST_BPS = 5.0 / 10000.0

TIER_SPECS = {
    "T1": {
        "max_tokens": 48,
        "gate_tau": 0.18,
        "latency_s": 0.9,
        "template": ["session_ohlc", "bist100_level", "vol_spike", "price_shock"],
    },
    "T3": {
        "max_tokens": 140,
        "gate_tau": 0.0,
        "latency_s": 2.1,
        "template": [
            "session_ohlc",
            "bist100_level",
            "vol_spike",
            "price_shock",
            "usdtry_up",
            "usdtry_down",
            "gold_up",
            "gold_down",
            "hist_analog",
            "sector_note",
        ],
    },
    "T10": {
        "max_tokens": 420,
        "gate_tau": 0.0,
        "latency_s": 3.4,
        "template": [
            "session_ohlc",
            "bist100_level",
            "vol_spike",
            "price_shock",
            "usdtry_up",
            "usdtry_down",
            "gold_up",
            "gold_down",
            "hist_analog",
            "sector_note",
            "macro_correlator",
            "hedge_sketch",
            "disclaimer",
            "kap_digest",
        ],
    },
}

FILLER = [f"filler_{i}" for i in range(80)]


def stack_features(samples, image_size: int = 24):
    tabular = np.stack([s.tabular for s in samples])
    text = np.stack([s.text for s in samples])
    images = np.stack([render_candles(s.ohlc, size=image_size).ravel() for s in samples])
    y_fwd = np.array([s.y_fwd for s in samples], dtype=int)
    y_fwd5 = np.array([s.y_fwd5 for s in samples], dtype=int)
    y_leak = np.array([s.y_leak for s in samples], dtype=int)
    rets = np.array([s.next_ret for s in samples], dtype=float)
    return tabular, text, images, y_fwd, y_fwd5, y_leak, rets


def delivered_payload(tier: str, relevant: set[str], text_vec: np.ndarray) -> list[str]:
    spec = TIER_SPECS[tier]
    live = []
    flags = {
        "usdtry_up": text_vec[0] > 0.5,
        "usdtry_down": text_vec[1] > 0.5,
        "gold_up": text_vec[2] > 0.5,
        "gold_down": text_vec[3] > 0.5,
        "vol_spike": text_vec[4] > 0.5,
        "price_shock": text_vec[5] > 0.5,
    }
    for tok in spec["template"]:
        if tok in flags and not flags[tok]:
            continue
        live.append(tok)
    # pad to the budgeted length with filler (simulates residual boilerplate)
    while len(live) < spec["max_tokens"] // 8:
        live.append(FILLER[len(live) % len(FILLER)])
    return live[: spec["max_tokens"]]


def evaluate_split(name, y_true, y_pred) -> dict:
    scores = binary_scores(y_true, y_pred)
    scores["n"] = int(len(y_true))
    scores["pos"] = int(y_true.sum())
    return scores


def backtest(pred: np.ndarray, rets: np.ndarray) -> dict:
    pos = np.where(pred == 1, 1.0, -1.0)
    net = pos * rets - COST_BPS
    equity = np.cumprod(1.0 + net)
    bh = np.cumprod(1.0 + rets)
    return {
        "hit_rate": float((pos * rets > 0).mean()),
        "mean_net": float(net.mean()),
        "sharpe": sharpe(net),
        "sharpe_bh": sharpe(rets),
        "total_net": float(equity[-1] - 1.0),
        "total_bh": float(bh[-1] - 1.0),
        "max_dd": float(((np.maximum.accumulate(equity) - equity) / np.maximum.accumulate(equity)).max()),
    }


def run() -> dict:
    cache = ROOT / "data" / "cache"
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    frames = download_public_market(cache)
    samples = build_samples(frames)
    train, val, test = chronological_split(samples)
    # train on train+val for final test numbers after model selection on val via early stopping
    fit_set = train + val

    Xtr_tab, Xtr_txt, Xtr_img, ytr_fwd, ytr_fwd5, ytr_leak, _ = stack_features(fit_set)
    Xte_tab, Xte_txt, Xte_img, yte_fwd, yte_fwd5, yte_leak, te_rets = stack_features(test)

    report: dict = {
        "n_total": len(samples),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "date_start": str(samples[0].date.date()),
        "date_end": str(samples[-1].date.date()),
        "test_start": str(test[0].date.date()),
        "test_end": str(test[-1].date.date()),
        "leak_rate_all": float(np.mean([s.y_leak for s in samples])),
        "fwd_pos_rate_test": float(yte_fwd.mean()),
    }

    # --- Diagnostic leaked-label task ---
    leak_rule = closed_form_vol_rule(Xte_tab)
    report["leak_closed_form"] = evaluate_split("closed_form", yte_leak, leak_rule)

    leak_rows = {}
    for seed in SEEDS:
        tab_m = fit_unimodal("tabular", Xtr_tab, ytr_leak, seed)
        vis_m = fit_unimodal("vision", Xtr_img, ytr_leak, seed)
        tab_p = (predict_proba(tab_m, Xte_tab) >= 0.5).astype(int)
        vis_p = (predict_proba(vis_m, Xte_img) >= 0.5).astype(int)
        leak_rows.setdefault("tabular", []).append(binary_scores(yte_leak, tab_p)["f1"])
        leak_rows.setdefault("vision", []).append(binary_scores(yte_leak, vis_p)["f1"])
        leak_rows.setdefault("tabular_acc", []).append(binary_scores(yte_leak, tab_p)["accuracy"])
        leak_rows.setdefault("vision_acc", []).append(binary_scores(yte_leak, vis_p)["accuracy"])
    report["leak_learned"] = {k: mean_ci(v) for k, v in leak_rows.items()}

    # --- Primary forward-return task ---
    fwd_f1: dict[str, list[float]] = {k: [] for k in [
        "majority", "text", "vision", "tabular", "concat", "mean", "gmu", "tfn", "gated", "mult"
    ]}
    fwd_acc: dict[str, list[float]] = {k: [] for k in fwd_f1}
    last_preds: dict[str, np.ndarray] = {}
    last_proba_gate = None
    last_g = None
    seed0_preds: dict[str, np.ndarray] | None = None
    seed0_g = None
    seed0_gate_proba = None

    majority = int(ytr_fwd.mean() >= 0.5)
    maj_pred = np.full_like(yte_fwd, majority)

    t0 = time.time()
    for seed in SEEDS:
        text_m = fit_unimodal("text", Xtr_txt, ytr_fwd, seed)
        vis_m = fit_unimodal("vision", Xtr_img, ytr_fwd, seed)
        tab_m = fit_unimodal("tabular", Xtr_tab, ytr_fwd, seed)

        p_txt = predict_proba(text_m, Xte_txt)
        p_vis = predict_proba(vis_m, Xte_img)
        p_tab = predict_proba(tab_m, Xte_tab)
        h_txt = np.stack([1 - p_txt, p_txt], axis=1)
        h_vis = np.stack([1 - p_vis, p_vis], axis=1)
        h_tab = np.stack([1 - p_tab, p_tab], axis=1)

        concat_x = np.concatenate([Xte_txt, Xte_img, Xte_tab], axis=1)
        concat_tr = np.concatenate([Xtr_txt, Xtr_img, Xtr_tab], axis=1)
        concat_m = fit_fusion("concat", concat_tr, ytr_fwd, seed)

        mean_te = (h_txt + h_vis) / 2.0
        mean_tr = (
            np.stack([1 - predict_proba(text_m, Xtr_txt), predict_proba(text_m, Xtr_txt)], axis=1)
            + np.stack([1 - predict_proba(vis_m, Xtr_img), predict_proba(vis_m, Xtr_img)], axis=1)
        ) / 2.0
        mean_m = fit_fusion("mean", mean_tr, ytr_fwd, seed)

        gmu_tr = gmu_features(
            np.stack([1 - predict_proba(text_m, Xtr_txt), predict_proba(text_m, Xtr_txt)], axis=1),
            np.stack([1 - predict_proba(vis_m, Xtr_img), predict_proba(vis_m, Xtr_img)], axis=1),
        )
        gmu_te = gmu_features(h_txt, h_vis)
        gmu_m = fit_fusion("gmu", gmu_tr, ytr_fwd, seed)

        tfn_tr = tfn_features(
            np.stack([1 - predict_proba(text_m, Xtr_txt), predict_proba(text_m, Xtr_txt)], axis=1),
            np.stack([1 - predict_proba(vis_m, Xtr_img), predict_proba(vis_m, Xtr_img)], axis=1),
        )
        tfn_te = tfn_features(h_txt, h_vis)
        tfn_m = fit_fusion("tfn", tfn_tr, ytr_fwd, seed)

        gated_tr = gated_features(
            np.stack([1 - predict_proba(text_m, Xtr_txt), predict_proba(text_m, Xtr_txt)], axis=1),
            np.stack([1 - predict_proba(vis_m, Xtr_img), predict_proba(vis_m, Xtr_img)], axis=1),
        )
        gated_te = gated_features(h_txt, h_vis)
        gated_m = fit_fusion("gated", gated_tr, ytr_fwd, seed)

        mult_tr = mult_features(Xtr_txt, Xtr_img)
        mult_te = mult_features(Xte_txt, Xte_img)
        mult_m = fit_fusion("mult", mult_tr, ytr_fwd, seed)

        preds = {
            "majority": maj_pred,
            "text": (p_txt >= 0.5).astype(int),
            "vision": (p_vis >= 0.5).astype(int),
            "tabular": (p_tab >= 0.5).astype(int),
            "concat": (predict_proba(concat_m, concat_x) >= 0.5).astype(int),
            "mean": (predict_proba(mean_m, mean_te) >= 0.5).astype(int),
            "gmu": (predict_proba(gmu_m, gmu_te) >= 0.5).astype(int),
            "tfn": (predict_proba(tfn_m, tfn_te) >= 0.5).astype(int),
            "gated": (predict_proba(gated_m, gated_te) >= 0.5).astype(int),
            "mult": (predict_proba(mult_m, mult_te) >= 0.5).astype(int),
        }
        for k, pred in preds.items():
            sc = binary_scores(yte_fwd, pred)
            fwd_f1[k].append(sc["f1"])
            fwd_acc[k].append(sc["accuracy"])
        last_preds = preds
        last_proba_gate = predict_proba(gated_m, gated_te)
        last_g = gated_te[:, -1]
        if seed == 0:
            seed0_preds = {k: v.copy() for k, v in preds.items()}
            seed0_g = last_g.copy()
            seed0_gate_proba = last_proba_gate.copy()

    report["train_seconds"] = time.time() - t0
    report["forward"] = {
        k: {"f1": mean_ci(fwd_f1[k]), "acc": mean_ci(fwd_acc[k])} for k in fwd_f1
    }
    # Single-seed reporting (McNemar, tiers, overlays) uses seed 0, not the last loop seed.
    assert seed0_preds is not None and seed0_g is not None and seed0_gate_proba is not None
    last_preds = seed0_preds
    last_g = seed0_g
    last_proba_gate = seed0_gate_proba
    mixer_keys = ["mean", "gmu", "tfn", "mult", "gated", "concat"]
    mcn = {}
    for a, b in [
        ("gated", "gmu"),
        ("mean", "gmu"),
        ("tfn", "gmu"),
        ("concat", "gmu"),
        ("mult", "gmu"),
        ("gated", "mean"),
        ("gated", "tabular"),
        ("gated", "vision"),
        ("gated", "mult"),
    ]:
        mcn[f"{a}_vs_{b}"] = mcnemar(yte_fwd, last_preds[a], last_preds[b])
    mcn["min_p_mixer_vs_gmu"] = min(mcn[f"{k}_vs_gmu"]["p"] for k in mixer_keys if k != "gmu")
    report["forward_seed0_mcnemar"] = mcn
    report["mcnemar_seed"] = 0

    # 5-day horizon, gated vs tabular (single seed 0 already in last_preds — refit quickly)
    tab5 = fit_unimodal("tabular5", Xtr_tab, ytr_fwd5, 0)
    vis5 = fit_unimodal("vision5", Xtr_img, ytr_fwd5, 0)
    txt5 = fit_unimodal("text5", Xtr_txt, ytr_fwd5, 0)
    p_t = predict_proba(txt5, Xte_txt)
    p_v = predict_proba(vis5, Xte_img)
    p_b = predict_proba(tab5, Xte_tab)
    g5 = gated_features(np.stack([1 - p_t, p_t], axis=1), np.stack([1 - p_v, p_v], axis=1))
    g5m = fit_fusion(
        "g5",
        gated_features(
            np.stack(
                [1 - predict_proba(txt5, Xtr_txt), predict_proba(txt5, Xtr_txt)],
                axis=1,
            ),
            np.stack(
                [1 - predict_proba(vis5, Xtr_img), predict_proba(vis5, Xtr_img)],
                axis=1,
            ),
        ),
        ytr_fwd5,
        0,
    )
    pred5 = {
        "tabular": (p_b >= 0.5).astype(int),
        "vision": (p_v >= 0.5).astype(int),
        "gated": (predict_proba(g5m, g5) >= 0.5).astype(int),
    }
    report["forward5"] = {k: binary_scores(yte_fwd5, v) for k, v in pred5.items()}

    # --- Tiers ---
    tier_out = {}
    for tier, spec in TIER_SPECS.items():
        if spec["gate_tau"] > 0:
            mask = np.abs(last_g - 0.5) > spec["gate_tau"]
        else:
            mask = np.ones(len(test), dtype=bool)
        pred = last_preds["gated"].copy()
        # T1 abstains (falls back to majority) when the gate is indecisive
        if spec["gate_tau"] > 0:
            pred = np.where(mask, pred, majority)
        scores = binary_scores(yte_fwd, pred)
        covered_acc = (
            float((pred[mask] == yte_fwd[mask]).mean()) if mask.any() else float("nan")
        )
        bt = backtest(pred, te_rets)
        noises = []
        for s, vec in zip(test, Xte_txt):
            payload = delivered_payload(tier, s.relevant_pool, vec)
            noises.append(information_noise(s.raw_tokens, payload, s.relevant_pool))
        tier_out[tier] = {
            "coverage": float(mask.mean()),
            "latency_s": spec["latency_s"],
            "latency_is_declared_budget": True,
            "accuracy": scores["accuracy"],
            "accuracy_covered": covered_acc,
            "f1": scores["f1"],
            "mean_tokens": float(np.mean([n.n_delivered for n in noises])),
            "in_old": float(np.mean([n.in_old for n in noises])),
            "in_new": float(np.mean([n.in_new for n in noises])),
            "compression": float(np.mean([n.compression for n in noises])),
            "in_is_templated_filler": True,
            "backtest": bt,
        }
    report["tiers"] = tier_out
    report["backtest_buy_hold"] = backtest(np.ones_like(yte_fwd), te_rets)
    report["backtest_gated"] = backtest(last_preds["gated"], te_rets)
    report["backtest_tabular"] = backtest(last_preds["tabular"], te_rets)
    report["backtest_vision"] = backtest(last_preds["vision"], te_rets)
    report["backtest_gmu"] = backtest(last_preds["gmu"], te_rets)

    # Sentiment / technical proxy accuracies on the test window (annotator-free proxies)
    # Sentiment proxy: text polarity vs subsequent return sign
    sent_true = yte_fwd
    sent_pred = (Xte_txt[:, 0] - Xte_txt[:, 1] + Xte_txt[:, 2] - Xte_txt[:, 3] > 0).astype(int)
    report["sentiment_accuracy_proxy"] = float((sent_pred == sent_true).mean())
    kap_signed = Xte_txt[:, -1] if Xte_txt.shape[1] >= 16 else np.zeros(len(Xte_txt))
    report["sentiment_accuracy_kap"] = float(((kap_signed > 0).astype(int) == sent_true).mean())
    # Mean vision accuracy over seeds — do not quote last-seed vision as "technical proxy".
    report["technical_accuracy_proxy"] = float(report["forward"]["vision"]["acc"]["mean"])
    report["technical_accuracy_tabular"] = float(report["forward"]["tabular"]["acc"]["mean"])
    report["technical_accuracy_seed0_vision"] = float((last_preds["vision"] == yte_fwd).mean())
    report["gated_proba_mean"] = float(np.mean(last_proba_gate))
    report["gate_stats"] = {
        "mean": float(np.mean(last_g)),
        "std": float(np.std(last_g)),
        "frac_text_priority": float(np.mean(last_g > 0.5)),
        "mean_abs_dev": float(np.mean(np.abs(last_g - 0.5))),
    }

    out = results_dir / "public_benchmark.json"
    out.write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))
    print(f"\nWrote {out}")
    return report


if __name__ == "__main__":
    run()

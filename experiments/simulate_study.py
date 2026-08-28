"""Three-condition dry-run on the eight sealed scenarios — not human NASA-TLX.

Conditions use real public features and the fitted public mixers:
  raw       — sign of that *name's* session return vs that name's week gold
  unimodal  — text-only MLP, XU100 scenarios only (index mixer)
  vesta     — scalar gate, XU100 scenarios only

Constituent scenarios are abstains for the model arms: the public mixer is
index-level and must not be scored against a single-name week gold.

NASA-TLX / trust stay blank. study/responses.csv is not written.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys_path_src = str(ROOT / "src")

import sys

sys.path.insert(0, sys_path_src)

from vesta.charts import render_candles  # noqa: E402
from vesta.data import build_samples, chronological_split, download_public_market  # noqa: E402
from vesta.models import fit_unimodal, gated_features, fit_fusion, predict_proba  # noqa: E402


def _call(dec: str, proba: float, pid: str, cond: str, sc_id: str, elapsed: float) -> dict:
    return {
        "participant_id": pid,
        "scenario_id": sc_id,
        "condition": cond,
        "time_cap_min": "3",
        "decision": dec,
        "time_s": f"{elapsed:.2f}",
        "tlx_mental": "",
        "tlx_physical": "",
        "tlx_temporal": "",
        "tlx_performance": "",
        "tlx_effort": "",
        "tlx_frustration": "",
        "trust": "",
        "notes": "MODEL PILOT not a human. NASA-TLX left blank on purpose.",
        "proba": f"{proba:.4f}",
    }


def main() -> None:
    gold = json.loads((ROOT / "study" / "gold.json").read_text())
    scenarios = json.loads((ROOT / "study" / "scenarios.json").read_text())
    frames = download_public_market(ROOT / "data" / "cache")
    samples = build_samples(frames)
    train, val, test = chronological_split(samples)
    fit = train + val
    Xtr_txt = np.stack([s.text for s in fit])
    Xtr_img = np.stack([render_candles(s.ohlc, 24).ravel() for s in fit])
    ytr = np.array([s.y_fwd for s in fit])
    t0 = time.time()
    text_m = fit_unimodal("t", Xtr_txt, ytr, 0)
    vis_m = fit_unimodal("v", Xtr_img, ytr, 0)
    gm = fit_fusion(
        "g",
        gated_features(
            np.stack([1 - predict_proba(text_m, Xtr_txt), predict_proba(text_m, Xtr_txt)], 1),
            np.stack([1 - predict_proba(vis_m, Xtr_img), predict_proba(vis_m, Xtr_img)], 1),
        ),
        ytr,
        0,
    )
    fit_s = time.time() - t0
    date_to_s = {s.date.strftime("%Y-%m-%d"): s for s in test}
    rows = []
    for sc in scenarios:
        s = date_to_s.get(sc["date"])
        # raw: that name's session-return sign vs that name's sealed week gold
        raw_dec = "up" if float(sc["session_ret"]) > 0 else "down"
        rows.append(_call(raw_dec, abs(float(sc["session_ret"])), "vesta_raw_pilot", "raw", sc["scenario_id"], 0.01))
        index_only = sc.get("ticker") == "XU100.IS"
        if s is None or not index_only:
            # Public mixers are XU100-level. Do not score them on a constituent's week gold.
            rows.append(_call("abstain", 0.5, "vesta_text_pilot", "unimodal", sc["scenario_id"], fit_s))
            rows.append(_call("abstain", 0.5, "vesta_gated_pilot", "vesta", sc["scenario_id"], fit_s))
            continue
        t1 = time.time()
        xt = s.text.reshape(1, -1)
        xv = render_candles(s.ohlc, 24).ravel().reshape(1, -1)
        pt = float(predict_proba(text_m, xt)[0])
        pv = float(predict_proba(vis_m, xv)[0])
        ht = np.array([[1 - pt, pt]])
        hv = np.array([[1 - pv, pv]])
        pg = float(predict_proba(gm, gated_features(ht, hv))[0])
        infer_s = time.time() - t1
        rows.append(_call("up" if pt >= 0.5 else "down", pt, "vesta_text_pilot", "unimodal", sc["scenario_id"], infer_s))
        rows.append(_call("up" if pg >= 0.5 else "down", pg, "vesta_gated_pilot", "vesta", sc["scenario_id"], infer_s))

    out = ROOT / "study" / "pilot_model_responses.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {}
    for cond in ("raw", "unimodal", "vesta"):
        sub = [r for r in rows if r["condition"] == cond]
        n = hits = 0
        for r in sub:
            if r["decision"] not in {"up", "down"}:
                continue
            n += 1
            hits += int(r["decision"] == gold[r["scenario_id"]]["correct_week"])
        summary[cond] = {"n": n, "week_hit_rate": hits / n if n else None, "hits": hits}
    dest = ROOT / "results" / "study_model_pilot.json"
    dest.write_text(
        json.dumps(
            {
                "wrote": str(out),
                "conditions": summary,
                "human_responses_empty": True,
                "index_mixer_only_on_xu100": True,
                "note": (
                    "raw is scored on all 8 names vs that name's 5-day gold. "
                    "unimodal/vesta use the public XU100 mixer and only the XU100 scenarios."
                ),
            },
            indent=2,
        )
    )
    print(json.dumps({"wrote": str(out), "conditions": summary}, indent=2))
    print("Human responses.csv is still empty.")


if __name__ == "__main__":
    main()

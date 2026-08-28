"""Zero-shot DePlot (Pix2Struct) and MatCha-ChartQA on candlestick screenshots.

CPU-only, not fine-tuned. Default is a stride subsample of the chronological
test window so a laptop can finish; pass ``--every 1`` for all 308 days.
Caches are keyed by date and resume after interruption.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vesta.charts import render_screenshot  # noqa: E402
from vesta.data import build_samples, chronological_split, download_public_market  # noqa: E402
from vesta.metrics import binary_scores  # noqa: E402
from vesta.vlm_parse import parse_deplot, parse_matcha_pair  # noqa: E402


def generate(model, processor, image: Image.Image, prompt: str, max_new: int) -> str:
    import torch

    inputs = processor(images=image, text=prompt, return_tensors="pt")
    with torch.no_grad():
        pred = model.generate(**inputs, max_new_tokens=max_new)
    return processor.decode(pred[0], skip_special_tokens=True)


def load_pix2struct(model_id: str):
    import torch
    from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

    torch.set_num_threads(max(1, torch.get_num_threads()))
    print(f"Loading {model_id}", flush=True)
    processor = Pix2StructProcessor.from_pretrained(model_id)
    model = Pix2StructForConditionalGeneration.from_pretrained(model_id).eval()
    return model, processor


def _load_date_cache(path: Path, test_dates: list[str]) -> dict:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        return raw
    return {test_dates[i]: raw[i] for i in range(min(len(raw), len(test_dates)))}


def _score_calls(y: np.ndarray, calls: list[int | None], majority: int) -> dict:
    pred = np.array([majority if c is None else c for c in calls], dtype=int)
    parsed = int(sum(c is not None for c in calls))
    return {
        "parsed": parsed,
        "parse_rate": parsed / max(len(calls), 1),
        **binary_scores(y, pred),
        "raw_pred": pred.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=4, help="Keep every Nth test day (1 = full 308).")
    ap.add_argument("--models", default="deplot,matcha", help="Comma list: deplot,matcha")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()
    wanted = {m.strip() for m in args.models.split(",") if m.strip()}

    cache = ROOT / "data" / "cache" / "vlm"
    cache.mkdir(parents=True, exist_ok=True)
    frames = download_public_market(ROOT / "data" / "cache")
    samples = build_samples(frames)
    train, val, test = chronological_split(samples)
    majority = int(np.mean([s.y_fwd for s in train + val]) >= 0.5)
    all_dates = [s.date.strftime("%Y-%m-%d") for s in test]
    idxs = list(range(0, len(test), max(1, args.every)))
    subset = [test[i] for i in idxs]
    dates = [s.date.strftime("%Y-%m-%d") for s in subset]
    y = np.array([s.y_fwd for s in subset], dtype=int)
    trend = np.array([int(s.ohlc[-1, 3] >= s.ohlc[0, 3]) for s in subset], dtype=int)
    print(f"Subsample {len(subset)}/{len(test)} (every={args.every})", flush=True)

    report: dict = {
        "n_test_full": len(test),
        "n_eval": len(subset),
        "every": args.every,
        "dates": dates,
        "majority": majority,
        "fine_tuned": False,
        "device": "cpu",
        "note": (
            "Zero-shot CPU. DePlot = Pix2Struct chart-to-table. "
            "MatCha-ChartQA is asked for first and last y-values. "
            "Not a fine-tune. Window-trend is sign(last close − first close) of the same OHLC."
        ),
        "window_trend": {
            **binary_scores(y, trend),
            "agreement_with_label": float((trend == y).mean()),
        },
        "majority_on_subset": binary_scores(y, np.full_like(y, majority)),
    }

    dep_path = cache / "deplot_captions.json"
    dep_caps = _load_date_cache(dep_path, all_dates)
    mch_path = cache / "matcha_pairs.json"
    pairs = _load_date_cache(mch_path, all_dates)

    need_dep = [s for s in subset if s.date.strftime("%Y-%m-%d") not in dep_caps] if "deplot" in wanted else []
    need_mch = [s for s in subset if s.date.strftime("%Y-%m-%d") not in pairs] if "matcha" in wanted else []

    if not args.score_only and (need_dep or need_mch):
        if need_dep:
            model, proc = load_pix2struct("google/deplot")
            t0 = time.time()
            for i, s in enumerate(need_dep):
                key = s.date.strftime("%Y-%m-%d")
                img = render_screenshot(s.ohlc, title=f"XU100 {key}")
                if i == 0:
                    img.save(cache / "sample_screenshot.png")
                text = generate(
                    model,
                    proc,
                    img,
                    "Generate underlying data table of the figure below:",
                    80,
                )
                dep_caps[key] = text
                dep_path.write_text(json.dumps(dep_caps, ensure_ascii=False))
                if i % 5 == 0 or i + 1 == len(need_dep):
                    print(
                        f"  deplot {i+1}/{len(need_dep)} {(time.time()-t0):.0f}s {text[:70]!r}",
                        flush=True,
                    )
            del model, proc

        if need_mch:
            model, proc = load_pix2struct("google/matcha-chartqa")
            t0 = time.time()
            for i, s in enumerate(need_mch):
                key = s.date.strftime("%Y-%m-%d")
                img = render_screenshot(s.ohlc, title=f"XU100 {key}")
                a = generate(model, proc, img, "What is the first y-value?", 8)
                b = generate(model, proc, img, "What is the last y-value?", 8)
                pairs[key] = {"first": a, "last": b}
                mch_path.write_text(json.dumps(pairs, ensure_ascii=False))
                if i % 5 == 0 or i + 1 == len(need_mch):
                    print(
                        f"  matcha {i+1}/{len(need_mch)} {(time.time()-t0):.0f}s {a!r} -> {b!r}",
                        flush=True,
                    )
            del model, proc

    if "deplot" in wanted:
        caps = [dep_caps.get(d, "") for d in dates]
        calls = [parse_deplot(c) for c in caps]
        block = {
            "model_id": "google/deplot",
            "family": "Pix2Struct",
            **_score_calls(y, calls, majority),
            "vs_window_trend": binary_scores(
                trend,
                np.array([majority if c is None else c for c in calls], dtype=int),
            ),
            "examples": caps[:4],
            "n_cached": sum(1 for d in dates if d in dep_caps),
        }
        report["deplot_pix2struct"] = block
        print("deplot", {k: block[k] for k in ("parsed", "accuracy", "f1", "n_cached")})

    if "matcha" in wanted:
        raw = [pairs.get(d, {"first": "", "last": ""}) for d in dates]
        calls = [parse_matcha_pair(p.get("first", ""), p.get("last", "")) for p in raw]
        block = {
            "model_id": "google/matcha-chartqa",
            "family": "MatCha",
            **_score_calls(y, calls, majority),
            "vs_window_trend": binary_scores(
                trend,
                np.array([majority if c is None else c for c in calls], dtype=int),
            ),
            "examples": raw[:4],
            "n_cached": sum(1 for d in dates if d in pairs),
        }
        report["matcha_chartqa"] = block
        print("matcha", {k: block[k] for k in ("parsed", "accuracy", "f1", "n_cached")})

    dest = ROOT / "results" / "vlm_baselines.json"
    dest.write_text(json.dumps(report, indent=2, default=float))
    print("Wrote", dest, "n_eval", len(subset))


if __name__ == "__main__":
    main()

"""Frozen ImageNet ViT-B/16 CLS probe on matplotlib candlestick screenshots.

A sklearn MLP head predicts next-day direction. Embeddings are cached by date.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vesta.charts import render_screenshot  # noqa: E402
from vesta.data import build_samples, chronological_split, download_public_market  # noqa: E402
from vesta.metrics import binary_scores, mean_ci  # noqa: E402
from vesta.models import fit_unimodal, predict_proba  # noqa: E402

MODEL_ID = "google/vit-base-patch16-224"


def _embed_images(images: list[Image.Image], batch: int = 4) -> np.ndarray:
    import torch
    from transformers import ViTImageProcessor, ViTModel

    proc = ViTImageProcessor.from_pretrained(MODEL_ID)
    model = ViTModel.from_pretrained(MODEL_ID).eval()
    vecs = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            chunk = [im.convert("RGB").resize((224, 224)) for im in images[i : i + batch]]
            enc = proc(images=chunk, return_tensors="pt")
            out = model(**enc).last_hidden_state[:, 0]
            vecs.append(out.numpy())
            if (i // batch) % 25 == 0:
                print(f"  vit {i}/{len(images)}", flush=True)
    return np.concatenate(vecs, axis=0).astype(np.float32)


def main() -> None:
    cache = ROOT / "data" / "cache" / "vit"
    cache.mkdir(parents=True, exist_ok=True)
    npz_path = cache / "xu100_cls.npz"
    frames = download_public_market(ROOT / "data" / "cache")
    samples = build_samples(frames)
    train, val, test = chronological_split(samples)
    all_s = train + val + test
    dates = [s.date.strftime("%Y-%m-%d") for s in all_s]
    if npz_path.exists():
        packed = np.load(npz_path, allow_pickle=True)
        saved = packed["dates"].tolist()
        vecs = packed["vectors"]
        have = {d: i for i, d in enumerate(saved)}
        print(f"Cached ViT CLS {len(have)}", flush=True)
        missing_idx = [i for i, d in enumerate(dates) if d not in have]
        if missing_idx:
            print(f"Rendering {len(missing_idx)} new screenshots", flush=True)
            imgs = [render_screenshot(all_s[i].ohlc, title=f"XU100 {dates[i]}") for i in missing_idx]
            extra = _embed_images(imgs)
            date_arr = saved + [dates[i] for i in missing_idx]
            vecs = np.concatenate([vecs, extra], axis=0)
            np.savez_compressed(npz_path, dates=np.array(date_arr), vectors=vecs)
            have = {d: i for i, d in enumerate(date_arr)}
        date_to_vec = {d: vecs[have[d]] for d in have}
    else:
        print(f"Rendering {len(all_s)} screenshots for {MODEL_ID}", flush=True)
        imgs = [render_screenshot(s.ohlc, title=f"XU100 {d}") for s, d in zip(all_s, dates)]
        vecs = _embed_images(imgs)
        np.savez_compressed(npz_path, dates=np.array(dates), vectors=vecs)
        date_to_vec = {d: v for d, v in zip(dates, vecs)}

    dim = next(iter(date_to_vec.values())).shape[0]
    zero = np.zeros(dim, dtype=np.float32)

    def stack(ss):
        X = np.stack([date_to_vec.get(s.date.strftime("%Y-%m-%d"), zero) for s in ss])
        y = np.array([s.y_fwd for s in ss], dtype=int)
        return X, y

    Xtr, ytr = stack(train + val)
    Xte, yte = stack(test)
    f1s, accs = [], []
    for seed in range(5):
        m = fit_unimodal("vitb16", Xtr, ytr, seed)
        pred = (predict_proba(m, Xte) >= 0.5).astype(int)
        sc = binary_scores(yte, pred)
        f1s.append(sc["f1"])
        accs.append(sc["accuracy"])
    report = {
        "model": MODEL_ID,
        "frozen_backbone": True,
        "n_test": int(len(test)),
        "dim": int(dim),
        "acc": mean_ci(accs),
        "f1": mean_ci(f1s),
        "seed0": binary_scores(yte, (predict_proba(fit_unimodal("vitb16", Xtr, ytr, 0), Xte) >= 0.5).astype(int)),
        "note": "Frozen ImageNet ViT-B/16 CLS plus a sklearn MLP head.",
    }
    dest = ROOT / "results" / "vit_baseline.json"
    dest.write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()

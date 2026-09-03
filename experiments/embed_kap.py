"""Multilingual MiniLM embeddings of daily KAP list text."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vesta.data import build_samples, chronological_split, download_public_market  # noqa: E402
from vesta.kap import download_kap  # noqa: E402
from vesta.metrics import binary_scores, mean_ci  # noqa: E402
from vesta.models import fit_unimodal, predict_proba  # noqa: E402


MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _embed_texts(texts: list[str], model_id: str = MODEL_ID, batch: int = 32) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval()
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            enc = tok(chunk, padding=True, truncation=True, max_length=128, return_tensors="pt")
            out = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
            vecs.append(pooled.numpy())
    return np.concatenate(vecs, axis=0).astype(np.float32)


def main() -> None:
    cache = ROOT / "data" / "cache"
    out = ROOT / "data" / "vesta_public"
    rows = download_kap(cache)
    by_date: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        bit = (r.get("text") or r.get("subject") or "").strip()
        if bit:
            by_date[r["date"]].append(bit[:240])
    dates = sorted(by_date)
    docs = [" ".join(by_date[d][:8])[:800] or "no kap" for d in dates]
    print(f"Embedding {len(docs)} KAP days with {MODEL_ID}", flush=True)
    vecs = _embed_texts(docs)
    feat = {d: v.tolist() for d, v in zip(dates, vecs)}
    (out / "kap_daily_embeddings.json").write_text(json.dumps({"model": MODEL_ID, "dim": int(vecs.shape[1]), "n": len(feat)}))
    np.savez_compressed(out / "kap_daily_embeddings.npz", dates=np.array(dates), vectors=vecs)

    frames = download_public_market(cache)
    samples = build_samples(frames)
    train, val, test = chronological_split(samples)
    date_to_vec = {d: v for d, v in zip(dates, vecs)}
    zero = np.zeros(vecs.shape[1], dtype=np.float32)

    def stack(ss):
        X = np.stack([date_to_vec.get(s.date.strftime("%Y-%m-%d"), zero) for s in ss])
        y = np.array([s.y_fwd for s in ss], dtype=int)
        return X, y

    Xtr, ytr = stack(train + val)
    Xte, yte = stack(test)
    f1s, accs = [], []
    for seed in range(5):
        m = fit_unimodal("kap_minilm", Xtr, ytr, seed)
        pred = (predict_proba(m, Xte) >= 0.5).astype(int)
        sc = binary_scores(yte, pred)
        f1s.append(sc["f1"])
        accs.append(sc["accuracy"])
    report = {
        "model": MODEL_ID,
        "not_production_m3": True,
        "n_test": int(len(test)),
        "acc": mean_ci(accs),
        "f1": mean_ci(f1s),
        "seed0": binary_scores(yte, (predict_proba(fit_unimodal("kap_minilm", Xtr, ytr, 0), Xte) >= 0.5).astype(int)),
    }
    dest = ROOT / "results" / "kap_embed_benchmark.json"
    dest.write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()

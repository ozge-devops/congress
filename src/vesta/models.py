"""Fusion modules and sklearn-backed classifiers used in the public replication."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def closed_form_vol_rule(tabular: np.ndarray) -> np.ndarray:
    """Deterministic 2σ realized-vol rule. tabular columns: [..., vol20, ..., vol_mean, vol_std]."""
    v20, vmean, vstd = tabular[:, 3], tabular[:, 9], tabular[:, 10]
    return (v20 > (vmean + 2.0 * vstd)).astype(int)


def _mlp(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    max_iter=250,
                    random_state=seed,
                    early_stopping=True,
                    validation_fraction=0.1,
                ),
            ),
        ]
    )


def _logreg(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=400, random_state=seed, C=0.5),
            ),
        ]
    )


def gmu_features(h_t: np.ndarray, h_v: np.ndarray) -> np.ndarray:
    """Bimodal GMU-style features (Arevalo et al.): z * h_v + (1-z) * h_t.

    The gate is driven by the *difference* of unimodal scores so that it
    actually varies across samples (a mean of [p, 1-p] pairs is constant).
    """
    logit = 4.0 * (h_v[:, 1:2] - h_t[:, 1:2]) + 0.5 * (h_v[:, 1:2] + h_t[:, 1:2] - 1.0)
    z = 1.0 / (1.0 + np.exp(-logit))
    return z * h_v + (1.0 - z) * h_t


def tfn_features(h_t: np.ndarray, h_v: np.ndarray) -> np.ndarray:
    """Compact tensor-fusion features (Zadeh et al.): outer product of [h; 1] then flatten."""
    t = np.concatenate([h_t, np.ones((h_t.shape[0], 1))], axis=1)
    v = np.concatenate([h_v, np.ones((h_v.shape[0], 1))], axis=1)
    # (B, Dt+1, Dv+1) — keep small dims
    out = np.einsum("bi,bj->bij", t, v)
    return out.reshape(h_t.shape[0], -1)


def gated_features(h_t: np.ndarray, h_v: np.ndarray) -> np.ndarray:
    """Standard scalar sigmoid gate (Arevalo/Jiang-style; not a contribution).

    g > 0.5 privileges text; g < 0.5 privileges vision. The logit uses the
    unimodal score gap so the gate is sample-dependent.
    """
    logit = 4.0 * (h_t[:, 1:2] - h_v[:, 1:2])
    g = 1.0 / (1.0 + np.exp(-logit))
    fused = g * np.tanh(h_t) + (1.0 - g) * np.tanh(h_v)
    return np.concatenate([fused, g], axis=1)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(np.clip(z, -40, 40))
    return e / np.clip(e.sum(axis=axis, keepdims=True), 1e-9, None)


def mult_features(text: np.ndarray, vision: np.ndarray) -> np.ndarray:
    """CPU MulT-style directional cross-attention (Tsai et al. 2019), one layer.

    Not the pretrained CMU-MOSI transformer. Text is two tokens of 8 from the
    16-d brief; vision is nine 8×8 patches of the 24×24 candle. Random
    projections are frozen (seed 0) so the MLP on top is the only learner,
    matching how GMU/TFN features are consumed.
    """
    text = np.asarray(text, dtype=np.float32)
    vision = np.asarray(vision, dtype=np.float32)
    b = text.shape[0]
    d_txt = text.shape[1]
    # two tokens
    split = d_txt // 2
    t0, t1 = text[:, :split], text[:, split : split * 2]
    if t0.shape[1] < 8:
        t0 = np.pad(t0, ((0, 0), (0, 8 - t0.shape[1])))
        t1 = np.pad(t1, ((0, 0), (0, 8 - t1.shape[1])))
    t_tok = np.stack([t0[:, :8], t1[:, :8]], axis=1)
    vis = vision.reshape(b, 24, 24) if vision.shape[1] == 576 else vision.reshape(b, -1)
    if vis.ndim == 2:
        # fallback: chunk into 9
        pad = int(np.ceil(vis.shape[1] / 9) * 9)
        vis = np.pad(vis, ((0, 0), (0, pad - vis.shape[1])))
        v_tok = vis.reshape(b, 9, -1)
    else:
        patches = [
            vis[:, i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8].reshape(b, 64)
            for i in range(3)
            for j in range(3)
        ]
        v_tok = np.stack(patches, axis=1)
    rng = np.random.default_rng(0)
    wt = rng.normal(0.0, 1.0 / np.sqrt(8), size=(8, 16)).astype(np.float32)
    wv = rng.normal(0.0, 1.0 / np.sqrt(v_tok.shape[-1]), size=(v_tok.shape[-1], 16)).astype(np.float32)
    t = t_tok @ wt
    v = v_tok @ wv
    scale = 16.0 ** -0.5
    attn_tv = _softmax((t @ np.transpose(v, (0, 2, 1))) * scale, axis=-1)
    t2 = attn_tv @ v
    attn_vt = _softmax((v @ np.transpose(t, (0, 2, 1))) * scale, axis=-1)
    v2 = attn_vt @ t
    return np.concatenate([t2.mean(axis=1), v2.mean(axis=1), t.mean(axis=1), v.mean(axis=1)], axis=1)


@dataclass
class Fitted:
    name: str
    pipeline: Pipeline
    kind: str


def fit_unimodal(name: str, X: np.ndarray, y: np.ndarray, seed: int) -> Fitted:
    pipe = _mlp(seed) if X.shape[1] >= 16 else _logreg(seed)
    pipe.fit(X, y)
    return Fitted(name=name, pipeline=pipe, kind="unimodal")


def fit_fusion(name: str, X: np.ndarray, y: np.ndarray, seed: int) -> Fitted:
    pipe = _mlp(seed)
    pipe.fit(X, y)
    return Fitted(name=name, pipeline=pipe, kind="fusion")


def predict_proba(model: Fitted, X: np.ndarray) -> np.ndarray:
    if hasattr(model.pipeline, "predict_proba"):
        p = model.pipeline.predict_proba(X)
        if p.shape[1] == 2:
            return p[:, 1]
        return p.ravel()
    return model.pipeline.predict(X).astype(float)

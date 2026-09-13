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
    """Bimodal GMU-style features (Arevalo et al.). Gate from the unimodal score gap."""
    logit = 4.0 * (h_v[:, 1:2] - h_t[:, 1:2]) + 0.5 * (h_v[:, 1:2] + h_t[:, 1:2] - 1.0)
    z = 1.0 / (1.0 + np.exp(-logit))
    return z * h_v + (1.0 - z) * h_t


def tfn_features(h_t: np.ndarray, h_v: np.ndarray) -> np.ndarray:
    """Compact tensor-fusion features (Zadeh et al.): outer product of [h; 1] then flatten."""
    t = np.concatenate([h_t, np.ones((h_t.shape[0], 1))], axis=1)
    v = np.concatenate([h_v, np.ones((h_v.shape[0], 1))], axis=1)
    out = np.einsum("bi,bj->bij", t, v)
    return out.reshape(h_t.shape[0], -1)


def gated_features(h_t: np.ndarray, h_v: np.ndarray) -> np.ndarray:
    """Scalar sigmoid gate on the unimodal score gap (Arevalo / Jiang)."""
    logit = 4.0 * (h_t[:, 1:2] - h_v[:, 1:2])
    g = 1.0 / (1.0 + np.exp(-logit))
    fused = g * np.tanh(h_t) + (1.0 - g) * np.tanh(h_v)
    return np.concatenate([fused, g], axis=1)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(np.clip(z, -40, 40))
    return e / np.clip(e.sum(axis=axis, keepdims=True), 1e-9, None)


def mult_features(text: np.ndarray, vision: np.ndarray) -> np.ndarray:
    """One-layer directional cross-attention (Tsai et al. 2019 style) on CPU.

    Text is two tokens from the 16-d brief; vision is nine 8×8 patches of the
    24×24 candle. Random projections are frozen (seed 0); the MLP on top trains.
    """
    text = np.asarray(text, dtype=np.float32)
    vision = np.asarray(vision, dtype=np.float32)
    b = text.shape[0]
    d_txt = text.shape[1]
    split = d_txt // 2
    t0, t1 = text[:, :split], text[:, split : split * 2]
    if t0.shape[1] < 8:
        t0 = np.pad(t0, ((0, 0), (0, 8 - t0.shape[1])))
        t1 = np.pad(t1, ((0, 0), (0, 8 - t1.shape[1])))
    t_tok = np.stack([t0[:, :8], t1[:, :8]], axis=1)
    vis = vision.reshape(b, 24, 24) if vision.shape[1] == 576 else vision.reshape(b, -1)
    if vis.ndim == 2:
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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


@dataclass
class LearnedGMU:
    """Arevalo GMU with fitted W_v, W_t, W_z on raw unimodal vectors."""

    W_t: np.ndarray
    W_v: np.ndarray
    W_z: np.ndarray
    w: np.ndarray
    b: float
    mu_t: np.ndarray
    sd_t: np.ndarray
    mu_v: np.ndarray
    sd_v: np.ndarray

    def _std(self, x_t: np.ndarray, x_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xt = (x_t - self.mu_t) / self.sd_t
        xv = (x_v - self.mu_v) / self.sd_v
        return xt, xv

    def hidden(self, x_t: np.ndarray, x_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xt, xv = self._std(np.asarray(x_t, dtype=np.float64), np.asarray(x_v, dtype=np.float64))
        h_t = np.tanh(xt @ self.W_t)
        h_v = np.tanh(xv @ self.W_v)
        z = _sigmoid(np.concatenate([xt, xv], axis=1) @ self.W_z)
        h = z * h_v + (1.0 - z) * h_t
        return h, z

    def predict_proba(self, x_t: np.ndarray, x_v: np.ndarray) -> np.ndarray:
        h, _ = self.hidden(x_t, x_v)
        return _sigmoid(h @ self.w + self.b)


def fit_learned_gmu(
    x_t: np.ndarray,
    x_v: np.ndarray,
    y: np.ndarray,
    seed: int,
    hidden: int = 8,
    steps: int = 280,
    lr: float = 0.07,
    l2: float = 1e-4,
) -> LearnedGMU:
    """Minibatch GD for Eqs. (gmuh)-(gmu) plus a linear head on h."""
    rng = np.random.default_rng(seed)
    x_t = np.asarray(x_t, dtype=np.float64)
    x_v = np.asarray(x_v, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu_t, sd_t = x_t.mean(0), x_t.std(0)
    mu_v, sd_v = x_v.mean(0), x_v.std(0)
    sd_t = np.where(sd_t < 1e-6, 1.0, sd_t)
    sd_v = np.where(sd_v < 1e-6, 1.0, sd_v)
    xt = (x_t - mu_t) / sd_t
    xv = (x_v - mu_v) / sd_v
    n, dt = xt.shape
    dv = xv.shape[1]
    W_t = rng.normal(0.0, 1.0 / np.sqrt(dt), size=(dt, hidden))
    W_v = rng.normal(0.0, 1.0 / np.sqrt(dv), size=(dv, hidden))
    W_z = rng.normal(0.0, 1.0 / np.sqrt(dt + dv), size=(dt + dv, 1))
    w = rng.normal(0.0, 0.1, size=(hidden,))
    b = 0.0
    batch = min(128, n)
    for step in range(steps):
        idx = rng.choice(n, size=batch, replace=False)
        t, v, yy = xt[idx], xv[idx], y[idx]
        a_t = t @ W_t
        a_v = v @ W_v
        h_t = np.tanh(a_t)
        h_v = np.tanh(a_v)
        cat = np.concatenate([t, v], axis=1)
        z = _sigmoid(cat @ W_z)
        h = z * h_v + (1.0 - z) * h_t
        logit = h @ w + b
        p = _sigmoid(logit)
        dlogit = (p - yy)
        dw = h.T @ dlogit / batch + l2 * w
        db = float(dlogit.mean())
        dh = dlogit[:, None] * w
        dz = np.sum(dh * (h_v - h_t), axis=1, keepdims=True)
        dh_v = dh * z
        dh_t = dh * (1.0 - z)
        da_v = dh_v * (1.0 - h_v ** 2)
        da_t = dh_t * (1.0 - h_t ** 2)
        dWz = cat.T @ (dz * z * (1.0 - z)) / batch + l2 * W_z
        dWv = v.T @ da_v / batch + l2 * W_v
        dWt = t.T @ da_t / batch + l2 * W_t
        W_t -= lr * dWt
        W_v -= lr * dWv
        W_z -= lr * dWz
        w -= lr * dw
        b -= lr * db
        if step in (80, 160):
            lr *= 0.5
    return LearnedGMU(
        W_t=W_t, W_v=W_v, W_z=W_z, w=w, b=b,
        mu_t=mu_t, sd_t=sd_t, mu_v=mu_v, sd_v=sd_v,
    )


def predict_proba(model: Fitted, X: np.ndarray) -> np.ndarray:
    if hasattr(model.pipeline, "predict_proba"):
        p = model.pipeline.predict_proba(X)
        if p.shape[1] == 2:
            return p[:, 1]
        return p.ravel()
    return model.pipeline.predict(X).astype(float)

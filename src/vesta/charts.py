"""Render OHLC windows to compact grayscale chart tensors."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def render_candles(ohlc: np.ndarray, size: int = 32) -> np.ndarray:
    """Return a size×size float image in [0, 1] of a candlestick window."""
    img = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)
    highs = ohlc[:, 1]
    lows = ohlc[:, 2]
    lo, hi = float(lows.min()), float(highs.max())
    span = hi - lo if hi > lo else 1.0
    n = len(ohlc)
    w = max(1, size // n)
    for i, (o, h, l, c) in enumerate(ohlc):
        x = int(i * size / n)
        y_h = int(size - 1 - (h - lo) / span * (size - 1))
        y_l = int(size - 1 - (l - lo) / span * (size - 1))
        y_o = int(size - 1 - (o - lo) / span * (size - 1))
        y_c = int(size - 1 - (c - lo) / span * (size - 1))
        draw.line([(x, y_h), (x, y_l)], fill=40, width=1)
        top, bot = min(y_o, y_c), max(y_o, y_c)
        fill = 30 if c >= o else 180
        draw.rectangle([x, top, min(size - 1, x + max(1, w - 1)), max(top + 1, bot)], fill=fill)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def render_screenshot(ohlc: np.ndarray, title: str = "BIST") -> "Image.Image":
    """RGB candlestick screenshot for Pix2Struct / MatCha (not the 24×24 MLP tensor)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ohlc = np.asarray(ohlc, dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 2.8), dpi=80)
    n = len(ohlc)
    for i, (o, h, l, c) in enumerate(ohlc):
        color = "#15803d" if c >= o else "#b91c1c"
        ax.plot([i, i], [l, h], color=color, lw=0.8)
        ax.add_patch(
            plt.Rectangle(
                (i - 0.35, min(o, c)),
                0.7,
                max(abs(c - o), (h - l) * 0.02 + 1e-9),
                facecolor=color,
                edgecolor=color,
                lw=0,
            )
        )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("bar")
    ax.set_ylabel("price")
    ax.set_xlim(-1, n)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return Image.fromarray(buf)

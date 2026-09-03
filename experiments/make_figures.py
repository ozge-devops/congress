"""Paper figures: architecture, leakage diagnostic, fusion comparison, noise metric, tiers."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
R = json.loads((ROOT / "results" / "public_benchmark.json").read_text())

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.dpi": 160,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.15)
    ax.axis("off")
    arr = dict(arrowstyle="->", lw=1.1, color="#111827")

    def box(x, y, w, h, text, fc, ec="#1f2937"):
        p = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
            facecolor=fc, edgecolor=ec, linewidth=1.1, zorder=2,
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, zorder=3)

    box(0.4, 4.65, 5.2, 1.15, "KAP list teasers\n+ portfolio", "#dbeafe")
    box(6.4, 4.65, 5.2, 1.15, "40 OHLC bars\n(image + numbers)", "#fce7f3")
    box(0.4, 3.05, 5.2, 1.15, "DataClaw0\nretrieve, score, brief", "#93c5fd")
    box(6.4, 3.05, 5.2, 1.15, "VisualClaw  |  Tabular\n24x24 + 3x3 patches", "#f9a8d4")
    box(3.4, 1.5, 5.2, 1.1, "Fusion (standard mixer)\nGMU / TFN / gate / mean", "#e5e7eb")
    box(1.4, 0.15, 9.2, 1.0, "T1 flash   |   T3 context   |   T10 briefing\n1 / 3 / 10 min budget", "#fde68a")

    ax.annotate("", xy=(3.0, 4.2), xytext=(3.0, 4.65), arrowprops=arr)
    ax.annotate("", xy=(9.0, 4.2), xytext=(9.0, 4.65), arrowprops=arr)
    ax.annotate("", xy=(5.05, 2.6), xytext=(3.0, 3.05), arrowprops=arr)
    ax.annotate("", xy=(6.95, 2.6), xytext=(9.0, 3.05), arrowprops=arr)
    ax.annotate("", xy=(6.0, 1.15), xytext=(6.0, 1.5), arrowprops=arr)
    fig.savefig(FIG / "architecture.pdf")
    fig.savefig(FIG / "architecture.png")
    plt.close()


def fig_leakage():
    labels = ["Closed-form\nvol rule", "Tabular\nOHLCV", "Vision\n(chart image)"]
    acc = [
        100.0 * R["leak_closed_form"]["accuracy"],
        100.0 * R["leak_learned"]["tabular_acc"]["mean"],
        100.0 * R["leak_learned"]["vision_acc"]["mean"],
    ]
    f1 = [
        100.0 * R["leak_closed_form"]["f1"],
        100.0 * R["leak_learned"]["tabular"]["mean"],
        100.0 * R["leak_learned"]["vision"]["mean"],
    ]
    f1_err = [0.0, 100.0 * R["leak_learned"]["tabular"]["ci95"], 100.0 * R["leak_learned"]["vision"]["ci95"]]
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.bar(x - w / 2, acc, w, label="Accuracy", color="#1d4ed8")
    ax.bar(x + w / 2, f1, w, yerr=f1_err, capsize=3, label="Macro-F1", color="#db2777")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Score on leaked 2-sigma vol label (%)")
    ax.legend(frameon=False)
    fig.savefig(FIG / "leakage.pdf")
    fig.savefig(FIG / "leakage.png")
    plt.close()


def fig_forward():
    order = ["mean", "tfn", "vision", "gated", "concat", "gmu", "mult", "text", "tabular", "majority"]
    pretty = {
        "majority": "Majority",
        "text": "Text-only",
        "tabular": "Tabular OHLCV",
        "vision": "Vision-only",
        "mean": "Mean fusion",
        "gmu": "GMU",
        "tfn": "TFN",
        "mult": "MulT-style",
        "gated": "Scalar gate",
        "concat": "Concat",
    }
    f1 = [100.0 * R["forward"][k]["f1"]["mean"] for k in order]
    err = [100.0 * R["forward"][k]["f1"]["ci95"] for k in order]
    colors = ["#9ca3af" if k in {"majority", "text", "tabular", "vision"} else "#1d4ed8" for k in order]
    colors[-1] = "#0f766e"
    fig, ax = plt.subplots(figsize=(7.1, 3.3))
    x = np.arange(len(order))
    ax.bar(x, f1, yerr=err, capsize=3, color=colors, edgecolor="white")
    ax.axhline(100.0 * R["forward"]["majority"]["f1"]["mean"], color="#6b7280", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty[k] for k in order], rotation=25, ha="right")
    ax.set_ylabel("Macro-F1 on next-day direction (%)")
    ax.set_ylim(0, 70)
    ax.text(
        0.99, 0.96,
        "I-shaped whiskers: 95% CI over five seeds",
        transform=ax.transAxes, ha="right", va="top", fontsize=7,
    )
    fig.savefig(FIG / "forward_f1.pdf")
    fig.savefig(FIG / "forward_f1.png")
    plt.close()


def fig_noise_and_tiers():
    tiers = ["T1", "T3", "T10"]
    old = [100.0 * R["tiers"][t]["in_old"] for t in tiers]
    new = [100.0 * R["tiers"][t]["in_new"] for t in tiers]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    x = np.arange(len(tiers))
    w = 0.36
    axes[0].bar(x - w / 2, old, w, label="IN_old (vs raw feed)", color="#93c5fd")
    axes[0].bar(x + w / 2, new, w, label="IN_new (vs delivered)", color="#b45309")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["T1 flash", "T3 context", "T10 briefing"], fontsize=8)
    axes[0].set_ylabel("Information noise (%)")
    axes[0].legend(frameon=False, fontsize=7)

    lat = [R["tiers"][t]["latency_s"] for t in tiers]
    cov = [100.0 * R["tiers"][t]["coverage"] for t in tiers]
    ax2 = axes[1]
    ax2.bar(x - w / 2, lat, w, label="Latency (s)", color="#1d4ed8")
    ax2b = ax2.twinx()
    ax2b.bar(x + w / 2, cov, w, label="Coverage (%)", color="#f59e0b")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["T1 flash", "T3 context", "T10 briefing"], fontsize=8)
    ax2.set_ylabel("Latency (s)")
    ax2b.set_ylabel("Coverage (%)")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "tiers.pdf")
    fig.savefig(FIG / "tiers.png")
    plt.close()


def fig_dataclw0_loop():
    """Retrieve-score loop for DataClaw0; a second hop only on T3/T10."""
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.2)
    ax.axis("off")
    arr = dict(arrowstyle="->", lw=1.1, color="#111827")

    def box(x, y, w, h, text, fc):
        p = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.1",
            facecolor=fc, edgecolor="#1f2937", linewidth=1.05, zorder=2,
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, zorder=3)

    box(1.6, 6.85, 6.8, 0.95, "KAP filings + user portfolio P", "#dbeafe")
    box(1.6, 5.35, 6.8, 0.95, "1. Embed documents and P (M3 / MiniLM)", "#93c5fd")
    box(1.6, 3.85, 6.8, 0.95, "2. Score R(d): similarity to P + gold / USD/TRY", "#93c5fd")
    box(1.6, 2.35, 6.8, 0.95, "3. Keep top-k as the text brief", "#93c5fd")
    box(1.6, 0.35, 6.8, 0.95, "Text vector to fusion", "#e5e7eb")

    ax.annotate("", xy=(5.0, 6.3), xytext=(5.0, 6.85), arrowprops=arr)
    ax.annotate("", xy=(5.0, 4.8), xytext=(5.0, 5.35), arrowprops=arr)
    ax.annotate("", xy=(5.0, 3.3), xytext=(5.0, 3.85), arrowprops=arr)
    ax.annotate("", xy=(5.0, 1.3), xytext=(5.0, 2.35), arrowprops=arr)

    # T3/T10 hop returns to retrieve/score.
    ax.annotate(
        "",
        xy=(8.55, 5.82),
        xytext=(8.55, 2.82),
        arrowprops=dict(
            arrowstyle="->",
            lw=1.1,
            color="#1d4ed8",
            connectionstyle="arc3,rad=-0.35",
        ),
    )
    ax.text(9.15, 4.1, "T3 / T10:\none more hop", ha="left", va="center", fontsize=7, color="#1d4ed8")
    fig.savefig(FIG / "dataclw0_loop.pdf")
    fig.savefig(FIG / "dataclw0_loop.png")
    plt.close()


def fig_visualclaw():
    """Example XU100 window: screenshot, 24×24 tensor, 3×3 patch tokens."""
    import csv
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from vesta.charts import render_candles, render_screenshot

    dates, ohlc = [], []
    with (ROOT / "data" / "cache" / "bist100.csv").open() as f:
        for row in csv.DictReader(f):
            dates.append(row["Date"][:10])
            ohlc.append([float(row[k]) for k in ("open", "high", "low", "close")])
    ohlc = np.asarray(ohlc, dtype=np.float32)
    # Public chart is bars t-40 … t-1; pick the first test-window close (27 May 2025).
    try:
        t = dates.index("2025-05-27")
    except ValueError:
        t = len(ohlc) - 1
    win = ohlc[t - 40 : t]
    shot = render_screenshot(win, title=f"XU100  {dates[t - 40]} to {dates[t - 1]}")
    tiny = render_candles(win, size=24)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))
    axes[0].imshow(np.asarray(shot))
    axes[0].set_title("(a) RGB screenshot (DePlot / MatCha)")
    axes[0].axis("off")

    axes[1].imshow(tiny, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title("(b) 24×24 MLP tensor")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    axes[2].imshow(tiny, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    for k in (8, 16):
        axes[2].axhline(k - 0.5, color="#dc2626", lw=0.7)
        axes[2].axvline(k - 0.5, color="#dc2626", lw=0.7)
    axes[2].set_title("(c) Nine 8×8 tokens (MulT)")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    fig.tight_layout()
    fig.savefig(FIG / "visualclaw.pdf")
    fig.savefig(FIG / "visualclaw.png")
    plt.close()


if __name__ == "__main__":
    fig_architecture()
    fig_dataclw0_loop()
    fig_visualclaw()
    fig_leakage()
    fig_forward()
    fig_noise_and_tiers()
    print("wrote", list(FIG.iterdir()))

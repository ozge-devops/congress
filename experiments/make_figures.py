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
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#1f2937"):
        p = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
            facecolor=fc, edgecolor=ec, linewidth=1.1,
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

    box(0.3, 4.3, 3.4, 1.3, "KAP / news stream\n+ portfolio graph", "#dbeafe")
    box(4.3, 4.3, 3.4, 1.3, "OHLCV candlesticks\n(image + numbers)", "#fce7f3")
    box(0.3, 2.5, 3.4, 1.3, "DataClaw0\nagentic RAG (text)", "#93c5fd")
    box(4.3, 2.5, 3.4, 1.3, "VisualClaw  |  Tabular\n24×24 MLP     OHLCV numbers", "#f9a8d4")
    box(8.3, 2.5, 3.4, 1.3, "Fusion (not a claim)\nGMU / TFN / gate / mean", "#e5e7eb")
    box(2.3, 0.7, 7.4, 1.3, "Temporal Orchestration Layer\nT1 flash  ·  T3 context  ·  T10 briefing", "#fde68a")

    ax.annotate("", xy=(2.0, 3.85), xytext=(2.0, 4.3), arrowprops=dict(arrowstyle="->", lw=1.1))
    ax.annotate("", xy=(6.0, 3.85), xytext=(6.0, 4.3), arrowprops=dict(arrowstyle="->", lw=1.1))
    ax.annotate("", xy=(8.3, 3.15), xytext=(3.7, 3.15), arrowprops=dict(arrowstyle="->", lw=1.1))
    ax.annotate("", xy=(8.3, 3.15), xytext=(7.7, 3.15), arrowprops=dict(arrowstyle="->", lw=1.1))
    ax.annotate("", xy=(6.0, 2.0), xytext=(10.0, 2.5), arrowprops=dict(arrowstyle="->", lw=1.1))
    ax.set_title("VESTA: perception, off-the-shelf fusion, time-budgeted delivery")
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
    ax.set_ylabel("Score on leaked 2σ-vol label (%)")
    ax.legend(frameon=False)
    ax.set_title("The old headline task is recoverable from the numbers")
    fig.savefig(FIG / "leakage.pdf")
    fig.savefig(FIG / "leakage.png")
    plt.close()


def fig_forward():
    order = ["majority", "text", "tabular", "vision", "mean", "gmu", "tfn", "mult", "gated", "concat"]
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
    ax.set_title("Leakage-free task: fusion does not significantly beat GMU")
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
    axes[0].set_xticklabels(tiers)
    axes[0].set_ylabel("Information noise (%)")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].set_title("IN_old vs IN_new")

    lat = [R["tiers"][t]["latency_s"] for t in tiers]
    cov = [100.0 * R["tiers"][t]["coverage"] for t in tiers]
    ax2 = axes[1]
    ax2.bar(x - w / 2, lat, w, label="Latency (s)", color="#1d4ed8")
    ax2b = ax2.twinx()
    ax2b.bar(x + w / 2, cov, w, label="Coverage (%)", color="#f59e0b")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tiers)
    ax2.set_ylabel("Latency (s)")
    ax2b.set_ylabel("Coverage (%)")
    ax2.set_title("Declared latency and coverage")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "tiers.pdf")
    fig.savefig(FIG / "tiers.png")
    plt.close()


if __name__ == "__main__":
    fig_architecture()
    fig_leakage()
    fig_forward()
    fig_noise_and_tiers()
    print("wrote", list(FIG.iterdir()))

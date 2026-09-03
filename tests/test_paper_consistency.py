"""Lock paper.tex and docs to the numbers in results/*.json and label_stats.json.

Fails if a rounded table figure in the manuscript drifts from the JSON that
produced it. Does not invent human-study scores.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    return json.loads((ROOT / name).read_text())


def _pct(x: float, nd: int = 1) -> str:
    return f"{100.0 * x:.{nd}f}"


def test_label_stats_match_codebook_and_paper():
    stats = _load("data/vesta_public/label_stats.json")
    paper = (ROOT / "paper" / "vesta.tex").read_text()
    codebook = (ROOT / "docs" / "LABEL_CODEBOOK.md").read_text()
    kap = stats["kap_polarity"]

    def compact(s: str) -> str:
        return s.replace("{,}", "").replace(",", "").replace(" ", "")

    for blob in (paper, codebook):
        c = compact(blob)
        assert str(kap["bullish"]) in c
        assert str(kap["bearish"]) in c
        assert str(kap["neutral"]) in c
        assert str(stats["with_kap_body"]) in c
    assert stats["n_events"] == 37046
    assert "6198" not in codebook
    assert "4709" not in compact(codebook)


def test_table2_matches_public_benchmark():
    r = _load("results/public_benchmark.json")
    paper = (ROOT / "paper" / "vesta.tex").read_text()
    fwd = r["forward"]
    checks = {
        "majority": (52.3, 34.3),
        "text": (52.1, 47.3),
        "tabular": (50.6, 41.8),
        "vision": (52.9, 51.4),
        "mean": (53.0, 52.4),
        "gmu": (50.6, 50.6),
        "tfn": (52.1, 51.7),
        "mult": (50.2, 48.4),
        "gated": (51.8, 51.4),
        "concat": (52.8, 51.1),
    }
    for key, (acc, f1) in checks.items():
        got_acc = round(100.0 * fwd[key]["acc"]["mean"], 1)
        got_f1 = round(100.0 * fwd[key]["f1"]["mean"], 1)
        assert got_acc == acc, f"{key} acc {got_acc} != {acc}"
        assert got_f1 == f1, f"{key} f1 {got_f1} != {f1}"
        assert f"{acc}" in paper
        assert f"{f1}" in paper
    m3 = _load("results/kap_m3_benchmark.json")
    vit = _load("results/vit_baseline.json")
    assert "51.1" in paper  # public BGE-M3 acc
    assert round(100.0 * m3["acc"]["mean"], 1) == 51.1
    assert round(100.0 * vit["acc"]["mean"], 1) == 50.9
    assert "50.9" in paper
    assert r["mcnemar_seed"] == 0
    assert "forward_seed0_mcnemar" in r


def test_leak_and_vol_window_wording():
    paper = (ROOT / "paper" / "vesta.tex").read_text()
    r = _load("results/public_benchmark.json")
    assert round(100.0 * r["leak_closed_form"]["f1"], 1) == 100.0
    assert round(100.0 * r["leak_learned"]["tabular_acc"]["mean"], 1) == 96.8
    assert round(100.0 * r["leak_learned"]["vision"]["mean"], 1) == 58.2
    test_pos = 100.0 * r["leak_closed_form"]["pos"] / r["leak_closed_form"]["n"]
    all_pos = 100.0 * r["leak_rate_all"]
    assert abs(test_pos - 8.1) < 0.05
    assert abs(all_pos - 9.4) < 0.05
    assert "8.1" in paper
    assert "trailing 30-day" not in paper
    assert "20-day" in paper and "60" in paper
    assert "loshchilov2019adamw" not in paper
    assert "always fire at 49.7" not in paper
    assert "Same macro-F1 values as Table" not in paper


def test_technical_proxy_is_mean_vision_not_last_seed():
    r = _load("results/public_benchmark.json")
    paper = (ROOT / "paper" / "vesta.tex").read_text()
    vis = round(100.0 * r["forward"]["vision"]["acc"]["mean"], 1)
    tech = round(100.0 * r["technical_accuracy_proxy"], 1)
    assert vis == tech == 52.9
    # §metrics must not attribute 50.6% to the vision encoder
    metrics = paper.split("Sentiment / technical accuracy")[1].split("Information noise")[0]
    assert "50.6" not in metrics or "tabular" in metrics.lower()
    assert "52.9" in paper
    assert "tab:proxy" in paper
    assert "38.4" not in paper
    assert "H4" not in paper
    figsrc = (ROOT / "experiments" / "make_figures.py").read_text()
    assert "H2/H4" not in figsrc
    assert 'set_title("Denominator change (H2)")' not in figsrc
    assert "visual input's OHLCV" not in paper
    assert "I-shaped whiskers" not in figsrc
    assert "yerr" not in figsrc
    assert "I-shaped whiskers" not in paper
    assert "Text (macro+KAP)" in figsrc
    assert "t-interval" not in paper.lower().replace(" ", "")
    assert "1.96" in paper
    assert "eq:pos" not in paper
    assert "0.18" in paper
    assert "W_z" in paper
    assert "occupies that gap" not in paper
    assert "first-class" not in paper
    assert "\u2014" not in paper
    assert "\u2013" not in paper
    assert "70B" not in (ROOT / "results" / "vit_baseline.json").read_text()
    assert "public_m3_not_infina" not in paper
    assert "H4" not in (ROOT / "docs" / "REVIEWER_MAP.md").read_text()


def test_tiers_disclose_declared_latency_and_covered_acc():
    r = _load("results/public_benchmark.json")
    paper = (ROOT / "paper" / "vesta.tex").read_text()
    assert r["tiers"]["T1"]["latency_is_declared_budget"] is True
    assert "declared" in paper.lower() or "budget" in paper.lower()
    assert "filler" in paper.lower() or "template" in paper.lower()
    t1 = r["tiers"]["T1"]
    assert abs(100.0 * t1["coverage"] - 32.5) < 1.0


def test_study_pilot_does_not_score_index_mixer_on_names():
    r = _load("results/study_model_pilot.json")
    assert r["human_responses_empty"] is True
    assert r.get("index_mixer_only_on_xu100") is True
    # unimodal/vesta n is the two XU100 scenarios, not eight names
    assert r["conditions"]["unimodal"]["n"] <= 2
    assert r["conditions"]["raw"]["n"] == 8
    paper = (ROOT / "paper" / "vesta.tex").read_text()
    assert "NASA-TLX" in paper
    assert "12-participant" in paper or "twelve" in paper.lower()


def test_vlm_full_308_and_not_finetuned():
    v = _load("results/vlm_baselines.json")
    paper = (ROOT / "paper" / "vesta.tex").read_text()
    assert v["n_eval"] == 308
    assert v["fine_tuned"] is False
    assert round(100.0 * v["deplot_pix2struct"]["accuracy"], 1) == 52.9
    assert round(100.0 * v["matcha_chartqa"]["accuracy"], 1) == 52.3
    assert "52.9" in paper and "52.3" in paper
    assert "not fine-tuned" in paper.lower() or "no fine-tune" in paper.lower()


if __name__ == "__main__":
    for fn in [
        test_label_stats_match_codebook_and_paper,
        test_table2_matches_public_benchmark,
        test_leak_and_vol_window_wording,
        test_technical_proxy_is_mean_vision_not_last_seed,
        test_tiers_disclose_declared_latency_and_covered_acc,
        test_study_pilot_does_not_score_index_mixer_on_names,
        test_vlm_full_308_and_not_finetuned,
    ]:
        fn()
        print("ok", fn.__name__)
    print("paper consistency checks passed")

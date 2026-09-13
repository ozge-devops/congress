"""Lock vesta.tex table cells to results/*.json."""

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
    assert "learned_gmu.json" in (ROOT / "docs" / "REVIEWER_MAP.md").read_text()
    assert "c03627c" in (ROOT / "docs" / "REVIEWER_MAP.md").read_text()


def test_learned_gmu_matches_json_and_paper():
    r = _load("results/learned_gmu.json")
    paper = (ROOT / "paper" / "vesta.tex").read_text()
    acc = round(100.0 * r["acc"]["mean"], 1)
    f1 = round(100.0 * r["f1"]["mean"], 1)
    p = r["seed0"]["vs_score_gmu"]["p"]
    assert acc == 52.5
    assert f1 == 40.3
    assert abs(p - 0.29) < 0.01
    assert "52.5" in paper
    assert "40.3" in paper
    assert "0.29" in paper
    assert "Learned GMU" in paper
    assert "fourteen rows" in paper
    assert "Human NASA-TLX responses are not claimed" in paper
    assert "Evaluating Time-Budgeted Multimodal Briefing" in paper
    assert "four protocol facts" in paper
    # CoMeSySo 2026 CFP: Introduction -- Methods -- Results -- Discussions
    assert "\\section{Introduction}" in paper
    assert "\\section{Methods}" in paper
    assert "\\section{Results}" in paper
    assert "\\section{Discussions}" in paper
    assert "\\section{Related Work}" not in paper
    assert "\\section{Experiments}" not in paper
    assert "\\section{Conclusion}" not in paper
    assert "\\section{System}" not in paper
    assert "Econometrics" in paper
    assert "Computational intelligence" in paper
    assert "CoMeSySo" in paper
    assert "no score-space mixer" in paper
    assert "p>0.46" not in paper
    assert "already in Table" not in paper
    assert "several seconds of wall-clock" not in paper
    cite = (ROOT / "CITATION.cff").read_text()
    assert "Evaluating Time-Budgeted Multimodal Briefing" in cite
    assert "Time-Budgeted Multimodal Agents" not in cite
    study = (ROOT / "study" / "README.md").read_text()
    assert "§6" not in study
    assert "sec:user" in study
    assert not (ROOT / "src" / "vesta" / "init.py").exists()
    assert not (ROOT / "src" / "vesta" / "vlm.parse.py").exists()
    readme = (ROOT / "README.md").read_text()
    assert "github.com/ozge-devops/congress" in readme
    assert "c03627c" in readme
    assert "LNCS draft" in readme
    assert "matches this PDF" in readme
    assert "does not match this PDF" not in readme
    assert "c03627c" not in paper
    assert "Time-Budgeted Multimodal Agents" not in paper
    assert "Agentic RAG" not in paper
    assert "Word is [paper/vesta.docx]" not in readme
    assert "templated analogue" in paper or "templated \\texttt{hist" in paper
    assert "one retrieval step over similar past days" not in paper
    assert "Dir.\\ (\\%)" in paper
    assert "54.9" in paper
    assert "4/8" in paper
    assert "0/2" in paper
    assert "1/2" in paper
    assert "skip that loop" in paper
    assert "not executed" in paper
    assert "by construction" in paper
    assert "no human annotators" in paper
    assert "not Tsai" in paper or "not MulT" in paper
    assert "fig:claw" not in paper
    assert "eq:rel" not in paper
    assert "chain depth" not in paper.lower()
    assert "MulT-style" not in paper
    assert "DataClaw0 retrieves portfolio-conditioned" not in readme
    assert "learned_gmu" in (ROOT / "src" / "vesta" / "engine.py").read_text()


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


def _pm(mean: float, ci: float, nd: int = 1) -> str:
    return f"{100.0 * mean:.{nd}f}\\pm{100.0 * ci:.{nd}f}"


def _row(*cells: str) -> str:
    return " & ".join(cells)


def test_vlm_full_308_and_not_finetuned():
    v = _load("results/vlm_baselines.json")
    paper = (ROOT / "paper" / "vesta.tex").read_text()
    assert v["n_eval"] == 308
    assert v["fine_tuned"] is False
    assert round(100.0 * v["deplot_pix2struct"]["accuracy"], 1) == 52.9
    assert round(100.0 * v["matcha_chartqa"]["accuracy"], 1) == 52.3
    assert "52.9" in paper and "52.3" in paper
    assert "not fine-tuned" in paper.lower() or "no fine-tune" in paper.lower()


def test_every_table_cell_matches_json():
    """Every numeric cell in the camera-ready tables is the rounded JSON value."""
    paper = (ROOT / "paper" / "vesta.tex").read_text()
    r = _load("results/public_benchmark.json")
    lg = _load("results/learned_gmu.json")
    m3 = _load("results/kap_m3_benchmark.json")
    mini = _load("results/kap_embed_benchmark.json")
    vit = _load("results/vit_baseline.json")
    vlm = _load("results/vlm_baselines.json")
    inv = _load("data/vesta_public/kap_inventory.json")
    agr = _load("results/agreement.json")

    fwd = r["forward"]
    table_fwd = {
        "Mean fusion": (fwd["mean"]["acc"], fwd["mean"]["f1"]),
        "TFN": (fwd["tfn"]["acc"], fwd["tfn"]["f1"]),
        "Vision-only ($24\\times24$ MLP)": (fwd["vision"]["acc"], fwd["vision"]["f1"]),
        "Scalar gate (score-space)": (fwd["gated"]["acc"], fwd["gated"]["f1"]),
        "Concatenation": (fwd["concat"]["acc"], fwd["concat"]["f1"]),
        "Text-only (MiniLM KAP)": (mini["acc"], mini["f1"]),
        "GMU (score-space)": (fwd["gmu"]["acc"], fwd["gmu"]["f1"]),
        "Vision-only (frozen ViT-B/16)": (vit["acc"], vit["f1"]),
        "Text-only (public BGE-M3)": (m3["acc"], m3["f1"]),
        "1-layer attention (not MulT)": (fwd["mult"]["acc"], fwd["mult"]["f1"]),
        "Text-only (macro + KAP)": (fwd["text"]["acc"], fwd["text"]["f1"]),
        "Tabular OHLCV": (fwd["tabular"]["acc"], fwd["tabular"]["f1"]),
        "Learned GMU ($W_v,W_t,W_z$)": (lg["acc"], lg["f1"]),
        "Majority class": (fwd["majority"]["acc"], fwd["majority"]["f1"]),
    }
    f1s = []
    for name, (acc, f1) in table_fwd.items():
        cell = (
            f"{name} & ${_pm(acc['mean'], acc['ci95'])}$ & "
            f"${_pm(f1['mean'], f1['ci95'])}$"
        )
        assert cell in paper, cell
        f1s.append(round(100.0 * f1["mean"], 1))
    assert f1s == sorted(f1s, reverse=True)

    leak = r["leak_learned"]
    assert _row("Closed-form OHLCV rule", "$100.0$", "$100.0$") in paper
    assert (
        _row(
            "Tabular MLP (same numbers)",
            f"${_pm(leak['tabular_acc']['mean'], leak['tabular_acc']['ci95'])}$",
            f"${_pm(leak['tabular']['mean'], leak['tabular']['ci95'])}$",
        )
        in paper
    )
    assert (
        _row(
            "Vision MLP (chart image)",
            f"${_pm(leak['vision_acc']['mean'], leak['vision_acc']['ci95'])}$",
            f"${_pm(leak['vision']['mean'], leak['vision']['ci95'])}$",
        )
        in paper
    )

    vis_acc, vis_ci = fwd["vision"]["acc"]["mean"], fwd["vision"]["acc"]["ci95"]
    tab_acc, tab_ci = fwd["tabular"]["acc"]["mean"], fwd["tabular"]["acc"]["ci95"]
    assert (
        _row(
            "Macro flags (USD/TRY, gold, vol, shock)",
            f"${_pct(r['sentiment_accuracy_proxy'])}$",
        )
        in paper
    )
    assert (
        _row(
            "KAP polarity (lexicon on public list text)",
            f"${_pct(r['sentiment_accuracy_kap'])}$",
        )
        in paper
    )
    assert _row("Vision $24\\times24$ MLP", f"${_pm(vis_acc, vis_ci)}$") in paper
    assert _row("Tabular OHLCV", f"${_pm(tab_acc, tab_ci)}$") in paper

    def vlm_row(name: str, acc: float, f1: float, trend: str) -> str:
        return _row(name, f"${_pct(acc)}$", f"${_pct(f1)}$", trend)

    assert vlm_row("Majority", vlm["majority_on_subset"]["accuracy"], vlm["majority_on_subset"]["f1"], "n/a") in paper
    assert (
        vlm_row(
            "Numeric window trend",
            vlm["window_trend"]["accuracy"],
            vlm["window_trend"]["f1"],
            "$100.0$",
        )
        in paper
    )
    assert (
        vlm_row(
            "DePlot (Pix2Struct)",
            vlm["deplot_pix2struct"]["accuracy"],
            vlm["deplot_pix2struct"]["f1"],
            f"${_pct(vlm['deplot_pix2struct']['vs_window_trend']['accuracy'])}$",
        )
        in paper
    )
    assert (
        vlm_row(
            "MatCha-ChartQA",
            vlm["matcha_chartqa"]["accuracy"],
            vlm["matcha_chartqa"]["f1"],
            f"${_pct(vlm['matcha_chartqa']['vs_window_trend']['accuracy'])}$",
        )
        in paper
    )
    assert "307/308" in paper
    assert "166/308" in paper

    def tier_row(name: str, spec: dict) -> str:
        return _row(
            name,
            f"{100.0 * spec['coverage']:.1f}",
            f"{spec['latency_s']:.1f}",
            f"{100.0 * spec['accuracy_covered']:.1f}",
            f"{spec['mean_tokens']:.0f}",
            f"{100.0 * spec['in_old']:.1f}\\%",
            f"{100.0 * spec['in_new']:.1f}\\%",
        )

    assert tier_row("T1 flash", r["tiers"]["T1"]) in paper
    assert tier_row("T3 context", r["tiers"]["T3"]) in paper
    assert tier_row("T10 briefing", r["tiers"]["T10"]) in paper
    assert "55.2\\%" in paper

    def bt_row(name: str, hit: float, sharpe: float, ret: float, dd: float) -> str:
        sign = "+" if ret >= 0 else "-"
        return _row(
            name,
            f"{100.0 * hit:.1f}",
            f"${sharpe:.2f}$",
            f"${sign}{abs(100.0 * ret):.1f}\\%$",
            f"${100.0 * dd:.1f}\\%$",
        )

    bh = r["backtest_buy_hold"]
    assert bt_row("Buy-and-hold", bh["hit_rate"], bh["sharpe_bh"], bh["total_bh"], bh["max_dd"]) in paper
    vis = r["backtest_vision"]
    assert bt_row("Vision-only overlay", vis["hit_rate"], vis["sharpe"], vis["total_net"], vis["max_dd"]) in paper
    gat = r["backtest_gated"]
    assert bt_row("Scalar-gate overlay", gat["hit_rate"], gat["sharpe"], gat["total_net"], gat["max_dd"]) in paper
    tab = r["backtest_tabular"]
    assert bt_row("Tabular overlay", tab["hit_rate"], tab["sharpe"], tab["total_net"], tab["max_dd"]) in paper
    gmu = r["backtest_gmu"]
    assert bt_row("GMU overlay", gmu["hit_rate"], gmu["sharpe"], gmu["total_net"], gmu["max_dd"]) in paper

    f5 = r["forward5"]
    assert "59.1" in paper and "39.9" in paper
    assert abs(100.0 * f5["tabular"]["accuracy"] - 59.1) < 0.05
    assert abs(100.0 * f5["tabular"]["f1"] - 39.9) < 0.05
    assert "51.9" in paper and "49.0" in paper
    assert "50.3" in paper and "49.3" in paper
    assert abs(round(100.0 * f5["vision"]["accuracy"], 1) - 51.9) < 0.05
    assert abs(round(100.0 * f5["vision"]["f1"], 1) - 49.0) < 0.05
    assert abs(round(100.0 * f5["gated"]["accuracy"], 1) - 50.3) < 0.05
    assert abs(round(100.0 * f5["gated"]["f1"], 1) - 49.3) < 0.05

    assert abs(r["forward_seed0_mcnemar"]["min_p_mixer_vs_gmu"] - 0.46) < 0.01
    assert abs(lg["seed0"]["vs_score_gmu"]["p"] - 0.29) < 0.01
    assert abs(agr["kap_10k"]["fleiss_abc"]["kappa"] - 0.50) < 0.005
    assert abs(agr["chart_10k"]["a_vs_b"]["kappa"] - 0.52) < 0.005
    assert inv["n_filings"] == 39890
    assert "39{,}890" in paper
    assert "2 January 2018" in paper
    assert "27 August 2026" in paper


def test_kap_example_rows_match_corpus():
    import pandas as pd

    paper = (ROOT / "paper" / "vesta.tex").read_text()
    df = pd.read_parquet(ROOT / "data" / "vesta_public" / "events.parquet")
    df["date_s"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    rows = [
        ("2018-06-14", "BIMAS.IS", "bullish", "Payların Geri Alınmasına İlişkin Bildirim"),
        ("2018-06-19", "ASELS.IS", "bullish", "Temettü Ödemesi"),
        ("2018-05-25", "YKBNK.IS", "bearish", "idari para cezasının iptaline"),
        ("2020-08-11", "FROTO.IS", "bearish", "Rekabet Kurulu Soruşturma Kararı"),
    ]
    for date, ticker, polar, snippet in rows:
        hit = df[(df["date_s"] == date) & (df["ticker"] == ticker)]
        assert len(hit) == 1, (date, ticker)
        rec = hit.iloc[0]
        assert rec["kap_polarity"] == polar
        assert snippet in str(rec["kap_text"])
        assert date in paper
        assert ticker.replace(".IS", "") in paper
        assert snippet in paper
        assert polar in paper
    bimas = df[(df["date_s"] == "2018-06-14") & (df["ticker"] == "BIMAS.IS")].iloc[0]
    assert abs(float(bimas["session_ret"]) - 0.0633) < 5e-4
    assert "+6.33" in paper
    assert "+1.65" in paper
    assert "+0.55" in paper
    assert "33.8" in paper
    assert "+0.12" not in paper
    assert "ticker-day record" in paper
    assert "kap\\_polarity" in paper or "kap_polarity" in paper
    stored = str(bimas["brief"])
    assert stored == (
        "2018-06-14 BIMAS.IS. Session return +6.33%. USD/TRY +1.65%, gold +0.55%. "
        "Ann. realized vol 33.8%, RSI 48. Brief polarity bullish; chart cue none. "
        "Payların Geri Alınmasına İlişkin Bildirim. 13 Haziran 2018 Tarihli Pay Geri Alım İşlemleri"
    )
    assert stored in paper.replace("\\%", "%")
    assert stored.split("Brief polarity")[1].split(";")[0].strip() == bimas["text_polarity"]
    assert "İşlemleri''. That" in paper
    assert "İşlemleri.''" not in paper
    assert "1-layer attn" not in paper


def test_stored_briefs_match_text_polarity():
    import pandas as pd

    df = pd.read_parquet(ROOT / "data" / "vesta_public" / "events.parquet", columns=["brief", "text_polarity"])
    got = df["brief"].str.extract(r"Brief polarity (\w+)", expand=False)
    mism = int((got != df["text_polarity"]).sum())
    assert mism == 0, f"{mism} briefs still disagree with text_polarity"


if __name__ == "__main__":
    for fn in [
        test_label_stats_match_codebook_and_paper,
        test_table2_matches_public_benchmark,
        test_leak_and_vol_window_wording,
        test_technical_proxy_is_mean_vision_not_last_seed,
        test_tiers_disclose_declared_latency_and_covered_acc,
        test_study_pilot_does_not_score_index_mixer_on_names,
        test_vlm_full_308_and_not_finetuned,
        test_learned_gmu_matches_json_and_paper,
        test_every_table_cell_matches_json,
        test_kap_example_rows_match_corpus,
        test_stored_briefs_match_text_polarity,
    ]:
        fn()
        print("ok", fn.__name__)
    print("paper consistency checks passed")

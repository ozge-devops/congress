"""Gradio demo for the public VESTA briefing path (Hugging Face Space / local)."""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr
from PIL import Image

from vesta.engine import VestaEngine
from vesta.tiers import MIXERS, TIER_SPECS

_ENGINE: VestaEngine | None = None
_LOAD_ERROR: str | None = None


def get_engine() -> VestaEngine:
    global _ENGINE, _LOAD_ERROR
    if _ENGINE is not None:
        return _ENGINE
    engine = VestaEngine(root=ROOT)
    try:
        engine.load()
    except Exception as exc:
        _LOAD_ERROR = str(exc)
        raise
    _ENGINE = engine
    return engine


def _dates() -> list[str]:
    try:
        engine = get_engine()
    except Exception:
        return ["2025-05-27"]
    test = engine.dates("test")
    extra = ["2018-06-14", "2023-01-02", "2025-05-27"]
    ordered = [d for d in extra if d in engine.by_date]
    for d in test:
        if d not in ordered:
            ordered.append(d)
    return ordered or ["2025-05-27"]


def _tickers() -> list[str]:
    try:
        return get_engine().tickers()
    except Exception:
        return ["XU100", "THYAO.IS", "BIMAS.IS"]


def _mixers() -> list[str]:
    return [m for m in MIXERS if m in ("gated", "mean", "gmu", "vision", "text", "tabular", "concat", "tfn", "mult")]


def _fmt_call(payload: dict) -> str:
    if payload.get("silent"):
        return "T1 silent  |g-0.5| ≤ 0.18"
    call = payload.get("call") or "—"
    p = payload.get("p_up")
    return f"{call.upper()}   p(up)={p:.3f}" if isinstance(p, float) else str(call)


def _fmt_hop(hop: dict | None) -> str:
    if not hop:
        return "Hop is attached on T3/T10 only. It does not enter the 16-d mixer."
    prior = hop.get("prior") or {}
    peer = hop.get("peer") or {}
    cosine = prior.get("cosine")
    cosine_s = f"{cosine:.3f}" if isinstance(cosine, float) else "n/a"
    lines = [
        f"prior day: {prior.get('hop_date') or 'none'}",
        f"cosine: {cosine_s}",
        f"peer: {peer.get('ticker') or 'none'}  {peer.get('date') or ''}",
        "alpha=1, no portfolio, beta unused, enters_mixer=false",
    ]
    teaser = (peer.get("kap_teaser") or "").strip()
    if teaser:
        lines.append(teaser[:400])
    return "\n".join(lines)


def _fmt_mixers(mixers: dict) -> str:
    rows = ["mixer\tp(up)\tcall"]
    for name, row in mixers.items():
        rows.append(f"{name}\t{row.get('p_up')}\t{row.get('call')}")
    return "\n".join(rows)


def run_brief(date: str, ticker: str, tier: str, mixer: str):
    try:
        engine = get_engine()
    except Exception as exc:
        err = f"Engine failed to load: {exc}"
        return err, err, err, None, err
    date = (date or "").strip()
    ticker = (ticker or "XU100").strip() or "XU100"
    try:
        payload = engine.brief(date, tier=tier, mixer=mixer, ticker=ticker)
        png = engine.vision_png(date, kind="screenshot")
        chart = Image.open(io.BytesIO(png))
    except KeyError:
        msg = (
            f"No XU100 sample on {date}. "
            "Pick a session from the date list (panel 24 May 2018 to 19 August 2026)."
        )
        return msg, "no call", "no hop", None, msg
    except Exception as exc:
        msg = f"Brief failed: {exc}"
        return msg, "error", "error", None, msg

    if payload.get("silent"):
        brief = (
            f"T1 stays silent on {date}: |g-0.5|={abs((payload.get('gate_g') or 0.5) - 0.5):.3f} ≤ 0.18. "
            "Raise the tier to T3 or T10 to read the brief."
        )
    else:
        brief = payload.get("brief") or "Empty brief (no event row for that ticker-day)."

    header = [
        _fmt_call(payload),
        f"split {payload.get('split')}   gate g={payload.get('gate_g')} keeps {payload.get('gate_keeps')}",
        f"call_scope={payload.get('call_scope')}   brief_scope={payload.get('brief_scope')}",
        f"chart cue: {payload.get('chart_signal')}   polarity: {payload.get('text_polarity')}",
        f"declared latency {payload.get('declared_latency_s')} s",
    ]
    realized = payload.get("realized") or {}
    if realized:
        header.append(
            f"realized next-day sign {realized.get('y_direction_1d')}  "
            f"ret {realized.get('next_ret_1d')}"
        )
    return brief, "\n".join(header), _fmt_hop(payload.get("hop")), chart, _fmt_mixers(payload.get("mixers") or {})


def build_demo() -> gr.Blocks:
    dates = _dates()
    tickers = _tickers()
    theme = gr.themes.Soft(primary_hue="stone", secondary_hue="amber")
    with gr.Blocks(title="VESTA briefing", theme=theme) as demo:
        gr.Markdown(
            "# VESTA\n"
            "Time-budgeted multimodal briefing for BIST retail users. "
            "16-d KAP bag, VisualClaw, seed-0 mixers, T1/T3/T10."
        )
        with gr.Row():
            with gr.Column(scale=1):
                date = gr.Dropdown(choices=dates, value="2025-05-27", label="Session date", allow_custom_value=True)
                ticker = gr.Dropdown(choices=tickers, value="XU100", label="Ticker", allow_custom_value=True)
                tier = gr.Radio(choices=list(TIER_SPECS), value="T3", label="Time budget")
                mixer = gr.Dropdown(choices=_mixers(), value="gated", label="Mixer")
                go = gr.Button("Build brief", variant="primary")
                gr.Markdown(
                    "Examples: `2025-05-27` XU100 T3 (headline test day); "
                    "`2018-06-14` BIMAS.IS T3 (paper Table 1 teaser); "
                    "`2023-01-02` XU100 (second holdout). "
                    "T1 may stay silent."
                )
            with gr.Column(scale=2):
                call = gr.Textbox(label="Call", lines=5)
                brief = gr.Textbox(label="Brief", lines=8)
                hop = gr.Textbox(label="M3 hop", lines=6)
                chart = gr.Image(label="40-bar chart (t-40 … t-1)", type="pil")
                mixers = gr.Textbox(label="All mixers", lines=10)
        go.click(run_brief, inputs=[date, ticker, tier, mixer], outputs=[brief, call, hop, chart, mixers])
        demo.load(run_brief, inputs=[date, ticker, tier, mixer], outputs=[brief, call, hop, chart, mixers])
        gr.Markdown(
            "Source: [github.com/ozge-devops/congress](https://github.com/ozge-devops/congress)."
        )
    return demo


demo = build_demo()


if __name__ == "__main__":
    from gradio import networking

    networking.url_ok = lambda _url: True
    port = int(os.environ.get("GRADIO_SERVER_PORT") or os.environ.get("PORT") or 7865)
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
        share=False,
        inbrowser=False,
    )

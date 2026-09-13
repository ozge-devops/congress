"""T1 / T3 / T10 payload templates. Shared by the paper tables and the HTTP API."""

from __future__ import annotations

import numpy as np

TIER_SPECS = {
    "T1": {
        "max_tokens": 48,
        "gate_tau": 0.18,
        "latency_s": 0.9,
        "minutes": 1,
        "label": "flash",
        "template": ["session_ohlc", "bist100_level", "vol_spike", "price_shock"],
    },
    "T3": {
        "max_tokens": 140,
        "gate_tau": 0.0,
        "latency_s": 2.1,
        "minutes": 3,
        "label": "context",
        "template": [
            "session_ohlc",
            "bist100_level",
            "vol_spike",
            "price_shock",
            "usdtry_up",
            "usdtry_down",
            "gold_up",
            "gold_down",
            "hist_analog",
            "sector_note",
        ],
    },
    "T10": {
        "max_tokens": 420,
        "gate_tau": 0.0,
        "latency_s": 3.4,
        "minutes": 10,
        "label": "briefing",
        "template": [
            "session_ohlc",
            "bist100_level",
            "vol_spike",
            "price_shock",
            "usdtry_up",
            "usdtry_down",
            "gold_up",
            "gold_down",
            "hist_analog",
            "sector_note",
            "macro_correlator",
            "hedge_sketch",
            "disclaimer",
            "kap_digest",
        ],
    },
}

FILLER = [f"filler_{i}" for i in range(80)]

TOKEN_HELP = {
    "session_ohlc": "Session OHLC flags for day t",
    "bist100_level": "Index level cue",
    "vol_spike": "Realized-vol spike vs trailing mean",
    "price_shock": "Session |return| above 2%",
    "usdtry_up": "USD/TRY jumped",
    "usdtry_down": "USD/TRY dropped",
    "gold_up": "Gold jumped",
    "gold_down": "Gold dropped",
    "hist_analog": "Retrieved prior KAP day (M3 cosine, α=1, β unused; no portfolio)",
    "sector_note": "Sector context for the ticker",
    "macro_correlator": "Macro explanation sketch (T10)",
    "hedge_sketch": "Hedge sketch (T10)",
    "disclaimer": "Retail-use disclaimer",
    "kap_digest": "Same-day KAP list digest",
}

MIXERS = ("gated", "gmu", "tfn", "mean", "concat", "mult", "text", "vision", "tabular")


def delivered_payload(tier: str, relevant: set[str], text_vec: np.ndarray) -> list[str]:
    spec = TIER_SPECS[tier]
    live = []
    flags = {
        "usdtry_up": text_vec[0] > 0.5,
        "usdtry_down": text_vec[1] > 0.5,
        "gold_up": text_vec[2] > 0.5,
        "gold_down": text_vec[3] > 0.5,
        "vol_spike": text_vec[4] > 0.5,
        "price_shock": text_vec[5] > 0.5,
    }
    for tok in spec["template"]:
        if tok in flags and not flags[tok]:
            continue
        live.append(tok)
    while len(live) < spec["max_tokens"] // 8:
        live.append(FILLER[len(live) % len(FILLER)])
    return live[: spec["max_tokens"]]

"""Silver-label codebook for the public VESTA corpus.

Labels are deterministic functions of public OHLCV and a KAP/macro brief.

PRIMARY
    y_direction_1d, y_direction_5d  : sign of future BIST/stock return
    y_excess_1d                     : next return above trailing 20-day mean

DIAGNOSTIC (function of the tabular vector at t)
    y_leak_vol                      : 20-day realized vol > mean+2σ of 60-day vol

PERCEPTION (rule-based)
    kap_polarity                    : bullish | bearish | neutral from KAP text only
    text_polarity                   : brief stance (macro + KAP lexicon)
    chart_signal                    : breakout | support_hold | divergence | none
    anomaly_flag                    : alias of y_leak_vol
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Polarity = Literal["bullish", "bearish", "neutral"]
ChartSignal = Literal["breakout", "support_hold", "divergence", "none"]

POLARITY_ID = {"bearish": 0, "neutral": 1, "bullish": 2}
CHART_ID = {"none": 0, "breakout": 1, "support_hold": 2, "divergence": 3}


def realized_vol(returns: np.ndarray, win: int = 20, annualize: int = 252) -> float:
    if len(returns) < win:
        return float("nan")
    sl = returns[-win:]
    sl = sl[np.isfinite(sl)]
    if len(sl) < max(5, win // 2):
        return float("nan")
    return float(np.std(sl, ddof=1) * np.sqrt(annualize))


def leak_vol_flag(vol_now: float, vol_hist: np.ndarray) -> int:
    hist = vol_hist[np.isfinite(vol_hist)]
    if len(hist) < 20 or not np.isfinite(vol_now):
        return 0
    mu, sd = float(np.mean(hist)), float(np.std(hist, ddof=1))
    if sd < 1e-12:
        return 0
    return int(vol_now > mu + 2.0 * sd)


def rsi_series(close: np.ndarray, n: int = 14) -> np.ndarray:
    close = np.asarray(close, dtype=float)
    delta = np.diff(close, prepend=close[0])
    gain = np.clip(delta, 0, None)
    loss = np.clip(-delta, 0, None)
    out = np.full_like(close, 50.0, dtype=float)
    if len(close) <= n:
        return out
    ag = float(np.mean(gain[1 : n + 1]))
    al = float(np.mean(loss[1 : n + 1]))
    for i in range(n, len(close)):
        ag = (ag * (n - 1) + gain[i]) / n
        al = (al * (n - 1) + loss[i]) / n
        rs = ag / al if al > 1e-12 else 1e6
        out[i] = 100.0 - 100.0 / (1.0 + rs)
    return out


def chart_signal(ohlc: np.ndarray) -> ChartSignal:
    """Dominant technical cue on the last bar of an OHLC window (lookback, 4).

    Priority: divergence > breakout > support_hold > none.
    All thresholds use *prior* bars so the label is a function of the visible
    chart, matching the original VisualClaw annotation axis.
    """
    if ohlc is None or len(ohlc) < 25:
        return "none"
    o, h, l, c = ohlc[:, 0], ohlc[:, 1], ohlc[:, 2], ohlc[:, 3]
    prior_high = float(np.max(h[-21:-1]))
    prior_low = float(np.min(l[-21:-1]))
    close, prev = float(c[-1]), float(c[-2])
    low_t, high_t = float(l[-1]), float(h[-1])

    rsi = rsi_series(c)
    # Price vs RSI: last 10-bar swing vs previous 10-bar swing
    p_hh = float(np.max(c[-10:])) >= float(np.max(c[-20:-10])) - 1e-12
    p_ll = float(np.min(c[-10:])) <= float(np.min(c[-20:-10])) + 1e-12
    r_hh = float(np.max(rsi[-10:])) >= float(np.max(rsi[-20:-10])) - 1e-9
    r_ll = float(np.min(rsi[-10:])) <= float(np.min(rsi[-20:-10])) + 1e-9
    bear_div = p_hh and (not r_hh) and close >= prev
    bull_div = p_ll and (not r_ll) and close <= prev
    if bear_div or bull_div:
        return "divergence"

    if close > prior_high and prev <= prior_high:
        return "breakout"

    near_support = low_t <= prior_low * 1.004
    recovered = close > prev and close >= low_t + 0.45 * max(high_t - low_t, 1e-9)
    if near_support and recovered:
        return "support_hold"
    return "none"


def text_polarity(
    usd_ret: float,
    gold_ret: float,
    session_ret: float,
    rsi: float,
    news_score: float = 0.0,
) -> Polarity:
    """Stance of the *textual brief*, not the future return.

    USD/TRY jump and gold jump load as risk-off for a TRY equity reader;
    session strength and oversold RSI load as constructive. News lexicon
    score (if any) is added before thresholding.
    """
    score = 0.0
    score -= 2.0 if usd_ret > 0.01 else (1.0 if usd_ret > 0.005 else 0.0)
    score += 2.0 if usd_ret < -0.01 else (1.0 if usd_ret < -0.005 else 0.0)
    score -= 1.0 if gold_ret > 0.008 else 0.0
    score += 1.0 if gold_ret < -0.008 else 0.0
    score += 1.5 if session_ret > 0.015 else (0.5 if session_ret > 0.005 else 0.0)
    score -= 1.5 if session_ret < -0.015 else (0.5 if session_ret < -0.005 else 0.0)
    if rsi >= 70:
        score -= 0.5
    if rsi <= 30:
        score += 0.5
    score += float(np.clip(news_score, -3, 3))
    if score >= 1.0:
        return "bullish"
    if score <= -1.0:
        return "bearish"
    return "neutral"


NEWS_LEXICON = {
    "bullish": (
        "upgrade",
        "beat",
        "record",
        "growth",
        "profit",
        "rally",
        "surge",
        "approval",
        "dividend",
        "yükseliş",
        "kâr",
        "kar payı",
        "temettü",
        "rekor",
        "büyüme",
        "olumlu",
        "alım",
        "sermaye artırımı",
        "bedelsiz",
        "ihale",
        "onay",
        "sözleşme",
        "geri alım",
        "not artırımı",
        "not görünümü olumlu",
        "yeni iş ilişkisi",
        "imza",
        "kazanç",
        "net kâr",
        "net kar",
        "bedelsiz sermaye",
        "pay geri alım",
    ),
    "bearish": (
        "downgrade",
        "miss",
        "loss",
        "probe",
        "fine",
        "crash",
        "plunge",
        "default",
        "lawsuit",
        "düşüş",
        "zarar",
        "soruşturma",
        "olumsuz",
        "satış",
        "iflas",
        "dava",
        "ceza",
        "iptal",
        "gecikme",
        "kıdem",
        "grev",
        "düzeltme bildirimi",
        "not indirimi",
        "not görünümü olumsuz",
        "konkordato",
        "negatif",
        "net zarar",
        "karşılık ayrılması",
        "işten çıkarma",
        "ertelenmesi",
        "fesih",
    ),
}

# Subject-type priors for KAP polarity.
KAP_SUBJECT_PRIOR = {
    "kar payı": 1.5,
    "kâr payı": 1.5,
    "temettü": 1.5,
    "payların geri alınmasına": 1.2,
    "geri alım": 1.2,
    "bedelsiz": 1.0,
    "sermaye artırımı": 0.8,
    "yeni iş ilişkisi": 1.0,
    "kredi derecelendirmesi": 0.0,
    "özel durum açıklaması": 0.0,
    "finansal rapor": 0.0,
    "birleşme": 0.3,
    "soruşturma": -1.5,
    "ceza": -1.2,
    "dava": -0.8,
}


def news_score(headlines: list[str]) -> float:
    s = 0.0
    for h in headlines:
        t = (h or "").lower()
        s += sum(1.0 for w in NEWS_LEXICON["bullish"] if w in t)
        s -= sum(1.0 for w in NEWS_LEXICON["bearish"] if w in t)
    return s


def kap_score(texts: list[str], subjects: list[str] | None = None) -> float:
    """KAP-only stance. Does not look at USD/TRY, gold, RSI, or future returns."""
    s = news_score(texts)
    for sub in subjects or []:
        t = (sub or "").lower()
        for needle, prior in KAP_SUBJECT_PRIOR.items():
            if needle in t:
                s += prior
    return float(s)


def kap_polarity(texts: list[str], subjects: list[str] | None = None) -> Polarity:
    score = kap_score(texts, subjects)
    if score >= 1.0:
        return "bullish"
    if score <= -1.0:
        return "bearish"
    return "neutral"


SUBJECT_BULL = (
    "kar payı",
    "kâr payı",
    "temettü",
    "payların geri alınmasına",
    "geri alım",
    "bedelsiz",
    "sermaye artırımı",
    "yeni iş ilişkisi",
    "not artırımı",
)
SUBJECT_BEAR = (
    "soruşturma",
    "idari para cezası",
    "iflas",
    "konkordato",
    "not indirimi",
    "net zarar",
    "fesih",
    "grev",
    "işten çıkarma",
)


def kap_polarity_b(subjects: list[str]) -> Polarity:
    """Codebook B: subject-title taxonomy only. No token lexicon, no FX."""
    bull = bear = 0
    for sub in subjects:
        t = (sub or "").lower()
        if any(s in t for s in SUBJECT_BULL):
            bull += 1
        if any(s in t for s in SUBJECT_BEAR):
            bear += 1
    if bull > bear and bull > 0:
        return "bullish"
    if bear > bull and bear > 0:
        return "bearish"
    return "neutral"


def kap_polarity_c(texts: list[str]) -> Polarity:
    """Codebook C: body/teaser tokens only, stricter threshold, no subject priors."""
    score = news_score(texts)
    if score >= 2.0:
        return "bullish"
    if score <= -2.0:
        return "bearish"
    return "neutral"


def chart_signal_b(ohlc: np.ndarray) -> ChartSignal:
    """Independent VisualClaw codebook: 10-bar levels, 5-bar RSI swings."""
    if ohlc is None or len(ohlc) < 20:
        return "none"
    o, h, l, c = ohlc[:, 0], ohlc[:, 1], ohlc[:, 2], ohlc[:, 3]
    prior_high = float(np.max(h[-11:-1]))
    prior_low = float(np.min(l[-11:-1]))
    close, prev = float(c[-1]), float(c[-2])
    low_t, high_t = float(l[-1]), float(h[-1])
    rsi = rsi_series(c)
    p_hh = float(np.max(c[-5:])) >= float(np.max(c[-10:-5])) - 1e-12
    p_ll = float(np.min(c[-5:])) <= float(np.min(c[-10:-5])) + 1e-12
    r_hh = float(np.max(rsi[-5:])) >= float(np.max(rsi[-10:-5])) - 1e-9
    r_ll = float(np.min(rsi[-5:])) <= float(np.min(rsi[-10:-5])) + 1e-9
    if (p_hh and not r_hh and close >= prev) or (p_ll and not r_ll and close <= prev):
        return "divergence"
    if close > prior_high:
        return "breakout"
    near_support = low_t <= prior_low * 1.008
    recovered = close > prev and close >= (low_t + high_t) / 2.0
    if near_support and recovered:
        return "support_hold"
    return "none"


def render_brief(
    ticker: str,
    date: str,
    session_ret: float,
    usd_ret: float,
    gold_ret: float,
    vol20: float,
    rsi: float,
    polarity: str,
    chart: str,
    headlines: list[str],
) -> str:
    news = " ".join(headlines[:3]) if headlines else "No contemporaneous headline in the public scrape."
    return (
        f"{date} {ticker}. Session return {session_ret:+.2%}. "
        f"USD/TRY {usd_ret:+.2%}, gold {gold_ret:+.2%}. "
        f"Ann. realized vol {vol20:.1%}, RSI {rsi:.0f}. "
        f"Brief polarity {polarity}; chart cue {chart}. {news}"
    )

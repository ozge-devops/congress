"""Sanity checks for the silver-label codebook (no future peeking on perception axes)."""

from __future__ import annotations nj

import numpy as np

from vesta.labeling import (
    chart_signal,
    chart_signal_b,
    kap_polarity,
    kap_polarity_b,
    kap_polarity_c,
    leak_vol_flag,
    news_score,
    text_polarity,
)
from vesta.metrics import cohen_kappa
from vesta.models import mult_features


def _flat_ohlc(n=40, start=100.0, drift=0.0, shock=0.0):
    c = start * np.cumprod(1.0 + np.full(n, drift))
    c[-1] *= 1.0 + shock
    o = np.roll(c, 1)
    o[0] = start
    h = np.maximum(o, c) * 1.004
    l = np.minimum(o, c) * 0.996
    return np.stack([o, h, l, c], axis=1)


def test_breakout_uses_prior_high_not_future():
    ohlc = _flat_ohlc(shock=0.08)
    assert chart_signal(ohlc) == "breakout"


def test_support_hold():
    ohlc = _flat_ohlc(drift=-0.002)
    # last bar: tag the low then close strong
    ohlc[-1, 2] = ohlc[-21:-1, 2].min() * 0.999
    ohlc[-1, 3] = ohlc[-2, 3] * 1.012
    ohlc[-1, 1] = ohlc[-1, 3]
    ohlc[-1, 0] = ohlc[-2, 3]
    assert chart_signal(ohlc) == "support_hold"


def test_polarity_lira_shock_is_bearish():
    assert text_polarity(usd_ret=0.03, gold_ret=0.0, session_ret=0.0, rsi=50) == "bearish"


def test_polarity_lira_ease_is_bullish():
    assert text_polarity(usd_ret=-0.03, gold_ret=0.0, session_ret=0.02, rsi=35) == "bullish"


def test_news_lexicon():
    assert news_score(["Bank beats estimates, record profit"]) > 0
    assert news_score(["Probe after loss and plunge"]) < 0


def test_kap_polarity_dividend_is_bullish():
    assert kap_polarity(["Kar payı dağıtım işlemlerine ilişkin bildirim, temettü"], ["Kar Payı Dağıtım İşlemlerine İlişkin Bildirim"]) == "bullish"


def test_kap_polarity_probe_is_bearish():
    assert kap_polarity(["SPK soruşturma ve idari para cezası"], ["Özel Durum Açıklaması (Genel)"]) == "bearish"


def test_kap_polarity_blank_is_neutral():
    assert kap_polarity([""], ["Şirket Genel Bilgi Formu"]) == "neutral"


def test_codebook_b_uses_subject_only():
    assert kap_polarity_b(["Kar Payı Dağıtım İşlemlerine İlişkin Bildirim"]) == "bullish"
    assert kap_polarity_b(["SPK Soruşturma"]) == "bearish"
    assert kap_polarity_b(["Şirket Genel Bilgi Formu"]) == "neutral"


def test_codebook_c_ignores_subject_title():
    # subject-only dividend words are not passed in
    assert kap_polarity_c(["rutin kupon ödemesi tamamlandı"]) == "neutral"


def test_cohen_kappa_perfect_and_chance():
    k = cohen_kappa(["a", "a", "b", "b"], ["a", "a", "b", "b"])
    assert k["kappa"] == 1.0
    k2 = cohen_kappa(["a", "b", "a", "b"], ["b", "a", "b", "a"])
    assert k2["kappa"] < 0.0


def test_mult_features_shape():
    rng = np.random.default_rng(0)
    text = rng.normal(size=(12, 16)).astype(np.float32)
    vis = rng.random((12, 576)).astype(np.float32)
    feat = mult_features(text, vis)
    assert feat.shape == (12, 64)


def test_chart_b_not_identical_to_a_on_10bar_breakout():
    ohlc = _flat_ohlc(n=40, drift=0.0)
    # 10-bar high broken but not 20-bar high
    ohlc[-12:-1, 1] = 101.0
    ohlc[-1, 3] = 101.5
    ohlc[-1, 1] = 101.6
    ohlc[-25:-12, 1] = 103.0
    assert chart_signal_b(ohlc) in {"breakout", "divergence", "none", "support_hold"}


def test_ohlc_window_excludes_day_t():
    """Public chart is t-LOOKBACK … t-1; leak flag uses day-t statistics."""
    from vesta.data import LOOKBACK

    n = LOOKBACK + 5
    ohlc = _flat_ohlc(n=n, start=100.0, drift=0.001)
    window = ohlc[:LOOKBACK]
    assert len(window) == LOOKBACK
    assert np.allclose(window[-1], ohlc[LOOKBACK - 1])
    assert not np.allclose(window[-1], ohlc[-1])


def test_leak_flag_closed_form():
    rng = np.random.default_rng(0)
    hist = 0.20 + 0.02 * rng.normal(size=60)
    mu, sd = float(hist.mean()), float(hist.std(ddof=1))
    assert leak_vol_flag(mu + 2.5 * sd, hist) == 1
    assert leak_vol_flag(mu + 0.5 * sd, hist) == 0


if __name__ == "__main__":
    tests = [
        test_breakout_uses_prior_high_not_future,
        test_support_hold,
        test_polarity_lira_shock_is_bearish,
        test_polarity_lira_ease_is_bullish,
        test_news_lexicon,
        test_kap_polarity_dividend_is_bullish,
        test_kap_polarity_probe_is_bearish,
        test_kap_polarity_blank_is_neutral,
        test_codebook_b_uses_subject_only,
        test_codebook_c_ignores_subject_title,
        test_cohen_kappa_perfect_and_chance,
        test_mult_features_shape,
        test_chart_b_not_identical_to_a_on_10bar_breakout,
        test_ohlc_window_excludes_day_t,
        test_leak_flag_closed_form,
    ]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print("all labeling tests passed")

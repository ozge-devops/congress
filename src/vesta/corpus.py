"""Build and silver-label the public VESTA event corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from vesta.data import LOOKBACK, VOL_DIST_WIN, VOL_WIN, _flatten_ohlcv, _rsi, download_public_market
from vesta.kap import index_by_ticker_date
from vesta.labeling import (
    CHART_ID,
    POLARITY_ID,
    chart_signal,
    kap_polarity,
    kap_score,
    leak_vol_flag,
    news_score,
    render_brief,
    rsi_series,
    text_polarity,
)


CONSTITUENTS = {
    "THYAO.IS": "airlines",
    "PGSUS.IS": "airlines",
    "TAVHL.IS": "airlines",
    "GARAN.IS": "banks",
    "AKBNK.IS": "banks",
    "YKBNK.IS": "banks",
    "ISCTR.IS": "banks",
    "KCHOL.IS": "holdings",
    "SAHOL.IS": "holdings",
    "SISE.IS": "holdings",
    "ASELS.IS": "defence",
    "TUPRS.IS": "energy",
    "PETKM.IS": "energy",
    "EREGL.IS": "industrials",
    "BIMAS.IS": "retail",
    "MGROS.IS": "retail",
    "TCELL.IS": "telecom",
    "TTKOM.IS": "telecom",
    "FROTO.IS": "auto",
    "TOASO.IS": "auto",
    "ARCLK.IS": "consumer",
    "AEFES.IS": "consumer",
    "ULKER.IS": "consumer",
    "SASA.IS": "materials",
    "EKGYO.IS": "real_estate",
    "ENKAI.IS": "construction",
    "TTRAK.IS": "industrials",
}


def download_constituents(cache_dir: Path) -> dict[str, pd.DataFrame]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    missing = []
    for ticker in CONSTITUENTS:
        csv_path = cache_dir / f"{ticker.replace('.', '_')}.csv"
        if csv_path.exists():
            out[ticker] = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            continue
        missing.append(ticker)
    if missing:
        raw = yf.download(
            missing,
            start="2018-01-01",
            progress=False,
            auto_adjust=True,
            group_by="ticker",
            threads=True,
        )
        if len(missing) == 1:
            t = missing[0]
            df = _flatten_ohlcv(raw, t)
            df.to_csv(cache_dir / f"{t.replace('.', '_')}.csv")
            out[t] = df
        else:
            for t in missing:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        sub = raw[t].copy()
                    else:
                        sub = raw.copy()
                    df = _flatten_ohlcv(sub, t)
                    if df.empty:
                        continue
                    df.to_csv(cache_dir / f"{t.replace('.', '_')}.csv")
                    out[t] = df
                except Exception:
                    continue
    return out


def fetch_recent_news(tickers: list[str], cache_path: Path) -> dict[str, list[dict]]:
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    bag: dict[str, list[dict]] = {}
    for t in tickers:
        items = []
        try:
            news = yf.Ticker(t).news or []
        except Exception:
            news = []
        for n in news:
            content = n.get("content") or n
            title = (
                content.get("title")
                or n.get("title")
                or (content.get("canonicalUrl") or {}).get("title")
                or ""
            )
            pub = (
                content.get("pubDate")
                or content.get("displayTime")
                or n.get("providerPublishTime")
                or ""
            )
            if title:
                items.append({"title": str(title), "published": str(pub)[:10], "ticker": t})
        bag[t] = items
    cache_path.write_text(json.dumps(bag, ensure_ascii=False, indent=2))
    return bag


def _news_for(date: pd.Timestamp, ticker: str, news_bag: dict[str, list[dict]]) -> list[str]:
    day = date.strftime("%Y-%m-%d")
    hits = []
    for item in news_bag.get(ticker, []):
        pub = (item.get("published") or "")[:10]
        if pub == day:
            hits.append(item["title"])
    return hits


def _prepare_index(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["ret1"] = x["close"].pct_change()
    x["vol20"] = x["ret1"].rolling(VOL_WIN).std() * np.sqrt(252)
    x["rsi"] = _rsi(x["close"])
    return x


def _windows_ok(df: pd.DataFrame, i: int) -> bool:
    if i < LOOKBACK + VOL_DIST_WIN or i + 5 >= len(df):
        return False
    w = df.iloc[i - LOOKBACK : i]
    return not w[["open", "high", "low", "close"]].isna().any().any()


def build_events(
    index_frames: dict[str, pd.DataFrame],
    stock_frames: dict[str, pd.DataFrame],
    news_bag: dict[str, list[dict]],
    kap_rows: list[dict] | None = None,
) -> pd.DataFrame:
    bist = _prepare_index(index_frames["bist100"])
    usd = index_frames["usdtry"].reindex(bist.index).ffill()
    gold = index_frames["gold"].reindex(bist.index).ffill()
    usd_ret = usd["close"].pct_change()
    gold_ret = gold["close"].pct_change()
    kap_index = index_by_ticker_date(kap_rows or [])
    kap_by_date: dict[str, list[dict]] = {}
    for item in kap_rows or []:
        kap_by_date.setdefault(item["date"], []).append(item)
    rows = []

    def add_row(
        *,
        ticker: str,
        sector: str,
        df: pd.DataFrame,
        i: int,
        keep_all: bool,
    ) -> None:
        if not _windows_ok(df, i):
            return
        t = df.index[i]
        if t not in bist.index:
            return
        close = df["close"]
        # Same convention as vesta.data.build_samples: bars t-LOOKBACK to t-1, features at t.
        ohlc = df.iloc[i - LOOKBACK : i][["open", "high", "low", "close"]].to_numpy(dtype=np.float32)
        r1 = float(df["ret1"].iloc[i]) if pd.notna(df["ret1"].iloc[i]) else 0.0
        v20 = float(df["vol20"].iloc[i]) if pd.notna(df["vol20"].iloc[i]) else float("nan")
        rsi_i = float(df["rsi"].iloc[i]) if pd.notna(df["rsi"].iloc[i]) else 50.0
        r_usd = float(usd_ret.loc[t]) if pd.notna(usd_ret.loc[t]) else 0.0
        r_gold = float(gold_ret.loc[t]) if pd.notna(gold_ret.loc[t]) else 0.0
        hist_vol = df["vol20"].iloc[max(0, i - VOL_DIST_WIN) : i].to_numpy(dtype=float)
        y_leak = leak_vol_flag(v20, hist_vol)
        day = t.strftime("%Y-%m-%d")
        kap_hits = kap_index.get((ticker, day), [])
        if ticker == "XU100.IS":
            kap_hits = kap_by_date.get(day, [])
        has_kap = len(kap_hits) > 0
        if not keep_all:
            if abs(r1) < 0.02 and y_leak == 0 and not (rsi_i >= 75 or rsi_i <= 25) and not has_kap:
                return
        next_ret = float(close.iloc[i + 1] / close.iloc[i] - 1)
        next_ret5 = float(close.iloc[i + 5] / close.iloc[i] - 1)
        trail = float(df["ret1"].iloc[i - 20 : i].mean()) if i >= 20 else 0.0
        headlines = _news_for(t, ticker, news_bag)
        kap_texts = [k["text"] for k in kap_hits if k.get("text")]
        kscore = kap_score(kap_texts, [k.get("subject") or "" for k in kap_hits])
        nscore = news_score(headlines) + kscore
        chart = chart_signal(ohlc)
        pol = text_polarity(r_usd, r_gold, r1, rsi_i, nscore)
        kpol = kap_polarity(kap_texts, [k.get("subject") or "" for k in kap_hits])
        brief_bits = headlines + kap_texts[:4]
        brief = render_brief(
            ticker,
            day,
            r1,
            r_usd,
            r_gold,
            v20 if np.isfinite(v20) else 0.0,
            rsi_i,
            pol,
            chart,
            brief_bits,
        )
        eid = hashlib.md5(f"{ticker}|{t.date()}".encode()).hexdigest()[:12]
        rsi_win = rsi_series(ohlc[:, 3])
        rows.append(
            {
                "event_id": eid,
                "date": t.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "sector": sector,
                "session_ret": r1,
                "usdtry_ret": r_usd,
                "gold_ret": r_gold,
                "vol20": v20 if np.isfinite(v20) else None,
                "rsi": rsi_i,
                "rsi_window_last": float(rsi_win[-1]),
                "close": float(close.iloc[i]),
                "next_ret_1d": next_ret,
                "next_ret_5d": next_ret5,
                "y_direction_1d": int(next_ret > 0),
                "y_direction_5d": int(next_ret5 > 0),
                "y_excess_1d": int(next_ret > trail),
                "y_leak_vol": y_leak,
                "anomaly_flag": y_leak,
                "text_polarity": pol,
                "text_polarity_id": POLARITY_ID[pol],
                "chart_signal": chart,
                "chart_signal_id": CHART_ID[chart],
                "kap_polarity": kpol,
                "kap_polarity_id": POLARITY_ID[kpol],
                "kap_score": kscore,
                "n_headlines": len(headlines),
                "headlines": " || ".join(headlines),
                "n_kap": len(kap_hits),
                "has_kap": has_kap,
                "kap_subjects": " || ".join(sorted({k.get("subject") or "" for k in kap_hits})),
                "kap_text": " || ".join(kap_texts[:8]),
                "kap_indices": ",".join(str(k.get("disclosure_index")) for k in kap_hits[:12]),
                "brief": brief,
                "label_source": "rule_v1_public",
                "leakage_in_primary": False,
                "leakage_in_y_leak_vol": True,
                "ohlc_open": json.dumps(ohlc[:, 0].round(4).tolist()),
                "ohlc_high": json.dumps(ohlc[:, 1].round(4).tolist()),
                "ohlc_low": json.dumps(ohlc[:, 2].round(4).tolist()),
                "ohlc_close": json.dumps(ohlc[:, 3].round(4).tolist()),
            }
        )

    for i in range(len(bist)):
        add_row(ticker="XU100.IS", sector="index", df=bist, i=i, keep_all=True)

    for ticker, df0 in stock_frames.items():
        df = _prepare_index(df0.reindex(bist.index))
        for i in range(len(df)):
            add_row(
                ticker=ticker,
                sector=CONSTITUENTS.get(ticker, "other"),
                df=df,
                i=i,
                keep_all=False,
            )

    events = pd.DataFrame(rows).drop_duplicates(subset=["ticker", "date"]).sort_values(["date", "ticker"])
    dates = sorted(events["date"].unique())
    n = len(dates)
    train_cut = dates[int(n * 0.70)]
    val_cut = dates[int(n * 0.85)]

    def split_of(d: str) -> str:
        if d < train_cut:
            return "train"
        if d < val_cut:
            return "val"
        return "test"

    events["split"] = events["date"].map(split_of)
    return events.reset_index(drop=True)


def stratified_annotation_sample(events: pd.DataFrame, n: int = 250, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    grouped = events.groupby(["text_polarity", "chart_signal"], dropna=False)
    quotas = max(1, n // max(len(grouped), 1))
    for _, g in grouped:
        k = min(len(g), quotas)
        idx = rng.choice(g.index.to_numpy(), size=k, replace=False)
        parts.append(g.loc[idx])
    sample = pd.concat(parts, ignore_index=True)
    if len(sample) < n:
        rest = events[~events["event_id"].isin(sample["event_id"])]
        extra = rest.sample(n=min(n - len(sample), len(rest)), random_state=seed)
        sample = pd.concat([sample, extra], ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=seed).head(n).reset_index(drop=True)
    keep = [
        "event_id",
        "date",
        "ticker",
        "sector",
        "split",
        "brief",
        "headlines",
        "kap_text",
        "n_kap",
        "session_ret",
        "rsi",
        "text_polarity",
        "kap_polarity",
        "chart_signal",
        "y_leak_vol",
        "y_direction_1d",
    ]
    keep = [c for c in keep if c in sample.columns]
    out = sample[keep].copy()
    out = out.rename(
        columns={
            "text_polarity": "silver_text_polarity",
            "kap_polarity": "silver_kap_polarity",
            "chart_signal": "silver_chart_signal",
            "y_leak_vol": "silver_anomaly_flag",
        }
    )
    out["annotator_text_polarity"] = ""
    out["annotator_chart_signal"] = ""
    out["annotator_notes"] = ""
    out["annotator_id"] = ""
    return out


def make_10k(events: pd.DataFrame, n: int = 10_000) -> pd.DataFrame:
    """Paper-sized slice: every index day plus the largest |session| constituent moves."""
    idx = events[events["ticker"] == "XU100.IS"]
    rest = events[events["ticker"] != "XU100.IS"].copy()
    rest["abs_ret"] = rest["session_ret"].abs()
    need = max(0, n - len(idx))
    top = rest.nlargest(need, "abs_ret").drop(columns=["abs_ret"])
    out = pd.concat([idx, top], ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    return out


def summarize(events: pd.DataFrame) -> dict:
    def dist(col: str) -> dict:
        return {str(k): int(v) for k, v in events[col].value_counts().sort_index().items()}

    return {
        "n_events": int(len(events)),
        "n_tickers": int(events["ticker"].nunique()),
        "date_start": str(events["date"].min()),
        "date_end": str(events["date"].max()),
        "split": dist("split"),
        "text_polarity": dist("text_polarity"),
        "chart_signal": dist("chart_signal"),
        "y_direction_1d": dist("y_direction_1d"),
        "y_leak_vol_rate": float(events["y_leak_vol"].mean()),
        "with_headlines": int((events["n_headlines"] > 0).sum()) if "n_headlines" in events else 0,
        "with_kap": int((events["n_kap"] > 0).sum()) if "n_kap" in events else 0,
        "n_kap_filings_linked": int(events["n_kap"].sum()) if "n_kap" in events else 0,
        "kap_polarity": dist("kap_polarity") if "kap_polarity" in events.columns else {},
        "index_events": int((events["ticker"] == "XU100.IS").sum()),
        "constituent_events": int((events["ticker"] != "XU100.IS").sum()),
    }

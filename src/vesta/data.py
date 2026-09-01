"""Public BIST100 market series (Yahoo). KAP list ingest lives in vesta.kap."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


LOOKBACK = 40
FWD = 1
VOL_WIN = 20
VOL_DIST_WIN = 60


def load_kap_daily_features(path: Path | None = None) -> dict[str, np.ndarray]:
    """Eight KAP-list features keyed by calendar date, or empty if the file is missing."""
    path = path or Path("data/vesta_public/kap_daily_features.json")
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {k: np.asarray(v, dtype=np.float32) for k, v in raw.items()}


def _flatten_ohlcv(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in raw.columns]
    out = raw[keep].copy()
    out.columns = [c.lower() for c in out.columns]
    return out.dropna(subset=["close"])


def download_public_market(cache_dir: Path) -> dict[str, pd.DataFrame]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    tickers = {"bist100": "XU100.IS", "usdtry": "USDTRY=X", "gold": "GC=F"}
    for name, ticker in tickers.items():
        csv_path = cache_dir / f"{name}.csv"
        if csv_path.exists():
            frames[name] = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            continue
        raw = yf.download(ticker, start="2018-01-01", progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            raise RuntimeError(f"Failed to download {ticker}")
        df = _flatten_ohlcv(raw, ticker)
        df.to_csv(csv_path)
        frames[name] = df
    return frames


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


@dataclass
class Sample:
    date: pd.Timestamp
    ohlc: np.ndarray  # (LOOKBACK, 4) OHLC
    tabular: np.ndarray
    text: np.ndarray
    y_fwd: int
    y_fwd5: int
    y_leak: int
    next_ret: float
    next_ret5: float
    raw_tokens: int
    relevant_pool: set[str]


def build_samples(frames: dict[str, pd.DataFrame], kap_features: dict[str, np.ndarray] | None = None) -> list[Sample]:
    bist = frames["bist100"]
    usd = frames["usdtry"].reindex(bist.index).ffill()
    gold = frames["gold"].reindex(bist.index).ffill()
    kap_features = kap_features if kap_features is not None else load_kap_daily_features()
    zeros = np.zeros(8, dtype=np.float32)

    close = bist["close"]
    ret1 = close.pct_change()
    vol20 = ret1.rolling(VOL_WIN).std() * np.sqrt(252)
    vol_mean = vol20.rolling(VOL_DIST_WIN).mean()
    vol_std = vol20.rolling(VOL_DIST_WIN).std()
    leak = (vol20 > (vol_mean + 2.0 * vol_std)).astype(int)
    rsi = _rsi(close)
    usd_ret = usd["close"].pct_change()
    gold_ret = gold["close"].pct_change()
    hl_range = (bist["high"] - bist["low"]) / close
    vol_z = (bist["volume"] - bist["volume"].rolling(20).mean()) / (
        bist["volume"].rolling(20).std().replace(0, np.nan)
    )

    samples: list[Sample] = []
    idx = bist.index
    for i in range(LOOKBACK + VOL_DIST_WIN, len(idx) - 5):
        window = bist.iloc[i - LOOKBACK : i]
        if window[["open", "high", "low", "close"]].isna().any().any():
            continue
        ohlc = window[["open", "high", "low", "close"]].to_numpy(dtype=np.float32)
        t = idx[i]
        r_usd = float(usd_ret.iloc[i]) if pd.notna(usd_ret.iloc[i]) else 0.0
        r_gold = float(gold_ret.iloc[i]) if pd.notna(gold_ret.iloc[i]) else 0.0
        r1 = float(ret1.iloc[i]) if pd.notna(ret1.iloc[i]) else 0.0
        r5 = float(close.iloc[i] / close.iloc[i - 5] - 1) if i >= 5 else 0.0
        r20 = float(close.iloc[i] / close.iloc[i - 20] - 1) if i >= 20 else 0.0
        v20 = float(vol20.iloc[i]) if pd.notna(vol20.iloc[i]) else 0.0
        rsi_i = float(rsi.iloc[i]) if pd.notna(rsi.iloc[i]) else 50.0
        hlr = float(hl_range.iloc[i]) if pd.notna(hl_range.iloc[i]) else 0.0
        vz = float(vol_z.iloc[i]) if pd.notna(vol_z.iloc[i]) else 0.0
        vmean = float(vol_mean.iloc[i]) if pd.notna(vol_mean.iloc[i]) else 0.0
        vstd = float(vol_std.iloc[i]) if pd.notna(vol_std.iloc[i]) else 0.0

        tabular = np.array(
            [r1, r5, r20, v20, rsi_i / 100.0, hlr, vz, r_usd, r_gold, vmean, vstd],
            dtype=np.float32,
        )
        text = np.array(
            [
                1.0 if r_usd > 0.005 else 0.0,
                1.0 if r_usd < -0.005 else 0.0,
                1.0 if r_gold > 0.005 else 0.0,
                1.0 if r_gold < -0.005 else 0.0,
                1.0 if v20 > vmean + vstd else 0.0,
                1.0 if abs(r1) > 0.02 else 0.0,
                1.0 if rsi_i > 70 else 0.0,
                1.0 if rsi_i < 30 else 0.0,
            ],
            dtype=np.float32,
        )
        kap_vec = kap_features.get(pd.Timestamp(t).strftime("%Y-%m-%d"), zeros)
        text = np.concatenate([text, kap_vec]).astype(np.float32)

        next_ret = float(close.iloc[i + 1] / close.iloc[i] - 1)
        next_ret5 = float(close.iloc[i + 5] / close.iloc[i] - 1)
        y_fwd = int(next_ret > 0)
        y_fwd5 = int(next_ret5 > 0)
        y_leak = int(leak.iloc[i])

        relevant: set[str] = set()
        if r_usd > 0.005:
            relevant.add("usdtry_up")
        elif r_usd < -0.005:
            relevant.add("usdtry_down")
        if r_gold > 0.005:
            relevant.add("gold_up")
        elif r_gold < -0.005:
            relevant.add("gold_down")
        if v20 > vmean + vstd:
            relevant.add("vol_spike")
        if abs(r1) > 0.02:
            relevant.add("price_shock")
        if float(kap_vec[4]) > 0.5 or float(kap_vec[5]) > 0.5:
            relevant.add("kap_digest")
        relevant.add("bist100_level")
        relevant.add("session_ohlc")

        samples.append(
            Sample(
                date=t,
                ohlc=ohlc,
                tabular=tabular,
                text=text,
                y_fwd=y_fwd,
                y_fwd5=y_fwd5,
                y_leak=y_leak,
                next_ret=next_ret,
                next_ret5=next_ret5,
                raw_tokens=720,
                relevant_pool=relevant,
            )
        )
    return samples


def chronological_split(
    samples: list[Sample], train=0.70, val=0.15
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    n = len(samples)
    i_train = int(n * train)
    i_val = int(n * (train + val))
    return samples[:i_train], samples[i_train:i_val], samples[i_val:]

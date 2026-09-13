"""In-process VESTA stack used by the HTTP API: samples, seed-0 mixers, corpus."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from vesta.charts import render_candles, render_screenshot
from vesta.data import Sample, build_samples, chronological_split, download_public_market
from vesta.labeling import chart_signal, render_brief, text_polarity
from vesta.metrics import information_noise
from vesta.models import (
    Fitted,
    closed_form_vol_rule,
    fit_fusion,
    fit_unimodal,
    gated_features,
    gmu_features,
    mult_features,
    predict_proba,
    tfn_features,
)
from vesta.hop import load_m3, retrieve_prior_day, same_date_peer
from vesta.tiers import MIXERS, TIER_SPECS, TOKEN_HELP, delivered_payload

ROOT = Path(__file__).resolve().parents[2]


def _date_key(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@dataclass
class MixerCall:
    mixer: str
    p_up: float
    call: str
    gate_g: float | None


class VestaEngine:
    def __init__(self, root: Path | None = None, seed: int = 0):
        self.root = Path(root) if root else ROOT
        self.seed = seed
        self.ready = False
        self.error: str | None = None
        self.samples: list[Sample] = []
        self.by_date: dict[str, Sample] = {}
        self.split_of: dict[str, str] = {}
        self.majority = 1
        self.models: dict[str, Fitted] = {}
        self._events: pd.DataFrame | None = None
        self._hop_dates: list[str] = []
        self._hop_vecs: np.ndarray | None = None
        self.meta: dict = {}

    def load(self) -> None:
        cache = self.root / "data" / "cache"
        frames = download_public_market(cache)
        samples = build_samples(frames)
        train, val, test = chronological_split(samples)
        fit_set = train + val
        self.samples = samples
        self.by_date = {_date_key(s.date): s for s in samples}
        self.split_of = {}
        for s in train:
            self.split_of[_date_key(s.date)] = "train"
        for s in val:
            self.split_of[_date_key(s.date)] = "val"
        for s in test:
            self.split_of[_date_key(s.date)] = "test"

        y = np.array([s.y_fwd for s in fit_set], dtype=int)
        self.majority = int(y.mean() >= 0.5)
        X_tab = np.stack([s.tabular for s in fit_set])
        X_txt = np.stack([s.text for s in fit_set])
        X_img = np.stack([render_candles(s.ohlc, size=24).ravel() for s in fit_set])

        text_m = fit_unimodal("text", X_txt, y, self.seed)
        vis_m = fit_unimodal("vision", X_img, y, self.seed)
        tab_m = fit_unimodal("tabular", X_tab, y, self.seed)
        self.models = {"text": text_m, "vision": vis_m, "tabular": tab_m}

        p_txt = predict_proba(text_m, X_txt)
        p_vis = predict_proba(vis_m, X_img)
        h_t = np.stack([1 - p_txt, p_txt], axis=1)
        h_v = np.stack([1 - p_vis, p_vis], axis=1)

        self.models["concat"] = fit_fusion(
            "concat", np.concatenate([X_txt, X_img, X_tab], axis=1), y, self.seed
        )
        self.models["mean"] = fit_fusion("mean", (h_t + h_v) / 2.0, y, self.seed)
        self.models["gmu"] = fit_fusion("gmu", gmu_features(h_t, h_v), y, self.seed)
        self.models["tfn"] = fit_fusion("tfn", tfn_features(h_t, h_v), y, self.seed)
        self.models["gated"] = fit_fusion("gated", gated_features(h_t, h_v), y, self.seed)
        self.models["mult"] = fit_fusion("mult", mult_features(X_txt, X_img), y, self.seed)
        self._hop_dates, self._hop_vecs = load_m3(self.root / "data" / "vesta_public" / "kap_daily_embeddings_m3.npz")

        self.meta = {
            "n_total": len(samples),
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "date_start": _date_key(samples[0].date),
            "date_end": _date_key(samples[-1].date),
            "test_start": _date_key(test[0].date),
            "test_end": _date_key(test[-1].date),
            "majority_up": bool(self.majority),
            "seed": self.seed,
        }
        self.ready = True
        self.error = None

    def dates(self, split: str | None = None) -> list[str]:
        keys = sorted(self.by_date)
        if not split or split == "all":
            return keys
        return [d for d in keys if self.split_of.get(d) == split]

    def sample(self, date: str) -> Sample:
        s = self.by_date.get(date)
        if s is None:
            raise KeyError(date)
        return s

    def scores(self, sample: Sample) -> dict[str, MixerCall]:
        img = render_candles(sample.ohlc, size=24).ravel()[None, :]
        txt = sample.text[None, :]
        tab = sample.tabular[None, :]
        p_txt = float(predict_proba(self.models["text"], txt)[0])
        p_vis = float(predict_proba(self.models["vision"], img)[0])
        p_tab = float(predict_proba(self.models["tabular"], tab)[0])
        h_t = np.array([[1.0 - p_txt, p_txt]], dtype=np.float32)
        h_v = np.array([[1.0 - p_vis, p_vis]], dtype=np.float32)
        gated = gated_features(h_t, h_v)
        g = float(gated[0, -1])

        def call(name: str, p: float, gate: float | None = None) -> MixerCall:
            return MixerCall(
                mixer=name,
                p_up=float(p),
                call="up" if p >= 0.5 else "down",
                gate_g=gate,
            )

        out = {
            "text": call("text", p_txt),
            "vision": call("vision", p_vis),
            "tabular": call("tabular", p_tab),
            "concat": call(
                "concat",
                float(
                    predict_proba(
                        self.models["concat"],
                        np.concatenate([txt, img, tab], axis=1),
                    )[0]
                ),
            ),
            "mean": call("mean", float(predict_proba(self.models["mean"], (h_t + h_v) / 2.0)[0])),
            "gmu": call("gmu", float(predict_proba(self.models["gmu"], gmu_features(h_t, h_v))[0])),
            "tfn": call("tfn", float(predict_proba(self.models["tfn"], tfn_features(h_t, h_v))[0])),
            "gated": call("gated", float(predict_proba(self.models["gated"], gated)[0]), g),
            "mult": call("mult", float(predict_proba(self.models["mult"], mult_features(txt, img))[0])),
        }
        return out

    def brief(self, date: str, tier: str = "T3", mixer: str = "gated", ticker: str = "XU100") -> dict:
        if tier not in TIER_SPECS:
            raise ValueError(f"tier must be one of {list(TIER_SPECS)}")
        if mixer not in MIXERS:
            raise ValueError(f"mixer must be one of {list(MIXERS)}")
        sample = self.sample(date)
        scores = self.scores(sample)
        chosen = scores[mixer]
        spec = TIER_SPECS[tier]
        g = scores["gated"].gate_g if scores["gated"].gate_g is not None else 0.5
        silent = bool(spec["gate_tau"] > 0 and abs(g - 0.5) <= spec["gate_tau"])
        payload = delivered_payload(tier, sample.relevant_pool, sample.text)
        noise = information_noise(sample.raw_tokens, payload, sample.relevant_pool)
        leak = int(closed_form_vol_rule(sample.tabular[None, :])[0])
        event = self.lookup_event(ticker, date)
        if event:
            session_ret = float(event["session_ret"])
            usd_ret = float(event["usdtry_ret"])
            gold_ret = float(event["gold_ret"])
            vol20 = float(event["vol20"] or 0.0)
            rsi = float(event["rsi"] or 50.0)
            polarity = str(event["text_polarity"])
            chart = str(event["chart_signal"])
            kap = (event.get("kap_text") or "").strip()
            headlines = [part.strip() for part in kap.split(" || ") if part.strip()] if kap else []
            brief_ticker = str(event["ticker"])
        else:
            session_ret = float(sample.tabular[0])
            usd_ret = float(sample.tabular[7])
            gold_ret = float(sample.tabular[8])
            vol20 = float(sample.tabular[3])
            rsi = float(sample.tabular[4] * 100.0)
            polarity = text_polarity(
                usd_ret=usd_ret, gold_ret=gold_ret, session_ret=session_ret, rsi=rsi
            )
            chart = chart_signal(sample.ohlc)
            headlines = []
            brief_ticker = "XU100.IS"
        text = render_brief(
            ticker=brief_ticker,
            date=date,
            session_ret=session_ret,
            usd_ret=usd_ret,
            gold_ret=gold_ret,
            vol20=vol20,
            rsi=rsi,
            polarity=polarity,
            chart=chart,
            headlines=headlines,
        )
        live_tokens = [t for t in payload if not t.startswith("filler_")]
        return {
            "date": date,
            "ticker": ticker,
            "split": self.split_of.get(date),
            "tier": tier,
            "tier_label": spec["label"],
            "budget_minutes": spec["minutes"],
            "declared_latency_s": spec["latency_s"],
            "mixer": mixer,
            "silent": silent,
            "reason": (
                "T1 abstains when |g-0.5| <= 0.18"
                if silent
                else "emit fused call"
            ),
            "call": None if silent else chosen.call,
            "p_up": round(chosen.p_up, 4),
            "gate_g": round(g, 4),
            "gate_keeps": "text" if g >= 0.5 else "vision",
            "mixers": {
                name: {"p_up": round(c.p_up, 4), "call": c.call, "gate_g": c.gate_g}
                for name, c in scores.items()
            },
            "call_scope": "XU100_index_mixer",
            "brief_scope": "event_row" if event else "index_sample",
            "vol_flag_2sigma": bool(leak),
            "chart_signal": chart,
            "text_polarity": polarity,
            "brief": None if silent else text,
            "tokens": live_tokens if not silent else [],
            "token_help": {k: TOKEN_HELP[k] for k in live_tokens if k in TOKEN_HELP},
            "delivery": {
                "n_delivered": int(noise.n_delivered),
                "n_raw": int(noise.n_raw),
                "in_old": round(noise.in_old, 4),
                "in_new": round(noise.in_new, 4),
                "compression": round(noise.compression, 4),
            },
            "event": event,
            "hop": (
                {
                    "prior": retrieve_prior_day(date, self._hop_dates, self._hop_vecs),
                    "peer": same_date_peer(self.events_frame(), date),
                    "alpha": 1.0,
                    "beta_fitted": False,
                    "portfolio": False,
                    "enters_mixer": False,
                }
                if tier in {"T3", "T10"} and not silent
                else None
            ),
            "realized": {
                "y_direction_1d": int(sample.y_fwd),
                "next_ret_1d": round(float(sample.next_ret), 6),
            },
        }

    def vision_png(self, date: str, kind: str = "screenshot") -> bytes:
        sample = self.sample(date)
        if kind == "tensor":
            arr = (render_candles(sample.ohlc, size=24) * 255).astype(np.uint8)
            img = Image.fromarray(arr, mode="L").resize((240, 240), Image.NEAREST)
            return _png_bytes(img.convert("RGB"))
        if kind == "tokens":
            arr = render_candles(sample.ohlc, size=24)
            img = (
                Image.fromarray((arr * 255).astype(np.uint8), mode="L")
                .resize((240, 240), Image.NEAREST)
                .convert("RGB")
            )
            canvas = ImageDraw.Draw(img)
            for k in (80, 160):
                canvas.line([(0, k), (239, k)], fill=(220, 38, 38), width=2)
                canvas.line([(k, 0), (k, 239)], fill=(220, 38, 38), width=2)
            return _png_bytes(img)
        shot = render_screenshot(sample.ohlc, title=f"XU100  {date}")
        return _png_bytes(shot)

    def events_frame(self) -> pd.DataFrame:
        if self._events is None:
            path = self.root / "data" / "vesta_public" / "events.parquet"
            cols = [
                "event_id",
                "date",
                "ticker",
                "sector",
                "session_ret",
                "usdtry_ret",
                "gold_ret",
                "vol20",
                "rsi",
                "y_direction_1d",
                "y_leak_vol",
                "text_polarity",
                "chart_signal",
                "has_kap",
                "kap_subjects",
                "kap_text",
                "kap_polarity",
                "brief",
                "split",
            ]
            df = pd.read_parquet(path, columns=cols)
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            self._events = df
        return self._events

    def tickers(self) -> list[str]:
        names = sorted(self.events_frame()["ticker"].dropna().unique().tolist())
        return ["XU100"] + names

    def lookup_event(self, ticker: str, date: str) -> dict | None:
        if ticker in {"XU100", "BIST100"}:
            want = "XU100.IS"
        elif ticker.endswith(".IS"):
            want = ticker
        else:
            want = f"{ticker}.IS"
        df = self.events_frame()
        hit = df[(df["ticker"] == want) & (df["date"] == date)]
        if hit.empty:
            hit = df[(df["ticker"] == ticker) & (df["date"] == date)]
        if hit.empty:
            return None
        row = hit.iloc[0].to_dict()
        for k, v in list(row.items()):
            if isinstance(v, (np.floating, float)):
                row[k] = float(v)
            elif isinstance(v, (np.integer,)):
                row[k] = int(v)
            elif pd.isna(v):
                row[k] = None
        return row

    def search_events(
        self,
        ticker: str | None = None,
        date: str | None = None,
        has_kap: bool | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> dict:
        df = self.events_frame()
        if ticker:
            if ticker in {"XU100", "BIST100"}:
                want = "XU100.IS"
            elif ticker.endswith(".IS"):
                want = ticker
            else:
                want = f"{ticker}.IS"
            df = df[df["ticker"] == want]
        if date:
            df = df[df["date"] == date]
        if has_kap is not None:
            df = df[df["has_kap"] == has_kap]
        total = int(len(df))
        page = df.iloc[offset : offset + limit]
        rows = []
        for rec in page.to_dict(orient="records"):
            rec.pop("kap_text", None)
            rec.pop("brief", None)
            for k, v in list(rec.items()):
                if isinstance(v, (np.floating, float)):
                    rec[k] = None if pd.isna(v) else float(v)
                elif isinstance(v, (np.integer,)):
                    rec[k] = int(v)
                elif pd.isna(v):
                    rec[k] = None
            rows.append(rec)
        return {"total": total, "offset": offset, "limit": limit, "items": rows}

    def paper_results(self, name: str | None = None) -> dict:
        folder = self.root / "results"
        files = {
            "public_benchmark": "public_benchmark.json",
            "agreement": "agreement.json",
            "vlm": "vlm_baselines.json",
            "vit": "vit_baseline.json",
            "kap_m3": "kap_m3_benchmark.json",
            "kap_minilm": "kap_embed_benchmark.json",
            "learned_gmu": "learned_gmu.json",
            "study_pilot": "study_model_pilot.json",
            "hop": "hop.json",
            "second_window": "second_window.json",
        }
        if name is None:
            pub = json.loads((folder / files["public_benchmark"]).read_text())
            return {
                "tables": list(files),
                "headline": {
                    "test_start": pub.get("test_start"),
                    "test_end": pub.get("test_end"),
                    "n_test": pub.get("n_test"),
                    "mean_fusion_acc": pub.get("forward", {}).get("mean", {}).get("acc"),
                    "mean_fusion_f1": pub.get("forward", {}).get("mean", {}).get("f1"),
                    "t1_coverage": pub.get("tiers", {}).get("T1", {}).get("coverage"),
                },
            }
        if name not in files:
            raise KeyError(name)
        return json.loads((folder / files[name]).read_text())

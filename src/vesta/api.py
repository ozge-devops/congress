"""VESTA HTTP API: time-budgeted BIST briefing over the public corpus."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from vesta import __version__
from vesta.engine import VestaEngine
from vesta.tiers import MIXERS, TIER_SPECS

STATIC = Path(__file__).resolve().parent / "static"
engine = VestaEngine()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        engine.load()
    except Exception as exc:  # pragma: no cover - surfaced via /health
        engine.ready = False
        engine.error = str(exc)
    yield


app = FastAPI(
    title="VESTA API",
    version=__version__,
    description=(
        "Time-budgeted multimodal briefing for BIST retail users. "
        "Public path: 16-d macro+KAP bag, VisualClaw, seed-0 score-space mixers. "
        "T1 may stay silent; T3 and T10 always emit templated tokens. "
        "Learned GMU is a paper ablation (GET /v1/results/learned_gmu)."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class BriefBody(BaseModel):
    date: str = Field(examples=["2025-05-27"])
    ticker: str = Field(default="XU100", examples=["THYAO.IS"])
    tier: Literal["T1", "T3", "T10"] = "T3"
    mixer: str = "gated"


def _need_ready() -> None:
    if not engine.ready:
        raise HTTPException(status_code=503, detail=engine.error or "engine not ready")


@app.get("/", include_in_schema=False)
def console():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok" if engine.ready else "loading_or_error",
        "ready": engine.ready,
        "version": __version__,
        "error": engine.error,
        "meta": engine.meta if engine.ready else None,
    }


@app.get("/v1/meta")
def meta():
    _need_ready()
    return {
        "version": __version__,
        "task": "next-day BIST100 direction",
        "mixers": list(MIXERS),
        "tiers": {
            k: {
                "minutes": v["minutes"],
                "label": v["label"],
                "gate_tau": v["gate_tau"],
                "declared_latency_s": v["latency_s"],
                "max_tokens": v["max_tokens"],
            }
            for k, v in TIER_SPECS.items()
        },
        "index": engine.meta,
        "tickers": engine.tickers(),
        "docs": "/docs",
    }


@app.get("/v1/dates")
def dates(split: Literal["all", "train", "val", "test"] = "all"):
    _need_ready()
    keys = engine.dates(split)
    return {"split": split, "n": len(keys), "dates": keys}


@app.get("/v1/tickers")
def tickers():
    _need_ready()
    return {"tickers": engine.tickers()}


def _brief(date: str, tier: str, mixer: str, ticker: str) -> dict:
    _need_ready()
    mixer = mixer.lower()
    try:
        return engine.brief(date=date, tier=tier, mixer=mixer, ticker=ticker)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"No XU100 sample on {exc.args[0]} (weekend, holiday, or outside the panel).",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/v1/brief")
def brief_get(
    date: str = Query(..., examples=["2025-05-27"]),
    ticker: str = "XU100",
    tier: Literal["T1", "T3", "T10"] = "T3",
    mixer: str = "gated",
):
    return _brief(date, tier, mixer, ticker)


@app.post("/v1/brief")
def brief_post(body: BriefBody):
    return _brief(body.date, body.tier, body.mixer, body.ticker)


@app.get("/v1/predict")
def predict(
    date: str = Query(..., examples=["2025-05-27"]),
    mixer: str = "gated",
):
    payload = _brief(date, "T3", mixer, "XU100")
    return {
        "date": date,
        "split": payload["split"],
        "mixer": mixer,
        "call": payload["call"],
        "p_up": payload["p_up"],
        "gate_g": payload["gate_g"],
        "mixers": payload["mixers"],
        "realized": payload["realized"],
    }


@app.get("/v1/mixers")
def mixers(date: str = Query(..., examples=["2025-05-27"])):
    payload = _brief(date, "T3", "gated", "XU100")
    return {
        "date": date,
        "split": payload["split"],
        "mixers": payload["mixers"],
        "realized": payload["realized"],
    }


@app.get("/v1/vision")
def vision(
    date: str = Query(..., examples=["2025-05-27"]),
    kind: Literal["screenshot", "tensor", "tokens"] = "screenshot",
):
    _need_ready()
    try:
        png = engine.vision_png(date, kind=kind)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"No XU100 sample on {exc.args[0]}.") from exc
    return Response(content=png, media_type="image/png")


@app.get("/v1/results")
def results_index():
    _need_ready()
    return engine.paper_results()


@app.get("/v1/results/{name}")
def results_table(name: str):
    _need_ready()
    try:
        return engine.paper_results(name)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail="Unknown table. Use public_benchmark, agreement, vlm, vit, kap_m3, kap_minilm, learned_gmu, study_pilot.",
        )


@app.get("/v1/events")
def events(
    ticker: str | None = None,
    date: str | None = None,
    has_kap: bool | None = None,
    limit: int = Query(25, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    _need_ready()
    return engine.search_events(
        ticker=ticker, date=date, has_kap=has_kap, limit=limit, offset=offset
    )


@app.get("/v1/events/{ticker}/{date}")
def event_one(ticker: str, date: str):
    _need_ready()
    row = engine.lookup_event(ticker, date)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No corpus row for {ticker} on {date}.")
    return row


def main() -> None:
    import uvicorn

    uvicorn.run("vesta.api:app", host="0.0.0.0", port=43187, reload=False)


if __name__ == "__main__":
    main()

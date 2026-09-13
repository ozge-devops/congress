"""HTTP API smoke tests against the public XU100 panel."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from vesta.api import app


def test_health_brief_vision_and_404():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["ready"] is True, body

        meta = client.get("/v1/meta")
        assert meta.status_code == 200
        assert "gated" in meta.json()["mixers"]
        assert "T1" in meta.json()["tiers"]

        dates = client.get("/v1/dates", params={"split": "test"})
        assert dates.status_code == 200
        assert "2025-05-27" in dates.json()["dates"]

        brief = client.get(
            "/v1/brief",
            params={"date": "2025-05-27", "tier": "T3", "mixer": "gated", "ticker": "XU100"},
        )
        assert brief.status_code == 200, brief.text
        payload = brief.json()
        assert payload["date"] == "2025-05-27"
        assert payload["silent"] is False
        assert payload["call"] in ("up", "down")
        assert "brief" in payload and payload["brief"]
        assert payload["mixers"]["gated"]["call"] in ("up", "down")

        t1 = client.get("/v1/brief", params={"date": "2025-05-27", "tier": "T1"})
        assert t1.status_code == 200
        assert t1.json()["tier"] == "T1"

        missing = client.get("/v1/brief", params={"date": "2010-01-01"})
        assert missing.status_code == 404

        vis = client.get("/v1/vision", params={"date": "2025-05-27", "kind": "tensor"})
        assert vis.status_code == 200
        assert vis.headers["content-type"].startswith("image/png")
        assert vis.content[:8] == b"\x89PNG\r\n\x1a\n"

        results = client.get("/v1/results")
        assert results.status_code == 200
        assert "headline" in results.json()
        assert "learned_gmu" in results.json()["tables"]
        gmu = client.get("/v1/results/learned_gmu")
        assert gmu.status_code == 200
        assert abs(gmu.json()["acc"]["mean"] - 0.524675) < 1e-4

        events = client.get("/v1/events", params={"ticker": "THYAO.IS", "limit": 3})
        assert events.status_code == 200
        assert events.json()["total"] > 0

        posted = client.post(
            "/v1/brief",
            json={"date": "2025-05-27", "ticker": "THYAO.IS", "tier": "T10", "mixer": "mean"},
        )
        assert posted.status_code == 200
        assert posted.json()["ticker"] == "THYAO.IS"
        assert posted.json()["mixer"] == "mean"
        assert posted.json()["call_scope"] == "XU100_index_mixer"
        assert posted.json()["brief_scope"] == "event_row"
        assert "THYAO.IS" in (posted.json()["brief"] or "")

        bimas = client.get(
            "/v1/brief",
            params={"date": "2018-06-14", "ticker": "BIMAS.IS", "tier": "T3"},
        )
        assert bimas.status_code == 200
        body = bimas.json()
        assert body["text_polarity"] == "bullish"
        assert "+6.33" in body["brief"]
        assert "Brief polarity bullish" in body["brief"]


if __name__ == "__main__":
    test_health_brief_vision_and_404()
    print("ok test_health_brief_vision_and_404")

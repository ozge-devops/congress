"""Download public KAP disclosure lists (no paid API key)."""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from html import unescape
from pathlib import Path

import pykap
import requests

KAP_URL = "https://www.kap.org.tr/tr/api/disclosure/members/byCriteria"
BODY_URL = "https://www.kap.org.tr/tr/api/notification/attachment-detail/{idx}"
HEADERS = {
    "Referer": "https://www.kap.org.tr/tr/bildirim-sorgu",
    "User-Agent": "vesta-research/0.2 (academic replication)",
    "Content-Type": "application/json",
}

NOISE_SUBJECTS = (
    "şirket genel bilgi formu",
    "pay bazında devre kesici",
    "borçlanma araçları, yatırım fonları ve varant itfa",
    "borsada işlem gören tipe dönüşüm",
    "sorumluluk beyanı",
    "ihraç tavanına ilişkin bildirim",
    "pay dışında sermaye piyasası aracı işlemlerine ilişkin bildirim",
)

# BIST tickers without .IS suffix
DEFAULT_TICKERS = [
    "THYAO",
    "PGSUS",
    "TAVHL",
    "GARAN",
    "AKBNK",
    "YKBNK",
    "ISCTR",
    "KCHOL",
    "SAHOL",
    "SISE",
    "ASELS",
    "TUPRS",
    "PETKM",
    "EREGL",
    "BIMAS",
    "MGROS",
    "TCELL",
    "TTKOM",
    "FROTO",
    "TOASO",
    "ARCLK",
    "AEFES",
    "ULKER",
    "SASA",
    "EKGYO",
    "ENKAI",
    "TTRAK",
]


def _oid_map() -> dict[str, str]:
    df = pykap.get_bist_companies()
    out = {}
    for _, row in df.iterrows():
        t = str(row["ticker"]).strip().upper()
        oid = str(row["company_id"]).strip()
        if t and oid:
            out[t] = oid
    return out


def _parse_publish(s: str) -> str | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        if len(s) >= 10 and s[2] == ".":
            return datetime.strptime(s[:10], "%d.%m.%Y").strftime("%Y-%m-%d")
        return datetime.strptime(s[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        return None


def _query(oid: str, start: date, end: date, tries: int = 4) -> list[dict]:
    body = {
        "fromDate": start.isoformat(),
        "toDate": end.isoformat(),
        "mkkMemberOidList": [oid],
        "subjectList": [],
        "inactiveMkkMemberOidList": [],
        "bdkMemberOidList": [],
        "fromSrc": False,
        "disclosureIndexList": [],
    }
    last_exc: Exception | None = None
    for attempt in range(tries):
        try:
            r = requests.post(KAP_URL, json=body, headers=HEADERS, timeout=45)
            if r.status_code >= 500:
                time.sleep(0.8 * (attempt + 1))
                last_exc = requests.HTTPError(f"{r.status_code} on {start}..{end}")
                continue
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list):
                return []
            return data
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            time.sleep(0.8 * (attempt + 1))
    if last_exc:
        raise last_exc
    return []


def _split_fetch(oid: str, start: date, end: date, cap: int = 2000) -> list[dict]:
    rows = _query(oid, start, end)
    if len(rows) < cap or (end - start).days <= 14:
        return rows
    mid = start + (end - start) / 2
    time.sleep(0.12)
    left = _split_fetch(oid, start, mid, cap)
    time.sleep(0.12)
    right = _split_fetch(oid, mid + timedelta(days=1), end, cap)
    return left + right


def fetch_ticker_range(ticker: str, oid: str, start: date, end: date) -> list[dict]:
    # Year-sized windows: the 8-year span returns HTTP 500 from KAP.
    raw: list[dict] = []
    y = start.year
    while y <= end.year:
        a = date(y, 1, 1)
        b = date(y, 12, 31)
        if a < start:
            a = start
        if b > end:
            b = end
        raw.extend(_split_fetch(oid, a, b))
        time.sleep(0.15)
        y += 1
    out = []
    seen = set()
    for item in raw:
        idx = item.get("disclosureIndex")
        if idx in seen:
            continue
        seen.add(idx)
        pub = _parse_publish(item.get("publishDate") or "")
        summary = (item.get("summary") or "").strip()
        subject = (item.get("subject") or "").strip()
        out.append(
            {
                "ticker": ticker,
                "date": pub,
                "publish_datetime": item.get("publishDate"),
                "disclosure_index": idx,
                "subject": subject,
                "summary": summary,
                "text": (f"{subject}. {summary}".strip() if summary else subject),
                "disclosure_class": item.get("disclosureClass"),
                "stock_codes": item.get("stockCodes"),
                "url": f"https://www.kap.org.tr/tr/Bildirim/{idx}" if idx else "",
            }
        )
    return [r for r in out if r["date"]]


def download_kap(
    cache_dir: Path,
    tickers: list[str] | None = None,
    start: date = date(2018, 1, 1),
    end: date | None = None,
) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "kap_disclosures.jsonl"
    if out_path.exists() and out_path.stat().st_size > 10_000:
        return [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]

    end = end or date.today()
    oids = _oid_map()
    tickers = tickers or DEFAULT_TICKERS
    all_rows: list[dict] = []
    for t in tickers:
        oid = oids.get(t)
        if not oid:
            print(f"  skip {t}: no KAP oid")
            continue
        print(f"  KAP {t} …", flush=True)
        try:
            rows = fetch_ticker_range(t, oid, start, end)
        except Exception as exc:
            print(f"  FAIL {t}: {exc}")
            continue
        print(f"    {len(rows)} filings")
        all_rows.extend(rows)
        time.sleep(0.2)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in all_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return all_rows


def inventory(rows: list[dict]) -> dict:
    """Counts for the public KAP list cache (subjects + teasers, not HTML bodies)."""
    from collections import Counter

    tickers = Counter(r["ticker"] for r in rows)
    dates = [r["date"] for r in rows if r.get("date")]
    indices = {r.get("disclosure_index") for r in rows}
    return {
        "source": "https://www.kap.org.tr/tr/api/disclosure/members/byCriteria",
        "fields": "subject + summary teaser (list API, not paid disclosureDetail HTML)",
        "n_filings": len(rows),
        "n_unique_disclosure_index": len(indices),
        "n_tickers": len(tickers),
        "date_start": min(dates) if dates else None,
        "date_end": max(dates) if dates else None,
        "by_ticker": dict(tickers.most_common()),
    }


def is_noise_subject(subject: str) -> bool:
    s = (subject or "").lower()
    return any(n in s for n in NOISE_SUBJECTS)


def strip_kap_html(html: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", html or "", flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[[A-Z0-9_]+\]", " ", text)
    text = re.sub(r"[A-Za-z0-9_.]+\|", " ", text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:4000]


def fetch_one_body(idx: int, tries: int = 3) -> dict:
    headers = {
        **HEADERS,
        "Referer": f"https://www.kap.org.tr/tr/Bildirim/{idx}",
    }
    last_exc: Exception | None = None
    for attempt in range(tries):
        try:
            r = requests.get(BODY_URL.format(idx=idx), headers=headers, timeout=25)
            if r.status_code >= 500:
                time.sleep(0.6 * (attempt + 1))
                last_exc = requests.HTTPError(f"{r.status_code} body {idx}")
                continue
            if r.status_code == 404:
                return {"disclosure_index": idx, "body_text": "", "ok": False}
            r.raise_for_status()
            data = r.json()
            item = data[0] if isinstance(data, list) and data else data
            html_parts = item.get("disclosureBody") or []
            html = " ".join(str(p) for p in html_parts if p)
            basic = ((item.get("disclosure") or {}).get("disclosureBasic") or {})
            return {
                "disclosure_index": idx,
                "title": basic.get("title") or "",
                "summary": basic.get("summary") or "",
                "body_text": strip_kap_html(html),
                "ok": True,
            }
        except (requests.RequestException, ValueError, IndexError, TypeError) as exc:
            last_exc = exc
            time.sleep(0.6 * (attempt + 1))
    return {"disclosure_index": idx, "body_text": "", "ok": False, "error": str(last_exc)}


def load_bodies(cache_dir: Path) -> dict[int, dict]:
    path = cache_dir / "kap_bodies.jsonl"
    out: dict[int, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        idx = row.get("disclosure_index")
        if idx is not None:
            out[int(idx)] = row
    return out


def download_bodies(
    cache_dir: Path,
    indices: list[int],
    max_workers: int = 8,
    retry_empty: bool = False,
) -> dict[int, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "kap_bodies.jsonl"
    have = load_bodies(cache_dir)
    todo = []
    seen: set[int] = set()
    for raw in indices:
        i = int(raw)
        if i in seen:
            continue
        seen.add(i)
        row = have.get(i)
        if row is None:
            todo.append(i)
            continue
        if retry_empty and not (row.get("ok") and (row.get("body_text") or "").strip()):
            if row.get("ok") is False and not row.get("error"):
                continue  # explicit 404
            todo.append(i)
    if not todo:
        return have
    print(f"  KAP bodies: {len(have)} cached, {len(todo)} to fetch", flush=True)
    lock = threading.Lock()

    def _write(row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line)

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(fetch_one_body, idx): idx for idx in todo}
        for fut in as_completed(futs):
            row = fut.result()
            have[int(row["disclosure_index"])] = row
            _write(row)
            done += 1
            if done % 200 == 0 or done == len(todo):
                ok = sum(1 for r in have.values() if r.get("ok") and r.get("body_text"))
                print(f"    bodies {done}/{len(todo)} ({ok} non-empty in cache)", flush=True)
    return have


def index_by_ticker_date(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    bag: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        key = (r["ticker"] + ".IS", r["date"])
        bag.setdefault(key, []).append(r)
        # also bare ticker
        bag.setdefault((r["ticker"], r["date"]), []).append(r)
    return bag

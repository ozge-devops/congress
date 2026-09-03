"""Parse DePlot / MatCha chart-to-table dumps into an up/down call."""

from __future__ import annotations

import re


def parse_deplot(text: str) -> int | None:
    """Return 1 if the extracted series ends above its start, else 0.

    DePlot sometimes swaps the header to ``price | bar`` while still emitting
    ``bar_index | price`` rows. Use the column whose values look like BIST
    index levels (thousands), and skip the bar-index column.
    """
    if not text:
        return None
    cleaned = text.replace(",", " ").replace("<0x0A>", "\n")
    pairs = re.findall(r"(\d+\.?\d*)\s*[|\t]\s*(\d+\.?\d*)", cleaned)
    col0 = [float(a) for a, _ in pairs]
    col1 = [float(b) for _, b in pairs]
    prices = [v for v in col1 if v >= 100]
    if len(prices) < 2:
        prices = [v for v in col0 if v >= 100]
    if len(prices) < 2:
        nums = [float(x) for x in re.findall(r"\d{3,6}(?:\.\d+)?", cleaned)]
        prices = [v for v in nums if v >= 100]
    if len(prices) < 2:
        return None
    if abs(prices[-1] - prices[0]) < 1e-6:
        return None
    return int(prices[-1] > prices[0])


def parse_number(text: str) -> float | None:
    nums = re.findall(r"\d+\.?\d*", (text or "").replace(",", ""))
    vals = [float(x) for x in nums if float(x) > 0]
    return vals[-1] if vals else None


def parse_matcha_pair(first: str, last: str) -> int | None:
    a, b = parse_number(first), parse_number(last)
    if a is None or b is None or a == b:
        return None
    return int(b > a)

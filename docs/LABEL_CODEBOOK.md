# VESTA-Public label codebook (rule_v1)

Silver labels for the public replication corpus in `data/vesta_public/`.
The labels are deterministic functions of public OHLCV and KAP list text.

## Files

| File | What it is |
|------|------------|
| `events.parquet` / `events.csv` | Full event table (37,046 rows) including 40-bar OHLC windows |
| `events_10k.parquet` / `events_10k.csv` | Paper-sized slice: every index day + largest constituent moves |
| `label_stats.json` / `label_stats_10k.json` | Class counts, KAP linkage, date range |
| `kap_inventory.json` | 39,890 public KAP list filings (27 tickers) |
| `human_annotation_sample.csv` | 250 stratified rows; annotator columns are codebook B (`annotator_id=codebook_b`) |

Index series: every XU100 session after the lookback. The public chart is the
40 sessions **strictly before** day \(t\); tabular features and `y_leak_vol`
are computed at \(t\). Constituents: a day is
kept if `|session return| ≥ 2%`, or the diagnostic vol flag fires, or
RSI ≥ 75 / ≤ 25, **or a KAP list item published that calendar date**.
Chronological split is by **calendar date** (70 / 15 / 15), so the same day
never sits in two splits.

## KAP columns (public list API)

Source: `POST https://www.kap.org.tr/tr/api/disclosure/members/byCriteria`
(year-sized windows; an 8-year query returns HTTP 500). Cached at
`data/cache/kap_disclosures.jsonl`.

| Column | Meaning |
|--------|---------|
| `n_kap` | Number of list items for that ticker on that calendar date (XU100 rows pool **all** 27 names that day) |
| `has_kap` | `n_kap > 0` |
| `kap_subjects` | Unique subjects, ` \|\| `-joined |
| `kap_text` | Up to eight `subject. summary` teasers |
| `kap_indices` | Up to 12 KAP disclosure indices (URL: `https://www.kap.org.tr/tr/Bildirim/{index}`) |
| `kap_body` | Stripped HTML from the public attachment-detail API when cached |
| `n_kap_bodies` | How many of those indices have a body in cache |
| `kap_daily_features.csv` | Per-calendar-day 8-d mixer features (index-level bag) |

These are list teasers from kap.org.tr. Lexicon hits on `kap_text` / `kap_body` feed
`kap_polarity` and the mixed `text_polarity`; they do not enter `y_direction_*`.

## KAP-only silver sentiment (original text axis)

**`kap_polarity`** ∈ `{bullish, bearish, neutral}`

Score KAP subjects + teasers (+ HTML body when cached). No USD/TRY, gold, RSI,
or future return. Subject priors: dividend / buyback / bonus issue tilt bullish;
probe / fine / lawsuit tilt bearish. Thresholds: `≥ 1` bullish, `≤ −1` bearish.

Full-panel counts: bullish 7,172; bearish 3,742; neutral 26,132. Most KAP days
are routine filings, so neutral is the honest majority. HTML bodies are cached
for 25,734 filings; 15,212 event rows have a stripped `kap_body`.

## Primary labels (do not leak from the chart)

| Column | Meaning |
|--------|---------|
| `y_direction_1d` | `1` iff next-session simple return `> 0` |
| `y_direction_5d` | same, five sessions ahead |
| `y_excess_1d` | next return above the trailing 20-day mean return |

These columns are not recoverable from today's OHLCV by a closed-form rule.

## Diagnostic label (function of the tabular vector at t)

| Column | Meaning |
|--------|---------|
| `y_leak_vol` / `anomaly_flag` | 20-day realized vol `>` mean + 2σ of the prior 60 vol observations |

Train the headline model on `y_direction_*`. Report `y_leak_vol` only as the
closed-form sanity check (rule = 100% F1).

## Silver perception labels (original three axes)

**`text_polarity`** ∈ `{bullish, bearish, neutral}`

Score the *brief*, not the future return:

- USD/TRY up (lira weaker) → bearish; down → bullish
- Gold up → risk-off (bearish); down → constructive
- Session return magnitude
- RSI ≥ 70 / ≤ 30 as a small tilt
- Yahoo headline lexicon if a same-calendar-day title exists
- KAP list subject/teaser lexicon (Turkish + English) if a filing published that day

Thresholds: `≥ 1` bullish, `≤ −1` bearish, else neutral.

**`chart_signal`** ∈ `{breakout, support_hold, divergence, none}`

Computed on the last bar of the 40-day window, using **prior** bars only:

1. **divergence** if a 10-bar price higher-high (lower-low) is not matched by RSI
2. else **breakout** if close prints a 20-bar high
3. else **support_hold** if the low tags the 20-bar low and the bar recovers
4. else **none**

Priority is the order above. This axis *is* a function of the visible chart,
which is what VisualClaw was originally scored on.

## Codebook B on the annotation sample

`human_annotation_sample.csv` annotator columns are filled by **codebook B**
(`annotator_id=codebook_b`): subject-title taxonomy for text, 10-bar VisualClaw
for chart. Three-codebook κ is in
[`docs/AGREEMENT.md`](AGREEMENT.md) and `results/agreement.json`.

A later human pass should overwrite `annotator_id` with initials and should
not look at `y_direction_1d`.

## Scope

KAP text is public list teasers plus a cached HTML-body subset.
Yahoo headlines cover 41 event rows (recent scrape).
Polarity is silver codebook output; this package has no three-annotator gold set.

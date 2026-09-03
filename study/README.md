# User-study materials

This folder is the executable protocol for §6 of `paper/vesta.tex`.
`responses.csv` is a header only; no participant answers have been collected.

## Design

- 12 retail-account holders, within-subject.
- 8 BIST scenarios in `scenarios.json` (drawn from the public test window).
- Three time caps: 1 / 3 / 10 minutes.
- Three information conditions: raw KAP+OHLC feed, unimodal (text or chart only), VESTA tier matching the cap.
- Latin-square the condition order.

## Primary outcomes

1. Direction call (up / down / abstain) vs the subsequent week's realized move (`study/gold.json`, experimenter-only).
2. Time-to-decision (seconds).
3. NASA-TLX (six 0-20 items in `instrument.csv`).
4. Trust (7-point).

## How to run

Interactive (one participant; writes `responses.csv`):

```bash
PYTHONPATH=src python experiments/run_study.py p01
```

Model dry-run (NASA-TLX left blank):

```bash
PYTHONPATH=src python experiments/simulate_study.py
```

The file `pilot_model_responses.csv` is a model dry-run on the eight
sealed scenarios (real week-direction gold). The public mixer is XU100-level:

| condition | participant_id | scored on | week hit |
|-----------|----------------|-----------|----------|
| raw (session-return sign) | vesta_raw_pilot | 8 names | 4/8 = 0.5 |
| unimodal text MLP | vesta_text_pilot | 2 XU100 weeks | 0/2 |
| gated VESTA | vesta_gated_pilot | 2 XU100 weeks | 1/2 |

NASA-TLX is blank. Constituent scenarios are abstains for the index mixer.
Do not copy this file into `responses.csv`.

1. Consent: `consent.md`.
2. Show one scenario at a time from `scenarios.json`. Keep `gold.json` sealed.
3. Log answers in `responses.csv` (header only until the study is run).
4. Score with `PYTHONPATH=src python experiments/score_study.py` once responses exist.

`gold.json` is sealed: it contains next-week return signs used only after the session.

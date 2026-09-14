# VESTA

Evaluation protocol for time-budgeted multimodal briefing for BIST retail users.

Peri Gunes (Infina Software), Ozge Zelal Kucuk (Istanbul Aydin University),
Harun Benli (Infina Software).

CoMeSySo 2026 manuscript and replication package.
Public repository: [github.com/ozge-devops/congress](https://github.com/ozge-devops/congress).
Commit `c03627c` is an earlier LNCS draft (title *Time-Budgeted Multimodal Agents*,
Agentic RAG keywords, merged McNemar `p>0.46`). Current `main` matches this PDF.

Camera-ready PDF: [paper/vesta.pdf](paper/vesta.pdf).
Source: [paper/vesta.tex](paper/vesta.tex).
Word (optional): `bash paper/export_docx.sh` (pandoc).

## HTTP API

A FastAPI service over the public corpus: next-day XU100 call, T1/T3/T10
payload, VisualClaw PNG, KAP event lookup, and the paper JSON tables.

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m vesta.api
```

Then open `http://127.0.0.1:43187/` (console) or `/docs` (OpenAPI).

| Method | Path | What it does |
| --- | --- | --- |
| GET | `/health` | Ready flag and panel dates |
| GET | `/v1/meta` | Mixers, tiers, tickers |
| GET | `/v1/dates?split=test` | Session calendar |
| GET | `/v1/brief?date=2025-05-27&tier=T3&mixer=gated&ticker=THYAO.IS` | Briefing |
| POST | `/v1/brief` | Same, JSON body |
| GET | `/v1/predict?date=2025-05-27` | Direction only |
| GET | `/v1/mixers?date=2025-05-27` | All mixer scores |
| GET | `/v1/vision?date=2025-05-27&kind=screenshot` | PNG (`screenshot`, `tensor`, `tokens`) |
| GET | `/v1/events?ticker=THYAO.IS` | Silver-labeled KAP rows |
| GET | `/v1/results` | Paper table index |
| GET | `/v1/results/public_benchmark` | Leak, forward, tier, overlay JSON |
| GET | `/v1/results/learned_gmu` | Learned Arevalo GMU ($W_v,W_t,W_z$) |
| GET | `/v1/results/hop` | M3 day-level hop ($\alpha=1$, no portfolio) |
| GET | `/v1/results/second_window` | 2023 chronological holdout on XU100 |
| GET | `/v1/results/study_pilot` | Model dry-run on the sealed study gold |
| GET | `/v1/results/vlm` | Zero-shot DePlot / MatCha JSON |
| GET | `/v1/tickers` | Public panel tickers |

T1 stays silent when `|g-0.5| ≤ 0.18`. Mixers are seed-0 sklearn fits on the
chronological train+val split. GitHub Actions runs `tests/test_api.py`.

VESTA has three layers. DataClaw0 runs an M3 day-level hop on cached KAP
vectors ($\alpha=1$, no portfolio; analogue hit 49.8% on 305 test days).
The hop does not enter the 16-d mixer. VisualClaw reads a 40-bar candlestick (bars \(t-40,\ldots,t-1\))
as a \(24\times24\) image. Score-space mixers combine the two streams
(GMU, tensor fusion, 1-layer attention, mean, concat). Learned GMU is a paper
ablation (`GET /v1/results/learned_gmu`). The Temporal Orchestration Layer
spends a 1-, 3- or 10-minute budget (T1 / T3 / T10) by changing how much
text is shown; the mixer stays fixed.

The headline task is next-day BIST100 direction. A \(2\sigma\) realized-volatility
flag on the tabular vector at day \(t\) is a diagnostic: a closed-form rule
scores 100% F1; the chart, which omits today's candle, scores 58.2% F1.
Actionability is a long/short overlay against buy-and-hold. Coverage, declared
latency, tokens, and length-normalized information noise are reported per tier.

Label codebook: [docs/LABEL_CODEBOOK.md](docs/LABEL_CODEBOOK.md).
Agreement: [docs/AGREEMENT.md](docs/AGREEMENT.md).
Bibliography check: [docs/BIBLIOGRAPHY_AUDIT.md](docs/BIBLIOGRAPHY_AUDIT.md).

## Data

`data/vesta_public/` is the silver-labeled panel: 37,046 events, 27 BIST names
plus XU100, 24 May 2018 to 19 August 2026. The canonical table is
`events.parquet` (the 55 MB `events.csv` export is gitignored). Splits are
chronological by calendar date (70/15/15). 19,645 rows have a same-day KAP
list teaser; 15,212 have a cached HTML body from kap.org.tr. Public BGE-M3
in the forward table uses the same list text.

Rebuild from Yahoo + KAP if you need to:

```bash
pip install -r requirements.txt
PYTHONPATH=src python experiments/build_corpus.py
PYTHONPATH=src python experiments/enrich_corpus.py
PYTHONPATH=src python tests/test_labeling.py
```

## Tables in the paper

The JSON under `results/` is copied into the LaTeX tables. To regenerate:

```bash
PYTHONPATH=src python experiments/run_public_benchmark.py   # leak, forward, tiers, overlay
PYTHONPATH=src python experiments/run_learned_gmu.py        # learned Arevalo GMU
PYTHONPATH=src python experiments/run_hop.py                # M3 day-level hop
PYTHONPATH=src python experiments/run_second_window.py      # 2023 holdout on the same index
PYTHONPATH=src python experiments/label_agreement.py
PYTHONPATH=src python experiments/embed_kap.py              # MiniLM
PYTHONPATH=src python experiments/embed_kap_m3.py           # public BGE-M3
PYTHONPATH=src python experiments/run_vit_baseline.py       # frozen ViT-B/16
PYTHONPATH=src python experiments/run_vlm_baselines.py --every 1 --models deplot
PYTHONPATH=src python experiments/make_figures.py
PYTHONPATH=src python tests/test_paper_consistency.py
```

On the public test window (27 May 2025 to 19 August 2026, 308 index days):

- No score-space mixer beats score-space GMU on next-day direction
  (minimum McNemar \(p=0.46\) on seed 0). Mean fusion is the five-seed
  point estimate: 53.0% accuracy / 52.4% macro-F1. Learned Arevalo GMU
  is \(52.5\pm0.5\) / \(40.3\pm4.4\) (seed-0 McNemar \(p=0.29\)).
- Proxy accuracy vs next-day direction: macro flags 54.9%, vision
  \(52.9\pm2.3\)%, tabular \(50.6\pm0.0\)%, KAP polarity 49.7%. Codebook
  Fleiss \(\kappa=0.50\) on the 10k slice; chart A/B Cohen \(\kappa=0.52\).
- T1 covers 32.5% of days on seed 0. \(\mathrm{IN}_{\mathrm{new}}\) is
  60.3 / 82.1 / 93.6% on templated token bags.
- Buy-and-hold Sharpe in that window is 1.69; every seed-0 overlay is lower.
- A second chronological cut (train before 2023-01-01, test 2023, n=248)
  is `results/second_window.json`. Mean fusion is 53.3±1.7 against majority
  51.2. Seed-0 vision overlay Sharpe is 1.81 versus buy-and-hold 0.99.
  That does not reverse the 2025-2026 bull table.

DePlot / MatCha are zero-shot on all 308 test screenshots (no fine-tune).
They need the optional CPU torch stack (`requirements-vlm.txt`) and are slow.

Python 3.10 is enough for the sklearn tables. First run downloads XU100, USDTRY
and gold into `data/cache/`; after that it stays offline.

## Paper (CoMeSySo 2026)

The camera-ready PDF is Springer LNNS using `llncs.cls`. The body follows the
CoMeSySo CFP: Introduction, Methods, Results, Discussions. Track fit is
Econometrics, Computational Intelligence, and Software Engineering in
Intelligent Systems. Submission deadline (extended): 25 September 2026.

```bash
cd paper
pdflatex vesta.tex && bibtex vesta && pdflatex vesta.tex && pdflatex vesta.tex
```

`llncs.cls` is Springer LNCS/LNNS 2.26. For Word: `bash paper/export_docx.sh`
(needs pandoc). For the OpenPublish ZIP (tex + figures + PDF):

```bash
bash paper/pack_comesyso.sh
```

## Hugging Face Space

`app.py` is a Gradio console on the same public engine (date, ticker, T1/T3/T10,
mixer, hop, 40-bar chart).

```bash
pip install -r requirements.txt
PYTHONPATH=src python app.py
```

Open `http://127.0.0.1:7865`. Public viewer:
[huggingface.co/spaces/ozgezelal/vesta](https://huggingface.co/spaces/ozgezelal/vesta).
New Gradio Spaces on Hugging Face cpu-basic require a PRO plan.

## What is still missing

A three-annotator gold set, a 12-investor NASA-TLX study (`study/` has the
protocol; `responses.csv` is a header only), full KAP disclosure-detail pages,
and a GPU Pix2Struct/MatCha fine-tune. The 2023 holdout is in the paper; it is
not a second exchange. The forward table still has score-space mixers;
the learned-GMU row fits $W_v,W_t,W_z$ on the raw streams and does not beat
score-space GMU. The public chart window is bars \(t-40,\ldots,t-1\); the
tabular features and the vol diagnostic are at day \(t\). Human NASA-TLX
responses are not claimed.

Questions: pgunes@infina.com.tr, hbenli@infina.com.tr, ozelalkucuk@stu.aydin.edu.tr

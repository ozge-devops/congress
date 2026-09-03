# VESTA

Time-budgeted multimodal briefing for BIST retail users.

Peri Gunes (Infina Software), Ozge Zelal Kucuk (Istanbul Aydin University),
Harun Benli (Infina Software).

Camera-ready PDF: [paper/vesta.pdf](paper/vesta.pdf).
Source: [paper/vesta.tex](paper/vesta.tex). Word is [paper/vesta.docx](paper/vesta.docx)
if you cannot open LaTeX; the LNCS page breaks are only in the PDF.

VESTA has three layers. DataClaw0 retrieves portfolio-conditioned KAP list
text. VisualClaw reads a 40-bar candlestick (bars \(t-40,\ldots,t-1\)) as a
\(24\times24\) image. A standard sigmoid mixer combines the two streams and is
compared with GMU, tensor fusion, a MulT-style block, mean, and concat. The
Temporal Orchestration Layer spends a 1-, 3-, or 10-minute budget (T1 / T3 /
T10) by changing chain depth, not the mixer.

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
plus XU100, 24 May 2018 to 19 August 2026. Splits are chronological by calendar
date (70/15/15). 19,645 rows have a same-day KAP list teaser; 15,212 have a
cached HTML body from kap.org.tr. Public BGE-M3 in Table 2 uses the same list
text (`public_m3_not_infina`).

Rebuild from Yahoo + KAP if you need to:

```bash
pip install -r requirements.txt
PYTHONPATH=src python experiments/build_corpus.py
PYTHONPATH=src python experiments/enrich_corpus.py
PYTHONPATH=src python tests/test_labeling.py
```

## Tables in the paper

The JSON under `results/` is what we copied into the LaTeX tables. To regenerate:

```bash
PYTHONPATH=src python experiments/run_public_benchmark.py   # Tables 1-4
PYTHONPATH=src python experiments/label_agreement.py
PYTHONPATH=src python experiments/embed_kap.py              # MiniLM
PYTHONPATH=src python experiments/embed_kap_m3.py           # public BGE-M3
PYTHONPATH=src python experiments/run_vit_baseline.py       # frozen ViT-B/16
PYTHONPATH=src python experiments/run_vlm_baselines.py --every 1 --models deplot
PYTHONPATH=src python experiments/make_figures.py
PYTHONPATH=src python tests/test_paper_consistency.py
```

On the public test window (27 May 2025 to 19 August 2026, 308 index days):

- No mixer beats GMU on next-day direction (McNemar \(p > 0.46\) on seed 0).
  Mean fusion is the point estimate: 53.0% accuracy / 52.4% macro-F1.
- Proxy accuracy vs next-day direction: macro flags 54.9%, KAP polarity 49.7%,
  vision \(52.9\pm2.3\)%, tabular \(50.6\pm0.0\)%. Codebook Fleiss \(\kappa=0.50\)
  on the 10k slice; chart A/B Cohen \(\kappa=0.52\).
- T1 covers 32.5% of days on seed 0. \(\mathrm{IN}_{\mathrm{new}}\) is 60-94%
  on templated token bags.
- Buy-and-hold Sharpe in that window is 1.69; every seed-0 overlay is lower.

DePlot / MatCha are zero-shot on all 308 test screenshots (no fine-tune).
They need the optional CPU torch stack (`requirements-vlm.txt`) and are slow.

Python 3.10 is enough for the sklearn tables. First run downloads XU100, USDTRY
and gold into `data/cache/`; after that it stays offline.

## Paper

```bash
cd paper
pdflatex vesta.tex && bibtex vesta && pdflatex vesta.tex && pdflatex vesta.tex
```

`llncs.cls` in that folder is Springer LNCS 2.26. For Word: `bash paper/export_docx.sh`
(needs pandoc).

## What is still missing

A three-annotator gold set, a 12-investor NASA-TLX study (`study/` has the
protocol; `responses.csv` is a header only), Infina KAP bodies, and a GPU
Pix2Struct/MatCha fine-tune. Table 2 mixers fuse unimodal scores rather than
a learned \(W_g\). The public chart window is bars \(t-40,\ldots,t-1\); the
tabular features and the vol diagnostic are at day \(t\).

Questions: ozelalkucuk@stu.aydin.edu.tr, pgunes@infina.com.tr, hbenli@infina.com.tr.

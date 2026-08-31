# VESTA

Peri Gunes (Infina Software),Harun Benli (Infina Software),Ozge Zelal Kucuk (Istanbul Aydin University),

Camera-ready PDF:[paper/vesta.pdf](paper/vesta.pdf)  
LaTeX:[paper/vesta.tex](paper/vesta.tex)  
Word (layout is not LNCS):[paper/vesta.docx](paper/vesta.docx)

This is the public code and BIST/KAP data for our LNCS revision after AIxIA 2026
(submission 84).Contact:pgunes@infina.com.tr,hbenli@infina.com.tr.
,ozelalkucuk@stu.aydin.edu.tr

## What changed after review

The 92.4% “anomaly” F1 is withdrawn.That label is a 2σ realized-volatility
flag on the *tabular* vector at day t(closed-form rule 100% F1;tabular MLP
96.8% accuracy).The public chart is bars t−40…t−1,so vision on the same
flag is 58.2% F1 — lag-1 recovery, not anomaly detection.We keep the flag
as `y_leak_vol`(Table 1).The task in the paper is next-day BIST100
direction, `y_direction_1d`.

Cross-Modal Gated Fusion is not a contribution. Table 2 runs GMU, tensor
fusion, a MulT-style block, mean, concat, a score-space scalar gate, and a
tabular OHLCV baseline. On the public test window (27 May 2025 – 19 August
2026, 308 index days) none of them beats GMU (McNemar p > 0.46 on seed 0).
Mean fusion is the point estimate,53.0%. Mean vs gate is p = 1.00 on that
seed. We dropped the sentence that mean fusion “collapses toward the weaker
modality”.

Sentiment / technical proxies are now a table: macro flags 54.9% vs next-day
direction(one split,no seed CI),KAP polarity 49.7%, vision 52.9±2.3%,
tabular 50.6±0.0%. Codebook agreement on the 10k slice is Fleiss κ = 0.50,
not the 0.81 we had claimed with three human annotators.There are no three
human annotators in this package.

The 1 / 3 / 10 minute layers are in Table 3 (coverage, declared latency,
tokens, old and new information-noise).The revised noise index uses
delivered length (Sun et al., 2019).T1 covers 32.5% of days on seed 0.
Table 4 is a long/short overlay against buy-and-hold.In that bull window
buy-and-hold Sharpe is 1.69;every overlay we tried is worse.The 12-person
NASA-TLX study is specified in `study/` and has not been run.
`responses.csv` is still only a header.

Every entry in `paper/vesta.bib` has a DOI or arXiv id.The TKDE 2023,
NeurIPS 2024 hallucination, CVPR 2025,Central Bank Review, Infina
whitepaper,and ACL 2023 “CMGF” items are gone. Checklist:
[docs/BIBLIOGRAPHY_AUDIT.md](docs/BIBLIOGRAPHY_AUDIT.md).Point-by-point map:
[docs/REVIEWER_MAP.md](docs/REVIEWER_MAP.md). Product screenshots are out.
Fig. 3 (tiers) reads left to right.

## Data

`data/vesta_public/` is the silver-labeled panel:37,046 events,27 BIST names
plus XU100, 24 May 2018 – 19 August 2026.Splits are chronological by calendar
date (70/15/15). 19,645 rows have a same-day KAP list teaser;15,212 have a
cached HTML body.This is the public KAP website,not Infina’s production HTML.Public BGE-M3 in Table 2 is the same list text (`public_m3_not_infina`).

```bash
pip install -r requirements.txt
PYTHONPATH=src python experiments/build_corpus.py
PYTHONPATH=src python experiments/enrich_corpus.py
PYTHONPATH=src python tests/test_labeling.py
```

Label codebook:[docs/LABEL_CODEBOOK.md](docs/LABEL_CODEBOOK.md).  
Agreement:[docs/AGREEMENT.md](docs/AGREEMENT.md).

## Reproduce the tables

The JSON under `results/` is what we copied into the LaTeX tables.

```bash
PYTHONPATH=src python experiments/run_public_benchmark.py
PYTHONPATH=src python experiments/label_agreement.py
PYTHONPATH=src python experiments/embed_kap.py
PYTHONPATH=src python experiments/embed_kap_m3.py
PYTHONPATH=src python experiments/run_vit_baseline.py
PYTHONPATH=src python experiments/run_vlm_baselines.py --every 1 --models deplot
PYTHONPATH=src python experiments/make_figures.py
PYTHONPATH=src python tests/test_paper_consistency.py
```

DePlot / MatCha are zero-shot on all 308 test screenshots.They need the
optional CPU torch stack (`requirements-vlm.txt`)and are slow.

Python 3.10 is enough for the sklearn tables. The first run downloads XU100,
USDTRY and gold into `data/cache/`.

## Paper

```bash
cd paper
pdflatex vesta.tex && bibtex vesta && pdflatex vesta.tex && pdflatex vesta.tex
```

`llncs.cls` is Springer LNCS 2.26.Word:`bash paper/export_docx.sh` (pandoc).

## What is still missing

Three human annotators,a 12-investor study,Infina KAP bodies,and a GPU
Pix2Struct/MatCha fine-tune.The public chart window is bars t−40…t−1; the
tabular features and the vol diagnostic are at day t.Mixer rows in Table 2
fuse unimodal scores,not a learned Wg.

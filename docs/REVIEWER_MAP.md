# Referee map (AIxIA 2026, submission 84)

How each numbered request was handled in the LNCS revision (`paper/vesta.tex`).

| # | Request | Where it is answered |
|---|---------|----------------------|
| 1 | Remove the LNCS template sentence from the abstract (H2, H4) | Abstract now starts at “In high-volatility emerging markets…”. |
| 2 | Verify every reference (H3, H4) | New `.bib` built from ACL Anthology, PMLR, Springer, ACM, OpenReview, Journal of Finance. Audit in `docs/BIBLIOGRAPHY_AUDIT.md`. Invented venues (fake TKDE 2023, fake NeurIPS 2024 hallucination paper, fake CVPR 2025, fake Central Bank Review, Infina “whitepaper”, fake ACL 2023 CMGF) are gone. |
| 3 | Repository link (H4) | https://github.com/ozge-devops/congress (public `main`). |
| 4 | Un-mirror Fig. 4 (H4) | Old Fig. 4 is deleted. New Fig. 3 (`tiers.pdf`) is drawn left-to-right. Product screenshots (old Figs. 2 and 5) are gone; H4 called them illegible marketing. |
| 5 | Move Sentiment / Technical accuracy into a numeric table with CIs (H4) | New Table (proxy accuracy vs \(y_{t+1}\)): macro 54.9%, KAP 49.7% (single-split rules, no seed CI), vision \(52.9\pm2.3\), tabular \(50.6\pm0.0\). Codebook Fleiss κ=0.50 (10k); not the withdrawn 0.81 human figure. |
| 6 | Report the ablation numbers (H4) | Table 2. Mean fusion macro-F1 \(52.4\pm2.3\) vs scalar gate \(51.4\pm1.8\), seed-0 McNemar \(p=1.00\). The sentence “mean fusion collapses toward the weaker modality” is withdrawn. |
| 7 | External baseline (H1, H2, H4) | **Run:** GMU, TFN, CPU MulT-style, tabular OHLCV, closed-form vol rule, MiniLM and public BGE-M3 KAP embeddings (`public_m3_not_infina`), frozen ViT-B/16 CLS probe, zero-shot DePlot and MatCha-ChartQA on **all 308** test screenshots. Not fine-tuned. Mixers are score-space + MLP, not learned \(W_g\). |
| 8 | Measure the 1/3/10-minute layers (H2, H4) | Table 3: coverage, *declared* latency, accuracy among emitted calls, tokens, \(\mathrm{IN}_{\mathrm{old}}\), \(\mathrm{IN}_{\mathrm{new}}\) on templated filler bags. T1 covers 32.5% (seed 0). |
| 9 | Decision quality / user study (H2, H4) | Table 4 is a seed-0 paper-trading overlay. `study/` holds consent, eight sealed scenarios, NASA-TLX instrument, empty `responses.csv`. Model pilot: raw 4/8 names; index mixers only on 2 XU100 weeks (text 0/2, gated 1/2). NASA-TLX blank. Not 12 humans. |
| 10 | Do not claim CMGF as a contribution (H4) | Title no longer contains “via Cross-Modal Gated Fusion”. Abstract and §3.3 call Eqs. (3)-(4) a standard sigmoid mixer. Closest citations: Arevalo et al. GMU and Jiang & Ji. |
| 11 | Label leakage (H4) | §4.1 + Table 1. Closed-form rule = 100% F1. Tabular MLP = 96.8% acc. Vision = 58.2% F1. Primary task replaced by next-day direction. Jiang, Kelly & Xiu is cited *together with* the tabular baseline, as requested. |
| 12 | Information-noise denominator (H2) | Eq. (5): \(\mathrm{IN}_{\mathrm{new}}=N_{\mathrm{irrel}}/N_{\mathrm{del}}\), citing Sun et al. 2019. Table 3 shows \(\mathrm{IN}_{\mathrm{old}}<7\%\) while \(\mathrm{IN}_{\mathrm{new}}\) is 60-94%. |

## Scores that were withdrawn

- 92.4% anomaly F1
- 95.9% “noise reduction” computed as \(1-\mathrm{IN}_{\mathrm{old}}\)
- “CMGF is the core innovation”
- Qualitative-only 1/3/10-minute claims

## On Dietterich (1998) and Harvey, Liu, Zhu (2016)

Included. Chronological holdout ⇒ McNemar on seed 0, not a paired \(t\)-test on five random seeds (Dietterich). Thirteen public rows ⇒ we do not declare a winner after an uncorrected hunt (Harvey et al.). Mixer-vs-GMU \(p\)-values (minimum 0.46) and the multiple-testing caution are in §4.3.

## What we could not do in this revision pass

- Three *human* annotators. Three independent *codebooks* plus a pretrained multilingual sentiment model (rater D, κ≈0 vs A; `not_a_human`). `annotator_id=codebook_b`.
- Fine-tune Pix2Struct / MatCha (no GPU). Zero-shot DePlot and MatCha-ChartQA are run on all 308 test screenshots.
- Recruit 12 retail investors. Eight real scenarios: raw 4/8; index mixers only on the two XU100 weeks; NASA-TLX blank.
- Infina production KAP HTML. Public BGE-M3 *is* run on KAP list concatenations (`public_m3_not_infina: true`).
- Same-day chart and tabular window. Public vision is \(t-40,\ldots,t-1\); the vol diagnostic is tabular at \(t\).

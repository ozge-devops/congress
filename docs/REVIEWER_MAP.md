# Response to the AIxIA 2026 referees (submission 84)

This note is for us and for the next PC. The LNCS manuscript is `paper/vesta.tex`.
Numbers come from `results/*.json`, not from the withdrawn draft.

1. **Template sentence in the abstract.** Removed. The abstract starts at
   “In high-volatility emerging markets…”.

2. **References.** Every item in `paper/vesta.bib` has a DOI or arXiv id and
   was checked against ACL Anthology, PMLR, Springer, ACM, OpenReview, or the
   journal page. Dropped: the TKDE 2023 item, the NeurIPS 2024 hallucination
   paper, CVPR 2025, Central Bank Review 2024, the Infina whitepaper, Chen et
   al. ACL 2023 “CMGF”, LangChain-as-a-paper, and CLIP as the InfoNCE cite.
   Details: `docs/BIBLIOGRAPHY_AUDIT.md`.

3. **Repository.** https://github.com/ozge-devops/congress

4. **Mirrored figure / product screenshots.** Old Fig. 4 is gone. Fig. 3
   (`tiers.pdf`) reads left to right. The product screenshots are out.

5. **Sentiment / technical accuracy in a table.** Proxy table vs \(y_{t+1}\):
   macro flags 54.9%, KAP polarity 49.7% (one split, no seed CI), vision
   \(52.9\pm2.3\), tabular \(50.6\pm0.0\). Codebook Fleiss \(\kappa=0.50\) on
   the 10k slice. Not the withdrawn 0.81 from three human annotators.

6. **Ablation numbers.** Table 2: mean fusion macro-F1 \(52.4\pm2.3\) vs
   scalar gate \(51.4\pm1.8\), seed-0 McNemar \(p=1.00\). We no longer say
   that mean fusion collapses toward the weaker modality.

7. **External baselines.** Run: GMU, TFN, a CPU MulT-style mixer, tabular
   OHLCV, the closed-form vol rule, MiniLM, public BGE-M3 on KAP list text
   (not Infina HTML), frozen ViT-B/16, zero-shot DePlot and MatCha-ChartQA on
   all 308 test screenshots. Mixers fuse unimodal scores; they are not a
   learned \(W_g\).

8. **1 / 3 / 10 minute layers.** Table 3: coverage, declared latency, accuracy
   among emitted calls, tokens, \(\mathrm{IN}_{\mathrm{old}}\) and
   \(\mathrm{IN}_{\mathrm{new}}\) on templated filler bags. T1 covers 32.5%
   of days on seed 0. Latency is a budget, not GPU wall-clock.

9. **Decision quality / user study.** Table 4 is a seed-0 paper-trading
   overlay. `study/` has consent, eight scenarios, and the NASA-TLX sheet.
   `responses.csv` is a header only. Model pilot: raw 4/8 names; index mixers
   only on the two XU100 weeks (text 0/2, gated 1/2). Not 12 humans.

10. **CMGF is not a contribution.** Title no longer says “via Cross-Modal
    Gated Fusion”. Eqs. (3)–(4) are a standard sigmoid mixer (Arevalo GMU,
    Jiang & Ji).

11. **Label leakage.** §4.1 and Table 1. Closed-form rule 100% F1; tabular
    MLP 96.8% acc.; vision 58.2% F1 (the chart is \(t-40,\ldots,t-1\)). The
    primary task is next-day direction. Jiang, Kelly & Xiu is cited together
    with the tabular baseline.

12. **Information-noise denominator.** Eq. (5): \(\mathrm{IN}_{\mathrm{new}}=N_{\mathrm{irrel}}/N_{\mathrm{del}}\)
    (Sun et al. 2019). Table 3: \(\mathrm{IN}_{\mathrm{old}}<7\%\),
    \(\mathrm{IN}_{\mathrm{new}}\) 60–94%.

Withdrawn numbers: 92.4% anomaly F1; 95.9% “noise reduction”; “CMGF is the
core innovation”; qualitative-only 1/3/10-minute claims.

Dietterich (1998): McNemar on seed 0 of one chronological holdout, not a
paired \(t\)-test on five random seeds. Harvey, Liu & Zhu (2016): thirteen
public rows, no winner after an uncorrected hunt. Mixer-vs-GMU \(p\) values
are at least 0.46.

Still open: three human annotators, a 12-investor NASA-TLX study, Infina KAP
HTML, and a GPU Pix2Struct/MatCha fine-tune. We would rather leave those
blank than restore the 92% figure.

# Paper to repository

Where the camera-ready claims in `paper/vesta.tex` are produced.

| Claim | Source |
|-------|--------|
| Leak, forward, McNemar, tiers, overlay | `results/public_benchmark.json` |
| Learned Arevalo GMU ($W_v,W_t,W_z$) | `results/learned_gmu.json` |
| MiniLM KAP probe | `results/kap_embed_benchmark.json` |
| Public BGE-M3 KAP probe | `results/kap_m3_benchmark.json` |
| Frozen ViT-B/16 | `results/vit_baseline.json` |
| DePlot / MatCha, n=308, no fine-tune | `results/vlm_baselines.json` |
| Event counts, KAP polarity, bodies | `data/vesta_public/label_stats.json` |
| Codebook Fleiss/Cohen kappa | `results/agreement.json` |
| Figures | `experiments/make_figures.py`, `paper/figures/` |
| NASA-TLX protocol, empty responses | `study/` |
| Model dry-run on sealed gold | `results/study_model_pilot.json` |
| Number lock | `tests/test_paper_consistency.py` |
| Stored `brief` == `text_polarity` | `experiments/refresh_briefs.py` |
| M3 day-level hop ($\alpha=1$, no portfolio) | `results/hop.json` |
| GitHub `ozge-devops/congress` `c03627c` | Older LNCS snapshot; current `main` is this PDF |

Public KAP text is list teasers from kap.org.tr. Score-space mixers fuse
unimodal class scores; `results/learned_gmu.json` fits Arevalo's maps on the
raw streams. The public hop is M3 day-level cosine (`results/hop.json`; $\alpha=1$, no
portfolio; $\beta$ unused). No `fig:claw` / `eq:rel` in the camera-ready.
The analogue does not enter the mixer; `IN_new` rises by construction. The 1-layer
attention row is not Tsai MulT. Fleiss is three codebooks; there are no
human annotators. The chart window is bars t-40 to t-1.

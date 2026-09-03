# Paper to repository

Where the camera-ready claims in `paper/vesta.tex` are produced.

| Claim | Source |
|-------|--------|
| Tables 1-4, McNemar, tiers, overlay | `results/public_benchmark.json` |
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

Public KAP text is list teasers from kap.org.tr. Mixers fuse unimodal class scores. The chart window is bars t-40 to t-1.

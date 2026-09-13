#!/usr/bin/env bash
# Build the CoMeSySo / OpenPublish LaTeX ZIP: sources, figures, final PDF.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$ROOT/comesyso-vesta.zip}"
cd "$ROOT"
pdflatex -interaction=nonstopmode vesta.tex >/dev/null
bibtex vesta >/dev/null
pdflatex -interaction=nonstopmode vesta.tex >/dev/null
pdflatex -interaction=nonstopmode vesta.tex >/dev/null
zip -q -r "$OUT" \
  vesta.tex vesta.bib vesta.pdf llncs.cls splncs04.bst \
  figures/architecture.pdf figures/architecture.png \
  figures/visualclaw_a.pdf figures/visualclaw_a.png \
  figures/visualclaw_b.pdf figures/visualclaw_b.png \
  figures/visualclaw_c.pdf figures/visualclaw_c.png \
  figures/leakage.pdf figures/leakage.png \
  figures/forward_f1.pdf figures/forward_f1.png \
  figures/tiers.pdf figures/tiers.png
echo "Wrote $OUT"
ls -lh "$OUT"

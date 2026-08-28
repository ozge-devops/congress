#!/usr/bin/env bash
# Export vesta.tex to Word. LNCS PDF remains the camera-ready artifact;
# this .docx is for coauthors who cannot open LaTeX. Figures must be PNG
# because pandoc's docx writer does not embed PDF images.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

python3 - <<'PY' "$ROOT" "$TMP"
import re, sys
from pathlib import Path
src, dst = Path(sys.argv[1]) / "vesta.tex", Path(sys.argv[2]) / "vesta.tex"
text = src.read_text(encoding="utf-8")
text = re.sub(r"figures/([A-Za-z0-9_-]+)\.pdf", r"figures/\1.png", text)
dst.write_text(text, encoding="utf-8")
PY

pandoc "$TMP/vesta.tex" \
  -o "$ROOT/vesta.docx" \
  --from=latex \
  --resource-path="$ROOT:$ROOT/figures" \
  --bibliography="$ROOT/vesta.bib" \
  --citeproc

echo "Wrote $ROOT/vesta.docx"
ls -lh "$ROOT/vesta.docx"

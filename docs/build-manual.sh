#!/usr/bin/env bash
# Build INSTALL.md into a typeset PDF.
#   apt-get install pandoc texlive-latex-recommended texlive-latex-extra \
#                   texlive-fonts-recommended lmodern
set -euo pipefail
cd "$(dirname "$0")/.."
pandoc INSTALL.md -o docs/MedROAD_V3_Installation.pdf \
  --pdf-engine=pdflatex \
  --include-in-header=docs/pandoc-header.tex \
  --toc --toc-depth=2 \
  --highlight-style=tango \
  -V geometry:"margin=2.2cm" -V fontsize=10pt \
  -V colorlinks=true -V linkcolor=linkblue -V urlcolor=linkblue \
  -V title="MedROAD V3" \
  -V subtitle="Installation and Operation Manual" \
  -V author="Pierre Boulanger \\ University of Alberta" \
  -V date="$(date +'%B %Y')"
echo "wrote docs/MedROAD_V3_Installation.pdf"

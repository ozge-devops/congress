.PHONY: data corpus enrich experiment gmu figures paper test agreement study api comesyso

data:
	PYTHONPATH=src python3 -c "from pathlib import Path; from vesta.data import download_public_market; download_public_market(Path('data/cache'))"

corpus:
	PYTHONPATH=src python3 experiments/build_corpus.py

enrich:
	PYTHONPATH=src python3 experiments/enrich_corpus.py

experiment:
	PYTHONPATH=src python3 experiments/run_public_benchmark.py

gmu:
	PYTHONPATH=src python3 experiments/run_learned_gmu.py

figures:
	PYTHONPATH=src python3 experiments/make_figures.py

agreement:
	PYTHONPATH=src python3 experiments/label_agreement.py

study:
	PYTHONPATH=src python3 experiments/build_study.py

test:
	PYTHONPATH=src python3 tests/test_labeling.py
	PYTHONPATH=src python3 tests/test_paper_consistency.py
	PYTHONPATH=src python3 tests/test_vlm_parse.py
	PYTHONPATH=src python3 tests/test_api.py

comesyso:
	bash paper/pack_comesyso.sh

api:
	PYTHONPATH=src python3 -m vesta.api

paper: figures
	cd paper && pdflatex -interaction=nonstopmode vesta.tex && bibtex vesta && pdflatex -interaction=nonstopmode vesta.tex && pdflatex -interaction=nonstopmode vesta.tex

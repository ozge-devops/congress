.PHONY: data corpus enrich experiment figures paper test agreement study

data:
	PYTHONPATH=src python3 -c "from pathlib import Path; from vesta.data import download_public_market; download_public_market(Path('data/cache'))"

corpus:
	PYTHONPATH=src python3 experiments/build_corpus.py

enrich:
	PYTHONPATH=src python3 experiments/enrich_corpus.py

experiment:
	PYTHONPATH=src python3 experiments/run_public_benchmark.py
 
figures: experiment
	PYTHONPATH=src python3 experiments/make_figures.py

agreement:
	PYTHONPATH=src python3 experiments/label_agreement.py

study:
	PYTHONPATH=src python3 experiments/build_study.py

test:
	PYTHONPATH=src python3 tests/test_labeling.py
	PYTHONPATH=src python3 tests/test_paper_consistency.py

paper: figures
	cd paper && pdflatex -interaction=nonstopmode vesta.tex && bibtex vesta && pdflatex -interaction=nonstopmode vesta.tex && pdflatex -interaction=nonstopmode vesta.tex

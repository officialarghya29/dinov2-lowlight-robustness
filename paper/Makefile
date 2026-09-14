# Paper build. Works with any TeX Live 2023+; on Overleaf just upload the folder.
MAIN = main

.PHONY: all clean tables figures

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib results/*.tex
	pdflatex -interaction=nonstopmode $(MAIN).tex >/dev/null
	bibtex $(MAIN) >/dev/null || true
	pdflatex -interaction=nonstopmode $(MAIN).tex >/dev/null
	pdflatex -interaction=nonstopmode $(MAIN).tex >/dev/null
	@echo "built $(MAIN).pdf"

tables:
	python3 ../make_paper_tables.py --results ../results_pilot --out results

figures:
	python3 ../make_paper_figures.py --results ../results_pilot --out figures

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out

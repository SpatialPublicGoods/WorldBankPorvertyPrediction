main.pdf: main.tex bibliography.bbl
	xelatex main.tex
	xelatex main.tex

bibliography.bbl: main.bcf bibliography.bib
	biber main

main.bcf: main.tex
	xelatex main.tex

clean:
	del main.aux bibliography.bbl main.bcf main.log main.synctex.gz main.toc main.run.xml

.PHONY: clean
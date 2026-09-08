@echo off
REM ============================================================
REM  Script de compilação do TG (Windows)
REM  Requer: MiKTeX ou TeX Live com abntex2
REM
REM  Sequência: pdflatex → bibtex → pdflatex → pdflatex
REM  (necessária para referências e sumário corretos)
REM ============================================================

echo [1/4] Primeira passagem do pdflatex...
pdflatex -interaction=nonstopmode main.tex

echo [2/4] Processando referencias (BibTeX)...
bibtex main

echo [3/4] Segunda passagem do pdflatex...
pdflatex -interaction=nonstopmode main.tex

echo [4/4] Terceira passagem do pdflatex...
pdflatex -interaction=nonstopmode main.tex

echo.
echo ============================================================
echo  Compilacao concluida. Arquivo gerado: main.pdf
echo ============================================================
pause

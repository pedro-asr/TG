@echo off
REM ============================================================
REM  Script de compilação do projeto LaTeX (tg_v2)
REM  Execute a partir da pasta tg_v2\
REM ============================================================
echo Compilando tese.tex...
pdflatex -interaction=nonstopmode tese.tex
bibtex tese
pdflatex -interaction=nonstopmode tese.tex
pdflatex -interaction=nonstopmode tese.tex
echo.
echo Compilação concluída! Arquivo: tese.pdf
pause

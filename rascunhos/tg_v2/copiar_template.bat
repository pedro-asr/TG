@echo off
REM ============================================================
REM  Script para copiar os arquivos de infraestrutura do template ITA
REM  para a pasta tg_v2.
REM
REM  Execute este arquivo UMA VEZ, a partir da pasta D:\tg\rascunhos\
REM  (dê dois cliques em tg_v2\copiar_template.bat)
REM ============================================================

SET TEMPLATE=..\ITA_Template_2024_v1__Copy_
SET DESTINO=%~dp0

echo Copiando arquivos do template ITA...

REM Classe principal
copy "%TEMPLATE%\ita.cls"           "%DESTINO%"
copy "%TEMPLATE%\abntcite.sty"      "%DESTINO%"
copy "%TEMPLATE%\abntex-abrev.sty"  "%DESTINO%"
copy "%TEMPLATE%\makeglo.sty"       "%DESTINO%"
copy "%TEMPLATE%\makeglo.ist"       "%DESTINO%"

REM Pasta abnt (estilos bibliográficos)
xcopy /E /I /Y "%TEMPLATE%\abnt"  "%DESTINO%abnt"

REM Pasta ita (logo, babel, defines)
xcopy /E /I /Y "%TEMPLATE%\ita"   "%DESTINO%ita"

echo.
echo Pronto! Os arquivos de infraestrutura foram copiados.
echo Agora você pode compilar tese.tex com: pdflatex + bibtex + pdflatex + pdflatex
pause

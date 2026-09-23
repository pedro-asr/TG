#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnóstico de resíduos dos 4 modelos de ML (SVM, MLP, LSTM, XGBoost) sobre o
conjunto de teste definitivo — testa se ainda sobra autocorrelação (estrutura
temporal não capturada) nos resíduos, via Ljung-Box (e Box-Pierce, para
comparação), em duas granularidades:

  - horária: série de resíduos na ordem cronológica do teste (12 horas
    diurnas/dia, sem grade contígua entre dias — lags que cruzam a fronteira
    entre dias misturam horas diferentes, mas o teste continua válido para
    "é ruído branco?").
  - diária: resíduo médio por dia de teste (41 dias), testando diretamente a
    dependência dia-a-dia que as defasagens D-1/D-2/D-3/D-7 tentam capturar.
  - horária (viés-hora removido): a autocorrelação horária pode vir de duas
    fontes bem diferentes — (a) viés fixo por hora do dia (mesmo perfil todo
    dia, corrigível com melhor codificação de hora, sem dado novo) ou (b)
    persistência do regime climático dentro do próprio dia de teste (não
    conhecida day-ahead, exigiria dado novo). Para separar as duas: calcula o
    perfil médio de resíduo por hora (pooled nos dias de teste), subtrai esse
    perfil de cada resíduo, e roda Ljung-Box de novo nos lags curtos [1,2,3]
    sobre o resíduo já sem o viés fixo — se a autocorrelação cair para não
    significativa, a causa era (a); se continuar forte, é (b).

Não re-treina nada: reaproveita `comparacao_final.gerar_previsoes` (que só faz
inferência dos modelos já salvos em disco) e `dados.carregar_split`.

Saídas:
  - `resultados/diagnostico_residuos_ljungbox.csv` (tabela completa, 3 granularidades)
  - `resultados/diagnostico_residuos_acf.png` (ACF dos resíduos horários, 4 painéis)
  - `resultados/diagnostico_residuos_perfil_hora.png` (perfil médio de resíduo por
    hora do dia ± erro padrão, 4 painéis)
  - interpretação (p<0.05 por modelo/granularidade) impressa no console
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comparacao_final import (  # noqa: E402
    COR_EIXO, COR_ENFASE, COR_GRADE, COR_MODELO_DESTAQUE, COR_PERSISTENCIA,
    COR_SUPERFICIE, COR_TEXTO_PRIMARIO, COR_TEXTO_SECUNDARIO, MODELOS_ML,
    gerar_previsoes,
)
from dados import carregar_split  # noqa: E402

SAIDA_DIR = Path(__file__).resolve().parent / "resultados"
SAIDA_TABELA = SAIDA_DIR / "diagnostico_residuos_ljungbox.csv"
SAIDA_ACF = SAIDA_DIR / "diagnostico_residuos_acf.png"
SAIDA_PERFIL_HORA = SAIDA_DIR / "diagnostico_residuos_perfil_hora.png"
SAIDA_CORRELACAO_TABELA = SAIDA_DIR / "diagnostico_residuos_correlacao.csv"
SAIDA_CORRELACAO_GRAFICO = SAIDA_DIR / "diagnostico_residuos_correlacao.png"

LAGS_HORARIOS = [1, 2, 3, 6, 12, 24]
LAGS_DIARIOS = [1, 2, 3, 7]
LAGS_SEM_VIES_HORA = [1, 2, 3]
NIVEL_SIGNIFICANCIA = 0.05


def calcular_residuos(teste, previsoes):
    """Resíduo = previsto - observado (mesma convenção do MBE em avaliacao.py)."""
    y_true = teste["P_ac"].values
    return {nome: previsoes[nome] - y_true for nome in MODELOS_ML}


def serie_diaria(teste, residuo):
    serie = pd.Series(residuo, index=teste.index)
    return serie.groupby(serie.index.date).mean()


def perfil_por_hora(teste, residuo):
    """Média, desvio-padrão e nº de dias do resíduo em cada hora do dia (pooled
    nos dias de teste) — a estimativa do viés fixo por hora, se houver."""
    serie = pd.Series(residuo, index=teste.index)
    grupo = serie.groupby(serie.index.hour)
    return grupo.mean(), grupo.std(), grupo.count()


def remover_vies_hora(teste, residuo, media_por_hora):
    """Resíduo menos o perfil médio da própria hora — isola o que resta depois
    de descontar qualquer viés fixo por hora do dia."""
    return residuo - media_por_hora.loc[teste.index.hour].values


def testar_ljungbox(residuos, teste):
    linhas = []
    for nome, resid in residuos.items():
        horario = acorr_ljungbox(resid, lags=LAGS_HORARIOS, boxpierce=True, return_df=True)
        for lag, linha in horario.iterrows():
            linhas.append({
                "modelo": nome, "granularidade": "horária", "lag": lag,
                "lb_stat": linha["lb_stat"], "lb_pvalue": linha["lb_pvalue"],
                "bp_stat": linha["bp_stat"], "bp_pvalue": linha["bp_pvalue"],
            })

        diario = serie_diaria(teste, resid)
        resultado_diario = acorr_ljungbox(diario.values, lags=LAGS_DIARIOS, boxpierce=True,
                                           return_df=True)
        for lag, linha in resultado_diario.iterrows():
            linhas.append({
                "modelo": nome, "granularidade": "diária", "lag": lag,
                "lb_stat": linha["lb_stat"], "lb_pvalue": linha["lb_pvalue"],
                "bp_stat": linha["bp_stat"], "bp_pvalue": linha["bp_pvalue"],
            })

        media_hora, _, _ = perfil_por_hora(teste, resid)
        resid_sem_vies = remover_vies_hora(teste, resid, media_hora)
        resultado_sem_vies = acorr_ljungbox(resid_sem_vies, lags=LAGS_SEM_VIES_HORA,
                                             boxpierce=True, return_df=True)
        for lag, linha in resultado_sem_vies.iterrows():
            linhas.append({
                "modelo": nome, "granularidade": "horária (viés-hora removido)", "lag": lag,
                "lb_stat": linha["lb_stat"], "lb_pvalue": linha["lb_pvalue"],
                "bp_stat": linha["bp_stat"], "bp_pvalue": linha["bp_pvalue"],
            })
    return pd.DataFrame(linhas)


def imprimir_interpretacao(tabela):
    print("\nInterpretação (H0: sem autocorrelação nos resíduos; teste Ljung-Box):")
    for (nome, granularidade), grupo in tabela.groupby(["modelo", "granularidade"], sort=False):
        lags_significativos = grupo.loc[grupo["lb_pvalue"] < NIVEL_SIGNIFICANCIA, "lag"].tolist()
        if lags_significativos:
            print(f"  {nome:12s} [{granularidade}]: rejeita H0 nos lags {lags_significativos} "
                  f"(p<{NIVEL_SIGNIFICANCIA}) -> sobra estrutura, há margem de melhoria.")
        else:
            print(f"  {nome:12s} [{granularidade}]: não rejeita H0 em nenhum lag testado "
                  f"-> resíduo ~ruído branco nessa granularidade.")


def gerar_grafico_acf(residuos, caminho_saida):
    """ACF dos resíduos horários dos 4 modelos, small multiples — mesma cor em todos
    os painéis (1 série por painel, sem identidade cruzada a comparar entre eles,
    mesmo padrão da dispersão em comparacao_final.py). Bandas tracejadas = limite
    de significância a 95% (±1.96/√n, aproximação para ruído branco)."""
    fig, eixos = plt.subplots(1, len(MODELOS_ML), figsize=(19, 4.0), sharey=True,
                               facecolor=COR_SUPERFICIE)

    for eixo, nome in zip(eixos, MODELOS_ML):
        resid = residuos[nome]
        n = len(resid)
        valores_acf = acf(resid, nlags=max(LAGS_HORARIOS), fft=True)
        limite = 1.96 / np.sqrt(n)

        eixo.set_facecolor(COR_SUPERFICIE)
        eixo.axhspan(-limite, limite, color=COR_GRADE, alpha=0.6, zorder=1)
        eixo.axhline(0, color=COR_EIXO, linewidth=1.0, zorder=2)
        lags = np.arange(len(valores_acf))
        eixo.bar(lags[1:], valores_acf[1:], color=COR_ENFASE, width=0.6, zorder=3)

        eixo.set_title(nome, fontsize=11, color=COR_TEXTO_SECUNDARIO)
        eixo.set_xlabel("Lag (horas de teste)", fontsize=9, color=COR_TEXTO_SECUNDARIO)
        eixo.grid(True, axis="y", color=COR_GRADE, linewidth=0.8, zorder=0)
        for spine in eixo.spines.values():
            spine.set_color(COR_EIXO)
        eixo.tick_params(colors=COR_TEXTO_SECUNDARIO, labelsize=8)

    eixos[0].set_ylabel("Autocorrelação", fontsize=9, color=COR_TEXTO_SECUNDARIO)
    fig.suptitle("ACF dos resíduos (granularidade horária) — banda = limite de "
                 "significância a 95%", fontsize=12, color=COR_TEXTO_PRIMARIO)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(caminho_saida, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def gerar_grafico_perfil_hora(residuos, teste, caminho_saida):
    """Resíduo médio por hora do dia ± erro padrão (1.96×SE), pooled nos dias de
    teste — separa viés fixo por hora (curva consistente, fora da faixa de zero)
    de ruído (barras cruzando zero sem padrão)."""
    fig, eixos = plt.subplots(1, len(MODELOS_ML), figsize=(19, 4.0), sharey=True,
                               facecolor=COR_SUPERFICIE)

    for eixo, nome in zip(eixos, MODELOS_ML):
        media, desvio, n = perfil_por_hora(teste, residuos[nome])
        erro = 1.96 * desvio / np.sqrt(n)

        eixo.set_facecolor(COR_SUPERFICIE)
        eixo.axhline(0, color=COR_EIXO, linewidth=1.0, zorder=2)
        eixo.bar(media.index, media.values, yerr=erro.values, color=COR_ENFASE,
                  width=0.6, capsize=3, error_kw={"ecolor": COR_TEXTO_SECUNDARIO,
                                                   "elinewidth": 1.0}, zorder=3)

        eixo.set_title(nome, fontsize=11, color=COR_TEXTO_SECUNDARIO)
        eixo.set_xlabel("Hora do dia", fontsize=9, color=COR_TEXTO_SECUNDARIO)
        eixo.grid(True, axis="y", color=COR_GRADE, linewidth=0.8, zorder=0)
        for spine in eixo.spines.values():
            spine.set_color(COR_EIXO)
        eixo.tick_params(colors=COR_TEXTO_SECUNDARIO, labelsize=8)

    eixos[0].set_ylabel("Resíduo médio (kW)", fontsize=9, color=COR_TEXTO_SECUNDARIO)
    fig.suptitle("Perfil do resíduo por hora do dia (± erro padrão a 95%)",
                 fontsize=12, color=COR_TEXTO_PRIMARIO)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(caminho_saida, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def calcular_correlacao_residuos(residuos):
    """Correlação de Pearson entre os resíduos horários dos 4 modelos — se os
    modelos erram nos mesmos momentos (correlação alta), o ganho esperado de
    combiná-los em ensemble é baixo; se erram de forma mais independente, ainda
    há espaço para reduzir RMSE combinando as previsões."""
    return pd.DataFrame(residuos)[MODELOS_ML].corr()


def gerar_grafico_correlacao(correlacao, caminho_saida):
    """Mapa de calor divergente (polaridade: -1 a +1), par azul/laranja já
    validado na paleta categórica do projeto (comparacao_final.py) + branco no
    meio — valores anotados em cada célula."""
    cmap = LinearSegmentedColormap.from_list(
        "diverging_bet01", [COR_MODELO_DESTAQUE, COR_SUPERFICIE, COR_PERSISTENCIA])

    fig, eixo = plt.subplots(figsize=(5.6, 5.0), facecolor=COR_SUPERFICIE)
    imagem = eixo.imshow(correlacao.values, cmap=cmap, vmin=-1, vmax=1)

    eixo.set_xticks(range(len(correlacao.columns)))
    eixo.set_xticklabels(correlacao.columns, rotation=30, ha="right", fontsize=9,
                          color=COR_TEXTO_SECUNDARIO)
    eixo.set_yticks(range(len(correlacao.index)))
    eixo.set_yticklabels(correlacao.index, fontsize=9, color=COR_TEXTO_SECUNDARIO)

    for i in range(len(correlacao.index)):
        for j in range(len(correlacao.columns)):
            valor = correlacao.values[i, j]
            cor_texto = COR_SUPERFICIE if abs(valor) > 0.6 else COR_TEXTO_PRIMARIO
            eixo.text(j, i, f"{valor:.2f}", ha="center", va="center", fontsize=10,
                       color=cor_texto)

    barra = fig.colorbar(imagem, ax=eixo, fraction=0.046, pad=0.04)
    barra.ax.tick_params(colors=COR_TEXTO_SECUNDARIO, labelsize=8)
    barra.outline.set_visible(False)

    eixo.set_title("Correlação entre resíduos horários dos 4 modelos", fontsize=11,
                    color=COR_TEXTO_PRIMARIO, pad=12)
    for spine in eixo.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main():
    SAIDA_DIR.mkdir(exist_ok=True)
    _, teste = carregar_split()
    previsoes = gerar_previsoes(teste)
    residuos = calcular_residuos(teste, previsoes)

    tabela = testar_ljungbox(residuos, teste)
    print(tabela.to_string(index=False))
    tabela.to_csv(SAIDA_TABELA, index=False)
    print(f"\nTabela salva em: {SAIDA_TABELA}")

    imprimir_interpretacao(tabela)

    gerar_grafico_acf(residuos, SAIDA_ACF)
    print(f"\nGráfico (ACF resíduos) salvo em: {SAIDA_ACF}")

    gerar_grafico_perfil_hora(residuos, teste, SAIDA_PERFIL_HORA)
    print(f"Gráfico (perfil por hora) salvo em: {SAIDA_PERFIL_HORA}")

    correlacao = calcular_correlacao_residuos(residuos)
    print("\nCorrelação entre resíduos horários dos 4 modelos:")
    print(correlacao.to_string())
    correlacao.to_csv(SAIDA_CORRELACAO_TABELA)
    print(f"Tabela (correlação) salva em: {SAIDA_CORRELACAO_TABELA}")

    gerar_grafico_correlacao(correlacao, SAIDA_CORRELACAO_GRAFICO)
    print(f"Gráfico (correlação) salvo em: {SAIDA_CORRELACAO_GRAFICO}")

    return tabela


if __name__ == "__main__":
    main()

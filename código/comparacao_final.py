#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolidação final — carrega os 4 modelos já treinados (SVM, MLP, LSTM, XGBoost) e os
baselines de persistência, todos sobre o mesmo split de teste (`dados.carregar_split`),
e gera:
  - tabela comparativa de métricas, incl. R² (console + `resultados/comparacao_final_tabela.csv`)
  - gráfico previsto vs. observado para 5 dias de teste representativos (mín., Q1,
    mediana, Q3 e máx. do RMSE diário do modelo vencedor) —
    `resultados/comparacao_final_previsto_vs_observado.png`
  - gráfico de dispersão previsto vs. observado dos 4 modelos, com R² anotado —
    `resultados/comparacao_final_dispersao.png`
  - gráfico de barras comparando RMSE de todos os modelos e baselines —
    `resultados/comparacao_final_rmse_barras.png`

Todos seguem a skill `dataviz` do projeto (paleta categórica validada, ênfase modelo
vs. baseline, pequenos múltiplos). Não re-treina nada: cada modelo é lido do disco e
usado só para inferência.
"""
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from avaliacao import avaliar, tabela_comparativa  # noqa: E402
from baselines import persistencia_ingenua, persistencia_inteligente  # noqa: E402
from dados import CAPACIDADE_KWP, carregar_split  # noqa: E402
from modelos.lstm import LSTMRegressorKeras  # noqa: E402
from modelos.mlp import MLPRegressorKeras  # noqa: E402

import tensorflow as tf  # noqa: E402

MODELOS_DIR = Path(__file__).resolve().parent / "modelos"
SAIDA_DIR = Path(__file__).resolve().parent / "resultados"
SAIDA_TABELA = SAIDA_DIR / "comparacao_final_tabela.csv"
SAIDA_GRAFICO_SERIE = SAIDA_DIR / "comparacao_final_previsto_vs_observado.png"
SAIDA_GRAFICO_DISPERSAO = SAIDA_DIR / "comparacao_final_dispersao.png"
SAIDA_GRAFICO_BARRAS = SAIDA_DIR / "comparacao_final_rmse_barras.png"

MODELO_DESTAQUE = "LSTM"  # melhor modelo até aqui — ver PROXIMOS_PASSOS.md
MODELOS_ML = ["SVM (SVR)", "MLP", "LSTM", "XGBoost"]
BASELINES = ["persistencia_ingenua", "persistencia_inteligente"]

# Paleta categórica validada (skill dataviz): slot 1 azul, slot 2 laranja — 2 cores
# categóricas + tinta neutra para "observado" ficam bem dentro do teto seguro para
# small multiples (all-pairs valida os 3 primeiros slots da paleta default).
COR_OBSERVADO = "#0b0b0b"      # tinta primária (não é uma série categórica)
COR_MODELO_DESTAQUE = "#2a78d6"  # slot 1 (azul) — também usado p/ "previsto" na dispersão
COR_PERSISTENCIA = "#eb6834"     # slot 2 (laranja)
COR_ENFASE = "#2a78d6"           # ênfase: modelos de ML (a história é "ML bate persistência")
COR_CONTEXTO = "#c3c2b7"         # de-ênfase (cinza): baselines de persistência
COR_SUPERFICIE = "#fcfcfb"
COR_GRADE = "#e1e0d9"
COR_EIXO = "#c3c2b7"
COR_TEXTO_SECUNDARIO = "#52514e"
COR_TEXTO_PRIMARIO = "#0b0b0b"


def carregar_svm():
    return joblib.load(MODELOS_DIR / "svm_bet01.joblib")


def carregar_mlp():
    modelo = MLPRegressorKeras.__new__(MLPRegressorKeras)
    modelo.modelo_ = tf.keras.models.load_model(MODELOS_DIR / "mlp_bet01.keras")
    scalers = joblib.load(MODELOS_DIR / "mlp_bet01_scalers.joblib")
    modelo.scaler_x_, modelo.scaler_y_ = scalers["scaler_x"], scalers["scaler_y"]
    return modelo


def carregar_lstm():
    modelo = LSTMRegressorKeras.__new__(LSTMRegressorKeras)
    modelo.modelo_ = tf.keras.models.load_model(MODELOS_DIR / "lstm_bet01.keras")
    scalers = joblib.load(MODELOS_DIR / "lstm_bet01_scalers.joblib")
    modelo.scaler_seq_ = scalers["scaler_seq"]
    modelo.scaler_estaticas_ = scalers["scaler_estaticas"]
    modelo.scaler_y_ = scalers["scaler_y"]
    return modelo


def carregar_xgboost():
    modelo = xgb.XGBRegressor()
    modelo.load_model(MODELOS_DIR / "xgboost_bet01.json")
    return modelo


def gerar_previsoes(teste):
    """Roda a inferência dos 4 modelos + 2 baselines sobre o teste, na ordem em que
    devem aparecer na tabela (agrupados por tipo, não por desempenho)."""
    colunas_x = [c for c in teste.columns if c != "P_ac"]
    X_teste = teste[colunas_x]

    previsoes = {
        "persistencia_ingenua": persistencia_ingenua(teste),
        "persistencia_inteligente": persistencia_inteligente(teste),
        "SVM (SVR)": carregar_svm().predict(X_teste),
        "MLP": carregar_mlp().predict(X_teste),
        "LSTM": carregar_lstm().predict(X_teste),
        "XGBoost": carregar_xgboost().predict(X_teste),
    }
    return {nome: np.clip(pred, 0, CAPACIDADE_KWP) for nome, pred in previsoes.items()}


def montar_tabela(y_true, previsoes):
    pred_ingenua = previsoes["persistencia_ingenua"]
    resultados = [
        avaliar(
            y_true, pred, potencia_instalada_kw=CAPACIDADE_KWP,
            y_pred_referencia=None if nome == "persistencia_ingenua" else pred_ingenua,
            nome=nome,
        )
        for nome, pred in previsoes.items()
    ]
    return tabela_comparativa(resultados)


def selecionar_dias_representativos(teste, y_pred_destaque):
    """Escolhe 5 dias de teste ao longo de toda a distribuição do RMSE diário do
    modelo em destaque (mín., Q1, mediana, Q3, máx.) — para ilustrar o desempenho sob
    o espectro completo de condições de céu, não só o melhor/pior caso isolado."""
    erro = pd.Series((np.asarray(y_pred_destaque) - teste["P_ac"].values) ** 2, index=teste.index)
    rmse_diario = np.sqrt(erro.groupby(teste.index.date).mean())

    alvos = {
        "Mín. RMSE": rmse_diario.quantile(0.0),
        "Q1 RMSE": rmse_diario.quantile(0.25),
        "Mediana RMSE": rmse_diario.quantile(0.50),
        "Q3 RMSE": rmse_diario.quantile(0.75),
        "Máx. RMSE": rmse_diario.quantile(1.0),
    }
    dias_usados = set()
    selecionados = []
    for rotulo, alvo in alvos.items():
        candidatos = (rmse_diario - alvo).abs().sort_values()
        dia = next(d for d in candidatos.index if d not in dias_usados)
        dias_usados.add(dia)
        selecionados.append((dia, rotulo))
    return selecionados


def gerar_grafico_serie(teste, previsoes, dias, caminho_saida):
    fig, eixos = plt.subplots(1, len(dias), figsize=(19, 3.6), sharey=True,
                               facecolor=COR_SUPERFICIE)

    for eixo, (dia, rotulo) in zip(eixos, dias):
        mascara = teste.index.date == dia
        horas = teste.index[mascara].hour
        eixo.set_facecolor(COR_SUPERFICIE)

        eixo.plot(horas, teste["P_ac"].values[mascara], color=COR_OBSERVADO,
                   linewidth=2.2, marker="o", markersize=4, label="Observado")
        eixo.plot(horas, previsoes[MODELO_DESTAQUE][mascara], color=COR_MODELO_DESTAQUE,
                   linewidth=1.8, marker="o", markersize=4, label=f"Previsto ({MODELO_DESTAQUE})")
        eixo.plot(horas, previsoes["persistencia_ingenua"][mascara], color=COR_PERSISTENCIA,
                   linewidth=1.8, marker="o", markersize=4, label="Previsto (persist. ingênua)")

        eixo.set_title(f"{rotulo} ({dia:%d/%m/%Y})", fontsize=10, color=COR_TEXTO_SECUNDARIO)
        eixo.set_xlabel("Hora do dia", fontsize=9, color=COR_TEXTO_SECUNDARIO)
        eixo.grid(True, color=COR_GRADE, linewidth=0.8)
        for spine in eixo.spines.values():
            spine.set_color(COR_EIXO)
        eixo.tick_params(colors=COR_TEXTO_SECUNDARIO, labelsize=8)

    eixos[0].set_ylabel("P_ac (kW)", fontsize=9, color=COR_TEXTO_SECUNDARIO)

    linhas, rotulos = eixos[0].get_legend_handles_labels()
    fig.legend(linhas, rotulos, loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Previsto vs. observado — espectro de dias de teste (mín. a máx. RMSE)",
                 fontsize=12, color=COR_TEXTO_PRIMARIO)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(caminho_saida, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def gerar_grafico_dispersao(y_true, previsoes, tabela, caminho_saida):
    """Dispersão previsto x observado dos 4 modelos de ML, com a diagonal y=x como
    referência (acerto perfeito) e RMSE/R² anotados — só 1 série por painel (o
    papel "previsto"), então a mesma cor se repete nos 4 painéis sem violar o teto
    de séries em small multiples (não há identidade cruzada entre painéis a comparar)."""
    fig, eixos = plt.subplots(1, len(MODELOS_ML), figsize=(19, 4.6), sharex=True, sharey=True,
                               facecolor=COR_SUPERFICIE)
    limite = max(y_true.max(), max(previsoes[m].max() for m in MODELOS_ML)) * 1.03

    for eixo, nome in zip(eixos, MODELOS_ML):
        eixo.set_facecolor(COR_SUPERFICIE)
        eixo.plot([0, limite], [0, limite], color=COR_EIXO, linewidth=1.2, linestyle="--",
                   zorder=1, label="Acerto perfeito (y=x)")
        eixo.scatter(y_true, previsoes[nome], color=COR_MODELO_DESTAQUE, s=14, alpha=0.5,
                     linewidths=0, zorder=2, label="Previsão")

        linha = tabela.loc[tabela["modelo"] == nome].iloc[0]
        eixo.text(0.04, 0.96, f"R² = {linha['R2']:.3f}\nRMSE = {linha['RMSE_kW']:.0f} kW",
                  transform=eixo.transAxes, ha="left", va="top", fontsize=9,
                  color=COR_TEXTO_SECUNDARIO)

        eixo.set_title(nome, fontsize=11, color=COR_TEXTO_SECUNDARIO)
        eixo.set_xlabel("Observado (kW)", fontsize=9, color=COR_TEXTO_SECUNDARIO)
        eixo.set_xlim(0, limite)
        eixo.set_ylim(0, limite)
        eixo.set_aspect("equal")
        eixo.grid(True, color=COR_GRADE, linewidth=0.8)
        for spine in eixo.spines.values():
            spine.set_color(COR_EIXO)
        eixo.tick_params(colors=COR_TEXTO_SECUNDARIO, labelsize=8)

    eixos[0].set_ylabel("Previsto (kW)", fontsize=9, color=COR_TEXTO_SECUNDARIO)

    linhas, rotulos = eixos[0].get_legend_handles_labels()
    fig.legend(linhas, rotulos, loc="lower center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Dispersão previsto vs. observado — conjunto de teste completo",
                 fontsize=12, color=COR_TEXTO_PRIMARIO)
    fig.tight_layout(rect=[0, 0.1, 1, 0.93])
    fig.savefig(caminho_saida, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def gerar_grafico_barras(tabela, caminho_saida):
    """Ranking de RMSE de todos os modelos e baselines — ênfase: modelos de ML no azul
    de destaque, baselines de persistência em cinza de contexto (a história é "ML bate
    a hipótese nula de persistência"), ordenado do melhor (menor RMSE) para o pior."""
    ordenado = tabela.sort_values("RMSE_kW", ascending=True).reset_index(drop=True)
    cores = [COR_ENFASE if m in MODELOS_ML else COR_CONTEXTO for m in ordenado["modelo"]]

    fig, eixo = plt.subplots(figsize=(9, 4.2), facecolor=COR_SUPERFICIE)
    eixo.set_facecolor(COR_SUPERFICIE)
    y_pos = np.arange(len(ordenado))[::-1]  # melhor no topo
    barras = eixo.barh(y_pos, ordenado["RMSE_kW"], color=cores, height=0.6)

    for y, valor in zip(y_pos, ordenado["RMSE_kW"]):
        eixo.text(valor + limite_texto(ordenado["RMSE_kW"]), y, f"{valor:.0f} kW",
                  va="center", ha="left", fontsize=9, color=COR_TEXTO_SECUNDARIO)

    eixo.set_yticks(y_pos)
    eixo.set_yticklabels(ordenado["modelo"], fontsize=9, color=COR_TEXTO_SECUNDARIO)
    eixo.set_xlabel("RMSE (kW) — menor é melhor", fontsize=9, color=COR_TEXTO_SECUNDARIO)
    eixo.set_xlim(0, ordenado["RMSE_kW"].max() * 1.18)
    eixo.grid(True, axis="x", color=COR_GRADE, linewidth=0.8)
    eixo.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        eixo.spines[spine].set_visible(False)
    eixo.spines["bottom"].set_color(COR_EIXO)
    eixo.tick_params(colors=COR_TEXTO_SECUNDARIO, labelsize=9, left=False)

    legenda = [
        plt.Rectangle((0, 0), 1, 1, color=COR_ENFASE, label="Modelos de ML"),
        plt.Rectangle((0, 0), 1, 1, color=COR_CONTEXTO, label="Baselines de persistência"),
    ]
    eixo.legend(handles=legenda, loc="upper right", frameon=False, fontsize=9)
    eixo.set_title("RMSE no conjunto de teste — modelos vs. baselines", fontsize=12,
                    color=COR_TEXTO_PRIMARIO, loc="left")
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def limite_texto(serie):
    return serie.max() * 0.012


def main():
    SAIDA_DIR.mkdir(exist_ok=True)
    _, teste = carregar_split()
    y_true = teste["P_ac"].values

    previsoes = gerar_previsoes(teste)

    tabela = montar_tabela(y_true, previsoes)
    print(tabela.to_string(index=False))
    tabela.to_csv(SAIDA_TABELA, index=False)
    print(f"\nTabela salva em: {SAIDA_TABELA}")

    dias = selecionar_dias_representativos(teste, previsoes[MODELO_DESTAQUE])
    gerar_grafico_serie(teste, previsoes, dias, SAIDA_GRAFICO_SERIE)
    print(f"Gráfico (série temporal) salvo em: {SAIDA_GRAFICO_SERIE}")

    gerar_grafico_dispersao(y_true, previsoes, tabela, SAIDA_GRAFICO_DISPERSAO)
    print(f"Gráfico (dispersão) salvo em: {SAIDA_GRAFICO_DISPERSAO}")

    gerar_grafico_barras(tabela, SAIDA_GRAFICO_BARRAS)
    print(f"Gráfico (barras RMSE) salvo em: {SAIDA_GRAFICO_BARRAS}")

    return tabela


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Passo 1 — Engenharia de features para previsão day-ahead (D+1) de P_ac (usina bet01).

Reconstrói a grade horária completa (24h/dia, incluindo noite) reusando as etapas de
dataset/formando_dataset/preprocessamento.py, calcula defasagens "mesma hora, N dias
atrás" via pivô data x hora (robusto à variação sazonal da duração do dia), anexa as
features determinísticas do horário-alvo e recorta para as horas diurnas de D+1
(linhas de modelagem). Nenhuma feature usa dado medido de D+1 — só defasagens (D e
anteriores) e grandezas conhecidas com antecedência (posição solar, calendário).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dataset" / "formando_dataset"))
from preprocessamento import (  # noqa: E402
    ENTRADA as ENTRADA_1MIN,
    COL_GS,
    etapa_alinhamento,
    etapa_outliers,
    etapa_reamostragem,
    etapa_faltantes,
    etapa_recorte_diurno,
)

SAIDA = Path(__file__).resolve().parent / "bet01_features_dayahead.csv"

# Variáveis medidas para as quais calculamos defasagens (histórico causal, D e antes)
VARS_LAG = ["P_ac", "temp_ambiente", "umidade_relativa", "velocidade_vento",
            "irradiacao_inclinada", "irradiacao_horizontal", "irradiacao_refletida", "T_inv"]
DIAS_LAG = [1, 2, 3, 7]
JANELAS_ROLLING = [3, 7]

# Features determinísticas do horário-alvo (conhecidas com antecedência, sem medição)
VARS_DETERMINISTICAS = ["elevacao_solar", "zenite_solar", "azimute_solar"]


def log(msg):
    print(msg, flush=True)


def construir_grade_horaria_completa():
    """Reproduz as etapas 1-4 de preprocessamento.py, mas SEM o recorte diurno,
    para obter a grade de 24h/dia necessária ao cálculo correto das defasagens."""
    df = etapa_alinhamento(ENTRADA_1MIN)
    df = etapa_outliers(df)
    df = etapa_reamostragem(df)
    df = etapa_faltantes(df)
    df = df.rename(columns={"SOL_IRRADIACAO_HORIZONTAL": "irradiacao_horizontal",
                             "SOL_IRRADIACAO_REFLETIDA": "irradiacao_refletida",
                             "TRACKER_DIRECAO_VENTO": "direcao_vento",
                             "SOL_TEMP_MODULO": "temp_modulo"})
    grade_completa = pd.date_range(df.index.min(), df.index.max(), freq="1h")
    df = df.reindex(grade_completa)
    df.index.name = "tempo"
    return df


def cyc(serie, periodo):
    ang = 2 * np.pi * serie / periodo
    return np.sin(ang), np.cos(ang)


def engenharia_lags(df):
    """Para cada variável em VARS_LAG, pivota (data x hora) e desloca ao longo do
    eixo de datas — dá 'mesma hora, N dias atrás' de forma robusta à duração do dia
    variar com a estação (diferente de um shift posicional sobre linhas diurnas)."""
    log("\n== ENGENHARIA DE LAGS (pivô data x hora) ==")
    df = df.copy()
    df["data"] = df.index.date
    df["hora"] = df.index.hour

    colunas_lag = {}
    for var in VARS_LAG:
        pivot = df.pivot(index="data", columns="hora", values=var).sort_index()
        for d in DIAS_LAG:
            colunas_lag[f"{var}_lag{d}d"] = pivot.shift(d).stack()
        for w in JANELAS_ROLLING:
            colunas_lag[f"{var}_media{w}d"] = pivot.shift(1).rolling(w).mean().stack()
        log(f"   {var}: lags {DIAS_LAG} + médias móveis {JANELAS_ROLLING}")

    features_lag = pd.DataFrame(colunas_lag)
    features_lag.index.names = ["data", "hora"]
    features_lag = features_lag.reset_index()
    features_lag["tempo"] = (pd.to_datetime(features_lag["data"].astype(str))
                              + pd.to_timedelta(features_lag["hora"], unit="h"))
    features_lag = features_lag.drop(columns=["data", "hora"]).set_index("tempo")
    return features_lag


def main():
    log("== RECONSTRUÇÃO DA GRADE HORÁRIA COMPLETA ==")
    grade = construir_grade_horaria_completa()
    log(f"   grade: {len(grade):,} horas, {grade.index.min()} a {grade.index.max()}")

    lags = engenharia_lags(grade)

    base = grade[VARS_DETERMINISTICAS + ["hora_do_dia", "dia_do_ano", "P_ac"]].copy()
    dataset = base.join(lags, how="left")

    dataset["hora_sin"], dataset["hora_cos"] = cyc(dataset["hora_do_dia"], 24)
    dataset["dia_ano_sin"], dataset["dia_ano_cos"] = cyc(dataset["dia_do_ano"], 365.25)
    dataset = dataset.drop(columns=["hora_do_dia", "dia_do_ano"])

    log("\n== RECORTE DIURNO DAS LINHAS-ALVO (D+1) ==")
    dataset = etapa_recorte_diurno(dataset)

    antes = len(dataset)
    dataset = dataset.dropna()
    log(f"\n== DESCARTE DE HISTÓRICO INSUFICIENTE ==")
    log(f"   linhas: {antes:,} -> {len(dataset):,} (descartadas {antes - len(dataset):,}: "
        f"primeiros ~7 dias sem histórico completo + cauda incompleta)")

    dataset.to_csv(SAIDA)
    log(f"\nArquivo salvo: {SAIDA}  (shape {dataset.shape})")
    log(f"Período final: {dataset.index.min()} a {dataset.index.max()}")
    return dataset


if __name__ == "__main__":
    main()

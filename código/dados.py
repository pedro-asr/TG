#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Carregamento e split cronológico do dataset de features — compartilhado pelos
baselines de persistência e por todos os modelos de ML (SVM, MLP, LSTM, XGBoost),
garantindo que todos sejam comparados exatamente sobre o mesmo conjunto de teste.
"""
from pathlib import Path

import pandas as pd

FEATURES_CSV = Path(__file__).resolve().parent / "bet01_features_dayahead.csv"
CAPACIDADE_KWP = 2500.0

LAT, LON, ALT, TZ = -3.2852895, -39.6487466, 0.0, "America/Fortaleza"

# PROVISÓRIO: o dataset atual só vai até 2026-07-09 (não há ago/set 2026 ainda).
# Split-alvo definitivo é por data (treino = até 31/07, teste = ago-set 2026, os
# meses mais recentes da usina) via `dividir_por_data`. Até esses dados chegarem,
# `dividir`/`carregar_split` usam um corte por fração como placeholder para validar
# o pipeline. Trocar as chamadas para `dividir_por_data` assim que o CSV de ago/set
# for adicionado.
FRACAO_TESTE = 0.15  # últimas ~15% das linhas (cronológico, sem embaralhar)
CORTE_DATA_ALVO = "2026-08-01"  # início do período de teste definitivo (ago-set 2026)


def carregar():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["tempo"]).set_index("tempo").sort_index()
    return df


def dividir(df, fracao_teste=FRACAO_TESTE):
    """[Placeholder] Split cronológico por fração: treino = início da série, teste = últimas linhas."""
    n_teste = int(len(df) * fracao_teste)
    treino, teste = df.iloc[:-n_teste].copy(), df.iloc[-n_teste:].copy()
    return treino, teste


def dividir_por_data(df, corte=CORTE_DATA_ALVO):
    """[Definitivo] Split por data: treino = tudo antes de `corte`, teste = tudo a partir dele
    (pensado para treino <= 31/07/2026 e teste = ago-set/2026, uma vez disponíveis)."""
    corte = pd.Timestamp(corte)
    treino, teste = df.loc[df.index < corte].copy(), df.loc[df.index >= corte].copy()
    return treino, teste


def carregar_split(fracao_teste=FRACAO_TESTE):
    return dividir(carregar(), fracao_teste)

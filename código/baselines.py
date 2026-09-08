#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baselines de persistência — §"Modelos de referência" do referencial teórico.
Representam a hipótese nula: qualquer modelo de ML só se justifica se bater estes.
"""
import warnings

import numpy as np
import pandas as pd
import pvlib

from avaliacao import avaliar, tabela_comparativa
from dados import CAPACIDADE_KWP, LAT, LON, ALT, TZ, carregar_split

LOCAL = pvlib.location.Location(LAT, LON, tz=TZ, altitude=ALT)


def clearsky_potencia(tempos):
    """Envelope físico simplificado de potência em céu claro: P = capacidade * GHI_clearsky/1000
    (modelo linear, ignora perdas térmicas/ôhmicas — suficiente para um benchmark)."""
    ghi = LOCAL.get_clearsky(pd.DatetimeIndex(tempos), model="ineichen")["ghi"]
    return (CAPACIDADE_KWP * ghi / 1000.0).clip(upper=CAPACIDADE_KWP).values


def persistencia_ingenua(teste):
    """P_ac(D+1,h) = P_ac(D,h) — já disponível como coluna P_ac_lag1d."""
    return teste["P_ac_lag1d"].values


def persistencia_inteligente(teste, janela_dias=3):
    """kt(D+1,h) = média(kt(D,h), ..., kt(D-janela+1,h)) * P_clearsky(D+1,h).
    Em horários de baixa elevação solar (alvo ou defasagens), kt fica mal definido
    (P_clearsky ~ 0); nesses casos cai para a persistência ingênua (P_ac_lag1d)."""
    p_clear_alvo = clearsky_potencia(teste.index)
    kts = []
    for d in range(1, janela_dias + 1):
        p_clear_lag = clearsky_potencia(teste.index - pd.Timedelta(days=d))
        p_ac_lag = teste[f"P_ac_lag{d}d"].values
        kts.append(p_ac_lag / np.where(p_clear_lag > 1e-6, p_clear_lag, np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        kt_medio = np.nanmean(np.vstack(kts), axis=0)
    pred = kt_medio * p_clear_alvo
    sem_kt = np.isnan(pred)
    pred[sem_kt] = teste["P_ac_lag1d"].values[sem_kt]
    return np.clip(pred, 0, CAPACIDADE_KWP)


def main():
    _, teste = carregar_split()
    y_true = teste["P_ac"].values

    pred_ingenua = persistencia_ingenua(teste)
    pred_inteligente = persistencia_inteligente(teste)

    resultados = [
        avaliar(y_true, pred_ingenua, potencia_instalada_kw=CAPACIDADE_KWP,
                nome="persistencia_ingenua"),
        avaliar(y_true, pred_inteligente, potencia_instalada_kw=CAPACIDADE_KWP,
                y_pred_referencia=pred_ingenua, nome="persistencia_inteligente"),
    ]
    print(tabela_comparativa(resultados).to_string(index=False))
    return resultados


if __name__ == "__main__":
    main()

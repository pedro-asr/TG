#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baselines de persistência v2 — §"Modelos de referência" do referencial teórico.
Representam a hipótese nula: qualquer modelo de ML só se justifica se bater estes.

Mudança em relação a `baselines.py` (v1, mantido intacto — decisão D6): o céu claro
passa a ser calculado com timestamps localizados em America/Fortaleza e no ponto médio
da hora (h:30). Na v1, o índice sem fuso era tratado pelo pvlib como UTC, o que
deslocava a curva de céu claro em +3 h (Fase 0 §3). A persistência ingênua é reutilizada
da v1 sem alteração; a inteligente da v1 é mantida só como linha de rastreabilidade.

Uso:  python -B baselines_v2.py            (grava resultados_v2/predicoes/predicoes_persistencias_v2.csv)
      python -B baselines_v2.py --rapido   (grava no sandbox resultados_v2/rapido/)
"""
import argparse
import sys
import warnings
from pathlib import Path

sys.dont_write_bytecode = True  # não regravar os .pyc versionados da v1 ao importar

import numpy as np
import pandas as pd
import pvlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from avaliacao import avaliar, tabela_comparativa  # noqa: E402  (v1, inalterado)
from baselines import persistencia_ingenua  # noqa: E402  (v1, inalterada)
from baselines import persistencia_inteligente as persistencia_inteligente_v1  # noqa: E402
from dados_v2 import (  # noqa: E402
    ALVO, CAPACIDADE_KWP, LAT, LON, ALT, TZ,
    caminho_predicoes, carregar_split, predicoes_formato_longo,
)
from saidas_v2 import ativar_modo_rapido, garantir_inexistentes  # noqa: E402

LOCAL = pvlib.location.Location(LAT, LON, tz=TZ, altitude=ALT)
JANELA_DIAS_KT = 3  # mesma janela da v1


def clearsky_potencia(tempos):
    """Envelope físico simplificado de potência em céu claro (mesmo modelo da v1):
    P = capacidade * GHI_clearsky/1000 (Ineichen), com o rótulo horário h (início do
    intervalo [h, h+1)) avaliado no ponto médio h:30 e localizado em `TZ`."""
    meio = (pd.DatetimeIndex(tempos) + pd.Timedelta(minutes=30)).tz_localize(TZ)
    ghi = LOCAL.get_clearsky(meio, model="ineichen")["ghi"]
    return (CAPACIDADE_KWP * ghi / 1000.0).clip(upper=CAPACIDADE_KWP).to_numpy()


def persistencia_inteligente(teste, janela_dias=JANELA_DIAS_KT):
    """kt(D+1,h) = média(kt(D,h), ..., kt(D-janela+1,h)) * P_clearsky(D+1,h).
    Mesma lógica da v1, com o céu claro corrigido. Onde kt fica mal definido
    (P_clearsky ~ 0), cai para a persistência ingênua (P_ac_lag1d)."""
    p_clear_alvo = clearsky_potencia(teste.index)
    kts = []
    for d in range(1, janela_dias + 1):
        p_clear_lag = clearsky_potencia(teste.index - pd.Timedelta(days=d))
        p_ac_lag = teste[f"P_ac_lag{d}d"].to_numpy()
        kts.append(p_ac_lag / np.where(p_clear_lag > 1e-6, p_clear_lag, np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        kt_medio = np.nanmean(np.vstack(kts), axis=0)
    pred = kt_medio * p_clear_alvo
    sem_kt = np.isnan(pred)
    pred[sem_kt] = teste["P_ac_lag1d"].to_numpy()[sem_kt]
    return np.clip(pred, 0, CAPACIDADE_KWP)


def previsoes_persistencias(teste):
    """{nome do modelo: previsão} das persistências (nomes usados no formato longo)."""
    return {
        "persistencia_ingenua": persistencia_ingenua(teste),
        "persistencia_inteligente": persistencia_inteligente(teste),
        "persistencia_inteligente_v1": persistencia_inteligente_v1(teste),  # rastreabilidade
    }


def main():
    parser = argparse.ArgumentParser(description="Persistências ingênua e inteligente (v2)")
    parser.add_argument("--rapido", action="store_true", help="grava no sandbox do modo rápido")
    args = parser.parse_args()
    if args.rapido:
        ativar_modo_rapido()
    saida = caminho_predicoes("persistencias")
    garantir_inexistentes([saida])

    _, teste, m = carregar_split()
    previsoes = previsoes_persistencias(teste)
    tabela = pd.concat([predicoes_formato_longo(teste, pred, nome)
                        for nome, pred in previsoes.items()], ignore_index=True)

    for cenario in ("teste_A", "teste_B"):
        mascara = m[cenario].to_numpy()
        y = teste.loc[mascara, ALVO].to_numpy()
        ref = previsoes["persistencia_ingenua"][mascara]
        resultados = [avaliar(y, pred[mascara], potencia_instalada_kw=CAPACIDADE_KWP, nome=nome,
                              y_pred_referencia=None if nome == "persistencia_ingenua" else ref)
                      for nome, pred in previsoes.items()]
        print(f"\nCenário {cenario[-1]} ({int(mascara.sum())} horas de teste):")
        print(tabela_comparativa(resultados).to_string(index=False))

    tabela.to_csv(saida, index=False)
    print(f"\nPrevisões salvas em: {saida}  ({len(tabela):,} linhas)")
    return tabela


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Carregamento e split cronológico do dataset de features v2 — compartilhado pelos
baselines e por todos os modelos v2 (regressão linear, SVR, MLP, LSTM, XGBoost),
garantindo que todos sejam comparados exatamente sobre o mesmo conjunto de teste e com
os mesmos filtros.

Mesmo corte (2026-08-01) e mesmas constantes físicas da v1 (importados de `dados.py`,
mantido intacto). As colunas `alvo_imputado` e `evento_operacional` usam dado medido de
D+1: servem só para montar as máscaras de filtragem (decisões D2/D3) e nunca entram
como features. Também concentra o formato longo das previsões de teste, comum a todos
os modelos.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # não regravar os .pyc versionados da v1 ao importar

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dados import CAPACIDADE_KWP, CORTE_DATA_ALVO, LAT, LON, ALT, TZ  # noqa: E402,F401
from saidas_v2 import DIR_RESULTADOS_V2, caminho_saida  # noqa: E402

# Entradas (sempre da pasta principal, também no modo rápido)
FEATURES_CSV = DIR_RESULTADOS_V2 / "dados" / "bet01_features_dayahead_v2.csv"
DICIONARIO_CSV = DIR_RESULTADOS_V2 / "dados" / "dicionario_features_v2.csv"

ALVO = "P_ac"
COLUNAS_FLAGS = ["alvo_imputado", "evento_operacional"]

# Formato longo das previsões de teste (um arquivo por script, todos com estas colunas)
COLUNAS_PREDICOES = ["tempo", "modelo", "semente", "y_obs", "y_pred",
                     "alvo_imputado", "evento_operacional"]
SEMENTE_DETERMINISTICO = -1  # modelos sem semente: persistências, regressão linear, SVR


def dicionario():
    """Dicionário de features gerado por `features_v2.py` (uma linha por feature)."""
    return pd.read_csv(DICIONARIO_CSV)


def colunas_features():
    """Features na ordem do dicionário — sem o alvo e sem as colunas de filtragem."""
    nomes = dicionario()["nome"].tolist()
    proibidas = set(nomes) & set([ALVO] + COLUNAS_FLAGS)
    if proibidas:
        raise ValueError(f"Alvo/flags no dicionário de features: {sorted(proibidas)}")
    return nomes


def grupos_features():
    """{grupo: [colunas]} na ordem do dicionário (variável de origem, posicao_solar,
    calendario) — usado na importância por permutação por grupo."""
    dic = dicionario()
    return {g: dic.loc[dic["grupo"] == g, "nome"].tolist() for g in dic["grupo"].unique()}


def validar(df):
    """O índice precisa ser estritamente crescente: o `validation_split` do Keras usa as
    ÚLTIMAS linhas passadas ao fit como validação, e o TimeSeriesSplit assume ordem
    cronológica. Também exige alvo/features sem NaN e flags booleanas."""
    if not (df.index.is_monotonic_increasing and df.index.is_unique):
        raise ValueError("Índice do dataset v2 não é estritamente crescente")
    colunas = colunas_features()
    faltando = sorted(set([ALVO] + COLUNAS_FLAGS + colunas) - set(df.columns))
    if faltando:
        raise ValueError(f"Colunas ausentes no dataset v2: {faltando}")
    if df[[ALVO] + colunas].isna().any().any():
        raise ValueError("NaN no alvo ou nas features do dataset v2")
    for flag in COLUNAS_FLAGS:
        if df[flag].dtype != bool:
            raise ValueError(f"Coluna de filtragem '{flag}' não é booleana ({df[flag].dtype})")


def carregar(caminho=FEATURES_CSV):
    df = pd.read_csv(caminho, parse_dates=["tempo"]).set_index("tempo")
    validar(df)
    return df


def dividir_por_data(df, corte=CORTE_DATA_ALVO):
    """Split por data: treino = tudo antes de `corte`, teste = tudo a partir dele."""
    corte = pd.Timestamp(corte)
    return df.loc[df.index < corte].copy(), df.loc[df.index >= corte].copy()


def mascaras(treino, teste):
    """Máscaras booleanas (Series alinhadas ao índice de cada conjunto):
    - 'treino_excluir': linhas fora do ajuste e do RMSE de CV (D2/D3);
    - 'teste_A': teste sem alvo imputado (cenário A);
    - 'teste_B': teste sem alvo imputado e sem evento operacional (cenário B)."""
    return {
        "treino_excluir": treino["alvo_imputado"] | treino["evento_operacional"],
        "teste_A": ~teste["alvo_imputado"],
        "teste_B": ~teste["alvo_imputado"] & ~teste["evento_operacional"],
    }


def carregar_split(corte=CORTE_DATA_ALVO):
    """Retorna (treino, teste, mascaras) — ver `mascaras`."""
    treino, teste = dividir_por_data(carregar(), corte)
    return treino, teste, mascaras(treino, teste)


def predicoes_formato_longo(teste, y_pred, modelo, semente=SEMENTE_DETERMINISTICO):
    """Previsões de teste no formato longo comum, já com clip em [0, CAPACIDADE_KWP].
    Inclui TODAS as linhas de teste (as flags permitem filtrar na comparação)."""
    y_pred = np.clip(np.asarray(y_pred, dtype=float).ravel(), 0, CAPACIDADE_KWP)
    if len(y_pred) != len(teste):
        raise ValueError(f"{len(y_pred)} previsões para {len(teste)} linhas de teste")
    tabela = pd.DataFrame({
        "tempo": teste.index,
        "modelo": modelo,
        "semente": int(semente),
        "y_obs": teste[ALVO].to_numpy(),
        "y_pred": y_pred,
        "alvo_imputado": teste["alvo_imputado"].to_numpy(),
        "evento_operacional": teste["evento_operacional"].to_numpy(),
    })
    return tabela[COLUNAS_PREDICOES]


def caminho_predicoes(nome):
    """`resultados_v2/predicoes/predicoes_<nome>_v2.csv` (sem checar existência)."""
    return caminho_saida("predicoes", f"predicoes_{nome}_v2.csv")


def main():
    treino, teste, m = carregar_split()
    colunas = colunas_features()
    print(f"Dataset v2: {len(treino) + len(teste):,} linhas, {len(colunas)} features "
          f"(índice estritamente crescente, sem NaN: OK)")
    print(f"Treino: {len(treino):,} linhas ({treino.index.min()} a {treino.index.max()}); "
          f"excluídas do ajuste: {int(m['treino_excluir'].sum())} -> efetivo "
          f"{int((~m['treino_excluir']).sum()):,}")
    print(f"Teste:  {len(teste):,} linhas ({teste.index.min()} a {teste.index.max()}); "
          f"cenário A: {int(m['teste_A'].sum())} | cenário B: {int(m['teste_B'].sum())}")
    print("Grupos de features:")
    for grupo, cols in grupos_features().items():
        print(f"   {grupo:<24} {len(cols):>2}  {cols}")


if __name__ == "__main__":
    main()

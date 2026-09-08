#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost para previsão day-ahead de P_ac — quarto e último dos 4 modelos
(SVM, MLP, LSTM, XGBoost).

Modelo baseado em árvores: usa o mesmo `bet01_features_dayahead.csv` tabular de
SVM/MLP diretamente, sem necessidade de normalização de features.

Nota de nome de arquivo: chamado `xgboost_model.py` (não `xgboost.py`) para não
sombrear o pacote `xgboost` ao rodar este script diretamente (o diretório do script
entra em sys.path[0], e um arquivo homônimo se auto-importaria em vez do pacote).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ajuste_hp import otimizar  # noqa: E402
from avaliacao import avaliar, tabela_comparativa  # noqa: E402
from baselines import persistencia_ingenua, persistencia_inteligente  # noqa: E402
from dados import CAPACIDADE_KWP, carregar_split  # noqa: E402

SAIDA_MODELO = Path(__file__).resolve().parent / "xgboost_bet01.json"
N_TRIALS = 40
SEED = 42


def construir_modelo(params):
    return xgb.XGBRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        min_child_weight=params["min_child_weight"],
        reg_lambda=params["reg_lambda"],
        reg_alpha=params["reg_alpha"],
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=-1,
    )


def espaco_busca(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


def main():
    treino, teste = carregar_split()
    colunas_x = [c for c in treino.columns if c != "P_ac"]

    X_treino, y_treino = treino[colunas_x], treino["P_ac"].values
    X_teste, y_teste = teste[colunas_x], teste["P_ac"].values

    print(f"Treino: {len(X_treino):,} linhas | Teste: {len(X_teste):,} linhas | "
          f"{len(colunas_x)} features")

    estudo = otimizar(construir_modelo, espaco_busca, X_treino, y_treino,
                       n_trials=N_TRIALS, nome_estudo="xgboost_bet01")
    print("\nMelhores hiperparâmetros:", estudo.best_params)
    print(f"RMSE (CV, treino): {estudo.best_value:.2f} kW")

    modelo = construir_modelo(estudo.best_params)
    modelo.fit(X_treino, y_treino)
    y_pred = np.clip(modelo.predict(X_teste), 0, CAPACIDADE_KWP)

    modelo.save_model(SAIDA_MODELO)
    print(f"\nModelo salvo em: {SAIDA_MODELO}")

    pred_ingenua = persistencia_ingenua(teste)
    pred_inteligente = persistencia_inteligente(teste)

    resultados = [
        avaliar(y_teste, pred_ingenua, potencia_instalada_kw=CAPACIDADE_KWP,
                nome="persistencia_ingenua"),
        avaliar(y_teste, pred_inteligente, potencia_instalada_kw=CAPACIDADE_KWP,
                y_pred_referencia=pred_ingenua, nome="persistencia_inteligente"),
        avaliar(y_teste, y_pred, potencia_instalada_kw=CAPACIDADE_KWP,
                y_pred_referencia=pred_ingenua, nome="XGBoost"),
    ]
    print()
    print(tabela_comparativa(resultados).to_string(index=False))

    print("\nImportância das features (top 10, ganho):")
    importancias = pd.Series(modelo.feature_importances_, index=colunas_x).sort_values(ascending=False)
    print(importancias.head(10).to_string())

    return modelo, resultados


if __name__ == "__main__":
    main()

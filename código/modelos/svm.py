#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVM (SVR) para previsão day-ahead de P_ac — primeiro dos 4 modelos (SVM, MLP, LSTM, XGBoost).
"""
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ajuste_hp import otimizar  # noqa: E402
from avaliacao import avaliar, tabela_comparativa  # noqa: E402
from baselines import persistencia_ingenua, persistencia_inteligente  # noqa: E402
from dados import CAPACIDADE_KWP, carregar_split  # noqa: E402

SAIDA_MODELO = Path(__file__).resolve().parent / "svm_bet01.joblib"
N_TRIALS = 1000


def construir_modelo(params):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=params["C"], epsilon=params["epsilon"], gamma=params["gamma"])),
    ])


def espaco_busca(trial):
    return {
        "C": trial.suggest_float("C", 1e-1, 1e3, log=True),
        "epsilon": trial.suggest_float("epsilon", 1e-2, 10 ** 1.5, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
    }


def main():
    treino, teste = carregar_split()
    colunas_x = [c for c in treino.columns if c != "P_ac"]

    X_treino, y_treino = treino[colunas_x], treino["P_ac"].values
    X_teste, y_teste = teste[colunas_x], teste["P_ac"].values

    print(f"Treino: {len(X_treino):,} linhas | Teste: {len(X_teste):,} linhas | "
          f"{len(colunas_x)} features")

    estudo = otimizar(construir_modelo, espaco_busca, X_treino, y_treino,
                       n_trials=N_TRIALS, nome_estudo="svm_bet01")
    print("\nMelhores hiperparâmetros:", estudo.best_params)
    print(f"RMSE (CV, treino): {estudo.best_value:.2f} kW")

    modelo = construir_modelo(estudo.best_params)
    modelo.fit(X_treino, y_treino)
    y_pred = np.clip(modelo.predict(X_teste), 0, CAPACIDADE_KWP)

    joblib.dump(modelo, SAIDA_MODELO)
    print(f"\nModelo salvo em: {SAIDA_MODELO}")

    pred_ingenua = persistencia_ingenua(teste)
    pred_inteligente = persistencia_inteligente(teste)

    resultados = [
        avaliar(y_teste, pred_ingenua, potencia_instalada_kw=CAPACIDADE_KWP,
                nome="persistencia_ingenua"),
        avaliar(y_teste, pred_inteligente, potencia_instalada_kw=CAPACIDADE_KWP,
                y_pred_referencia=pred_ingenua, nome="persistencia_inteligente"),
        avaliar(y_teste, y_pred, potencia_instalada_kw=CAPACIDADE_KWP,
                y_pred_referencia=pred_ingenua, nome="SVM (SVR)"),
    ]
    print()
    print(tabela_comparativa(resultados).to_string(index=False))
    return modelo, resultados


if __name__ == "__main__":
    main()

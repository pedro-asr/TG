#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVM (SVR) v2 para previsão day-ahead de P_ac — primeiro dos 4 modelos (SVM, MLP, LSTM,
XGBoost). Baseado em `svm.py` (v1, mantido intacto). Mudanças:
  - o alvo também é padronizado (`TransformedTargetRegressor` com `StandardScaler`), logo
    C e epsilon passam a valer em unidades padronizadas de P_ac;
  - espaço de busca, todos em escala log: C em [1e-1, 1e3], epsilon em [1e-3, 1],
    gamma em [1e-4, 1];
  - mesmo orçamento de trials das demais famílias (`ajuste_hp_v2.N_TRIALS`).
O SVR é determinístico: um único treino final (semente = -1 nas previsões).

Uso:  python -B modelos/svm_v2.py [--rapido] [--retomar]
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # não regravar os .pyc versionados da v1 ao importar

import joblib
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saidas_v2 import caminho_saida  # noqa: E402
from treino_v2 import executar  # noqa: E402

NOME = "svr"


def construir_modelo(params, semente=None, epocas=None):
    """`semente`/`epocas` são ignorados (SVR é determinístico e não iterativo por épocas)."""
    return TransformedTargetRegressor(
        regressor=Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=params["C"], epsilon=params["epsilon"],
                        gamma=params["gamma"])),
        ]),
        transformer=StandardScaler(),
    )


def espaco_busca(trial):
    return {
        "C": trial.suggest_float("C", 1e-1, 1e3, log=True),
        "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
    }


def arquivos_modelo(semente):
    return {"modelo": caminho_saida("modelos", f"{NOME}_bet01_v2.joblib")}


def salvar_modelo(modelo, arquivos):
    joblib.dump(modelo, arquivos["modelo"])


def carregar_modelo(arquivos):
    return joblib.load(arquivos["modelo"])


def main():
    return executar(NOME, construir_modelo, espaco_busca, deterministico=True,
                    arquivos_modelo=arquivos_modelo, salvar_modelo=salvar_modelo,
                    carregar_modelo=carregar_modelo, descricao="SVR (v2)")


if __name__ == "__main__":
    main()

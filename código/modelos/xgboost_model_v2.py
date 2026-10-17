#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost v2 para previsão day-ahead de P_ac — quarto e último dos 4 modelos
(SVM, MLP, LSTM, XGBoost). Baseado em `xgboost_model.py` (v1, mantido intacto): mesmo
espaço de busca e mesmas features tabulares, sem normalização. Mudanças: orçamento de
trials comum (`ajuste_hp_v2.N_TRIALS`), ajuste com a semente 42 e retreino final com as
sementes 0 a 9 (`random_state`, que governa `subsample`/`colsample_bytree`).

Paralelismo: `n_jobs=1` (na v1, -1). Com o histograma do XGBoost, a mesma semente gera
árvores diferentes conforme o número de threads (soma em ponto flutuante em outra
ordem), então -1 tornaria o resultado dependente da máquina; neste conjunto pequeno,
1 thread também não é mais lento.

Nota de nome de arquivo: `xgboost_model_v2.py` (não `xgboost_v2.py`), pelo mesmo motivo
da v1 — evitar qualquer sombra sobre o pacote `xgboost`.

Uso:  python -B modelos/xgboost_model_v2.py [--rapido] [--retomar]
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # não regravar os .pyc versionados da v1 ao importar

import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saidas_v2 import caminho_saida  # noqa: E402
from treino_v2 import executar  # noqa: E402

NOME = "xgboost"
N_JOBS = 1  # reprodutibilidade independente do número de núcleos (ver docstring)


def construir_modelo(params, semente, epocas=None):
    """`epocas` é ignorado (o número de árvores é o hiperparâmetro `n_estimators`)."""
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
        random_state=semente,
        n_jobs=N_JOBS,
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


def arquivos_modelo(semente):
    return {"modelo": caminho_saida("modelos", f"{NOME}_bet01_s{semente}_v2.json")}


def salvar_modelo(modelo, arquivos):
    modelo.save_model(arquivos["modelo"])


def carregar_modelo(arquivos):
    modelo = xgb.XGBRegressor()
    modelo.load_model(arquivos["modelo"])
    return modelo


def main():
    return executar(NOME, construir_modelo, espaco_busca, deterministico=False,
                    arquivos_modelo=arquivos_modelo, salvar_modelo=salvar_modelo,
                    carregar_modelo=carregar_modelo, descricao="XGBoost (v2)")


if __name__ == "__main__":
    main()

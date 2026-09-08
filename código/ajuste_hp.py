#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ajuste de hiperparâmetros compartilhado pelos 4 modelos (SVM, MLP, LSTM, XGBoost) via
otimização bayesiana (Optuna/TPE) com validação cruzada em série temporal (walk-forward,
`TimeSeriesSplit` — nunca k-fold aleatório, para não vazar futuro no treino de cada fold).
"""
import numpy as np
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

N_SPLITS_CV = 5
SEED = 42

optuna.logging.set_verbosity(optuna.logging.WARNING)


def validar_cv(construir_modelo, params, X, y, n_splits=N_SPLITS_CV):
    """RMSE médio em `n_splits` folds de walk-forward, para um dado conjunto de
    hiperparâmetros. `construir_modelo(params)` retorna um estimator sklearn-like
    (com fit/predict) ainda não treinado."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses = []
    for idx_treino, idx_val in tscv.split(X):
        modelo = construir_modelo(params)
        modelo.fit(X.iloc[idx_treino], y[idx_treino])
        pred = modelo.predict(X.iloc[idx_val])
        rmses.append(np.sqrt(mean_squared_error(y[idx_val], pred)))
    return float(np.mean(rmses))


def otimizar(construir_modelo, espaco_busca, X, y, n_trials=40, n_splits=N_SPLITS_CV,
             nome_estudo="estudo"):
    """`espaco_busca(trial)` amostra e retorna um dict de hiperparâmetros a partir de
    um `optuna.Trial`. Retorna o `optuna.Study` completo (acessar `.best_params`)."""
    def objetivo(trial):
        params = espaco_busca(trial)
        return validar_cv(construir_modelo, params, X, y, n_splits=n_splits)

    estudo = optuna.create_study(
        direction="minimize",
        study_name=nome_estudo,
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    estudo.optimize(objetivo, n_trials=n_trials, show_progress_bar=False)
    return estudo

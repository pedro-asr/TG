#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Métricas de avaliação compartilhadas entre todos os modelos (SVM, MLP, LSTM, XGBoost)
e os baselines de persistência — §"Métricas de Avaliação e Benchmarks".
"""
import numpy as np
import pandas as pd


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mbe(y_true, y_pred):
    """Mean Bias Error: viés médio (positivo = modelo superestima)."""
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def nrmse(y_true, y_pred, normalizacao):
    """RMSE normalizado por uma referência física (ex.: potência instalada, kW)."""
    return rmse(y_true, y_pred) / normalizacao


def r2(y_true, y_pred):
    """Coeficiente de determinação: fração da variância de y_true explicada por y_pred."""
    y_true = np.asarray(y_true)
    ss_res = np.sum((y_true - np.asarray(y_pred)) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot)


def skill_score(y_true, y_pred, y_pred_referencia):
    """Ganho relativo de RMSE frente a um modelo de referência (ex.: persistência).
    > 0: o modelo bate a referência. <= 0: não agrega valor informacional."""
    rmse_modelo = rmse(y_true, y_pred)
    rmse_ref = rmse(y_true, y_pred_referencia)
    if rmse_ref == 0:
        return float("nan")
    return 1.0 - (rmse_modelo / rmse_ref)


def avaliar(y_true, y_pred, *, potencia_instalada_kw, y_pred_referencia=None, nome="modelo"):
    """Calcula o conjunto padrão de métricas e retorna como dict (uma linha de tabela)."""
    resultado = {
        "modelo": nome,
        "n": len(y_true),
        "RMSE_kW": rmse(y_true, y_pred),
        "MAE_kW": mae(y_true, y_pred),
        "MBE_kW": mbe(y_true, y_pred),
        "nRMSE_%": 100 * nrmse(y_true, y_pred, potencia_instalada_kw),
        "R2": r2(y_true, y_pred),
    }
    if y_pred_referencia is not None:
        resultado["skill_score_vs_persistencia"] = skill_score(y_true, y_pred, y_pred_referencia)
    return resultado


def tabela_comparativa(resultados):
    """Recebe uma lista de dicts (saída de `avaliar`) e retorna um DataFrame ordenado por RMSE."""
    return pd.DataFrame(resultados).sort_values("RMSE_kW").reset_index(drop=True)

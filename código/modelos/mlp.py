#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLP (rede neural feedforward tradicional) para previsão day-ahead de P_ac —
segundo dos 4 modelos (SVM, MLP, LSTM, XGBoost).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silencia logs de inicialização do TF

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ajuste_hp import otimizar  # noqa: E402
from avaliacao import avaliar, tabela_comparativa  # noqa: E402
from baselines import persistencia_ingenua, persistencia_inteligente  # noqa: E402
from dados import CAPACIDADE_KWP, carregar_split  # noqa: E402

import tensorflow as tf  # noqa: E402

SAIDA_MODELO = Path(__file__).resolve().parent / "mlp_bet01.keras"
SAIDA_SCALERS = Path(__file__).resolve().parent / "mlp_bet01_scalers.joblib"
N_TRIALS = 30
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


class MLPRegressorKeras:
    """Wrapper sklearn-like (fit/predict) em torno de um Sequential do Keras, para
    ser reutilizado por `ajuste_hp.otimizar`/`validar_cv` sem alterar essa infraestrutura."""

    def __init__(self, n_camadas, unidades, dropout, taxa_aprendizado, batch_size,
                 epocas=150, paciencia=15):
        self.n_camadas = n_camadas
        self.unidades = unidades
        self.dropout = dropout
        self.taxa_aprendizado = taxa_aprendizado
        self.batch_size = batch_size
        self.epocas = epocas
        self.paciencia = paciencia

    def _construir_rede(self, n_features):
        modelo = tf.keras.Sequential([tf.keras.Input(shape=(n_features,))])
        for _ in range(self.n_camadas):
            modelo.add(tf.keras.layers.Dense(self.unidades, activation="relu"))
            if self.dropout > 0:
                modelo.add(tf.keras.layers.Dropout(self.dropout))
        modelo.add(tf.keras.layers.Dense(1, activation="linear"))
        modelo.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.taxa_aprendizado),
            loss="mse",
        )
        return modelo

    def fit(self, X, y):
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32").reshape(-1, 1)

        self.scaler_x_ = StandardScaler().fit(X)
        self.scaler_y_ = StandardScaler().fit(y)
        Xs = self.scaler_x_.transform(X)
        ys = self.scaler_y_.transform(y).ravel()

        self.modelo_ = self._construir_rede(n_features=Xs.shape[1])
        parada = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.paciencia, restore_best_weights=True)
        self.modelo_.fit(
            Xs, ys, validation_split=0.15, epochs=self.epocas,
            batch_size=self.batch_size, callbacks=[parada], verbose=0,
        )
        return self

    def predict(self, X):
        X = np.asarray(X, dtype="float32")
        Xs = self.scaler_x_.transform(X)
        pred_escalado = self.modelo_.predict(Xs, verbose=0).ravel()
        return self.scaler_y_.inverse_transform(pred_escalado.reshape(-1, 1)).ravel()


def construir_modelo(params):
    return MLPRegressorKeras(**params)


def espaco_busca(trial):
    return {
        "n_camadas": trial.suggest_int("n_camadas", 1, 3),
        "unidades": trial.suggest_categorical("unidades", [16, 32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "taxa_aprendizado": trial.suggest_float("taxa_aprendizado", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }


def main():
    treino, teste = carregar_split()
    colunas_x = [c for c in treino.columns if c != "P_ac"]

    X_treino, y_treino = treino[colunas_x], treino["P_ac"].values
    X_teste, y_teste = teste[colunas_x], teste["P_ac"].values

    print(f"Treino: {len(X_treino):,} linhas | Teste: {len(X_teste):,} linhas | "
          f"{len(colunas_x)} features")

    estudo = otimizar(construir_modelo, espaco_busca, X_treino, y_treino,
                       n_trials=N_TRIALS, nome_estudo="mlp_bet01")
    print("\nMelhores hiperparâmetros:", estudo.best_params)
    print(f"RMSE (CV, treino): {estudo.best_value:.2f} kW")

    modelo = construir_modelo(estudo.best_params)
    modelo.fit(X_treino, y_treino)
    y_pred = np.clip(modelo.predict(X_teste), 0, CAPACIDADE_KWP)

    modelo.modelo_.save(SAIDA_MODELO)
    joblib.dump({"scaler_x": modelo.scaler_x_, "scaler_y": modelo.scaler_y_}, SAIDA_SCALERS)
    print(f"\nModelo salvo em: {SAIDA_MODELO} (scalers em {SAIDA_SCALERS})")

    pred_ingenua = persistencia_ingenua(teste)
    pred_inteligente = persistencia_inteligente(teste)

    resultados = [
        avaliar(y_teste, pred_ingenua, potencia_instalada_kw=CAPACIDADE_KWP,
                nome="persistencia_ingenua"),
        avaliar(y_teste, pred_inteligente, potencia_instalada_kw=CAPACIDADE_KWP,
                y_pred_referencia=pred_ingenua, nome="persistencia_inteligente"),
        avaliar(y_teste, y_pred, potencia_instalada_kw=CAPACIDADE_KWP,
                y_pred_referencia=pred_ingenua, nome="MLP"),
    ]
    print()
    print(tabela_comparativa(resultados).to_string(index=False))
    return modelo, resultados


if __name__ == "__main__":
    main()

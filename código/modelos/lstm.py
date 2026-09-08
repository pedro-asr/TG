#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM para previsão day-ahead de P_ac — terceiro dos 4 modelos (SVM, MLP, LSTM, XGBoost).

Reaproveita o mesmo `bet01_features_dayahead.csv` de SVM/MLP (mesmas linhas, mesmo
split), para manter a comparação justa entre os 4 modelos. A "sequência" temporal da
LSTM é formada pelos 4 lags diários já calculados por variável (D-7, D-3, D-2, D-1,
em ordem cronológica) — uma sequência ordenada de snapshots, não espaçada
uniformemente, mas suficiente para a rede aprender a dinâmica recente sem duplicar a
engenharia de features. As médias móveis (3d/7d) e as features determinísticas
(posição solar, calendário) entram por um ramo denso separado (entrada "estática"),
concatenado à saída da LSTM antes da camada final.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ajuste_hp import otimizar  # noqa: E402
from avaliacao import avaliar, tabela_comparativa  # noqa: E402
from baselines import persistencia_ingenua, persistencia_inteligente  # noqa: E402
from dados import CAPACIDADE_KWP, carregar_split  # noqa: E402
from features import VARS_LAG  # noqa: E402

import tensorflow as tf  # noqa: E402

SAIDA_MODELO = Path(__file__).resolve().parent / "lstm_bet01.keras"
SAIDA_SCALERS = Path(__file__).resolve().parent / "lstm_bet01_scalers.joblib"
N_TRIALS = 30
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

# Ordem cronológica (mais antigo -> mais recente) dos lags disponíveis em features.py
LAGS_SEQUENCIA = ["lag7d", "lag3d", "lag2d", "lag1d"]
COLUNAS_ESTATICAS_DETERMINISTICAS = [
    "elevacao_solar", "zenite_solar", "azimute_solar",
    "hora_sin", "hora_cos", "dia_ano_sin", "dia_ano_cos",
]


def _colunas_estaticas():
    return ([f"{var}_media3d" for var in VARS_LAG]
            + [f"{var}_media7d" for var in VARS_LAG]
            + COLUNAS_ESTATICAS_DETERMINISTICAS)


class LSTMRegressorKeras:
    """Wrapper sklearn-like (fit/predict) em torno de um modelo funcional do Keras
    com dois ramos (sequência LSTM + estáticas densas), reutilizável por
    `ajuste_hp.otimizar`/`validar_cv` sem alterar essa infraestrutura."""

    def __init__(self, n_camadas_lstm, unidades_lstm, unidades_densas, dropout,
                 taxa_aprendizado, batch_size, epocas=150, paciencia=15):
        self.n_camadas_lstm = n_camadas_lstm
        self.unidades_lstm = unidades_lstm
        self.unidades_densas = unidades_densas
        self.dropout = dropout
        self.taxa_aprendizado = taxa_aprendizado
        self.batch_size = batch_size
        self.epocas = epocas
        self.paciencia = paciencia

    def _dividir_colunas(self, X):
        X = pd.DataFrame(X)
        seq = np.stack(
            [X[[f"{var}_{lag}" for lag in LAGS_SEQUENCIA]].values for var in VARS_LAG],
            axis=-1,
        )  # shape (n, len(LAGS_SEQUENCIA), len(VARS_LAG))
        estaticas = X[_colunas_estaticas()].values
        return seq.astype("float32"), estaticas.astype("float32")

    def _construir_rede(self, n_passos, n_vars_seq, n_estaticas):
        entrada_seq = tf.keras.Input(shape=(n_passos, n_vars_seq), name="sequencia")
        x = entrada_seq
        for _ in range(self.n_camadas_lstm - 1):
            x = tf.keras.layers.LSTM(self.unidades_lstm, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(self.unidades_lstm, return_sequences=False)(x)
        if self.dropout > 0:
            x = tf.keras.layers.Dropout(self.dropout)(x)

        entrada_estatica = tf.keras.Input(shape=(n_estaticas,), name="estaticas")
        ramo_estatico = tf.keras.layers.Dense(self.unidades_densas, activation="relu")(entrada_estatica)

        concat = tf.keras.layers.Concatenate()([x, ramo_estatico])
        saida = tf.keras.layers.Dense(self.unidades_densas, activation="relu")(concat)
        saida = tf.keras.layers.Dense(1, activation="linear")(saida)

        modelo = tf.keras.Model(inputs=[entrada_seq, entrada_estatica], outputs=saida)
        modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.taxa_aprendizado), loss="mse")
        return modelo

    def fit(self, X, y):
        seq, estaticas = self._dividir_colunas(X)
        y = np.asarray(y, dtype="float32").reshape(-1, 1)
        n, n_passos, n_vars = seq.shape

        self.scaler_seq_ = StandardScaler().fit(seq.reshape(-1, n_vars))
        seq_s = self.scaler_seq_.transform(seq.reshape(-1, n_vars)).reshape(n, n_passos, n_vars)

        self.scaler_estaticas_ = StandardScaler().fit(estaticas)
        estaticas_s = self.scaler_estaticas_.transform(estaticas)

        self.scaler_y_ = StandardScaler().fit(y)
        y_s = self.scaler_y_.transform(y).ravel()

        self.modelo_ = self._construir_rede(n_passos, n_vars, estaticas.shape[1])
        parada = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.paciencia, restore_best_weights=True)
        self.modelo_.fit(
            [seq_s, estaticas_s], y_s, validation_split=0.15, epochs=self.epocas,
            batch_size=self.batch_size, callbacks=[parada], verbose=0,
        )
        return self

    def predict(self, X):
        seq, estaticas = self._dividir_colunas(X)
        n, n_passos, n_vars = seq.shape
        seq_s = self.scaler_seq_.transform(seq.reshape(-1, n_vars)).reshape(n, n_passos, n_vars)
        estaticas_s = self.scaler_estaticas_.transform(estaticas)
        pred_s = self.modelo_.predict([seq_s, estaticas_s], verbose=0).ravel()
        return self.scaler_y_.inverse_transform(pred_s.reshape(-1, 1)).ravel()


def construir_modelo(params):
    return LSTMRegressorKeras(**params)


def espaco_busca(trial):
    return {
        "n_camadas_lstm": trial.suggest_int("n_camadas_lstm", 1, 2),
        "unidades_lstm": trial.suggest_categorical("unidades_lstm", [16, 32, 64]),
        "unidades_densas": trial.suggest_categorical("unidades_densas", [16, 32, 64]),
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
          f"sequência: {len(LAGS_SEQUENCIA)} passos x {len(VARS_LAG)} variáveis")

    estudo = otimizar(construir_modelo, espaco_busca, X_treino, y_treino,
                       n_trials=N_TRIALS, nome_estudo="lstm_bet01")
    print("\nMelhores hiperparâmetros:", estudo.best_params)
    print(f"RMSE (CV, treino): {estudo.best_value:.2f} kW")

    modelo = construir_modelo(estudo.best_params)
    modelo.fit(X_treino, y_treino)
    y_pred = np.clip(modelo.predict(X_teste), 0, CAPACIDADE_KWP)

    modelo.modelo_.save(SAIDA_MODELO)
    joblib.dump(
        {"scaler_seq": modelo.scaler_seq_, "scaler_estaticas": modelo.scaler_estaticas_,
         "scaler_y": modelo.scaler_y_},
        SAIDA_SCALERS,
    )
    print(f"\nModelo salvo em: {SAIDA_MODELO} (scalers em {SAIDA_SCALERS})")

    pred_ingenua = persistencia_ingenua(teste)
    pred_inteligente = persistencia_inteligente(teste)

    resultados = [
        avaliar(y_teste, pred_ingenua, potencia_instalada_kw=CAPACIDADE_KWP,
                nome="persistencia_ingenua"),
        avaliar(y_teste, pred_inteligente, potencia_instalada_kw=CAPACIDADE_KWP,
                y_pred_referencia=pred_ingenua, nome="persistencia_inteligente"),
        avaliar(y_teste, y_pred, potencia_instalada_kw=CAPACIDADE_KWP,
                y_pred_referencia=pred_ingenua, nome="LSTM"),
    ]
    print()
    print(tabela_comparativa(resultados).to_string(index=False))
    return modelo, resultados


if __name__ == "__main__":
    main()

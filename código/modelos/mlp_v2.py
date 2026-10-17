#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLP (rede neural feedforward tradicional) v2 para previsão day-ahead de P_ac —
segundo dos 4 modelos (SVM, MLP, LSTM, XGBoost). Baseado em `mlp.py` (v1, mantido
intacto): mesma arquitetura, mesmo espaço de busca, mesma parada antecipada. Mudanças:
orçamento de trials comum (`ajuste_hp_v2.N_TRIALS`), ajuste com a semente 42 e retreino
final com as sementes 0 a 9, cada treino iniciado com `tf.keras.utils.set_random_seed`.

Uso:  python -B modelos/mlp_v2.py [--rapido] [--retomar]
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silencia logs de inicialização do TF
sys.dont_write_bytecode = True  # não regravar os .pyc versionados da v1 ao importar

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saidas_v2 import caminho_saida  # noqa: E402
from treino_v2 import EPOCAS, executar  # noqa: E402

import tensorflow as tf  # noqa: E402

tf.config.experimental.enable_op_determinism()  # mesma semente -> mesmo modelo (CPU)

NOME = "mlp"
PACIENCIA = 15
FRACAO_VALIDACAO = 0.15  # últimas 15% das linhas do fit (ordem cronológica, ver dados_v2)


class MLPRegressorKeras:
    """Wrapper sklearn-like (fit/predict) em torno de um Sequential do Keras, para
    ser reutilizado por `ajuste_hp_v2.otimizar`/`validar_cv` e pelo retreino com sementes."""

    def __init__(self, n_camadas, unidades, dropout, taxa_aprendizado, batch_size,
                 semente=42, epocas=EPOCAS, paciencia=PACIENCIA):
        self.n_camadas = n_camadas
        self.unidades = unidades
        self.dropout = dropout
        self.taxa_aprendizado = taxa_aprendizado
        self.batch_size = batch_size
        self.semente = semente
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
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(self.semente)
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32").reshape(-1, 1)

        self.scaler_x_ = StandardScaler().fit(X)
        self.scaler_y_ = StandardScaler().fit(y)
        Xs = self.scaler_x_.transform(X)
        ys = self.scaler_y_.transform(y).ravel()

        self.modelo_ = self._construir_rede(n_features=Xs.shape[1])
        parada = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.paciencia, restore_best_weights=True)
        historico = self.modelo_.fit(
            Xs, ys, validation_split=FRACAO_VALIDACAO, epochs=self.epocas,
            batch_size=self.batch_size, callbacks=[parada], verbose=0,
        )
        self.epocas_treinadas_ = len(historico.history["loss"])
        return self

    def predict(self, X):
        X = np.asarray(X, dtype="float32")
        Xs = self.scaler_x_.transform(X)
        pred_escalado = self.modelo_.predict(Xs, verbose=0).ravel()
        return self.scaler_y_.inverse_transform(pred_escalado.reshape(-1, 1)).ravel()

    @classmethod
    def carregar(cls, caminho_modelo, caminho_scalers):
        """Reconstrói o wrapper treinado a partir dos arquivos salvos (só para predict)."""
        salvo = joblib.load(caminho_scalers)
        modelo = cls(**salvo["hiperparametros"])
        modelo.scaler_x_, modelo.scaler_y_ = salvo["scaler_x"], salvo["scaler_y"]
        modelo.modelo_ = tf.keras.models.load_model(caminho_modelo)
        return modelo


def construir_modelo(params, semente, epocas):
    return MLPRegressorKeras(**params, semente=semente, epocas=epocas)


def espaco_busca(trial):
    return {
        "n_camadas": trial.suggest_int("n_camadas", 1, 3),
        "unidades": trial.suggest_categorical("unidades", [16, 32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "taxa_aprendizado": trial.suggest_float("taxa_aprendizado", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }


def arquivos_modelo(semente):
    return {"modelo": caminho_saida("modelos", f"{NOME}_bet01_s{semente}_v2.keras"),
            "scalers": caminho_saida("modelos", f"{NOME}_bet01_s{semente}_scalers_v2.joblib")}


def salvar_modelo(modelo, arquivos):
    modelo.modelo_.save(arquivos["modelo"])
    hiperparametros = {k: getattr(modelo, k) for k in (
        "n_camadas", "unidades", "dropout", "taxa_aprendizado", "batch_size", "semente",
        "epocas", "paciencia")}
    joblib.dump({"scaler_x": modelo.scaler_x_, "scaler_y": modelo.scaler_y_,
                 "hiperparametros": hiperparametros,
                 "epocas_treinadas": modelo.epocas_treinadas_}, arquivos["scalers"])


def carregar_modelo(arquivos):
    return MLPRegressorKeras.carregar(arquivos["modelo"], arquivos["scalers"])


def main():
    return executar(NOME, construir_modelo, espaco_busca, deterministico=False,
                    arquivos_modelo=arquivos_modelo, salvar_modelo=salvar_modelo,
                    carregar_modelo=carregar_modelo, descricao="MLP (v2)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM v2 para previsão day-ahead de P_ac — terceiro dos 4 modelos (SVM, MLP, LSTM,
XGBoost). Baseado em `lstm.py` (v1, mantido intacto): mesma arquitetura de dois ramos,
mesmo espaço de busca, mesma parada antecipada, mesmo dataset/split dos demais modelos.

- Ramo sequencial (LSTM): os lags diários de cada variável de VARS_LAG, em ordem
  cronológica (D-6, D-2, D-1, D = lags de 7, 3, 2 e 1 dias) — uma sequência ordenada de
  snapshots da mesma hora, não espaçada uniformemente.
- Ramo estático (denso): as features determinísticas do horário-alvo (elevação,
  azimute seno/cosseno, hora e dia do ano seno/cosseno). Na v1 entravam também as médias
  móveis de 3 e 7 dias, removidas na v2 (registro M1).
As duas listas de colunas são derivadas do dicionário de features (`dados_v2.dicionario`),
não escritas à mão. Mudanças de treino como nos demais modelos v2: orçamento comum de
trials, ajuste com a semente 42 e retreino final com as sementes 0 a 9.

Uso:  python -B modelos/lstm_v2.py [--rapido] [--retomar]
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.dont_write_bytecode = True  # não regravar os .pyc versionados da v1 ao importar

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dados_v2 import dicionario  # noqa: E402
from saidas_v2 import caminho_saida  # noqa: E402
from treino_v2 import EPOCAS, executar  # noqa: E402

import tensorflow as tf  # noqa: E402

tf.config.experimental.enable_op_determinism()  # mesma semente -> mesmo modelo (CPU)

NOME = "lstm"
PACIENCIA = 15
FRACAO_VALIDACAO = 0.15  # últimas 15% das linhas do fit (ordem cronológica, ver dados_v2)


def estrutura_entradas():
    """Deriva do dicionário de features:
    - `sequencia`: DataFrame (passos x variáveis) com o NOME da coluna de cada posição,
      passos em ordem cronológica (maior defasagem primeiro);
    - `estaticas`: lista das features determinísticas.
    Exige que toda variável tenha os mesmos lags e que nenhuma feature fique de fora."""
    dic = dicionario()
    lags = dic[dic["tipo"] == "lag"].copy()
    lags["defasagem_dias"] = lags["defasagem_dias"].astype(int)
    variaveis = list(dict.fromkeys(lags["variavel"]))
    defasagens = sorted(lags["defasagem_dias"].unique(), reverse=True)
    sequencia = (lags.pivot(index="defasagem_dias", columns="variavel", values="nome")
                 .reindex(index=defasagens, columns=variaveis))
    if sequencia.isna().any().any():
        raise ValueError("Variáveis com conjuntos de lags diferentes — sequência da LSTM irregular")
    estaticas = dic.loc[dic["tipo"] == "deterministica", "nome"].tolist()
    fora = set(dic["nome"]) - set(sequencia.to_numpy().ravel()) - set(estaticas)
    if fora:
        raise ValueError(f"Features do dicionário sem lugar na LSTM: {sorted(fora)}")
    return sequencia, estaticas


COLUNAS_SEQUENCIA, COLUNAS_ESTATICAS = estrutura_entradas()


class LSTMRegressorKeras:
    """Wrapper sklearn-like (fit/predict) em torno de um modelo funcional do Keras
    com dois ramos (sequência LSTM + estáticas densas), reutilizável por
    `ajuste_hp_v2.otimizar`/`validar_cv` e pelo retreino com sementes."""

    def __init__(self, n_camadas_lstm, unidades_lstm, unidades_densas, dropout,
                 taxa_aprendizado, batch_size, semente=42, epocas=EPOCAS, paciencia=PACIENCIA):
        self.n_camadas_lstm = n_camadas_lstm
        self.unidades_lstm = unidades_lstm
        self.unidades_densas = unidades_densas
        self.dropout = dropout
        self.taxa_aprendizado = taxa_aprendizado
        self.batch_size = batch_size
        self.semente = semente
        self.epocas = epocas
        self.paciencia = paciencia

    def _dividir_colunas(self, X):
        X = pd.DataFrame(X)
        n_passos, n_vars = COLUNAS_SEQUENCIA.shape
        seq = (X[COLUNAS_SEQUENCIA.to_numpy().ravel()].to_numpy(dtype="float32")
               .reshape(len(X), n_passos, n_vars))  # (n, passos, variáveis)
        estaticas = X[COLUNAS_ESTATICAS].to_numpy(dtype="float32")
        return seq, estaticas

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
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(self.semente)
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
        historico = self.modelo_.fit(
            [seq_s, estaticas_s], y_s, validation_split=FRACAO_VALIDACAO, epochs=self.epocas,
            batch_size=self.batch_size, callbacks=[parada], verbose=0,
        )
        self.epocas_treinadas_ = len(historico.history["loss"])
        return self

    def predict(self, X):
        seq, estaticas = self._dividir_colunas(X)
        n, n_passos, n_vars = seq.shape
        seq_s = self.scaler_seq_.transform(seq.reshape(-1, n_vars)).reshape(n, n_passos, n_vars)
        estaticas_s = self.scaler_estaticas_.transform(estaticas)
        pred_s = self.modelo_.predict([seq_s, estaticas_s], verbose=0).ravel()
        return self.scaler_y_.inverse_transform(pred_s.reshape(-1, 1)).ravel()

    @classmethod
    def carregar(cls, caminho_modelo, caminho_scalers):
        """Reconstrói o wrapper treinado a partir dos arquivos salvos (só para predict)."""
        salvo = joblib.load(caminho_scalers)
        modelo = cls(**salvo["hiperparametros"])
        modelo.scaler_seq_ = salvo["scaler_seq"]
        modelo.scaler_estaticas_ = salvo["scaler_estaticas"]
        modelo.scaler_y_ = salvo["scaler_y"]
        modelo.modelo_ = tf.keras.models.load_model(caminho_modelo)
        return modelo


def construir_modelo(params, semente, epocas):
    return LSTMRegressorKeras(**params, semente=semente, epocas=epocas)


def espaco_busca(trial):
    return {
        "n_camadas_lstm": trial.suggest_int("n_camadas_lstm", 1, 2),
        "unidades_lstm": trial.suggest_categorical("unidades_lstm", [16, 32, 64]),
        "unidades_densas": trial.suggest_categorical("unidades_densas", [16, 32, 64]),
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
        "n_camadas_lstm", "unidades_lstm", "unidades_densas", "dropout", "taxa_aprendizado",
        "batch_size", "semente", "epocas", "paciencia")}
    joblib.dump({"scaler_seq": modelo.scaler_seq_, "scaler_estaticas": modelo.scaler_estaticas_,
                 "scaler_y": modelo.scaler_y_, "hiperparametros": hiperparametros,
                 "colunas_sequencia": COLUNAS_SEQUENCIA, "colunas_estaticas": COLUNAS_ESTATICAS,
                 "epocas_treinadas": modelo.epocas_treinadas_}, arquivos["scalers"])


def carregar_modelo(arquivos):
    return LSTMRegressorKeras.carregar(arquivos["modelo"], arquivos["scalers"])


def main():
    n_passos, n_vars = COLUNAS_SEQUENCIA.shape
    print(f"Sequência: {n_passos} passos x {n_vars} variáveis "
          f"(defasagens {list(COLUNAS_SEQUENCIA.index)} dias; variáveis {list(COLUNAS_SEQUENCIA.columns)})")
    print(f"Estáticas ({len(COLUNAS_ESTATICAS)}): {COLUNAS_ESTATICAS}")
    return executar(NOME, construir_modelo, espaco_busca, deterministico=False,
                    arquivos_modelo=arquivos_modelo, salvar_modelo=salvar_modelo,
                    carregar_modelo=carregar_modelo, descricao="LSTM (v2)")


if __name__ == "__main__":
    main()

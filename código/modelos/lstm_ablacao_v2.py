#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 5 (v2, opcional) — ablação da LSTM: o ganho vem da recorrência ou da arquitetura
de dois ramos (sequência + estáticas)?

Três variantes, todas com os MELHORES hiperparâmetros da LSTM do modo completo
(`resultados_v2/modelos/lstm_resumo_v2.json`), as mesmas sementes (0 a 9), o mesmo
treino efetivo (sem `alvo_imputado`/`evento_operacional`), o mesmo ramo estático e a
mesma cabeça densa — muda só o ramo da sequência:
  (a) original: a LSTM de `lstm_v2.py`, passos em ordem cronológica (D-6, D-2, D-1, D);
      deve reproduzir exatamente as previsões salvas na Fase 3 (controle);
  (b) sequencia_invertida: a mesma rede, com os passos em ordem cronológica inversa
      (D, D-1, D-2, D-6) — a LSTM lê o dia mais recente primeiro;
  (c) ramo_denso: a(s) camada(s) LSTM trocada(s) por Dense(unidades_lstm, relu) sobre a
      sequência achatada (4 passos x 8 variáveis = 32 valores), mesma quantidade de
      camadas e unidades; não há recorrência.
Leitura: (b) ≈ (a) indica que a ordem dos passos não importa para o modelo; (c) ≈ (a)
indica que o ganho não vem da recorrência, e sim das features e da arquitetura de dois
ramos. Comparação por métricas (média ± dp entre sementes, cenários A e B) e pelo teste
de Diebold-Mariano (HLN) sobre a previsão média das sementes, como em `comparacao_v2.py`.

Uso:  python -B modelos/lstm_ablacao_v2.py            (grava em resultados_v2/)
      python -B modelos/lstm_ablacao_v2.py --rapido   (5 épocas, 2 sementes; sandbox)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.dont_write_bytecode = True  # não regravar os .pyc versionados da v1 ao importar

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comparacao_v2 import (  # noqa: E402
    CENARIOS, REF_PRINCIPAL, SUFIXO_MEDIA, _media_dp, agregar_sementes, metricas, no_cenario,
    perdas_diarias, por_semente, teste_dm_hln,
)
from dados_v2 import (  # noqa: E402
    ALVO, COLUNAS_PREDICOES, SEMENTE_DETERMINISTICO, caminho_predicoes, carregar_split,
    colunas_features, predicoes_formato_longo,
)
from saidas_v2 import DIR_RESULTADOS_V2, ativar_modo_rapido, caminho_saida, garantir_inexistentes  # noqa: E402
from treino_v2 import EPOCAS, EPOCAS_RAPIDO, SEMENTES, SEMENTES_RAPIDO  # noqa: E402

import lstm_v2  # noqa: E402  (mesma pasta; também ativa o determinismo de operações do TF)
import tensorflow as tf  # noqa: E402

# Sempre os hiperparâmetros do modo completo (também no --rapido, que só encurta o treino)
RESUMO_LSTM = DIR_RESULTADOS_V2 / "modelos" / "lstm_resumo_v2.json"
PREVISOES_LSTM = DIR_RESULTADOS_V2 / "predicoes" / "predicoes_lstm_v2.csv"
TOLERANCIA_REPRODUCAO_KW = 1e-3


class LSTMSequenciaInvertida(lstm_v2.LSTMRegressorKeras):
    """(b) Mesma rede da LSTM original; os passos da sequência entram em ordem cronológica
    inversa (o scaler da sequência é por variável, logo não muda com a ordem)."""

    def _dividir_colunas(self, X):
        seq, estaticas = super()._dividir_colunas(X)
        return np.ascontiguousarray(seq[:, ::-1, :]), estaticas


class LSTMRamoDenso(lstm_v2.LSTMRegressorKeras):
    """(c) Sem recorrência: a sequência é achatada e passa por `n_camadas_lstm` camadas
    Dense(unidades_lstm, relu). Ramo estático, concatenação e cabeça são os mesmos de
    `lstm_v2.LSTMRegressorKeras._construir_rede`."""

    def _construir_rede(self, n_passos, n_vars_seq, n_estaticas):
        entrada_seq = tf.keras.Input(shape=(n_passos, n_vars_seq), name="sequencia")
        x = tf.keras.layers.Flatten()(entrada_seq)
        for _ in range(self.n_camadas_lstm):
            x = tf.keras.layers.Dense(self.unidades_lstm, activation="relu")(x)
        if self.dropout > 0:
            x = tf.keras.layers.Dropout(self.dropout)(x)

        # --- a partir daqui, idêntico à LSTM original ---
        entrada_estatica = tf.keras.Input(shape=(n_estaticas,), name="estaticas")
        ramo_estatico = tf.keras.layers.Dense(self.unidades_densas, activation="relu")(entrada_estatica)

        concat = tf.keras.layers.Concatenate()([x, ramo_estatico])
        saida = tf.keras.layers.Dense(self.unidades_densas, activation="relu")(concat)
        saida = tf.keras.layers.Dense(1, activation="linear")(saida)

        modelo = tf.keras.Model(inputs=[entrada_seq, entrada_estatica], outputs=saida)
        modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.taxa_aprendizado), loss="mse")
        return modelo


VARIANTES = {
    "lstm_original": lstm_v2.LSTMRegressorKeras,
    "lstm_sequencia_invertida": LSTMSequenciaInvertida,
    "lstm_ramo_denso": LSTMRamoDenso,
}
ROTULOS = {"lstm_original": "(a) LSTM original",
           "lstm_sequencia_invertida": "(b) LSTM, sequência invertida",
           "lstm_ramo_denso": "(c) Ramo denso (sem recorrência)"}


def rotulo(modelo):
    if modelo == REF_PRINCIPAL:
        return "Persistência inteligente"
    base = modelo.removesuffix(SUFIXO_MEDIA)
    return ROTULOS.get(base, modelo) + (" — média das sementes" if modelo.endswith(SUFIXO_MEDIA) else "")


def treinar_variantes(treino, teste, excluir, hiperparametros, sementes, epocas):
    colunas_x = colunas_features()
    X_treino, y_treino = treino[colunas_x][~excluir], treino[ALVO].to_numpy()[~excluir]
    X_teste = teste[colunas_x]
    tabelas, info = [], {}
    for variante, classe in VARIANTES.items():
        inicio, epocas_treinadas = time.time(), []
        for s in sementes:
            modelo = classe(**hiperparametros, semente=s, epocas=epocas).fit(X_treino, y_treino)
            tabelas.append(predicoes_formato_longo(teste, modelo.predict(X_teste), variante, s))
            epocas_treinadas.append(modelo.epocas_treinadas_)
        info[variante] = {"n_parametros": int(modelo.modelo_.count_params()),
                          "epocas_treinadas": epocas_treinadas,
                          "tempo_s": round(time.time() - inicio, 1)}
        print(f"   {ROTULOS[variante]:<34} {info[variante]['n_parametros']:>6,} parâmetros | épocas "
              f"treinadas {epocas_treinadas} | {info[variante]['tempo_s']:.0f}s", flush=True)
    return pd.concat(tabelas, ignore_index=True), info


def com_media_das_sementes(prev):
    medias = []
    for variante in VARIANTES:
        d = prev[prev["modelo"] == variante]
        m = d.groupby("tempo", as_index=False).agg(
            y_obs=("y_obs", "first"), y_pred=("y_pred", "mean"),
            alvo_imputado=("alvo_imputado", "first"),
            evento_operacional=("evento_operacional", "first"))
        m["modelo"] = variante + SUFIXO_MEDIA
        m["semente"] = SEMENTE_DETERMINISTICO
        medias.append(m[COLUNAS_PREDICOES])
    prev = pd.concat([prev, *medias], ignore_index=True)
    prev["hora"] = prev["tempo"].dt.hour
    prev["dia"] = prev["tempo"].dt.normalize()
    return prev


def tabela_metricas(prev, info):
    partes = []
    for cenario in CENARIOS:
        sub = no_cenario(prev, cenario)
        ref = sub[sub["modelo"] == REF_PRINCIPAL]
        rmse_ref = metricas(ref["y_obs"], ref["y_pred"])["RMSE_kW"]

        def funcao(d):
            r = metricas(d["y_obs"], d["y_pred"])
            r["skill_vs_inteligente"] = 1 - r["RMSE_kW"] / rmse_ref
            return r

        t = agregar_sementes(por_semente(sub[sub["modelo"] != REF_PRINCIPAL], ["modelo"], funcao),
                             ["modelo"], ["RMSE_kW", "MAE_kW", "MBE_kW", "nRMSE_%", "R2",
                                          "skill_vs_inteligente"])
        t.insert(0, "cenario", cenario)
        partes.append(t)
    tabela = pd.concat(partes, ignore_index=True)
    tabela.insert(2, "rotulo", tabela["modelo"].map(rotulo))
    base = tabela["modelo"].str.removesuffix(SUFIXO_MEDIA)
    tabela["n_parametros"] = base.map(lambda v: info[v]["n_parametros"])
    tabela["epocas_treinadas_media"] = base.map(lambda v: float(np.mean(info[v]["epocas_treinadas"])))
    return tabela


def tabela_dm(prev):
    medias = [v + SUFIXO_MEDIA for v in VARIANTES]
    pares = [(medias[0], medias[1]), (medias[0], medias[2]), (medias[1], medias[2])]
    pares += [(m, REF_PRINCIPAL) for m in medias]
    linhas = []
    for cenario in CENARIOS:
        sub = no_cenario(prev, cenario)
        for m1, m2 in pares:
            r = teste_dm_hln(perdas_diarias(sub, m1), perdas_diarias(sub, m2))
            linhas.append({"cenario": cenario, "modelo_1": m1, "modelo_2": m2,
                           "rotulo_1": rotulo(m1), "rotulo_2": rotulo(m2), **r,
                           "melhor": m1 if r["media_d_kW2"] < 0 else m2,
                           "significativo_5%": r["p_valor"] < 0.05})
    return pd.DataFrame(linhas)


def conferir_reproducao(prev_ablacao, sementes, epocas):
    """Variante (a) x previsões salvas da LSTM na Fase 3 (só faz sentido no modo completo)."""
    resumo = json.loads(RESUMO_LSTM.read_text(encoding="utf-8"))
    if epocas != resumo["epocas_max"] or sementes != resumo["sementes_retreino"]:
        print("   [controle] modo rápido: épocas/sementes diferentes da Fase 3 — reprodução não conferida")
        return None
    salvas = pd.read_csv(PREVISOES_LSTM, parse_dates=["tempo"]).set_index(["semente", "tempo"])["y_pred"]
    novas = (prev_ablacao[prev_ablacao["modelo"] == "lstm_original"]
             .set_index(["semente", "tempo"])["y_pred"])
    dif = float(np.max(np.abs(novas.sort_index().to_numpy() - salvas.sort_index().to_numpy())))
    ok = dif <= TOLERANCIA_REPRODUCAO_KW
    print(f"   [controle] (a) x previsões salvas da LSTM (Fase 3), 10 sementes: máx |dif| = "
          f"{dif:.2e} kW -> {'OK, reprodução exata' if ok else 'DIFERENTE — ver ambiente/versões'}")
    return dif


def main():
    parser = argparse.ArgumentParser(description="Ablação da LSTM (v2)")
    parser.add_argument("--rapido", action="store_true",
                        help="5 épocas e 2 sementes; grava no sandbox resultados_v2/rapido/")
    args = parser.parse_args()
    if args.rapido:
        ativar_modo_rapido()
    epocas = EPOCAS_RAPIDO if args.rapido else EPOCAS
    sementes = SEMENTES_RAPIDO if args.rapido else SEMENTES

    saidas = {"predicoes": caminho_predicoes("lstm_ablacao"),
              "metricas": caminho_saida("tabelas", "lstm_ablacao_metricas_v2.csv"),
              "dm": caminho_saida("tabelas", "lstm_ablacao_diebold_mariano_v2.csv"),
              "resumo": caminho_saida("modelos", "lstm_ablacao_resumo_v2.json")}
    garantir_inexistentes(saidas.values())

    hiperparametros = json.loads(RESUMO_LSTM.read_text(encoding="utf-8"))["melhores_hiperparametros"]
    treino, teste, m = carregar_split()
    excluir = m["treino_excluir"].to_numpy()
    print(f"Hiperparâmetros (melhor trial da LSTM, modo completo): {hiperparametros}")
    print(f"Treino efetivo: {int((~excluir).sum()):,} linhas | Teste: {len(teste)} | "
          f"sementes {sementes} | até {epocas} épocas\n")

    ablacao, info = treinar_variantes(treino, teste, excluir, hiperparametros, sementes, epocas)
    reproducao = conferir_reproducao(ablacao, sementes, epocas)

    persistencias = pd.read_csv(caminho_predicoes("persistencias"), parse_dates=["tempo"])
    prev = com_media_das_sementes(pd.concat(
        [ablacao, persistencias[persistencias["modelo"] == REF_PRINCIPAL]], ignore_index=True))
    metr = tabela_metricas(prev, info)
    dm = tabela_dm(prev)

    for cenario, descricao in CENARIOS.items():
        t = metr[metr["cenario"] == cenario]
        print(f"\n== Cenário {cenario} ({descricao}) ==")
        print(f"{'variante':<52}{'RMSE (kW)':>15}{'MAE (kW)':>15}{'MBE (kW)':>16}{'skill intelig. (%)':>20}")
        for _, l in t.iterrows():
            print(f"{l['rotulo']:<52}{_media_dp(l['RMSE_kW_media'], l['RMSE_kW_dp']):>15}"
                  f"{_media_dp(l['MAE_kW_media'], l['MAE_kW_dp']):>15}"
                  f"{_media_dp(l['MBE_kW_media'], l['MBE_kW_dp'], '{:+.1f}'):>16}"
                  f"{_media_dp(100 * l['skill_vs_inteligente_media'], 100 * l['skill_vs_inteligente_dp'], '{:+.1f}'):>20}")
        print("Diebold-Mariano (HLN, perdas diárias, previsão média das sementes):")
        for _, l in dm[dm["cenario"] == cenario].iterrows():
            print(f"   {l['rotulo_1']:<52} x {l['rotulo_2']:<52} DM* {l['DM_HLN']:+6.2f}  "
                  f"p = {l['p_valor']:.3f}{' *' if l['significativo_5%'] else ''}")

    ablacao.to_csv(saidas["predicoes"], index=False)
    metr.to_csv(saidas["metricas"], index=False)
    dm.to_csv(saidas["dm"], index=False)
    saidas["resumo"].write_text(json.dumps({
        "modo": "rapido" if args.rapido else "completo",
        "hiperparametros": hiperparametros, "sementes": sementes, "epocas_max": epocas,
        "variantes": {v: {"rotulo": ROTULOS[v], **info[v]} for v in VARIANTES},
        "reproducao_max_dif_kW": reproducao,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSalvos: {', '.join(c.name for c in saidas.values())}")


if __name__ == "__main__":
    main()

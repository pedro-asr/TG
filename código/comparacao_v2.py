#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolidação final (v2) — compara os modelos day-ahead da usina bet01 a partir das
previsões de teste SALVAS em `resultados_v2/predicoes/` (formato longo de `dados_v2`).
Nada é re-treinado. Única exceção: a importância por permutação recarrega os modelos
salvos, porque precisa reprever o teste com features embaralhadas (e confere que a
previsão sem permutação reproduz a previsão salva).

Gera em resultados_v2/tabelas/ e resultados_v2/figuras/ (sufixo _v2):
  1. métricas por modelo (RMSE, MAE, MBE, nRMSE sobre 2500 kWp, R²; média ± dp entre
     sementes) nos cenários A (teste sem alvo imputado) e B (também sem evento
     operacional), com skill score em RMSE contra a persistência inteligente
     (referência principal) e a ingênua (secundária); MLP, LSTM e XGBoost aparecem
     também pela previsão média das sementes;
  2. teste de Diebold-Mariano com correção de Harvey-Leybourne-Newbold sobre perdas
     diárias (previsão média das sementes para MLP, LSTM e XGBoost);
  3. erro por hora do dia (RMSE e MBE);
  4. erro por classe de céu (tercis do índice diário k_d no teste);
  5. importância por permutação no teste (10 repetições, RMSE; por feature e por grupo);
  6. RMSE por dobra de CV da melhor configuração de cada família (otimista);
  7. comparação v1 x v2 (lê `resultados/comparacao_final_tabela.csv`, só leitura);
  8. figuras: dispersão, previsto vs. observado (dias do mín. ao máx. RMSE), barras de
     RMSE e erro por hora do dia — seguindo a skill `dataviz` (cor fixa por família).

Uso:  python -B comparacao_v2.py                    (lê resultados_v2/predicoes/)
      python -B comparacao_v2.py --rapido           (lê e grava no sandbox resultados_v2/rapido/)
      python -B comparacao_v2.py --sem-importancia  (pula a importância por permutação)
"""
import argparse
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.dont_write_bytecode = True  # não regravar os .pyc versionados da v1 ao importar
# Saída redirecionada no Windows (ex.: `| Tee-Object`) usa cp1252: nunca abortar por um
# caractere fora dela — ele sai como '?' no console; as tabelas são gravadas em UTF-8.
sys.stdout.reconfigure(errors="replace")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from avaliacao import mae, mbe, nrmse, r2, rmse  # noqa: E402  (v1, inalterado)
from baselines_v2 import clearsky_potencia  # noqa: E402
from dados import FEATURES_CSV as FEATURES_CSV_V1  # noqa: E402  (v1, só leitura)
from dados_v2 import (  # noqa: E402
    ALVO, CAPACIDADE_KWP, COLUNAS_PREDICOES, CORTE_DATA_ALVO, SEMENTE_DETERMINISTICO,
    caminho_predicoes, carregar_split, colunas_features, grupos_features,
)
from saidas_v2 import ativar_modo_rapido, caminho_saida, garantir_inexistentes  # noqa: E402

TABELA_V1 = Path(__file__).resolve().parent / "resultados" / "comparacao_final_tabela.csv"

ARQUIVOS_PREVISOES = ["persistencias", "regressao_linear", "svr", "mlp", "lstm", "xgboost"]
FAMILIAS_ML = ["svr", "mlp", "lstm", "xgboost"]
ESTOCASTICOS = ["mlp", "lstm", "xgboost"]
SUFIXO_MEDIA = "_media_sementes"
LINEAR = "regressao_linear"
REF_PRINCIPAL = "persistencia_inteligente"
REF_SECUNDARIA = "persistencia_ingenua"

ORDEM_MODELOS = [REF_SECUNDARIA, REF_PRINCIPAL, "persistencia_inteligente_v1", LINEAR,
                 *FAMILIAS_ML, *[f + SUFIXO_MEDIA for f in ESTOCASTICOS]]
ROTULOS = {
    REF_SECUNDARIA: "Persistência ingênua",
    REF_PRINCIPAL: "Persistência inteligente",
    "persistencia_inteligente_v1": "Persistência inteligente (v1, céu claro deslocado)",
    LINEAR: "Regressão linear",
    "svr": "SVR", "mlp": "MLP", "lstm": "LSTM", "xgboost": "XGBoost",
    "mlp" + SUFIXO_MEDIA: "MLP (média das sementes)",
    "lstm" + SUFIXO_MEDIA: "LSTM (média das sementes)",
    "xgboost" + SUFIXO_MEDIA: "XGBoost (média das sementes)",
}
V1_PARA_V2 = {"persistencia_ingenua": REF_SECUNDARIA, "persistencia_inteligente": REF_PRINCIPAL,
              "SVM (SVR)": "svr", "MLP": "mlp", "LSTM": "lstm", "XGBoost": "xgboost"}

CENARIOS = {"A": "teste sem alvo imputado",
            "B": "teste sem alvo imputado e sem evento operacional"}
CENARIO_PRINCIPAL = "A"  # mesmas 491 horas do teste da v1; usado nas figuras e na permutação
CLASSES_CEU = ["nublado", "intermediario", "claro"]  # tercis de k_d no teste
N_REPETICOES_PERMUTACAO = 10
SEED = 42
TOLERANCIA_PREVISAO_KW = 1e-2
METRICAS = ["RMSE_kW", "MAE_kW", "MBE_kW", "nRMSE_%", "R2"]
OBS_CV = ("RMSE por dobra da configuração escolhida nestas mesmas dobras: estimativa "
          "otimista (exceto a regressão linear, que não tem ajuste de hiperparâmetros).")

# Cores (skill dataviz, paleta de referência, modo claro): cor fixa por família em todas
# as figuras; categóricas 1-4 validadas (validate_palette.js: CVD ΔE adjacente >= 9,1).
# Verde-água e amarelo ficam < 3:1 sobre a superfície -> legenda + tabela CSV como alívio.
COR_FAMILIA = {"svr": "#2a78d6", "mlp": "#eb6834", "lstm": "#1baf7a", "xgboost": "#eda100",
               LINEAR: "#52514e", REF_PRINCIPAL: "#898781"}
MARCADOR = {"svr": "o", "mlp": "s", "lstm": "^", "xgboost": "D", LINEAR: "v", REF_PRINCIPAL: "X"}
COR_OBSERVADO = "#0b0b0b"
COR_ENFASE = "#2a78d6"
COR_CONTEXTO = "#c3c2b7"
COR_SUPERFICIE = "#fcfcfb"
COR_GRADE = "#e1e0d9"
COR_EIXO = "#c3c2b7"
COR_REFERENCIA = "#898781"
COR_TEXTO_SECUNDARIO = "#52514e"
COR_TEXTO_PRIMARIO = "#0b0b0b"
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]


def log(msg=""):
    print(msg, flush=True)


def rotulo(modelo):
    return ROTULOS.get(modelo, modelo)


def ordenar(tabela, coluna="modelo"):
    ordem = {m: i for i, m in enumerate(ORDEM_MODELOS)}
    return tabela.sort_values(coluna, key=lambda s: s.map(ordem).fillna(len(ordem)), kind="stable")


# ------------------------------------------------------------------------------------------
# Carregamento das previsões salvas
# ------------------------------------------------------------------------------------------
def carregar_previsoes():
    """Lê todos os `predicoes_*_v2.csv`, confere que cobrem as mesmas horas com as mesmas
    observações/flags e acrescenta a previsão média das sementes de MLP, LSTM e XGBoost."""
    caminhos = {nome: caminho_predicoes(nome) for nome in ARQUIVOS_PREVISOES}
    faltando = [str(c) for c in caminhos.values() if not c.exists()]
    if faltando:
        raise FileNotFoundError("Previsões ausentes — rode antes os scripts correspondentes:\n  "
                                + "\n  ".join(faltando))
    prev = pd.concat([pd.read_csv(c, parse_dates=["tempo"]) for c in caminhos.values()],
                     ignore_index=True)
    if list(prev.columns) != COLUNAS_PREDICOES:
        raise ValueError(f"Colunas inesperadas nas previsões: {list(prev.columns)}")

    colunas_ref = ["y_obs", "alvo_imputado", "evento_operacional"]
    referencia = (prev[prev["modelo"] == REF_SECUNDARIA].set_index("tempo")[colunas_ref]
                  .sort_index())
    for (modelo, semente), d in prev.groupby(["modelo", "semente"]):
        if not d.set_index("tempo")[colunas_ref].sort_index().equals(referencia):
            raise ValueError(f"Previsões de {modelo} (semente {semente}) não cobrem as mesmas "
                             "horas/observações/flags das demais")

    medias = []
    for familia in ESTOCASTICOS:
        d = prev[prev["modelo"] == familia]
        m = d.groupby("tempo", as_index=False).agg(
            y_obs=("y_obs", "first"), y_pred=("y_pred", "mean"),
            alvo_imputado=("alvo_imputado", "first"),
            evento_operacional=("evento_operacional", "first"))
        m["modelo"] = familia + SUFIXO_MEDIA
        m["semente"] = SEMENTE_DETERMINISTICO
        medias.append(m[COLUNAS_PREDICOES])
    prev = pd.concat([prev, *medias], ignore_index=True)
    prev["hora"] = prev["tempo"].dt.hour
    prev["dia"] = prev["tempo"].dt.normalize()

    sementes = prev[prev["modelo"].isin(ESTOCASTICOS)].groupby("modelo")["semente"].nunique()
    log(f"Previsões: {referencia.shape[0]} horas de teste, {prev['modelo'].nunique()} modelos "
        f"(sementes: {sementes.to_dict()}); horas, observações e flags idênticas entre arquivos: OK")
    return prev


def no_cenario(prev, cenario):
    mascara = ~prev["alvo_imputado"]
    if cenario == "B":
        mascara &= ~prev["evento_operacional"]
    return prev[mascara]


# ------------------------------------------------------------------------------------------
# Métricas (média ± dp entre sementes)
# ------------------------------------------------------------------------------------------
def metricas(y, p):
    return {"RMSE_kW": rmse(y, p), "MAE_kW": mae(y, p), "MBE_kW": mbe(y, p),
            "nRMSE_%": 100 * nrmse(y, p, CAPACIDADE_KWP), "R2": r2(y, p)}


def por_semente(sub, chaves, funcao):
    """Aplica `funcao(d)` (-> dict) a cada grupo (chaves + semente)."""
    linhas = []
    for valores, d in sub.groupby(chaves + ["semente"], sort=False):
        linhas.append({**dict(zip(chaves + ["semente"], valores)), "n_horas": len(d), **funcao(d)})
    return pd.DataFrame(linhas)


def agregar_sementes(tabela, chaves, colunas):
    """Média e desvio-padrão amostral (ddof=1) entre sementes; dp = NaN com 1 semente."""
    g = tabela.groupby(chaves, sort=False)
    saida = g.agg(n_horas=("n_horas", "first"), n_sementes=("semente", "size"))
    for c in colunas:
        saida[f"{c}_media"] = g[c].mean()
        saida[f"{c}_dp"] = g[c].std(ddof=1)
    return saida.reset_index()


def tabela_metricas(prev):
    partes = []
    for cenario in CENARIOS:
        sub = no_cenario(prev, cenario)
        ref = {m: rmse(d["y_obs"], d["y_pred"]) for m, d in sub.groupby("modelo")
               if m in (REF_PRINCIPAL, REF_SECUNDARIA)}

        def funcao(d):
            r = metricas(d["y_obs"], d["y_pred"])
            r["skill_vs_inteligente"] = 1 - r["RMSE_kW"] / ref[REF_PRINCIPAL]
            r["skill_vs_ingenua"] = 1 - r["RMSE_kW"] / ref[REF_SECUNDARIA]
            return r

        tabela = agregar_sementes(por_semente(sub, ["modelo"], funcao), ["modelo"],
                                  METRICAS + ["skill_vs_inteligente", "skill_vs_ingenua"])
        tabela.insert(0, "cenario", cenario)
        partes.append(ordenar(tabela))
    tabela = pd.concat(partes, ignore_index=True)
    tabela.insert(2, "rotulo", tabela["modelo"].map(rotulo))
    return tabela


def _media_dp(media, dp, fmt="{:.1f}"):
    return fmt.format(media) if pd.isna(dp) else f"{fmt.format(media)} ± {fmt.format(dp)}"


def imprimir_metricas(tabela):
    for cenario, descricao in CENARIOS.items():
        t = tabela[tabela["cenario"] == cenario]
        log(f"\n== MÉTRICAS — cenário {cenario} ({descricao}; {int(t['n_horas'].iloc[0])} horas) ==")
        log(f"{'modelo':<52}{'sem.':>4} {'RMSE (kW)':>15} {'MAE (kW)':>15} {'MBE (kW)':>15} "
            f"{'nRMSE (%)':>13} {'R²':>15} {'skill intelig.':>16} {'skill ingênua':>16}")
        for _, l in t.iterrows():
            log(f"{l['rotulo']:<52}{int(l['n_sementes']):>4} "
                f"{_media_dp(l['RMSE_kW_media'], l['RMSE_kW_dp']):>15} "
                f"{_media_dp(l['MAE_kW_media'], l['MAE_kW_dp']):>15} "
                f"{_media_dp(l['MBE_kW_media'], l['MBE_kW_dp'], '{:+.1f}'):>15} "
                f"{_media_dp(l['nRMSE_%_media'], l['nRMSE_%_dp'], '{:.2f}'):>13} "
                f"{_media_dp(l['R2_media'], l['R2_dp'], '{:.3f}'):>15} "
                f"{_media_dp(100 * l['skill_vs_inteligente_media'], 100 * l['skill_vs_inteligente_dp'], '{:+.1f}'):>16} "
                f"{_media_dp(100 * l['skill_vs_ingenua_media'], 100 * l['skill_vs_ingenua_dp'], '{:+.1f}'):>16}")
    log("(skill em %: 1 - RMSE/RMSE_referência; média ± dp entre sementes)")


# ------------------------------------------------------------------------------------------
# Diebold-Mariano (correção HLN) sobre perdas diárias
# ------------------------------------------------------------------------------------------
def modelos_dm():
    return ["svr"] + [f + SUFIXO_MEDIA for f in ESTOCASTICOS]


def pares_dm():
    """Todos os modelos de ML entre si; cada modelo (inclusive a regressão linear) contra a
    persistência inteligente; cada modelo de ML contra a regressão linear."""
    ml = modelos_dm()
    return (list(combinations(ml, 2))
            + [(m, REF_PRINCIPAL) for m in [LINEAR, *ml]]
            + [(m, LINEAR) for m in ml])


def perdas_diarias(sub, modelo):
    """L_t = média dos erros quadráticos horários do dia t (kW²)."""
    d = sub[sub["modelo"] == modelo]
    return ((d["y_pred"] - d["y_obs"]) ** 2).groupby(d["dia"]).mean()


def teste_dm_hln(perda_1, perda_2):
    """d_t = L1_t - L2_t, horizonte h = 1 (uma previsão por dia):
    DM = média(d) / sqrt(gama_0 / n), com gama_0 = (1/n) Σ (d_t - média)² — a variância
    de longo prazo do DM original para h = 1 (sem termos de autocovariância);
    DM* = DM · sqrt((n - 1)/n) (Harvey, Leybourne e Newbold, 1997, com h = 1);
    p-valor bicaudal pela t de Student com n - 1 graus de liberdade.
    (Com h = 1, DM* coincide com a estatística t pareada das perdas diárias.)"""
    d = (perda_1 - perda_2).dropna()
    n = len(d)
    media = float(d.mean())
    desvios = (d - media).to_numpy()
    gama0 = float(np.mean(desvios ** 2))
    dm = media / np.sqrt(gama0 / n)
    dm_hln = dm * np.sqrt((n - 1) / n)
    p_valor = 2 * stats.t.sf(abs(dm_hln), df=n - 1)
    autocorr = float(np.sum(desvios[1:] * desvios[:-1]) / np.sum(desvios ** 2))
    return {"n_dias": n, "media_d_kW2": media, "DM": dm, "DM_HLN": dm_hln, "p_valor": p_valor,
            "autocorr_lag1_d": autocorr}


def tabela_diebold_mariano(prev):
    linhas = []
    for cenario in CENARIOS:
        sub = no_cenario(prev, cenario)
        for m1, m2 in pares_dm():
            r = teste_dm_hln(perdas_diarias(sub, m1), perdas_diarias(sub, m2))
            linhas.append({"cenario": cenario, "modelo_1": m1, "modelo_2": m2,
                           "rotulo_1": rotulo(m1), "rotulo_2": rotulo(m2), **r,
                           "melhor": m1 if r["media_d_kW2"] < 0 else m2,
                           "significativo_5%": r["p_valor"] < 0.05})
    return pd.DataFrame(linhas)


def imprimir_dm(tabela):
    log("\n== DIEBOLD-MARIANO (HLN, perdas diárias; d = L1 - L2 < 0 favorece o modelo 1) ==")
    for cenario in CENARIOS:
        t = tabela[tabela["cenario"] == cenario]
        log(f"Cenário {cenario} ({int(t['n_dias'].iloc[0])} dias):")
        for _, l in t.iterrows():
            log(f"   {l['rotulo_1']:<30} x {l['rotulo_2']:<30} DM* {l['DM_HLN']:+6.2f}  "
                f"p = {l['p_valor']:.3f}{' *' if l['significativo_5%'] else '  '}  "
                f"melhor: {rotulo(l['melhor']):<30} autocorr. lag 1 de d: {l['autocorr_lag1_d']:+.2f}")
    log("   (* p < 0,05)")


# ------------------------------------------------------------------------------------------
# Erro por hora do dia e por classe de céu
# ------------------------------------------------------------------------------------------
def tabela_erro_por_hora(prev):
    partes = []
    for cenario in CENARIOS:
        sub = no_cenario(prev, cenario)
        t = agregar_sementes(
            por_semente(sub, ["modelo", "hora"],
                        lambda d: {"RMSE_kW": rmse(d["y_obs"], d["y_pred"]),
                                   "MBE_kW": mbe(d["y_obs"], d["y_pred"])}),
            ["modelo", "hora"], ["RMSE_kW", "MBE_kW"])
        t.insert(0, "cenario", cenario)
        partes.append(ordenar(t.sort_values("hora", kind="stable")))
    tabela = pd.concat(partes, ignore_index=True)
    tabela.insert(2, "rotulo", tabela["modelo"].map(rotulo))
    return tabela


def classes_ceu_diarias(prev):
    """k_d = Σ P_ac observado / Σ P_ac de céu claro (modelo de `baselines_v2`: Ineichen,
    timestamps localizados, ponto médio da hora) no dia; classes = tercis de k_d no teste.
    k_d usa só as horas do cenário B (sem alvo imputado e sem evento operacional), para
    que uma parada da usina não rebaixe a classe de um dia de céu claro; a mesma
    classificação vale para os dois cenários."""
    base = no_cenario(prev, "B")
    base = base[base["modelo"] == REF_SECUNDARIA]
    diario = pd.DataFrame({"dia": base["dia"].to_numpy(), "p_obs": base["y_obs"].to_numpy(),
                           "p_ceu_claro": clearsky_potencia(base["tempo"])})
    diario = diario.groupby("dia").agg(soma_obs_kWh=("p_obs", "sum"),
                                       soma_ceu_claro_kWh=("p_ceu_claro", "sum"),
                                       n_horas=("p_obs", "size"))
    diario["k_d"] = diario["soma_obs_kWh"] / diario["soma_ceu_claro_kWh"]
    diario["classe_ceu"] = pd.qcut(diario["k_d"], 3, labels=CLASSES_CEU).astype(str)
    return diario.reset_index()


def tabela_erro_por_classe(prev, diario):
    classe_do_dia = diario.set_index("dia")["classe_ceu"]
    partes = []
    for cenario in CENARIOS:
        sub = no_cenario(prev, cenario).assign(classe_ceu=lambda d: d["dia"].map(classe_do_dia))
        sub = sub[sub["classe_ceu"].notna()]
        ref = {c: rmse(d["y_obs"], d["y_pred"])
               for c, d in sub[sub["modelo"] == REF_PRINCIPAL].groupby("classe_ceu")}

        def funcao(d):
            r = metricas(d["y_obs"], d["y_pred"])
            r["skill_vs_inteligente"] = 1 - r["RMSE_kW"] / ref[d["classe_ceu"].iloc[0]]
            r["n_dias"] = d["dia"].nunique()
            return r

        t = agregar_sementes(por_semente(sub, ["modelo", "classe_ceu"], funcao),
                             ["modelo", "classe_ceu"], METRICAS + ["skill_vs_inteligente", "n_dias"])
        t = t.drop(columns=["n_dias_dp"]).rename(columns={"n_dias_media": "n_dias"})
        t["classe_ceu"] = pd.Categorical(t["classe_ceu"], CLASSES_CEU, ordered=True)
        t.insert(0, "cenario", cenario)
        partes.append(ordenar(t.sort_values("classe_ceu", kind="stable")))
    tabela = pd.concat(partes, ignore_index=True)
    tabela.insert(2, "rotulo", tabela["modelo"].map(rotulo))
    return tabela


def imprimir_classes(diario, tabela):
    limites = diario.groupby("classe_ceu")["k_d"].agg(["min", "max", "size"]).reindex(CLASSES_CEU)
    log("\n== CLASSES DE CÉU (tercis de k_d no teste) ==")
    for classe, l in limites.iterrows():
        log(f"   {classe:<14} k_d de {l['min']:.3f} a {l['max']:.3f}  ({int(l['size'])} dias)")
    t = tabela[(tabela["cenario"] == CENARIO_PRINCIPAL)
               & tabela["modelo"].isin([REF_PRINCIPAL, LINEAR, *FAMILIAS_ML])]
    piv = t.pivot_table(index="rotulo", columns="classe_ceu", values="RMSE_kW_media",
                        observed=True, sort=False)
    log(f"   RMSE (kW, média entre sementes) por classe — cenário {CENARIO_PRINCIPAL}:")
    log("   " + piv.round(1).to_string().replace("\n", "\n   "))


# ------------------------------------------------------------------------------------------
# Importância por permutação (recarrega os modelos salvos)
# ------------------------------------------------------------------------------------------
def _resumo_familia(familia):
    caminho = caminho_saida("modelos", f"{familia}_resumo_v2.json")
    return json.loads(caminho.read_text(encoding="utf-8"))


def construir_preditores():
    """{modelo: função X -> previsão (kW, clip [0, CAPACIDADE_KWP])}. MLP, LSTM e XGBoost
    = média das previsões (já com clip) das sementes, como em `carregar_previsoes`."""
    from modelos import lstm_v2, mlp_v2, svm_v2, xgboost_model_v2

    def clip(p):
        return np.clip(np.asarray(p, dtype=float).ravel(), 0, CAPACIDADE_KWP)

    linear = joblib.load(caminho_saida("modelos", f"{LINEAR}_bet01_v2.joblib"))
    preditores = {LINEAR: lambda X, m=linear: clip(m.predict(X))}
    modulos = {"svr": svm_v2, "mlp": mlp_v2, "lstm": lstm_v2, "xgboost": xgboost_model_v2}
    for familia, modulo in modulos.items():
        sementes = _resumo_familia(familia)["sementes_retreino"]
        modelos = [modulo.carregar_modelo(modulo.arquivos_modelo(s)) for s in sementes]
        nome = familia if familia not in ESTOCASTICOS else familia + SUFIXO_MEDIA
        preditores[nome] = lambda X, ms=modelos: np.mean([clip(m.predict(X)) for m in ms], axis=0)
    return preditores


def tabela_importancia(prev, cenario=CENARIO_PRINCIPAL, n_repeticoes=N_REPETICOES_PERMUTACAO):
    """Aumento do RMSE (kW) quando uma feature — ou um grupo inteiro de features, de uma
    vez — é embaralhado entre as linhas de teste do cenário. As mesmas permutações de
    linhas são usadas para todas as features, grupos e modelos. Como todos os modelos
    preveem linha a linha, as `n_repeticoes` cópias permutadas são empilhadas e previstas
    numa única chamada (o overhead por chamada do Keras domina o custo)."""
    _, teste, _ = carregar_split()
    sub = no_cenario(prev, cenario)
    tempos = np.sort(sub.loc[sub["modelo"] == REF_SECUNDARIA, "tempo"].to_numpy())
    colunas = colunas_features()
    X = teste.loc[tempos, colunas]
    y = teste.loc[tempos, ALVO].to_numpy()
    alvos = ([("feature", c, [c]) for c in colunas]
             + [("grupo", g, cols) for g, cols in grupos_features().items()])
    rng = np.random.default_rng(SEED)
    permutacoes = [rng.permutation(len(X)) for _ in range(n_repeticoes)]
    empilhado = pd.concat([X] * n_repeticoes, ignore_index=True)

    log(f"\n== IMPORTÂNCIA POR PERMUTAÇÃO (cenário {cenario}, {len(X)} horas, "
        f"{n_repeticoes} repetições, {len(colunas)} features + {len(alvos) - len(colunas)} grupos) ==")
    linhas = []
    for nome, prever in construir_preditores().items():
        inicio = time.time()
        base = prever(X)
        salvo = sub[sub["modelo"] == nome].set_index("tempo").loc[tempos, "y_pred"].to_numpy()
        dif = float(np.max(np.abs(base - salvo)))
        if dif > TOLERANCIA_PREVISAO_KW:
            raise RuntimeError(f"{nome}: previsão dos modelos recarregados difere da salva ({dif:.3g} kW)")
        rmse_base = rmse(y, base)
        for tipo, alvo, cols in alvos:
            Xp = empilhado.copy()
            valores = X[cols].to_numpy()
            Xp[cols] = np.vstack([valores[perm] for perm in permutacoes])
            previsoes = prever(Xp).reshape(n_repeticoes, len(X))
            aumentos = [rmse(y, p) - rmse_base for p in previsoes]
            linhas.append({"cenario": cenario, "modelo": nome, "rotulo": rotulo(nome),
                           "tipo": tipo, "nome": alvo, "n_colunas": len(cols),
                           "rmse_base_kW": rmse_base,
                           "aumento_rmse_kW_media": float(np.mean(aumentos)),
                           "aumento_rmse_kW_dp": float(np.std(aumentos, ddof=1))})
        grupos = pd.DataFrame([l for l in linhas if l["modelo"] == nome and l["tipo"] == "grupo"])
        top = grupos.nlargest(3, "aumento_rmse_kW_media")
        log(f"   {rotulo(nome):<30} RMSE base {rmse_base:7.2f} kW (= previsão salva, máx |dif| "
            f"{dif:.1e}) | grupos mais importantes: "
            + ", ".join(f"{r['nome']} (+{r['aumento_rmse_kW_media']:.0f})" for _, r in top.iterrows())
            + f" | {time.time() - inicio:.0f}s")
    tabela = pd.DataFrame(linhas)
    tabela["aumento_rmse_%"] = 100 * tabela["aumento_rmse_kW_media"] / tabela["rmse_base_kW"]
    return tabela


# ------------------------------------------------------------------------------------------
# CV por dobra e comparação v1 x v2
# ------------------------------------------------------------------------------------------
def tabela_cv_por_dobra():
    linhas = []
    for familia in [LINEAR, *FAMILIAS_ML]:
        resumo = _resumo_familia(familia)
        for d in resumo["dobras_cv"]:
            linhas.append({"familia": familia, "rotulo": rotulo(familia), **d,
                           "rmse_cv_medio_kW": resumo["rmse_cv_medio_kW"],
                           "otimista": familia != LINEAR, "observacao": OBS_CV})
    tabela = pd.DataFrame(linhas)
    piv = tabela.pivot_table(index=["dobra", "inicio_validacao", "fim_validacao"],
                             columns="rotulo", values="rmse_kW", sort=False)
    piv.index = [f"{d} ({i[:10]} a {f[:10]})" for d, i, f in piv.index]
    log("\n== RMSE POR DOBRA DE CV (kW) — melhor configuração de cada família ==")
    log("   " + piv.round(1).to_string().replace("\n", "\n   "))
    log(f"   Obs.: {OBS_CV}")
    return tabela


def tabela_v1_v2(metricas_v2, prev):
    """Lado a lado com `resultados/comparacao_final_tabela.csv` (v1, só leitura), no cenário
    A — que tem exatamente as mesmas horas de teste da v1 (conferido abaixo)."""
    v1 = pd.read_csv(TABELA_V1).set_index("modelo")
    dados_v1 = pd.read_csv(FEATURES_CSV_V1, parse_dates=["tempo"], usecols=["tempo", ALVO])
    teste_v1 = dados_v1[dados_v1["tempo"] >= pd.Timestamp(CORTE_DATA_ALVO)].set_index("tempo")[ALVO]
    teste_v2 = prev[prev["modelo"] == REF_SECUNDARIA].set_index("tempo")["y_obs"].sort_index()
    teste_a = no_cenario(prev, "A")
    teste_a = teste_a[teste_a["modelo"] == REF_SECUNDARIA]
    mesmas_horas = (teste_v1.index.equals(teste_v2.index) and np.allclose(teste_v1, teste_v2)
                    and len(teste_a) == len(teste_v2))

    a = metricas_v2[metricas_v2["cenario"] == "A"].set_index("modelo")
    linhas = []
    for nome_v1, modelo in V1_PARA_V2.items():
        l1, l2 = v1.loc[nome_v1], a.loc[modelo]
        linha = {"modelo": modelo, "rotulo": rotulo(modelo), "n_v1": int(l1["n"]),
                 "n_v2": int(l2["n_horas"]), "n_sementes_v2": int(l2["n_sementes"]),
                 "RMSE_kW_v1": l1["RMSE_kW"], "RMSE_kW_v2": l2["RMSE_kW_media"],
                 "RMSE_kW_v2_dp": l2["RMSE_kW_dp"],
                 "RMSE_kW_v2_media_sementes": (a.loc[modelo + SUFIXO_MEDIA, "RMSE_kW_media"]
                                               if modelo in ESTOCASTICOS else l2["RMSE_kW_media"]),
                 "delta_RMSE_kW": l2["RMSE_kW_media"] - l1["RMSE_kW"],
                 "MAE_kW_v1": l1["MAE_kW"], "MAE_kW_v2": l2["MAE_kW_media"],
                 "MBE_kW_v1": l1["MBE_kW"], "MBE_kW_v2": l2["MBE_kW_media"],
                 "R2_v1": l1["R2"], "R2_v2": l2["R2_media"],
                 "skill_vs_ingenua_v1": l1.get("skill_score_vs_persistencia", np.nan),
                 "skill_vs_ingenua_v2": l2["skill_vs_ingenua_media"],
                 "skill_vs_inteligente_v2": l2["skill_vs_inteligente_media"]}
        linhas.append(linha)
    tabela = ordenar(pd.DataFrame(linhas))
    rastreio = a.loc["persistencia_inteligente_v1", "RMSE_kW_media"]

    log("\n== COMPARAÇÃO v1 x v2 (cenário A) ==")
    log(f"   Mesmas {len(teste_v2)} horas de teste e mesmas observações da v1: "
        f"{'OK' if mesmas_horas else 'NÃO — comparar com cautela'}")
    log(f"   Persistência inteligente com o céu claro da v1, recalculada na v2: {rastreio:.2f} kW "
        f"(v1: {v1.loc['persistencia_inteligente', 'RMSE_kW']:.2f} kW)")
    log(f"   {'modelo':<26}{'RMSE v1':>9}{'RMSE v2':>17}{'dif.':>8}{'skill ingênua v1 -> v2':>24}"
        f"{'skill intelig. v2':>19}")
    for _, l in tabela.iterrows():
        skill_v1 = "—" if pd.isna(l["skill_vs_ingenua_v1"]) else f"{100 * l['skill_vs_ingenua_v1']:+.1f}"
        log(f"   {l['rotulo']:<26}{l['RMSE_kW_v1']:9.1f}"
            f"{_media_dp(l['RMSE_kW_v2'], l['RMSE_kW_v2_dp']):>17}{l['delta_RMSE_kW']:+8.1f}"
            f"{skill_v1:>14} -> {100 * l['skill_vs_ingenua_v2']:+6.1f}"
            f"{100 * l['skill_vs_inteligente_v2']:+19.1f}")
    log("   (a persistência inteligente da v2 usa o céu claro corrigido — registro M8)")
    return tabela


# ------------------------------------------------------------------------------------------
# Figuras
# ------------------------------------------------------------------------------------------
def _estilizar(eixo, grade="both"):
    eixo.set_facecolor(COR_SUPERFICIE)
    if grade:
        eixo.grid(True, axis=grade, color=COR_GRADE, linewidth=0.8)
    eixo.set_axisbelow(True)
    for spine in eixo.spines.values():
        spine.set_color(COR_EIXO)
    eixo.tick_params(colors=COR_TEXTO_SECUNDARIO, labelsize=8)


def _salvar(fig, caminho):
    fig.savefig(caminho, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _serie_modelo(sub, modelo):
    return sub[sub["modelo"] == modelo].set_index("tempo").sort_index()


def _nome_figura(familia, prev):
    """Família estocástica -> previsão média das sementes (rótulo com o nº de sementes)."""
    if familia not in ESTOCASTICOS:
        return familia, rotulo(familia)
    n = prev.loc[prev["modelo"] == familia, "semente"].nunique()
    return familia + SUFIXO_MEDIA, f"{rotulo(familia)} (média de {n} sementes)"


def melhor_familia(metricas_v2):
    a = metricas_v2[(metricas_v2["cenario"] == CENARIO_PRINCIPAL)
                    & metricas_v2["modelo"].isin(FAMILIAS_ML)]
    return a.loc[a["RMSE_kW_media"].idxmin(), "modelo"]


def figura_dispersao(prev, caminho):
    """Previsto x observado das 4 famílias (uma série por painel, mesma cor), y = x como
    referência e RMSE/R² da série plotada anotados."""
    sub = no_cenario(prev, CENARIO_PRINCIPAL)
    fig, eixos = plt.subplots(1, len(FAMILIAS_ML), figsize=(19, 4.8), sharex=True, sharey=True,
                              facecolor=COR_SUPERFICIE)
    limite = CAPACIDADE_KWP * 1.03
    for eixo, familia in zip(eixos, FAMILIAS_ML):
        modelo, titulo = _nome_figura(familia, prev)
        s = _serie_modelo(sub, modelo)
        _estilizar(eixo)
        eixo.plot([0, limite], [0, limite], color=COR_REFERENCIA, linewidth=1.0, zorder=1,
                  label="Acerto perfeito (y = x)")
        eixo.scatter(s["y_obs"], s["y_pred"], color=COR_ENFASE, s=14, alpha=0.5, linewidths=0,
                     zorder=2, label="Previsão")
        m = metricas(s["y_obs"], s["y_pred"])
        eixo.text(0.04, 0.96, f"R² = {m['R2']:.3f}\nRMSE = {m['RMSE_kW']:.0f} kW",
                  transform=eixo.transAxes, ha="left", va="top", fontsize=9,
                  color=COR_TEXTO_SECUNDARIO)
        eixo.set_title(titulo, fontsize=11, color=COR_TEXTO_SECUNDARIO)
        eixo.set_xlabel("Observado (kW)", fontsize=9, color=COR_TEXTO_SECUNDARIO)
        eixo.set_xlim(0, limite)
        eixo.set_ylim(0, limite)
        eixo.set_aspect("equal")
    eixos[0].set_ylabel("Previsto (kW)", fontsize=9, color=COR_TEXTO_SECUNDARIO)
    linhas, rotulos = eixos[0].get_legend_handles_labels()
    fig.legend(linhas, rotulos, loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Dispersão previsto vs. observado — teste (cenário {CENARIO_PRINCIPAL}, "
                 f"{len(s)} horas)", fontsize=12, color=COR_TEXTO_PRIMARIO)
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    _salvar(fig, caminho)


def selecionar_dias_representativos(serie):
    """5 dias ao longo da distribuição do RMSE diário (mín., Q1, mediana, Q3, máx.)."""
    rmse_diario = np.sqrt(((serie["y_pred"] - serie["y_obs"]) ** 2).groupby(serie.index.normalize()).mean())
    alvos = {"Mín. RMSE": 0.0, "Q1 RMSE": 0.25, "Mediana RMSE": 0.5, "Q3 RMSE": 0.75, "Máx. RMSE": 1.0}
    usados, selecionados = set(), []
    for nome, q in alvos.items():
        candidatos = (rmse_diario - rmse_diario.quantile(q)).abs().sort_values()
        dia = next(d for d in candidatos.index if d not in usados)
        usados.add(dia)
        selecionados.append((dia, nome, rmse_diario[dia]))
    return selecionados


def figura_previsto_observado(prev, familia, caminho):
    sub = no_cenario(prev, CENARIO_PRINCIPAL)
    modelo, titulo_modelo = _nome_figura(familia, prev)
    serie = _serie_modelo(sub, modelo)
    persistencia = _serie_modelo(sub, REF_PRINCIPAL)
    dias = selecionar_dias_representativos(serie)
    fig, eixos = plt.subplots(1, len(dias), figsize=(19, 3.8), sharey=True, facecolor=COR_SUPERFICIE)
    for eixo, (dia, nome, rmse_dia) in zip(eixos, dias):
        s = serie[serie.index.normalize() == dia]
        p = persistencia[persistencia.index.normalize() == dia]
        _estilizar(eixo)
        eixo.plot(s.index.hour, s["y_obs"], color=COR_OBSERVADO, linewidth=2, marker="o",
                  markersize=5, label="Observado")
        eixo.plot(s.index.hour, s["y_pred"], color=COR_FAMILIA[familia], linewidth=2,
                  marker=MARCADOR[familia], markersize=5, label=f"Previsto — {titulo_modelo}")
        eixo.plot(p.index.hour, p["y_pred"], color=COR_FAMILIA[REF_PRINCIPAL], linewidth=2,
                  marker=MARCADOR[REF_PRINCIPAL], markersize=5,
                  label=f"Previsto — {rotulo(REF_PRINCIPAL).lower()}")
        eixo.set_title(f"{nome} ({dia:%d/%m/%Y}; {rmse_dia:.0f} kW)", fontsize=10,
                       color=COR_TEXTO_SECUNDARIO)
        eixo.set_xlabel("Hora do dia", fontsize=9, color=COR_TEXTO_SECUNDARIO)
    eixos[0].set_ylabel("P_ac (kW)", fontsize=9, color=COR_TEXTO_SECUNDARIO)
    linhas, rotulos = eixos[0].get_legend_handles_labels()
    fig.legend(linhas, rotulos, loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Previsto vs. observado — dias de teste do mín. ao máx. RMSE diário — "
                 f"{titulo_modelo}", fontsize=12, color=COR_TEXTO_PRIMARIO)
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    _salvar(fig, caminho)


def figura_rmse_barras(metricas_v2, caminho):
    """Ranking de RMSE (cenário principal): famílias de ML em destaque (com ± dp entre
    sementes), baselines em cinza de contexto; ordenado do menor para o maior RMSE."""
    modelos = [REF_SECUNDARIA, REF_PRINCIPAL, LINEAR, *FAMILIAS_ML]
    t = metricas_v2[(metricas_v2["cenario"] == CENARIO_PRINCIPAL) & metricas_v2["modelo"].isin(modelos)]
    t = t.sort_values("RMSE_kW_media").reset_index(drop=True)
    fig, eixo = plt.subplots(figsize=(9, 4.4), facecolor=COR_SUPERFICIE)
    _estilizar(eixo, grade="x")
    y_pos = np.arange(len(t))[::-1]
    cores = [COR_ENFASE if m in FAMILIAS_ML else COR_CONTEXTO for m in t["modelo"]]
    eixo.barh(y_pos, t["RMSE_kW_media"], color=cores, height=0.55)
    dp = t["RMSE_kW_dp"].fillna(0).to_numpy()
    tem_dp = t["RMSE_kW_dp"].notna().to_numpy()
    eixo.errorbar(t["RMSE_kW_media"][tem_dp], y_pos[tem_dp], xerr=dp[tem_dp], fmt="none",
                  ecolor=COR_TEXTO_PRIMARIO, elinewidth=1, capsize=3)
    folga = t["RMSE_kW_media"].max() * 0.012
    for y, media, desvio, n in zip(y_pos, t["RMSE_kW_media"], t["RMSE_kW_dp"], t["n_sementes"]):
        texto = f"{media:.0f} kW" if pd.isna(desvio) else f"{media:.0f} ± {desvio:.0f} kW ({n} sementes)"
        eixo.text(media + (0 if pd.isna(desvio) else desvio) + folga, y, texto, va="center",
                  ha="left", fontsize=9, color=COR_TEXTO_SECUNDARIO)
    eixo.set_yticks(y_pos)
    eixo.set_yticklabels([rotulo(m) for m in t["modelo"]], fontsize=9, color=COR_TEXTO_SECUNDARIO)
    eixo.set_xlabel("RMSE (kW) — menor é melhor", fontsize=9, color=COR_TEXTO_SECUNDARIO)
    eixo.set_xlim(0, (t["RMSE_kW_media"] + t["RMSE_kW_dp"].fillna(0)).max() * 1.35)
    for spine in ("top", "right", "left"):
        eixo.spines[spine].set_visible(False)
    eixo.tick_params(left=False)
    legenda = [plt.Rectangle((0, 0), 1, 1, color=COR_ENFASE, label="Modelos de ML"),
               plt.Rectangle((0, 0), 1, 1, color=COR_CONTEXTO,
                             label="Referências (persistências e regressão linear)")]
    eixo.legend(handles=legenda, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
                frameon=False, fontsize=9)
    eixo.set_title(f"RMSE no teste (cenário {CENARIO_PRINCIPAL}, {int(t['n_horas'].iloc[0])} horas)",
                   fontsize=12, color=COR_TEXTO_PRIMARIO, loc="left")
    fig.tight_layout()
    _salvar(fig, caminho)


def figura_erro_por_hora(tabela_hora, caminho):
    """RMSE e MBE por hora do dia em dois painéis (escalas diferentes -> nunca eixo duplo);
    cor e marcador fixos por modelo; MLP, LSTM e XGBoost pela média entre sementes."""
    modelos = [REF_PRINCIPAL, LINEAR, *FAMILIAS_ML]
    t = tabela_hora[(tabela_hora["cenario"] == CENARIO_PRINCIPAL) & tabela_hora["modelo"].isin(modelos)]
    fig, (eixo_rmse, eixo_mbe) = plt.subplots(1, 2, figsize=(15, 4.6), sharex=True,
                                              facecolor=COR_SUPERFICIE)
    for eixo, coluna, titulo in ((eixo_rmse, "RMSE_kW_media", "RMSE por hora do dia (kW)"),
                                 (eixo_mbe, "MBE_kW_media", "MBE por hora do dia (kW; > 0 = superestima)")):
        _estilizar(eixo)
        if coluna.startswith("MBE"):
            eixo.axhline(0, color=COR_REFERENCIA, linewidth=1, zorder=1)
        for modelo in modelos:
            s = t[t["modelo"] == modelo].sort_values("hora")
            eixo.plot(s["hora"], s[coluna], color=COR_FAMILIA[modelo], linewidth=2,
                      marker=MARCADOR[modelo], markersize=5, label=rotulo(modelo), zorder=2)
        eixo.set_title(titulo, fontsize=11, color=COR_TEXTO_SECUNDARIO, loc="left")
        eixo.set_xlabel("Hora do dia (início do intervalo horário)", fontsize=9,
                        color=COR_TEXTO_SECUNDARIO)
        eixo.set_xticks(sorted(t["hora"].unique()))
    linhas, rotulos = eixo_rmse.get_legend_handles_labels()
    fig.legend(linhas, rotulos, loc="lower center", ncol=len(modelos), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Erro por hora do dia — teste (cenário {CENARIO_PRINCIPAL}); "
                 "MLP, LSTM e XGBoost: média entre sementes", fontsize=12, color=COR_TEXTO_PRIMARIO)
    fig.tight_layout(rect=[0, 0.07, 1, 0.94])
    _salvar(fig, caminho)


# ------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Comparação final dos modelos (v2)")
    parser.add_argument("--rapido", action="store_true",
                        help="lê e grava no sandbox do modo rápido (resultados_v2/rapido/)")
    parser.add_argument("--sem-importancia", action="store_true",
                        help="não calcula a importância por permutação (a parte mais lenta)")
    args = parser.parse_args()
    if args.rapido:
        ativar_modo_rapido()

    tabelas = {nome: caminho_saida("tabelas", f"{nome}_v2.csv") for nome in (
        "metricas", "diebold_mariano", "erro_por_hora", "classes_ceu_diarias",
        "erro_por_classe_ceu", "cv_por_dobra")}
    tabelas["comparacao_v1_v2"] = caminho_saida("tabelas", "comparacao_v1_v2.csv")  # já termina em _v2
    if not args.sem_importancia:
        tabelas["importancia_permutacao"] = caminho_saida("tabelas", "importancia_permutacao_v2.csv")
    figuras = {nome: caminho_saida("figuras", f"{nome}_v2.png") for nome in (
        "dispersao", "previsto_vs_observado", "rmse_barras", "erro_por_hora")}
    garantir_inexistentes([*tabelas.values(), *figuras.values()])

    prev = carregar_previsoes()
    resultados = {"metricas": tabela_metricas(prev)}
    imprimir_metricas(resultados["metricas"])
    resultados["diebold_mariano"] = tabela_diebold_mariano(prev)
    imprimir_dm(resultados["diebold_mariano"])
    resultados["erro_por_hora"] = tabela_erro_por_hora(prev)
    resultados["classes_ceu_diarias"] = classes_ceu_diarias(prev)
    resultados["erro_por_classe_ceu"] = tabela_erro_por_classe(prev, resultados["classes_ceu_diarias"])
    imprimir_classes(resultados["classes_ceu_diarias"], resultados["erro_por_classe_ceu"])
    resultados["cv_por_dobra"] = tabela_cv_por_dobra()
    resultados["comparacao_v1_v2"] = tabela_v1_v2(resultados["metricas"], prev)
    if not args.sem_importancia:
        resultados["importancia_permutacao"] = tabela_importancia(prev)

    for nome, caminho in tabelas.items():
        resultados[nome].to_csv(caminho, index=False)

    destaque = melhor_familia(resultados["metricas"])
    figura_dispersao(prev, figuras["dispersao"])
    figura_previsto_observado(prev, destaque, figuras["previsto_vs_observado"])
    figura_rmse_barras(resultados["metricas"], figuras["rmse_barras"])
    figura_erro_por_hora(resultados["erro_por_hora"], figuras["erro_por_hora"])
    log(f"\nTabelas salvas em {tabelas['metricas'].parent}: "
        + ", ".join(c.name for c in tabelas.values()))
    log(f"Figuras salvas em {figuras['dispersao'].parent}: "
        + ", ".join(c.name for c in figuras.values())
        + f" (destaque na série temporal: {rotulo(destaque)}, menor RMSE médio no cenário "
          f"{CENARIO_PRINCIPAL})")
    return resultados


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Passo 1 (v2) — Engenharia de features para previsão day-ahead (D+1) de P_ac (usina bet01).

Baseado em `features.py` (v1, mantido intacto). Reconstrói a grade horária completa
(24h/dia) reusando as etapas de dataset/formando_dataset/preprocessamento.py (importadas
sem alteração), calcula defasagens "mesma hora, N dias atrás" via pivô data x hora,
anexa as features determinísticas do horário-alvo e recorta para as horas diurnas de
D+1. Mudanças em relação à v1 (detalhadas em resultados_v2/registro_mudancas_v2.md):
  - sem médias móveis: `{var}_media3d` era combinação linear exata de lag1d, lag2d e
    lag3d; `{var}_media7d` foi removida por decisão do autor após a Fase 1 (VIF > 100,
    pois 4 dos seus 7 termos já são lags presentes) — ficam só os lags de 1, 2, 3 e 7 dias;
  - posição solar e calendário calculados a partir do timestamp, no ponto médio da
    hora (h:30), com pvlib (NREL SPA) — na v1 eram médias horárias dos valores de 1 min,
    o que quebrava o azimute na passagem 0°/360° e dependia da cobertura do logger;
  - azimute como (seno, cosseno); zênite removido (= 90° - elevação);
  - recorte diurno: elevação solar aparente no ponto médio da hora > 0;
  - colunas `alvo_imputado` e `evento_operacional`, que usam dado MEDIDO de D+1 e
    servem SOMENTE para filtrar treino/avaliação — nunca entram como features.

Uso:  python -B features_v2.py               (gera os CSVs em resultados_v2/dados/)
      python -B features_v2.py --sem-salvar  (só calcula e roda as verificações)
"""
import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # não regravar os .pyc versionados da v1 ao importar

import numpy as np
import pandas as pd
import pvlib
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import VARS_LAG, DIAS_LAG, cyc  # noqa: E402  (v1; também expõe preprocessamento)
from preprocessamento import (  # noqa: E402
    ENTRADA as ENTRADA_1MIN,
    etapa_alinhamento,
    etapa_outliers,
    etapa_reamostragem,
    etapa_faltantes,
)
from dados import CAPACIDADE_KWP, CORTE_DATA_ALVO, LAT, LON, ALT, TZ  # noqa: E402
from saidas_v2 import caminho_saida, garantir_inexistentes  # noqa: E402

SAIDA = caminho_saida("dados", "bet01_features_dayahead_v2.csv")
SAIDA_DICIONARIO = caminho_saida("dados", "dicionario_features_v2.csv")

ALVO = "P_ac"

# Mesmo mapeamento de nomes de features.construir_grade_horaria_completa (inline na v1)
RENOMEAR_COLUNAS = {"SOL_IRRADIACAO_HORIZONTAL": "irradiacao_horizontal",
                    "SOL_IRRADIACAO_REFLETIDA": "irradiacao_refletida",
                    "TRACKER_DIRECAO_VENTO": "direcao_vento",
                    "SOL_TEMP_MODULO": "temp_modulo"}

# VARS_LAG e DIAS_LAG vêm da v1 (decisão D1: mesmo conjunto). Sem médias móveis na v2.

# Features determinísticas do horário-alvo (só dependem do timestamp)
VARS_POSICAO_SOLAR = ["elevacao_solar", "azimute_sin", "azimute_cos"]
VARS_CALENDARIO = ["hora_sin", "hora_cos", "dia_ano_sin", "dia_ano_cos"]
VARS_DETERMINISTICAS = VARS_POSICAO_SOLAR + VARS_CALENDARIO

# Colunas de filtragem (dado medido de D+1 — NUNCA usar como feature)
COLUNAS_FLAGS = ["alvo_imputado", "evento_operacional"]

# Critério de evento operacional (decisão D2), limiares ajustáveis.
# Núcleo: hora diurna com G_poa > LIMIAR_G_POA_WM2 e PR = P_ac/(CAPACIDADE_KWP*G_poa/1000)
# < LIMIAR_PR. Expansão: horas vizinhas (±1 h, mesmo dia) de uma hora-núcleo com
# G_poa > LIMIAR_G_POA_VIZINHA_WM2 e PR < LIMIAR_PR_VIZINHA (início/fim do evento dentro da hora).
LIMIAR_G_POA_WM2 = 200.0
LIMIAR_PR = 0.3
LIMIAR_G_POA_VIZINHA_WM2 = 50.0
LIMIAR_PR_VIZINHA = 0.7

N_AMOSTRAS_VERIFICACAO_LAGS = 200
N_DIAS_VERIFICACAO_CAUSALIDADE = 30
SEED = 42

DESCRICAO_VARIAVEIS = {
    "P_ac": ("Potência CA da usina (soma dos 10 inversores), média horária", "kW"),
    "temp_ambiente": ("Temperatura ambiente, média horária", "°C"),
    "umidade_relativa": ("Umidade relativa do ar, média horária", "%"),
    "velocidade_vento": ("Velocidade do vento, média horária", "m/s"),
    "irradiacao_inclinada": ("Irradiância no plano inclinado (POA), média horária", "W/m²"),
    "irradiacao_horizontal": ("Irradiância global horizontal (GHI), média horária", "W/m²"),
    "irradiacao_refletida": ("Irradiância refletida (albedômetro), média horária", "W/m²"),
    "T_inv": ("Temperatura interna média dos 10 inversores, média horária", "°C"),
}
DESCRICAO_DETERMINISTICAS = {
    "elevacao_solar": ("posicao_solar", "elevacao_solar",
                       "Elevação solar aparente (pvlib, NREL SPA) no ponto médio da hora-alvo", "°"),
    "azimute_sin": ("posicao_solar", "azimute_solar",
                    "Seno do azimute solar (0° = norte, sentido horário) no ponto médio da hora-alvo",
                    "adimensional"),
    "azimute_cos": ("posicao_solar", "azimute_solar",
                    "Cosseno do azimute solar (0° = norte, sentido horário) no ponto médio da hora-alvo",
                    "adimensional"),
    "hora_sin": ("calendario", "hora_do_dia", "Seno da hora do dia no ponto médio (período 24 h)",
                 "adimensional"),
    "hora_cos": ("calendario", "hora_do_dia", "Cosseno da hora do dia no ponto médio (período 24 h)",
                 "adimensional"),
    "dia_ano_sin": ("calendario", "dia_do_ano", "Seno do dia do ano (período 365,25 d)",
                    "adimensional"),
    "dia_ano_cos": ("calendario", "dia_do_ano", "Cosseno do dia do ano (período 365,25 d)",
                    "adimensional"),
}


def log(msg):
    print(msg, flush=True)


def construir_grade_horaria_completa():
    """Reproduz as etapas 1-4 de preprocessamento.py (importadas, sem alteração) SEM o
    recorte diurno, como na v1, e devolve também a máscara de P_ac ausente DEPOIS da
    reamostragem e ANTES de `etapa_faltantes` (origem da coluna `alvo_imputado`)."""
    df = etapa_alinhamento(ENTRADA_1MIN)
    df = etapa_outliers(df)
    df = etapa_reamostragem(df)
    pac_ausente = df[ALVO].isna()
    df = etapa_faltantes(df)
    df = df.rename(columns=RENOMEAR_COLUNAS)
    grade_completa = pd.date_range(df.index.min(), df.index.max(), freq="1h")
    df = df.reindex(grade_completa)
    df.index.name = "tempo"
    pac_ausente = pac_ausente.reindex(grade_completa, fill_value=True)
    pac_ausente.index.name = "tempo"
    return df, pac_ausente


def features_deterministicas(indice):
    """Posição solar e calendário do horário-alvo, calculados SÓ a partir do timestamp
    (conhecidos com antecedência), no ponto médio do intervalo horário [h, h+1).
    O rótulo horário marca o início do intervalo (resample com label='left')."""
    indice = pd.DatetimeIndex(indice)
    meio = indice + pd.Timedelta(minutes=30)
    pos = pvlib.solarposition.get_solarposition(meio.tz_localize(TZ), LAT, LON,
                                                altitude=ALT, method="nrel_numpy")
    azimute = np.deg2rad(pos["azimuth"].to_numpy())

    det = pd.DataFrame(index=indice)
    det.index.name = "tempo"
    det["elevacao_solar"] = pos["apparent_elevation"].to_numpy()
    det["azimute_sin"] = np.sin(azimute)
    det["azimute_cos"] = np.cos(azimute)
    det["hora_sin"], det["hora_cos"] = cyc(np.asarray(meio.hour + meio.minute / 60, dtype=float), 24)
    det["dia_ano_sin"], det["dia_ano_cos"] = cyc(np.asarray(meio.dayofyear, dtype=float), 365.25)
    return det


def engenharia_lags(df, verboso=True):
    """Para cada variável em VARS_LAG, pivota (data x hora) e desloca ao longo do
    eixo de datas — dá 'mesma hora, N dias atrás' de forma robusta à duração do dia
    variar com a estação (diferente de um shift posicional sobre linhas diurnas)."""
    if verboso:
        log("\n== ENGENHARIA DE LAGS (pivô data x hora) ==")
    df = df.copy()
    df["data"] = df.index.date
    df["hora"] = df.index.hour

    colunas_lag = {}
    for var in VARS_LAG:
        pivot = df.pivot(index="data", columns="hora", values=var).sort_index()
        for d in DIAS_LAG:
            colunas_lag[f"{var}_lag{d}d"] = pivot.shift(d).stack()
        if verboso:
            log(f"   {var}: lags {DIAS_LAG}")

    features_lag = pd.DataFrame(colunas_lag)
    features_lag.index.names = ["data", "hora"]
    features_lag = features_lag.reset_index()
    features_lag["tempo"] = (pd.to_datetime(features_lag["data"].astype(str))
                              + pd.to_timedelta(features_lag["hora"], unit="h"))
    features_lag = features_lag.drop(columns=["data", "hora"]).set_index("tempo")
    return features_lag


def calcular_features(grade, verboso=True):
    """Todas as features (determinísticas + defasagens) para cada hora da grade completa,
    na ordem final de colunas (antes do recorte diurno e do descarte de NaN)."""
    det = features_deterministicas(grade.index)
    lags = engenharia_lags(grade, verboso=verboso)
    return det.join(lags, how="left")


def marcar_eventos_operacionais(grade, elevacao):
    """Candidatos a evento operacional (decisão D2) em cada hora da grade: núcleo
    (G_poa alta e PR muito baixa) + expansão para as vizinhas de ±1 h no mesmo dia.
    Usa P_ac e G_poa MEDIDOS na própria hora-alvo (D+1) — só para filtragem/avaliação."""
    g_poa = grade["irradiacao_inclinada"]
    pr = grade[ALVO] / (CAPACIDADE_KWP * g_poa / 1000.0)
    diurno = elevacao > 0

    nucleo = diurno & (g_poa > LIMIAR_G_POA_WM2) & (pr < LIMIAR_PR)
    dia = grade.index.normalize()
    anterior_mesmo_dia = dia == (grade.index - pd.Timedelta(hours=1)).normalize()
    seguinte_mesmo_dia = dia == (grade.index + pd.Timedelta(hours=1)).normalize()
    vizinha = ((nucleo.shift(1, fill_value=False) & anterior_mesmo_dia)
               | (nucleo.shift(-1, fill_value=False) & seguinte_mesmo_dia))
    expansao = (diurno & vizinha & (g_poa > LIMIAR_G_POA_VIZINHA_WM2)
                & (pr < LIMIAR_PR_VIZINHA) & ~nucleo)
    return nucleo, expansao


def _dia_relativo(defasagem):
    """Rótulo do dia de origem de uma defasagem, com o dia-alvo = D+1 (lag 1 -> 'D')."""
    return "D" if defasagem == 1 else f"D-{defasagem - 1}"


def dicionario_features(colunas):
    """Uma linha por feature: nome, grupo (variável de origem, ou posicao_solar/calendario),
    descrição, unidade, tipo (lag / deterministica) e defasagem em dias."""
    linhas = []
    for nome in colunas:
        if nome in DESCRICAO_DETERMINISTICAS:
            grupo, variavel, descricao, unidade = DESCRICAO_DETERMINISTICAS[nome]
            linhas.append({"nome": nome, "grupo": grupo, "descricao": descricao, "unidade": unidade,
                           "tipo": "deterministica", "variavel": variavel, "defasagem_dias": np.nan})
            continue
        var, sufixo = nome.rsplit("_", 1)
        texto, unidade = DESCRICAO_VARIAVEIS[var]
        d = int(sufixo[3:-1])  # sufixo 'lag{d}d'
        descricao = (f"{texto}, na mesma hora do dia {_dia_relativo(d)} ({d} dia{'s' if d > 1 else ''}"
                     f" antes do dia-alvo D+1)")
        linhas.append({"nome": nome, "grupo": var, "descricao": descricao, "unidade": unidade,
                       "tipo": "lag", "variavel": var, "defasagem_dias": d})
    return pd.DataFrame(linhas)


# ------------------------------------------------------------------------------------------
# Verificações automáticas
# ------------------------------------------------------------------------------------------
def verificar_posto_e_vif(X):
    """Posto da matriz padronizada (deve ser igual ao número de colunas) e VIF de cada
    feature, VIF_j = [R^-1]_jj, com R a matriz de correlação."""
    Xs = StandardScaler().fit_transform(X.to_numpy(dtype=float))
    posto = int(np.linalg.matrix_rank(Xs))
    sv = np.linalg.svd(Xs, compute_uv=False)
    log(f"   [posto] {posto} de {X.shape[1]} features -> "
        f"{'OK' if posto == X.shape[1] else 'FALHOU (colinearidade exata)'}  "
        f"(número de condição {sv[0] / sv[-1]:.1f})")
    if posto < X.shape[1]:
        return pd.Series(np.inf, index=X.columns)
    vif = pd.Series(np.diag(np.linalg.inv(np.corrcoef(Xs, rowvar=False))), index=X.columns)
    vif = vif.sort_values(ascending=False)
    log(f"   [VIF] {int((vif > 10).sum())} features com VIF > 10; {int((vif > 100).sum())} com VIF > 100")
    log("   " + "\n   ".join(f"{n:<34} {v:10.2f}" for n, v in vif.items()))
    return vif


def verificar_lags(grade, dataset, n_amostras=N_AMOSTRAS_VERIFICACAO_LAGS, semente=SEED):
    """Em linhas sorteadas, cada `{var}_lag{d}d` deve ser igual ao valor de `var` na grade
    horária na mesma hora, d dias antes (consulta por tempo, independente do pivô)."""
    rng = np.random.default_rng(semente)
    amostra = dataset.index[rng.choice(len(dataset), size=n_amostras, replace=False)]
    divergencias = 0
    for var in VARS_LAG:
        for d in DIAS_LAG:
            esperado = grade[var].reindex(amostra - pd.Timedelta(days=d)).to_numpy()
            obtido = dataset.loc[amostra, f"{var}_lag{d}d"].to_numpy()
            divergencias += int((~np.isclose(obtido, esperado, rtol=0, atol=1e-9)).sum())
    n = n_amostras * len(VARS_LAG) * len(DIAS_LAG)
    log(f"   [lags] {n_amostras} linhas x {len(VARS_LAG) * len(DIAS_LAG)} colunas de lag = {n} "
        f"comparações com a grade em (t - d dias): {divergencias} divergências -> "
        f"{'OK' if divergencias == 0 else 'FALHOU'}")
    return divergencias


def verificar_causalidade(grade, dataset, colunas, n_dias=N_DIAS_VERIFICACAO_CAUSALIDADE,
                          semente=SEED):
    """Teste de perturbação: para dias-alvo sorteados (D+1), troca por ruído TODOS os dados
    medidos da grade a partir de D+1 00:00 (i.e., depois de 23h59 de D), recalcula as
    features e exige que as linhas do dia-alvo não mudem. Controle positivo: perturbar a
    partir de D 00:00 tem de mudar as features (prova que o teste detecta vazamento)."""
    base = calcular_features(grade, verboso=False)
    dias = np.array(sorted(dataset.index.normalize().unique()))
    rng = np.random.default_rng(semente)
    escolhidos = dias[np.sort(rng.choice(len(dias), size=min(n_dias, len(dias)), replace=False))]
    numericas = grade.select_dtypes(include=[np.number]).columns

    def mudancas(inicio_perturbacao, dia_alvo):
        pert = grade.copy()
        mascara = pert.index >= inicio_perturbacao
        pert.loc[mascara, numericas] = rng.normal(1e4, 1e3, size=(int(mascara.sum()), len(numericas)))
        linhas = dataset.index[dataset.index.normalize() == dia_alvo]
        a = base.loc[linhas, colunas].to_numpy()
        b = calcular_features(pert, verboso=False).loc[linhas, colunas].to_numpy()
        return int((~np.isclose(a, b, rtol=0, atol=1e-9, equal_nan=True)).sum())

    divergencias = sum(mudancas(dia, dia) for dia in escolhidos)
    controle = mudancas(escolhidos[0] - pd.Timedelta(days=1), escolhidos[0])
    log(f"   [causalidade] {len(escolhidos)} dias-alvo, dados >= 00h00 de D+1 trocados por ruído: "
        f"{divergencias} valores de feature alterados -> {'OK' if divergencias == 0 else 'FALHOU'}"
        f"  (controle: perturbar desde 00h00 de D altera {controle} valores — "
        f"{'teste sensível' if controle > 0 else 'TESTE SEM PODER'})")
    return divergencias, controle


def main():
    parser = argparse.ArgumentParser(description="Features day-ahead v2 (usina bet01)")
    parser.add_argument("--sem-salvar", action="store_true",
                        help="só calcula e verifica, sem gravar os CSVs")
    args = parser.parse_args()
    if not args.sem_salvar:
        garantir_inexistentes([SAIDA, SAIDA_DICIONARIO])

    log("== RECONSTRUÇÃO DA GRADE HORÁRIA COMPLETA ==")
    grade, pac_ausente = construir_grade_horaria_completa()
    log(f"   grade: {len(grade):,} horas, {grade.index.min()} a {grade.index.max()}")

    features = calcular_features(grade)
    colunas_x = list(features.columns)

    nucleo, expansao = marcar_eventos_operacionais(grade, features["elevacao_solar"])
    dataset = grade[[ALVO]].join(features)
    dataset["alvo_imputado"] = pac_ausente
    dataset["evento_operacional"] = nucleo | expansao

    log("\n== RECORTE DIURNO DAS LINHAS-ALVO (D+1): elevação solar no ponto médio > 0 ==")
    dataset = dataset[dataset["elevacao_solar"] > 0].copy()
    log(f"   {len(dataset):,} horas diurnas na grade")

    antes = len(dataset)
    dataset = dataset.dropna(subset=[ALVO] + colunas_x)
    log("\n== DESCARTE DE HISTÓRICO INSUFICIENTE / ALVO AUSENTE ==")
    log(f"   linhas: {antes:,} -> {len(dataset):,} (descartadas {antes - len(dataset):,}: "
        f"alvo ausente, primeiros ~7 dias sem histórico completo e lacunas de dados)")
    dataset = dataset[[ALVO] + colunas_x + COLUNAS_FLAGS]
    if not (dataset.index.is_monotonic_increasing and dataset.index.is_unique):
        raise ValueError("Índice do dataset v2 não é estritamente crescente")

    dicionario = dicionario_features(colunas_x)
    corte = pd.Timestamp(CORTE_DATA_ALVO)
    treino = dataset.index < corte
    excluir = dataset["alvo_imputado"] | dataset["evento_operacional"]

    log("\n== VERIFICAÇÕES AUTOMÁTICAS ==")
    log(f"   Matriz de treino efetivo (antes de {CORTE_DATA_ALVO}, sem alvo_imputado/"
        f"evento_operacional): {int((treino & ~excluir).sum()):,} linhas x {len(colunas_x)} features")
    verificar_posto_e_vif(dataset.loc[treino & ~excluir, colunas_x])
    verificar_lags(grade, dataset)
    verificar_causalidade(grade, dataset, colunas_x)
    sobrepostas = set(COLUNAS_FLAGS + [ALVO]) & set(dicionario["nome"])
    log(f"   [flags] flags/alvo fora da lista de features: "
        f"{'OK' if not sobrepostas else f'FALHOU {sobrepostas}'}")

    log("\n== RESUMO ==")
    log(f"   Features por grupo ({len(colunas_x)} no total):")
    for grupo, n in dicionario.groupby("grupo", sort=False).size().items():
        tipos = dicionario.loc[dicionario["grupo"] == grupo, "tipo"].value_counts().to_dict()
        log(f"      {grupo:<24} {n:>2}  {tipos}")
    log(f"   Linhas: {len(dataset):,} (treino {int(treino.sum()):,} | teste {int((~treino).sum()):,}); "
        f"{dataset.index.min()} a {dataset.index.max()}")
    nuc = nucleo.reindex(dataset.index)
    exp = expansao.reindex(dataset.index)
    for nome, m in (("alvo_imputado", dataset["alvo_imputado"]),
                    ("evento_operacional", dataset["evento_operacional"]),
                    ("   - núcleo (G_poa>200, PR<0,3)", nuc),
                    ("   - expansão (vizinhas ±1 h)", exp),
                    ("ambas as flags", dataset["alvo_imputado"] & dataset["evento_operacional"])):
        log(f"   {nome:<34} total {int(m.sum()):>4} | treino {int((m & treino).sum()):>4} | "
            f"teste {int((m & ~treino).sum()):>4}")
    log(f"   Treino efetivo (sem flags): {int((treino & ~excluir).sum()):,} | teste cenário A "
        f"(sem alvo imputado): {int((~treino & ~dataset['alvo_imputado']).sum())} | "
        f"cenário B (também sem evento): {int((~treino & ~excluir).sum())}")

    if args.sem_salvar:
        log("\n(--sem-salvar: nada foi gravado)")
        return dataset, dicionario

    dataset.to_csv(SAIDA)
    dicionario.to_csv(SAIDA_DICIONARIO, index=False)
    log(f"\nArquivo salvo: {SAIDA}  (shape {dataset.shape})")
    log(f"Arquivo salvo: {SAIDA_DICIONARIO}  ({len(dicionario)} features)")
    return dataset, dicionario


if __name__ == "__main__":
    main()

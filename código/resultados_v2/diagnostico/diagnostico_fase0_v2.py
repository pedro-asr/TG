#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 0 da v2 — diagnóstico SOMENTE LEITURA do pipeline v1 (usina bet01).

Importa (sem alterar) `preprocessamento.py`, `features.py`, `dados.py` e `baselines.py`,
reconstrói em memória as etapas da grade horária e responde às perguntas da Fase 0:
  1. `etapa_faltantes`: imputação de P_ac e horas-alvo imputadas (treino/teste);
  2. `etapa_recorte_diurno`: critério efetivamente aplicado pelo `features.py`;
  3. `etapa_reamostragem` e posição solar: rótulo horário, instante e convenção;
     inclui a checagem de fuso horário do céu claro de `baselines.py`;
  4. evento de 07/09/2026 (6h-8h) nos dados brutos de 1 min;
  5. candidatos a eventos operacionais (G_poa > 200 W/m² e PR horária < 0,3);
  6. correlação (Pearson/Spearman) entre as variáveis de VARS_LAG no lag de 1 dia;
  7. horário de emissão implícito da previsão.

Sem `--salvar`, apenas imprime. Com `--salvar`, grava as tabelas e a saída do console
nesta pasta (`resultados_v2/diagnostico/`), abortando se algum arquivo já existir.
"""
import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # não regravar os .pyc versionados da v1 ao importar

import numpy as np
import pandas as pd
import pvlib

DIR_DIAG = Path(__file__).resolve().parent
RAIZ_CODIGO = DIR_DIAG.parents[1]  # código/
RAIZ_DATASET = RAIZ_CODIGO.parent / "dataset" / "formando_dataset"
sys.path.insert(0, str(RAIZ_CODIGO))
sys.path.insert(0, str(RAIZ_DATASET))

import preprocessamento as pp  # noqa: E402
from variaveis_calculadas import RAW_ENTRADA, COLS_POT_INVERSOR  # noqa: E402
from features import VARS_LAG, DIAS_LAG, VARS_DETERMINISTICAS, engenharia_lags  # noqa: E402
from features import cyc  # noqa: E402
from dados import FEATURES_CSV, CAPACIDADE_KWP, CORTE_DATA_ALVO, LAT, LON, ALT, TZ  # noqa: E402
from baselines import clearsky_potencia  # noqa: E402

# Mesmo mapeamento de nomes de features.construir_grade_horaria_completa (inline na v1)
RENOMEAR_V1 = {"SOL_IRRADIACAO_HORIZONTAL": "irradiacao_horizontal",
               "SOL_IRRADIACAO_REFLETIDA": "irradiacao_refletida",
               "TRACKER_DIRECAO_VENTO": "direcao_vento",
               "SOL_TEMP_MODULO": "temp_modulo"}

# Critério de candidato a evento operacional (item 5 da Fase 0)
LIMIAR_G_POA_WM2 = 200.0
LIMIAR_PR = 0.3

EVENTO_DIA = "2026-09-07"

SAIDAS = {
    "imputados": "fase0_alvos_imputados_v2.csv",
    "azimute": "fase0_azimute_media_horaria_vs_ponto_medio_v2.csv",
    "evento_1min": "fase0_evento_20260907_1min_v2.csv",
    "eventos": "fase0_eventos_candidatos_v2.csv",
    "eventos_mes": "fase0_eventos_candidatos_por_mes_v2.csv",
    "pearson": "fase0_correlacao_lag1d_pearson_v2.csv",
    "spearman": "fase0_correlacao_lag1d_spearman_v2.csv",
    "pearson_anom": "fase0_correlacao_lag1d_anomalia_horaria_pearson_v2.csv",
    "console": "fase0_saida_console_v2.txt",
}

_LINHAS_LOG = []


def log(msg=""):
    print(msg, flush=True)
    _LINHAS_LOG.append(str(msg))


def secao(titulo):
    log("\n" + "=" * 92)
    log(titulo)
    log("=" * 92)


def caminhos_saida():
    """Resolve os caminhos de saída e aborta se qualquer um já existir (regra 3)."""
    caminhos = {k: DIR_DIAG / v for k, v in SAIDAS.items()}
    existentes = [str(c) for c in caminhos.values() if c.exists()]
    if existentes:
        raise FileExistsError("Arquivo(s) de saída já existem — abortando sem gravar nada:\n  "
                              + "\n  ".join(existentes))
    return caminhos


def comprimentos_lacunas(mascara_nan):
    """Para uma série booleana (True = NaN), retorna uma série alinhada com o comprimento
    da lacuna (run) a que cada posição pertence (0 fora de lacunas)."""
    grupos = (mascara_nan != mascara_nan.shift()).cumsum()
    tamanho = mascara_nan.groupby(grupos).transform("sum")
    return tamanho.where(mascara_nan, 0).astype(int)


def diferenca_circular(a, b):
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def posicao_solar(tempos_ingenuos_locais):
    tempos = pd.DatetimeIndex(tempos_ingenuos_locais).tz_localize(TZ)
    pos = pvlib.solarposition.get_solarposition(tempos, LAT, LON, altitude=ALT)
    pos.index = pd.DatetimeIndex(tempos_ingenuos_locais)
    return pos


# ------------------------------------------------------------------------------------------
# Carregamento (reconstrução em memória das etapas da v1)
# ------------------------------------------------------------------------------------------
def carregar_tudo():
    secao("CARREGAMENTO — etapas de preprocessamento.py reconstruídas em memória")
    um_min = pp.etapa_alinhamento(pp.ENTRADA)
    log(f"   1-min (variáveis calculadas): {len(um_min):,} linhas, "
        f"{um_min.index.min()} a {um_min.index.max()}")
    um_min_out = pp.etapa_outliers(um_min)
    horario_pre = pp.etapa_reamostragem(um_min_out)
    horario_pos = pp.etapa_faltantes(horario_pre)
    contagem_pac = um_min_out["P_ac"].resample("1h").count()

    grade = horario_pos.rename(columns=RENOMEAR_V1)
    grade = grade.reindex(pd.date_range(grade.index.min(), grade.index.max(), freq="1h"))
    grade.index.name = "tempo"
    pre = horario_pre.rename(columns=RENOMEAR_V1).reindex(grade.index)
    log(f"   grade horária: {len(grade):,} horas, {grade.index.min()} a {grade.index.max()}")

    v1 = pd.read_csv(FEATURES_CSV, parse_dates=["tempo"]).set_index("tempo").sort_index()
    log(f"   dataset v1 ({FEATURES_CSV.name}): {v1.shape[0]:,} linhas x {v1.shape[1]} colunas, "
        f"{v1.index.min()} a {v1.index.max()}")

    # Sanidade: a reconstrução em memória reproduz o CSV v1?
    lags = engenharia_lags(grade)
    base = grade[VARS_DETERMINISTICAS + ["hora_do_dia", "dia_do_ano", "P_ac"]].copy()
    rep = base.join(lags, how="left")
    rep["hora_sin"], rep["hora_cos"] = cyc(rep["hora_do_dia"], 24)
    rep["dia_ano_sin"], rep["dia_ano_cos"] = cyc(rep["dia_do_ano"], 365.25)
    rep = rep.drop(columns=["hora_do_dia", "dia_do_ano"])
    colunas_no_recorte = list(rep.columns)
    rep = pp.etapa_recorte_diurno(rep).dropna()
    mesmo_indice = rep.index.equals(v1.index)
    dif_max = float((rep[v1.columns] - v1).abs().max().max()) if mesmo_indice else float("nan")
    log(f"   reprodução em memória == CSV v1? índice igual: {mesmo_indice} | "
        f"maior |diferença| entre valores: {dif_max:.3g}")

    bruto = pd.read_csv(RAW_ENTRADA, parse_dates=["tempo"],
                        usecols=["tempo", "irradiacao_inclinada", "SOL_IRRADIACAO_HORIZONTAL",
                                 "SOL_IRRADIACAO_REFLETIDA"] + COLS_POT_INVERSOR)
    bruto = bruto.drop_duplicates(subset="tempo", keep="first").set_index("tempo").sort_index()
    log(f"   bruto 1-min ({RAW_ENTRADA.name}): {len(bruto):,} linhas")

    return {"um_min": um_min, "um_min_out": um_min_out, "pre": pre, "grade": grade,
            "contagem_pac": contagem_pac.reindex(grade.index).fillna(0).astype(int),
            "v1": v1, "colunas_no_recorte": colunas_no_recorte, "bruto": bruto}


# ------------------------------------------------------------------------------------------
# 1. etapa_faltantes
# ------------------------------------------------------------------------------------------
def diagnostico_faltantes(d):
    secao("1. etapa_faltantes — imputação de P_ac")
    log(f"   Código: para TODAS as colunas (inclusive P_ac): "
        f"interpolate(method='time', limit={pp.LIMITE_INTERPOLACAO_HORAS}, limit_area='inside').")
    log(f"   Antes dela, etapa_reamostragem já anula a média horária de P_ac quando há "
        f"< {pp.COBERTURA_MINIMA_1MIN} leituras de 1 min na hora.")

    pre_nan = d["pre"]["P_ac"].isna()
    pos_nan = d["grade"]["P_ac"].isna()
    imputado = pre_nan & ~pos_nan
    lacuna = comprimentos_lacunas(pre_nan)
    contagem = d["contagem_pac"]

    log(f"\n   Grade horária completa: {pre_nan.sum():,} horas com P_ac NaN após a reamostragem; "
        f"{imputado.sum():,} preenchidas por etapa_faltantes; {pos_nan.sum():,} continuam NaN.")
    causa_cobertura = pre_nan & (contagem > 0)
    log(f"   Origem do NaN: {causa_cobertura.sum():,} horas com cobertura parcial "
        f"(1 a {pp.COBERTURA_MINIMA_1MIN - 1} leituras) e {(pre_nan & (contagem == 0)).sum():,} "
        f"horas sem nenhuma leitura de P_ac.")

    tab = pd.DataFrame({"lacuna_h": lacuna[pre_nan], "imputado": imputado[pre_nan]})
    resumo = tab.groupby(pd.cut(tab["lacuna_h"], [0, 1, 2, 3, 6, 24, 10_000],
                                labels=["1 h", "2 h", "3 h", "4-6 h", "7-24 h", ">24 h"]),
                         observed=False)["imputado"].agg(horas_nan="size", horas_imputadas="sum")
    log("\n   Lacunas de P_ac (horas NaN agrupadas pelo comprimento da lacuna a que pertencem):")
    log(resumo.to_string())
    n_lacunas_longas = ((lacuna >= 2) & imputado).sum()
    log(f"\n   ATENÇÃO (semântica do pandas): `limit=1` não restringe a interpolação a lacunas de "
        f"1 h —\n   ele preenche a PRIMEIRA hora de toda lacuna interna, de qualquer tamanho. "
        f"Horas imputadas dentro\n   de lacunas >= 2 h: {n_lacunas_longas:,}. "
        f"(Lacunas nas bordas da série não são tocadas: limit_area='inside'.)")

    v1 = d["v1"]
    corte = pd.Timestamp(CORTE_DATA_ALVO)
    alvo_imp = imputado.reindex(v1.index).fillna(False).astype(bool)
    treino = v1.index < corte
    log(f"\n   Horas-alvo do dataset v1 com P_ac imputado (corte {CORTE_DATA_ALVO}):")
    log(f"      treino: {int(alvo_imp[treino].sum()):>4} de {int(treino.sum()):,} "
        f"({100 * alvo_imp[treino].mean():.2f}%)")
    log(f"      teste : {int(alvo_imp[~treino].sum()):>4} de {int((~treino).sum()):,} "
        f"({100 * alvo_imp[~treino].mean():.2f}%)")

    detalhe = pd.DataFrame({
        "P_ac_imputado_kW": v1.loc[alvo_imp, "P_ac"],
        "leituras_1min_na_hora": contagem.reindex(v1.index)[alvo_imp],
        "comprimento_lacuna_h": lacuna.reindex(v1.index)[alvo_imp],
        "conjunto": np.where(v1.index[alvo_imp] < corte, "treino", "teste"),
    })
    if alvo_imp.any():
        log("\n   Horas-alvo imputadas (detalhe):")
        log(detalhe.round(2).to_string())

    log("\n   Horas imputadas por etapa_faltantes nas demais colunas usadas pela v1 "
        "(total / em lacunas >= 2 h):")
    for col in VARS_LAG[1:] + ["elevacao_solar", "azimute_solar", "hora_do_dia"]:
        nan_c = d["pre"][col].isna()
        imp_c = nan_c & d["grade"][col].notna()
        log(f"      {col:<24} {int(imp_c.sum()):>4} / {int((imp_c & (comprimentos_lacunas(nan_c) >= 2)).sum()):>4}")

    # Lags alimentados por P_ac imputado (informativo para D3)
    pac_imp_d = {dd: imputado.reindex(v1.index - pd.Timedelta(days=dd)).fillna(False).values
                 for dd in DIAS_LAG}
    algum_lag = np.logical_or.reduce(list(pac_imp_d.values()))
    log(f"\n   Linhas v1 com ao menos um P_ac_lag{{1,2,3,7}}d vindo de hora imputada: "
        f"{int(algum_lag.sum()):,} (as médias móveis também herdam esses valores).")
    return {"imputado": imputado, "alvo_imputado": alvo_imp, "detalhe": detalhe}


# ------------------------------------------------------------------------------------------
# 2. etapa_recorte_diurno
# ------------------------------------------------------------------------------------------
def diagnostico_recorte(d):
    secao("2. etapa_recorte_diurno — critério efetivamente aplicado")
    cols = d["colunas_no_recorte"]
    log(f"   Código: se COL_GS='{pp.COL_GS}' estiver nas colunas -> G_horizontal MEDIDA > 0;"
        f" senão -> elevacao_solar > 0.")
    log(f"   No features.py, o DataFrame passado à etapa tem '{pp.COL_GS}'? "
        f"{pp.COL_GS in cols}  (a coluna foi renomeada para 'irradiacao_horizontal' e só os lags"
        f" entram no dataset).")
    log("   => O recorte da v1 é ASTRONÔMICO: média horária da elevação aparente de 1 min > 0.")
    log("      preprocessamento.main() usaria o critério MEDIDO, mas features.py não chama main().")

    grade, v1 = d["grade"], d["v1"]
    pos_meio = posicao_solar(grade.index + pd.Timedelta(minutes=30))
    elev_meio = pd.Series(pos_meio["apparent_elevation"].values, index=grade.index)
    disponivel = grade["P_ac"].notna()  # o dropna final da v1 exige alvo

    crit = pd.DataFrame({
        "v1: média horária da elevação > 0": grade["elevacao_solar"] > 0,
        "elevação no ponto médio (h:30) > 0": elev_meio > 0,
        "medido: G_horizontal > 0 (main())": grade["irradiacao_horizontal"] > 0,
    })
    log("\n   Horas da grade completa que passam em cada critério (todas / com P_ac disponível):")
    for nome, m in crit.items():
        log(f"      {nome:<38} {int(m.sum()):>6,} / {int((m & disponivel).sum()):>6,}")
    a, b = crit.iloc[:, 0], crit.iloc[:, 1]
    log(f"   Divergências v1 x ponto médio: {int((a & ~b).sum())} horas só na v1, "
        f"{int((~a & b).sum())} só no ponto médio.")
    log(f"   Menor elevacao_solar entre as linhas do dataset v1: {v1['elevacao_solar'].min():.4f}°")
    log("   Obs.: o dropna final remove linhas cujo ALVO (P_ac de D+1) está NaN. É uma seleção por"
        " dado de D+1,\n   porém inevitável (sem alvo não há treino/avaliação) e não usa o valor do alvo.")


# ------------------------------------------------------------------------------------------
# 3. etapa_reamostragem e posição solar
# ------------------------------------------------------------------------------------------
def diagnostico_reamostragem(d):
    secao("3. etapa_reamostragem e posição solar")
    um_min, grade = d["um_min_out"], d["grade"]

    seg = pd.Series(um_min.index.second).value_counts(normalize=True).head(3)
    log(f"   Segundo dos timestamps de 1 min (top 3): "
        + ", ".join(f":{int(s):02d} ({100 * f:.1f}%)" for s, f in seg.items()))

    hora = pd.Timestamp("2026-03-10 10:00")
    janela = um_min.loc[hora:hora + pd.Timedelta(minutes=59, seconds=59), "P_ac"]
    log(f"   Teste de rótulo: média 1-min em [{hora:%H:%M}, {hora + pd.Timedelta(hours=1):%H:%M}) = "
        f"{janela.mean():.4f} | grade[{hora}] = {grade.loc[hora, 'P_ac']:.4f} "
        f"(primeira/última leitura: {janela.index.min():%H:%M:%S} / {janela.index.max():%H:%M:%S})")
    log("   => resample('1h') com label='left', closed='left': o rótulo h marca o INÍCIO do "
        "intervalo [h, h+1).")
    log(f"   hora_do_dia (média horária de h + min/60 + s/3600) em {hora}: "
        f"{grade.loc[hora, 'hora_do_dia']:.4f} -> features cíclicas de hora já ficam ~no ponto médio.")

    # Posição solar: v1 = média horária dos valores de 1 min. Comparar com h:00 e h:30.
    v1 = d["v1"]
    idx = v1.index
    p00 = posicao_solar(idx)
    p30 = posicao_solar(idx + pd.Timedelta(minutes=30))
    elev = v1["elevacao_solar"].values
    log("\n   Posição solar da v1 = média horária dos valores de pvlib calculados a cada leitura de "
        "1 min\n   (variaveis_calculadas.py; elevação/zênite APARENTES, timestamps localizados em "
        f"{TZ}).")
    for nome, p in (("h:00 (início)", p00), ("h:30 (ponto médio)", p30)):
        dif = elev - p["apparent_elevation"].values
        log(f"      elevação v1 - pvlib em {nome:<18}: média {dif.mean():+7.3f}°, "
            f"RMS {np.sqrt(np.mean(dif ** 2)):6.3f}°, máx |.| {np.abs(dif).max():6.3f}°")
    grandes = np.abs(elev - p30["apparent_elevation"].values) > 2
    if grandes.any():
        n_leit = um_min["elevacao_solar"].resample("1h").count().reindex(idx)
        log(f"      linhas com |elevação v1 - h:30| > 2°: {int(grandes.sum())}; leituras de 1 min "
            f"nessas horas: {n_leit[grandes].to_dict()}")
        log("      (posição solar média de poucas leituras da hora — depende da cobertura do logger)")
    soma = v1["elevacao_solar"] + v1["zenite_solar"]
    log(f"      elevacao_solar + zenite_solar: mín {soma.min():.6f}, máx {soma.max():.6f} "
        "(colinearidade exata)")
    log("   => elevação calculada, na prática, no PONTO MÉDIO (média do intervalo ≈ valor em h:30; "
        "erro residual\n      vem da curvatura perto do meio-dia e das horas de nascer/pôr).")

    dif_az = diferenca_circular(v1["azimute_solar"].values, p30["azimuth"].values)
    ruins = np.abs(dif_az) > 5
    log(f"\n   Azimute: convenção do pvlib (0° = norte, sentido horário; 90° = leste, 270° = oeste)."
        f"\n   Exemplo {idx[0]}: v1 {v1['azimute_solar'].iloc[0]:.1f}° (ENE, manhã de maio, sol ao "
        f"norte).")
    log(f"   |azimute v1 - azimute pvlib em h:30| > 5°: {int(ruins.sum())} de {len(idx):,} linhas.")
    tab_az = pd.DataFrame({"hora": idx.hour, "mes": idx.strftime("%Y-%m"),
                           "azimute_v1": v1["azimute_solar"].values,
                           "azimute_pvlib_h30": p30["azimuth"].values,
                           "dif_circular": dif_az}, index=idx)[ruins]
    if len(tab_az):
        log("   Distribuição das linhas problemáticas por hora do dia: "
            + str(tab_az["hora"].value_counts().sort_index().to_dict()))
        log("   ... e por mês: " + str(tab_az["mes"].value_counts().sort_index().to_dict()))
        log("   Exemplos (hora do meio-dia solar, sol ao norte -> azimute passa por 0°/360°; a média"
            " aritmética\n   dos valores de 1 min cai perto de 180° = SUL, fisicamente errado):")
        log(tab_az.sort_values("dif_circular", key=np.abs, ascending=False).head(6)
            .round(1).to_string())

    # Fuso dos timestamps: centróide medido da irradiância vs. meio-dia solar calculado
    ghi = d["um_min_out"]["SOL_IRRADIACAO_HORIZONTAL"].clip(lower=0)
    hora_dec = um_min.index.hour + um_min.index.minute / 60 + um_min.index.second / 3600
    df_c = pd.DataFrame({"g": ghi.values, "gt": ghi.values * hora_dec,
                         "elev": um_min["elevacao_solar"].values,
                         "dia": um_min.index.normalize()}).dropna()
    por_dia = df_c.groupby("dia")
    centroide = (por_dia["gt"].sum() / por_dia["g"].sum()).median()
    instante_max = um_min["elevacao_solar"].groupby(um_min.index.normalize()).idxmax().dropna()
    transito = pd.Series(instante_max.dt.hour + instante_max.dt.minute / 60).median()
    log(f"\n   Fuso dos timestamps: centróide diário mediano da G_horizontal MEDIDA = {centroide:.2f} h;"
        f" meio-dia solar\n   calculado (máx. elevação) = {transito:.2f} h -> timestamps brutos estão em "
        f"hora local ({TZ}, UTC-3).")

    # baselines.clearsky_potencia recebe índice sem fuso
    dia = pd.date_range("2026-08-15 00:00", periods=24, freq="1h")
    p_base = clearsky_potencia(dia)
    p_local = (CAPACIDADE_KWP * pvlib.location.Location(LAT, LON, tz=TZ, altitude=ALT)
               .get_clearsky(dia.tz_localize(TZ), model="ineichen")["ghi"] / 1000).clip(
        upper=CAPACIDADE_KWP).values
    log("\n   baselines.clearsky_potencia (timestamps SEM fuso) vs. timestamps localizados em "
        f"{TZ}, 15/08/2026:")
    log("      hora    " + " ".join(f"{h:>5d}" for h in range(5, 19)))
    log("      v1      " + " ".join(f"{v:5.0f}" for v in p_base[5:19]))
    log("      local   " + " ".join(f"{v:5.0f}" for v in p_local[5:19]))
    log(f"   Pico v1 às {int(np.argmax(p_base))}h, pico local às {int(np.argmax(p_local))}h -> "
        "o pvlib trata índice sem fuso como UTC:\n   a curva de céu claro da persistência "
        "inteligente da v1 está DESLOCADA de +3 h (e avaliada em h:00, não h:30).")
    return {"azimute": tab_az}


# ------------------------------------------------------------------------------------------
# 4. evento de 07/09/2026
# ------------------------------------------------------------------------------------------
def diagnostico_evento(d):
    secao(f"4. Evento de {EVENTO_DIA} (6h-8h) nos dados brutos de 1 min")
    bruto = d["bruto"]
    jan = bruto.loc[f"{EVENTO_DIA} 05:00":f"{EVENTO_DIA} 10:00"].copy()
    inv = jan[COLS_POT_INVERSOR]
    jan["P_ac_soma"] = inv.sum(axis=1, min_count=1)
    jan["n_inv_sem_dado"] = inv.isna().sum(axis=1)
    jan["n_inv_gerando"] = (inv > 1).sum(axis=1)
    resumo = jan.resample("10min").agg({
        "irradiacao_inclinada": "mean", "SOL_IRRADIACAO_HORIZONTAL": "mean",
        "SOL_IRRADIACAO_REFLETIDA": "mean", "P_ac_soma": "mean",
        "n_inv_sem_dado": "max", "n_inv_gerando": "max",
    })
    resumo["leituras"] = jan["P_ac_soma"].resample("10min").count()
    resumo.columns = ["G_poa", "G_hor", "G_refl", "P_ac", "inv_NaN(máx)", "inv>1kW(máx)", "leituras"]
    log("   Médias de 10 min do bruto (P_ac = soma dos 10 inversores):")
    log(resumo.round(1).to_string())

    primeira = jan.index[(jan["n_inv_gerando"] > 0) & (jan.index.hour >= 6)]
    if len(primeira):
        log(f"\n   Primeira leitura com algum inversor > 1 kW após 6h: {primeira[0]}")
    zerado = jan.loc[f"{EVENTO_DIA} 06:00":f"{EVENTO_DIA} 08:59"]
    log(f"   06:00-08:59: {len(zerado)} leituras de 1 min; inversores com NaN em alguma leitura: "
        f"{int((zerado[COLS_POT_INVERSOR].isna().any()).sum())}; "
        f"máx P_ac {zerado['P_ac_soma'].max():.1f} kW; G_poa média {zerado['irradiacao_inclinada'].mean():.0f} W/m².")
    por_inv = zerado[COLS_POT_INVERSOR].agg(["min", "max", "count"]).T
    log("   Por inversor (06:00-08:59): " + "; ".join(
        f"{c[-2:]}: máx {r['max']:.1f} kW ({int(r['count'])} leit.)" for c, r in por_inv.iterrows()))

    grade, pre, cont = d["grade"], d["pre"], d["contagem_pac"]
    hrs = pd.date_range(f"{EVENTO_DIA} 05:00", periods=6, freq="1h")
    tab = pd.DataFrame({"P_ac_pre_faltantes": pre.loc[hrs, "P_ac"],
                        "P_ac_grade_v1": grade.loc[hrs, "P_ac"],
                        "leituras_1min": cont.loc[hrs],
                        "G_poa": grade.loc[hrs, "irradiacao_inclinada"],
                        "G_hor": grade.loc[hrs, "irradiacao_horizontal"]})
    log("\n   Grade horária (antes/depois de etapa_faltantes):")
    log(tab.round(2).to_string())
    return jan


# ------------------------------------------------------------------------------------------
# 5. candidatos a eventos operacionais
# ------------------------------------------------------------------------------------------
def diagnostico_eventos(d, imputado):
    secao(f"5. Candidatos a eventos operacionais: horas diurnas com G_poa > {LIMIAR_G_POA_WM2:.0f} "
          f"W/m² e PR = P_ac/(2500·G_poa/1000) < {LIMIAR_PR}")
    grade, v1, bruto = d["grade"], d["v1"], d["bruto"]
    diurno = grade["elevacao_solar"] > 0
    g = grade["irradiacao_inclinada"]
    pr = grade["P_ac"] / (CAPACIDADE_KWP * g / 1000.0)
    cand = diurno & (g > LIMIAR_G_POA_WM2) & (pr < LIMIAR_PR)
    log(f"   Universo: {int(diurno.sum()):,} horas diurnas na grade (v1: média horária da elevação > 0), "
        f"{int((diurno & (g > LIMIAR_G_POA_WM2) & pr.notna()).sum()):,} com G_poa > "
        f"{LIMIAR_G_POA_WM2:.0f} e P_ac disponível.")
    pr_ok = pr[diurno & (g > LIMIAR_G_POA_WM2)]
    log(f"   PR horária nessas horas: mediana {pr_ok.median():.3f}, p5 {pr_ok.quantile(.05):.3f}, "
        f"p95 {pr_ok.quantile(.95):.3f}.")

    # Diagnóstico por inversor a partir do bruto
    inv = bruto[COLS_POT_INVERSOR]
    media_inv = inv.resample("1h").mean().reindex(grade.index)
    cont_inv = inv.resample("1h").count().reindex(grade.index).fillna(0)
    n_sem_dado = (cont_inv < pp.COBERTURA_MINIMA_1MIN).sum(axis=1)
    n_parados = ((media_inv < 1) & (cont_inv >= pp.COBERTURA_MINIMA_1MIN)).sum(axis=1)

    corte = pd.Timestamp(CORTE_DATA_ALVO)
    tab = pd.DataFrame({
        "G_poa_Wm2": g, "G_hor_Wm2": grade["irradiacao_horizontal"], "P_ac_kW": grade["P_ac"],
        "PR": pr, "P_ac_imputado": imputado,
        "inv_sem_dado": n_sem_dado, "inv_parados": n_parados,
        "no_dataset_v1": grade.index.isin(v1.index),
    })[cand]
    tab["conjunto_v1"] = np.where(~tab["no_dataset_v1"], "-",
                                  np.where(tab.index < corte, "treino", "teste"))
    # Energia não gerada na hora, supondo a PR mediana das horas com G_poa acima do limiar
    tab["deficit_kWh"] = pr_ok.median() * CAPACIDADE_KWP * tab["G_poa_Wm2"] / 1000 - tab["P_ac_kW"]
    tab = tab.sort_values(["PR", "G_poa_Wm2"], ascending=[True, False])

    meses = pd.period_range(grade.index.min(), grade.index.max(), freq="M")
    por_mes = pd.DataFrame({
        "horas_diurnas": diurno.groupby(grade.index.to_period("M")).sum(),
        "candidatos": cand.groupby(grade.index.to_period("M")).sum(),
        "candidatos_no_dataset_v1": (cand & grade.index.isin(v1.index)).groupby(
            grade.index.to_period("M")).sum(),
        "dias_distintos": tab.groupby(tab.index.to_period("M")).apply(
            lambda x: x.index.normalize().nunique()),
    }).reindex(meses).fillna(0).astype(int)
    por_mes.index = por_mes.index.astype(str)
    log("\n   Contagem por mês:")
    log(por_mes.to_string())
    log(f"   Total: {len(tab)} horas em {tab.index.normalize().nunique()} dias; no dataset v1: "
        f"{int(tab['no_dataset_v1'].sum())} (treino {int((tab['conjunto_v1'] == 'treino').sum())}, "
        f"teste {int((tab['conjunto_v1'] == 'teste').sum())}).")
    log(f"   Composição: {int((tab['inv_sem_dado'] > 0).sum())} com >= 1 inversor sem dado na hora; "
        f"{int(tab['P_ac_imputado'].sum())} com P_ac imputado; inversores parados (média horária "
        f"< 1 kW): 10 -> {int((tab['inv_parados'] >= 10).sum())}, 1 a 9 -> "
        f"{int(tab['inv_parados'].between(1, 9).sum())}, nenhum -> {int((tab['inv_parados'] == 0).sum())}.")
    sem_parada = tab[tab["inv_parados"] == 0].index
    if len(sem_parada):
        mi = media_inv.loc[sem_parada]
        dispersao = (mi.min(axis=1) / mi.max(axis=1)).median()
        log(f"   Nas horas sem inversor parado, razão mediana (menor/maior potência média entre os 10 "
            f"inversores) = {dispersao:.2f}\n   -> todos gerando juntos em nível baixo (perfil de "
            "partida/parada parcial dentro da hora ou limitação de potência), não falha isolada.")

    # Horas de transição (início/fim do evento dentro da hora) que escapam do limiar
    vizinha = cand.shift(1, fill_value=False) | cand.shift(-1, fill_value=False)
    transicao = diurno & (g > LIMIAR_G_POA_WM2) & pr.between(LIMIAR_PR, 0.7, inclusive="left") & ~cand
    log(f"\n   Horas com {LIMIAR_PR} <= PR < 0,7 e G_poa > {LIMIAR_G_POA_WM2:.0f}: {int(transicao.sum())}, "
        f"das quais {int((transicao & vizinha).sum())} vizinhas (±1 h) de um candidato "
        "(transição do evento dentro da hora).")
    # Proposta de expansão: vizinhas (±1 h, mesmo dia) com G_poa > 50 W/m² e PR < 0,7
    mesmo_dia_ant = grade.index.normalize() == (grade.index - pd.Timedelta(hours=1)).normalize()
    mesmo_dia_pos = grade.index.normalize() == (grade.index + pd.Timedelta(hours=1)).normalize()
    vizinha_dia = ((cand.shift(1, fill_value=False) & mesmo_dia_ant)
                   | (cand.shift(-1, fill_value=False) & mesmo_dia_pos))
    expansao = diurno & vizinha_dia & (g > 50) & (pr < 0.7) & ~cand
    no_v1 = grade.index.isin(v1.index)
    log(f"   Expansão proposta (vizinha ±1 h no mesmo dia, G_poa > 50 W/m², PR < 0,7): +{int(expansao.sum())} "
        f"horas (no dataset v1: treino {int((expansao & no_v1 & (grade.index < corte)).sum())}, "
        f"teste {int((expansao & no_v1 & (grade.index >= corte)).sum())}). "
        f"Em {EVENTO_DIA}: {[f'{h:%H}h' for h in grade.index[expansao & (grade.index.normalize() == pd.Timestamp(EVENTO_DIA))]]}.")

    # Contaminação das defasagens de P_ac por horas de evento
    cand_v1 = {dd: cand.reindex(v1.index - pd.Timedelta(days=dd)).fillna(False).values
               for dd in DIAS_LAG}
    contaminada = np.logical_or.reduce(list(cand_v1.values()))
    treino_v1 = v1.index < corte
    log(f"   Linhas do dataset v1 com algum P_ac_lag{{1,2,3,7}}d vindo de hora candidata: "
        f"{int(contaminada.sum())} (treino {int((contaminada & treino_v1).sum())}, "
        f"teste {int((contaminada & ~treino_v1).sum())}); só lag1d: {int(cand_v1[1].sum())}.")

    p_dia = grade.loc[diurno, "P_ac"]
    p_1min = d["um_min"]["P_ac"]
    log(f"\n   P_ac de 1 min ANTES do clip de etapa_outliers: máx {p_1min.max():.0f} kW; leituras > "
        f"{CAPACIDADE_KWP:.0f} kW: {int((p_1min > CAPACIDADE_KWP).sum()):,} de {int(p_1min.notna().sum()):,}.")
    log(f"   Referência de capacidade: P_ac horária máx {p_dia.max():.0f} kW; horas diurnas com "
        f"P_ac >= 2450 kW: {int((p_dia >= 2450).sum())}; PR mediana {pr_ok.median():.2f} (> 0,9 sugere "
        "que 2500 kW\n   é o limite CA dos inversores e/ou há ganho bifacial — a PR aqui é só um "
        "índice relativo para triagem).")

    log("\n   15 casos mais severos (ordem: PR crescente, depois G_poa decrescente):")
    log(tab.head(15).drop(columns=["no_dataset_v1"]).round(3).to_string())
    return tab, por_mes


# ------------------------------------------------------------------------------------------
# 6. correlações entre VARS_LAG (lag 1d)
# ------------------------------------------------------------------------------------------
def diagnostico_correlacoes(d):
    secao("6. Correlação entre as variáveis de VARS_LAG no lag de 1 dia (linhas de TREINO da v1)")
    v1, grade = d["v1"], d["grade"]
    treino = v1.loc[v1.index < pd.Timestamp(CORTE_DATA_ALVO)]
    log(f"   VARS_LAG da v1: {VARS_LAG}")
    log("   Obs.: temp_modulo (SOL_TEMP_MODULO) NÃO está em VARS_LAG; T_inv é a temperatura interna"
        " média dos\n   10 inversores. temp_modulo entra aqui só para a comparação pedida.")
    tab = treino[[f"{v}_lag1d" for v in VARS_LAG]].copy()
    tab.columns = VARS_LAG
    tab["temp_modulo"] = grade["temp_modulo"].reindex(treino.index - pd.Timedelta(days=1)).values
    n_validas = tab.dropna().shape[0]
    log(f"   Linhas: {len(tab):,} (com temp_modulo disponível: {n_validas:,})")

    pearson = tab.corr(method="pearson")
    spearman = tab.corr(method="spearman")
    anomalia = tab.groupby(treino.index.hour).transform(lambda s: s - s.mean())
    pearson_anom = anomalia.corr(method="pearson")

    pd.set_option("display.width", 200)
    log("\n   Pearson:")
    log(pearson.round(3).to_string())
    log("\n   Spearman:")
    log(spearman.round(3).to_string())

    pares = [("temp_ambiente", "temp_modulo"), ("temp_ambiente", "T_inv"), ("temp_modulo", "T_inv"),
             ("irradiacao_inclinada", "irradiacao_horizontal"),
             ("irradiacao_inclinada", "irradiacao_refletida"),
             ("irradiacao_horizontal", "irradiacao_refletida"),
             ("P_ac", "irradiacao_inclinada"), ("P_ac", "irradiacao_horizontal"),
             ("temp_ambiente", "umidade_relativa")]
    log("\n   Pares em destaque          Pearson  Spearman  Pearson(anomalia por hora do dia)")
    for a, b in pares:
        log(f"   {a:>22} x {b:<22} {pearson.loc[a, b]:6.3f}   {spearman.loc[a, b]:6.3f}   "
            f"{pearson_anom.loc[a, b]:6.3f}")
    log("   (anomalia por hora = valor menos a média da mesma hora no treino: remove o ciclo diurno, "
        "que infla\n   todas as correlações entre grandezas solares)")
    return pearson, spearman, pearson_anom


# ------------------------------------------------------------------------------------------
# 7. horário de emissão implícito
# ------------------------------------------------------------------------------------------
def diagnostico_emissao(d, n_amostras=200, semente=0):
    secao("7. Horário de emissão implícito")
    v1, grade, um_min = d["v1"], d["grade"], d["um_min_out"]
    horas = v1.index.hour.value_counts().sort_index()
    log(f"   Horas-alvo presentes no dataset v1: {horas.to_dict()}")
    h_max = int(horas.index.max())

    rng = np.random.default_rng(semente)
    amostra = v1.index[rng.choice(len(v1), size=n_amostras, replace=False)]
    erros = 0
    for var in VARS_LAG:
        for dd in DIAS_LAG:
            esperado = grade[var].reindex(amostra - pd.Timedelta(days=dd)).values
            obtido = v1.loc[amostra, f"{var}_lag{dd}d"].values
            erros += int((~np.isclose(obtido, esperado, rtol=0, atol=1e-9)).sum())
    log(f"   {n_amostras} linhas sorteadas: {{var}}_lag{{d}}d == valor da grade em (t - d dias) "
        f"para todas as {len(VARS_LAG) * len(DIAS_LAG)} colunas de lag? divergências: {erros}")
    m3 = max(float((v1[f"{v}_media3d"] - v1[[f"{v}_lag{k}d" for k in (1, 2, 3)]].mean(axis=1))
                   .abs().max()) for v in VARS_LAG)
    log(f"   max |{{var}}_media3d - (lag1d+lag2d+lag3d)/3| em todas as variáveis: {m3:.2e} "
        "(combinação linear exata)")

    exemplo = v1.index[(v1.index.hour == h_max)][-1]
    origem = exemplo - pd.Timedelta(days=1)
    leituras = um_min.loc[origem:origem + pd.Timedelta(minutes=59, seconds=59)].index
    log(f"\n   Exemplo: alvo {exemplo} -> P_ac_lag1d = média de P_ac em [{origem:%d/%m %H:%M}, "
        f"{origem + pd.Timedelta(hours=1):%H:%M}) de D;\n   última leitura de 1 min usada: "
        f"{leituras.max()}.")
    log(f"   Nenhuma feature usa timestamp de D+1 (só lags >= 1 dia e grandezas determinísticas); a "
        f"mais recente\n   é a hora {h_max}h de D, cujo intervalo termina às {h_max + 1}h00 de D.")
    log(f"   => A previsão de D+1 só pode ser EMITIDA após ~{h_max + 1}h00 de D (fim da geração de D);"
        f" uma emissão\n   mais cedo (ex.: prazo da programação diária pela manhã de D) exigiria "
        "lags >= 2 dias para as\n   horas posteriores ao horário de emissão.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--salvar", action="store_true",
                        help="grava tabelas e saída do console em resultados_v2/diagnostico/")
    args = parser.parse_args()
    caminhos = caminhos_saida() if args.salvar else None

    d = carregar_tudo()
    falt = diagnostico_faltantes(d)
    diagnostico_recorte(d)
    ream = diagnostico_reamostragem(d)
    evento = diagnostico_evento(d)
    eventos, por_mes = diagnostico_eventos(d, falt["imputado"])
    pearson, spearman, pearson_anom = diagnostico_correlacoes(d)
    diagnostico_emissao(d)

    if args.salvar:
        falt["detalhe"].to_csv(caminhos["imputados"])
        ream["azimute"].to_csv(caminhos["azimute"])
        evento.to_csv(caminhos["evento_1min"])
        eventos.to_csv(caminhos["eventos"])
        por_mes.to_csv(caminhos["eventos_mes"])
        pearson.to_csv(caminhos["pearson"])
        spearman.to_csv(caminhos["spearman"])
        pearson_anom.to_csv(caminhos["pearson_anom"])
        caminhos["console"].write_text("\n".join(_LINHAS_LOG) + "\n", encoding="utf-8")
        log(f"\nSaídas gravadas em {DIR_DIAG}")


if __name__ == "__main__":
    main()

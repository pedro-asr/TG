#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figuras complementares para o texto do TG (v2), geradas SÓ a partir das tabelas, previsões
e resumos já salvos em `resultados_v2/` por `comparacao_v2.py` e `modelos/lstm_ablacao_v2.py`.
Nada é treinado nem reprevisto.

Gera em resultados_v2/figuras/ (sufixo _v2; aborta sem gravar nada se alguma já existir):
  1. importancia_grupos_v2.png    — importância por permutação por grupo de variáveis
                                    (um painel por modelo; escala própria em cada painel);
  2. erro_por_classe_ceu_v2.png   — RMSE e MBE por classe de céu (tercis de k_d no teste);
  3. diebold_mariano_v2.png       — matriz de p-valores do teste DM (HLN), cenários A e B;
  4. cv_por_dobra_v2.png          — RMSE por dobra de CV da melhor configuração (otimista);
  5. ablacao_lstm_v2.png          — RMSE das três variantes da ablação da LSTM, por semente e
                                    pela previsão média das sementes, cenários A e B.

Mesmas convenções visuais de `comparacao_v2.py` (skill `dataviz`): cor e marcador fixos por
família, importados de lá. Cenário principal A nas figuras 1, 2 e 4.

Uso:  python -B figuras_tg_v2.py            (lê e grava em resultados_v2/)
      python -B figuras_tg_v2.py --rapido   (lê e grava no sandbox resultados_v2/rapido/)
"""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # não regravar os .pyc versionados da v1 ao importar
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402  (backend Agg já fixado por comparacao_v2)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from comparacao_v2 import (  # noqa: E402  (constantes e utilitários de figura da Fase 4)
    CENARIO_PRINCIPAL, CLASSES_CEU, COR_FAMILIA, COR_REFERENCIA, COR_SUPERFICIE,
    COR_TEXTO_PRIMARIO, COR_TEXTO_SECUNDARIO, ESTOCASTICOS, FAMILIAS_ML, LINEAR, MARCADOR,
    N_REPETICOES_PERMUTACAO, REF_PRINCIPAL, SUFIXO_MEDIA, _estilizar, _salvar, metricas,
    no_cenario, rotulo,
)
from dados_v2 import caminho_predicoes  # noqa: E402
from saidas_v2 import ativar_modo_rapido, caminho_saida, garantir_inexistentes  # noqa: E402

# Modelos na ordem das figuras: referências primeiro, depois as famílias de ML.
MODELOS = [REF_PRINCIPAL, LINEAR, *FAMILIAS_ML]
MODELOS_PERMUTACAO = [LINEAR, "svr", *[f + SUFIXO_MEDIA for f in ESTOCASTICOS]]
MODELOS_DM = [REF_PRINCIPAL, LINEAR, "svr", *[f + SUFIXO_MEDIA for f in ESTOCASTICOS]]
ROTULO_CURTO = {REF_PRINCIPAL: "Persist.\ninteligente", LINEAR: "Regressão\nlinear",
                "svr": "SVR", "mlp": "MLP", "lstm": "LSTM", "xgboost": "XGBoost"}
ROTULO_CLASSE = {"nublado": "Nublado", "intermediario": "Intermediário", "claro": "Claro"}
ROTULO_GRUPO = {
    "posicao_solar": "Posição solar", "calendario": "Calendário", "P_ac": "P_ac (lags)",
    "temp_ambiente": "Temperatura ambiente", "umidade_relativa": "Umidade relativa",
    "velocidade_vento": "Velocidade do vento", "irradiacao_inclinada": "Irradiância inclinada",
    "irradiacao_horizontal": "Irradiância horizontal", "irradiacao_refletida": "Irradiância refletida",
    "T_inv": "T_inv",
}
VARIANTES_ABLACAO = ["lstm_original", "lstm_sequencia_invertida", "lstm_ramo_denso"]

# Matriz do DM: rampa ordinal de um só matiz (azul da paleta de referência, passos 250/450/600;
# validate_palette.js --ordinal: PASS sobre #fcfcfb), mais o cinza neutro para p >= 0,10.
FAIXAS_P = [(0.01, "#184f95", "p < 0,01"), (0.05, "#2a78d6", "0,01 ≤ p < 0,05"),
            (0.10, "#86b6ef", "0,05 ≤ p < 0,10"), (np.inf, "#f0efec", "p ≥ 0,10")]
TEXTO_SOBRE_ESCURO = "#fcfcfb"


def familia_de(modelo):
    """`lstm_media_sementes` -> `lstm` (cor e marcador são da família)."""
    return modelo.removesuffix(SUFIXO_MEDIA)


def virgula(valor, fmt="{:.3f}"):
    """Número com vírgula decimal, para os textos que entram no TG."""
    return fmt.format(valor).replace(".", ",")


def ler_tabela(nome):
    caminho = caminho_saida("tabelas", f"{nome}_v2.csv")
    if not caminho.exists():
        raise FileNotFoundError(f"{caminho} não existe — rode antes comparacao_v2.py")
    return pd.read_csv(caminho)


def _sem_moldura(eixo, lados=("top", "right")):
    for lado in lados:
        eixo.spines[lado].set_visible(False)


# ------------------------------------------------------------------------------------------
# 1. Importância por permutação por grupo
# ------------------------------------------------------------------------------------------
def figura_importancia_grupos(importancia, caminho):
    """Um painel por modelo, barras horizontais por grupo (± dp entre as 10 repetições).
    A ordem dos grupos é a mesma em todos os painéis (aumento médio relativo entre os
    modelos), para comparar a posição de cada grupo; a escala x é própria de cada painel,
    porque o calendário na regressão linear (+559 kW) achataria os demais."""
    t = importancia[(importancia["cenario"] == CENARIO_PRINCIPAL) & (importancia["tipo"] == "grupo")]
    ordem = (t.groupby("nome")["aumento_rmse_%"].mean().sort_values(ascending=False).index.tolist())
    fig, eixos = plt.subplots(1, len(MODELOS_PERMUTACAO), figsize=(19, 5.2), sharey=True,
                              facecolor=COR_SUPERFICIE)
    y_pos = np.arange(len(ordem))[::-1]
    for eixo, modelo in zip(eixos, MODELOS_PERMUTACAO):
        s = t[t["modelo"] == modelo].set_index("nome").loc[ordem]
        _estilizar(eixo, grade="x")
        eixo.barh(y_pos, s["aumento_rmse_kW_media"], height=0.62,
                  color=COR_FAMILIA[familia_de(modelo)])
        eixo.errorbar(s["aumento_rmse_kW_media"], y_pos, xerr=s["aumento_rmse_kW_dp"], fmt="none",
                      ecolor=COR_TEXTO_PRIMARIO, elinewidth=1, capsize=2)
        eixo.axvline(0, color=COR_REFERENCIA, linewidth=1)
        # rótulo só no grupo mais importante do painel (rotulagem seletiva)
        topo = s["aumento_rmse_kW_media"].idxmax()
        y_topo = y_pos[ordem.index(topo)]
        x_topo = s.loc[topo, "aumento_rmse_kW_media"] + s.loc[topo, "aumento_rmse_kW_dp"]
        eixo.text(x_topo, y_topo, f"  +{s.loc[topo, 'aumento_rmse_kW_media']:.0f} kW", va="center",
                  ha="left", fontsize=8, color=COR_TEXTO_SECUNDARIO)
        eixo.set_xlim(min(0, s["aumento_rmse_kW_media"].min() * 1.3), x_topo * 1.35)
        base = s["rmse_base_kW"].iloc[0]
        eixo.set_title(f"{rotulo(modelo)}\nRMSE sem permutação: {base:.0f} kW", fontsize=10,
                       color=COR_TEXTO_SECUNDARIO, loc="left")
        eixo.set_xlabel("Aumento do RMSE (kW)", fontsize=9, color=COR_TEXTO_SECUNDARIO)
        _sem_moldura(eixo, ("top", "right", "left"))
        eixo.tick_params(left=False)
    eixos[0].set_yticks(y_pos)
    eixos[0].set_yticklabels([ROTULO_GRUPO.get(g, g) for g in ordem], fontsize=9,
                             color=COR_TEXTO_SECUNDARIO)
    fig.suptitle(f"Importância por permutação por grupo de variáveis — teste (cenário "
                 f"{CENARIO_PRINCIPAL}; média ± dp de {N_REPETICOES_PERMUTACAO} repetições; "
                 "escala própria em cada painel)", fontsize=12, color=COR_TEXTO_PRIMARIO)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _salvar(fig, caminho)


# ------------------------------------------------------------------------------------------
# 2. Erro por classe de céu
# ------------------------------------------------------------------------------------------
def figura_erro_por_classe(tabela_classe, caminho):
    """RMSE e MBE por classe de céu em dois painéis (medidas diferentes -> nunca eixo
    duplo); barras agrupadas por classe, cor fixa por modelo; MLP, LSTM e XGBoost pela
    média ± dp entre sementes."""
    t = tabela_classe[(tabela_classe["cenario"] == CENARIO_PRINCIPAL)
                      & tabela_classe["modelo"].isin(MODELOS)]
    n_dias = t.groupby("classe_ceu")["n_dias"].first()
    x = np.arange(len(CLASSES_CEU))
    largura = 0.8 / len(MODELOS)
    fig, eixos = plt.subplots(1, 2, figsize=(15, 4.8), facecolor=COR_SUPERFICIE)
    paineis = (("RMSE_kW", "RMSE por classe de céu (kW)"),
               ("MBE_kW", "MBE por classe de céu (kW; > 0 = superestima)"))
    for eixo, (metrica, titulo) in zip(eixos, paineis):
        _estilizar(eixo, grade="y")
        for i, modelo in enumerate(MODELOS):
            s = t[t["modelo"] == modelo].set_index("classe_ceu").loc[CLASSES_CEU]
            deslocamento = (i - (len(MODELOS) - 1) / 2) * largura
            # borda na cor da superfície = espaço de 2 px entre barras vizinhas
            eixo.bar(x + deslocamento, s[f"{metrica}_media"], width=largura,
                     color=COR_FAMILIA[modelo], edgecolor=COR_SUPERFICIE, linewidth=1.5,
                     label=rotulo(modelo), zorder=2)
            if s[f"{metrica}_dp"].notna().any():
                eixo.errorbar(x + deslocamento, s[f"{metrica}_media"], yerr=s[f"{metrica}_dp"],
                              fmt="none", ecolor=COR_TEXTO_PRIMARIO, elinewidth=1, capsize=2,
                              zorder=3)
        if metrica == "MBE_kW":
            eixo.axhline(0, color=COR_REFERENCIA, linewidth=1, zorder=4)
        eixo.set_xticks(x)
        eixo.set_xticklabels([f"{ROTULO_CLASSE[c]}\n({int(n_dias[c])} dias)" for c in CLASSES_CEU],
                             fontsize=9, color=COR_TEXTO_SECUNDARIO)
        eixo.set_title(titulo, fontsize=11, color=COR_TEXTO_SECUNDARIO, loc="left")
        _sem_moldura(eixo)
    linhas, rotulos = eixos[0].get_legend_handles_labels()
    fig.legend(linhas, rotulos, loc="lower center", ncol=len(MODELOS), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"Erro por classe de céu (tercis do índice diário k_d) — teste (cenário "
                 f"{CENARIO_PRINCIPAL}); MLP, LSTM e XGBoost: média ± dp entre sementes",
                 fontsize=12, color=COR_TEXTO_PRIMARIO)
    fig.tight_layout(rect=[0, 0.07, 1, 0.93])
    _salvar(fig, caminho)


# ------------------------------------------------------------------------------------------
# 3. Matriz de p-valores do Diebold-Mariano
# ------------------------------------------------------------------------------------------
def _cor_p(p):
    return next((cor, i) for i, (limite, cor, _) in enumerate(FAIXAS_P) if p < limite)


def figura_diebold_mariano(dm, caminho):
    """Triângulo inferior: célula (linha, coluna) = par testado, com o p-valor e o modelo de
    menor perda; células em branco = par não testado (ver `comparacao_v2.pares_dm`)."""
    n = len(MODELOS_DM)
    indice = {m: i for i, m in enumerate(MODELOS_DM)}
    fig, eixos = plt.subplots(1, 2, figsize=(12.5, 6.6), facecolor=COR_SUPERFICIE)
    for eixo, cenario in zip(eixos, ("A", "B")):
        eixo.set_facecolor(COR_SUPERFICIE)
        t = dm[dm["cenario"] == cenario]
        for _, l in t.iterrows():
            i, j = sorted((indice[l["modelo_1"]], indice[l["modelo_2"]]), reverse=True)
            cor, faixa = _cor_p(l["p_valor"])
            eixo.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1, facecolor=cor,
                                         edgecolor=COR_SUPERFICIE, linewidth=2))
            p_txt = "p < 0,001" if l["p_valor"] < 0.001 else f"p = {virgula(l['p_valor'])}"
            texto = (f"{p_txt}\nmelhor:\n"
                     f"{ROTULO_CURTO[familia_de(l['melhor'])].replace(chr(10), ' ')}")
            eixo.text(j + 0.5, n - 0.5 - i, texto, ha="center", va="center", fontsize=8,
                      color=TEXTO_SOBRE_ESCURO if faixa <= 1 else COR_TEXTO_PRIMARIO)
        eixo.set_xlim(0, n - 1)
        eixo.set_ylim(0, n - 1)
        eixo.set_aspect("equal")
        rotulos = [ROTULO_CURTO[familia_de(m)] for m in MODELOS_DM]
        eixo.set_xticks(np.arange(n - 1) + 0.5)
        eixo.set_xticklabels(rotulos[:-1], fontsize=9, color=COR_TEXTO_SECUNDARIO)
        eixo.set_yticks(np.arange(n - 1) + 0.5)
        eixo.set_yticklabels(rotulos[1:][::-1], fontsize=9, color=COR_TEXTO_SECUNDARIO)
        eixo.tick_params(length=0)
        for spine in eixo.spines.values():
            spine.set_visible(False)
        n_dias = int(t["n_dias"].iloc[0])
        eixo.set_title(f"Cenário {cenario} ({n_dias} dias)", fontsize=11,
                       color=COR_TEXTO_SECUNDARIO, loc="left")
    legenda = [plt.Rectangle((0, 0), 1, 1, facecolor=cor, edgecolor=COR_REFERENCIA, linewidth=0.5,
                             label=texto) for _, cor, texto in FAIXAS_P]
    fig.legend(handles=legenda, loc="lower center", ncol=len(FAIXAS_P), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Teste de Diebold-Mariano (correção HLN) sobre perdas diárias — p-valores "
                 "bicaudais; MLP, LSTM e XGBoost pela previsão média das sementes",
                 fontsize=12, color=COR_TEXTO_PRIMARIO)
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    _salvar(fig, caminho)


# ------------------------------------------------------------------------------------------
# 4. RMSE por dobra de CV
# ------------------------------------------------------------------------------------------
def _rotulo_dobra(linha):
    inicio, fim = pd.Timestamp(linha["inicio_validacao"]), pd.Timestamp(linha["fim_validacao"])
    return f"Dobra {linha['dobra']}\n{inicio:%d/%m/%y} a\n{fim:%d/%m/%y}"


def figura_cv_por_dobra(cv, caminho):
    """Uma linha por família; o eixo x são as 5 dobras de validação em ordem temporal."""
    familias = [LINEAR, *FAMILIAS_ML]
    referencia = cv[cv["familia"] == LINEAR].sort_values("dobra")
    fig, eixo = plt.subplots(figsize=(11, 4.8), facecolor=COR_SUPERFICIE)
    _estilizar(eixo)
    for familia in familias:
        s = cv[cv["familia"] == familia].sort_values("dobra")
        eixo.plot(s["dobra"], s["rmse_kW"], color=COR_FAMILIA[familia], linewidth=2,
                  marker=MARCADOR[familia], markersize=8, label=rotulo(familia), zorder=2)
    eixo.set_xticks(referencia["dobra"])
    eixo.set_xticklabels([_rotulo_dobra(l) for _, l in referencia.iterrows()], fontsize=8,
                         color=COR_TEXTO_SECUNDARIO)
    eixo.set_ylabel("RMSE na dobra de validação (kW)", fontsize=9, color=COR_TEXTO_SECUNDARIO)
    eixo.set_ylim(0, cv["rmse_kW"].max() * 1.12)
    _sem_moldura(eixo)
    eixo.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=len(familias),
                frameon=False, fontsize=9)
    eixo.set_title("RMSE por dobra de validação cruzada (TimeSeriesSplit) — melhor configuração "
                   "de cada família", fontsize=12, color=COR_TEXTO_PRIMARIO, loc="left", pad=22)
    eixo.text(0, 1.02, "Estimativa otimista: os hiperparâmetros foram escolhidos nestas mesmas "
              "dobras (exceto a regressão linear, sem ajuste).", transform=eixo.transAxes,
              fontsize=8, color=COR_TEXTO_SECUNDARIO, va="bottom")
    fig.tight_layout()
    _salvar(fig, caminho)


# ------------------------------------------------------------------------------------------
# 5. Ablação da LSTM
# ------------------------------------------------------------------------------------------
def rmse_ablacao(prev):
    """RMSE por semente e pela previsão média das sementes, por variante e cenário."""
    linhas = []
    for cenario in ("A", "B"):
        sub = no_cenario(prev, cenario)
        for variante in VARIANTES_ABLACAO:
            v = sub[sub["modelo"] == variante]
            for semente, d in v.groupby("semente"):
                linhas.append({"cenario": cenario, "variante": variante, "semente": semente,
                               "RMSE_kW": metricas(d["y_obs"], d["y_pred"])["RMSE_kW"]})
            media = v.groupby("tempo")[["y_obs", "y_pred"]].mean()
            linhas.append({"cenario": cenario, "variante": variante, "semente": "media",
                           "RMSE_kW": metricas(media["y_obs"], media["y_pred"])["RMSE_kW"]})
    return pd.DataFrame(linhas)


def figura_ablacao(tabela, resumo, metricas_v2, caminho):
    """Pontos = sementes; barra = média ± dp entre sementes; losango vazado = previsão média
    das sementes; linha tracejada = persistência inteligente do mesmo cenário. Todas as
    variantes são LSTM, por isso uma única cor (a da família)."""
    cor = COR_FAMILIA["lstm"]
    y_pos = np.arange(len(VARIANTES_ABLACAO))[::-1]
    fig, eixos = plt.subplots(1, 2, figsize=(15, 4.2), sharey=True, facecolor=COR_SUPERFICIE)
    for eixo, cenario in zip(eixos, ("A", "B")):
        _estilizar(eixo, grade="x")
        t = tabela[tabela["cenario"] == cenario]
        for y, variante in zip(y_pos, VARIANTES_ABLACAO):
            v = t[t["variante"] == variante]
            sementes = v[v["semente"] != "media"]["RMSE_kW"]
            eixo.scatter(sementes, np.full(len(sementes), y), s=36, color=cor, alpha=0.55,
                         linewidths=0, zorder=2, label="Sementes (0 a 9)" if y == y_pos[0] else None)
            eixo.errorbar(sementes.mean(), y - 0.22, xerr=sementes.std(ddof=1), fmt="o",
                          color=COR_TEXTO_PRIMARIO, markersize=5, elinewidth=1.2, capsize=3,
                          zorder=3, label="Média ± dp entre sementes" if y == y_pos[0] else None)
            media = v[v["semente"] == "media"]["RMSE_kW"].iloc[0]
            eixo.scatter([media], [y + 0.22], marker="D", s=60, facecolors=COR_SUPERFICIE,
                         edgecolors=cor, linewidths=2, zorder=3,
                         label="Previsão média das sementes" if y == y_pos[0] else None)
            eixo.text(media, y + 0.22, f"   {virgula(media, '{:.1f}')} kW", va="center",
                      ha="left", fontsize=8, color=COR_TEXTO_SECUNDARIO)
        ref = metricas_v2[(metricas_v2["cenario"] == cenario)
                          & (metricas_v2["modelo"] == REF_PRINCIPAL)]["RMSE_kW_media"].iloc[0]
        eixo.axvline(ref, color=COR_REFERENCIA, linewidth=1.5, linestyle="--", zorder=1,
                     label=f"{rotulo(REF_PRINCIPAL)}")
        eixo.text(ref, y_pos[0] + 0.55, f" {virgula(ref, '{:.1f}')} kW", fontsize=8,
                  color=COR_TEXTO_SECUNDARIO, ha="left", va="center")
        n_horas = int(no_cenario_n_horas(cenario, metricas_v2))
        eixo.set_title(f"Cenário {cenario} ({n_horas} horas)", fontsize=11,
                       color=COR_TEXTO_SECUNDARIO, loc="left")
        eixo.set_xlabel("RMSE no teste (kW) — menor é melhor", fontsize=9, color=COR_TEXTO_SECUNDARIO)
        eixo.set_ylim(y_pos[-1] - 0.6, y_pos[0] + 0.8)
        _sem_moldura(eixo, ("top", "right", "left"))
        eixo.tick_params(left=False)
    eixos[0].set_yticks(y_pos)
    eixos[0].set_yticklabels([resumo["variantes"][v]["rotulo"] for v in VARIANTES_ABLACAO],
                             fontsize=9, color=COR_TEXTO_SECUNDARIO)
    linhas, rotulos = eixos[0].get_legend_handles_labels()
    fig.legend(linhas, rotulos, loc="lower center", ncol=len(linhas), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Ablação do ramo sequencial da LSTM — mesmos hiperparâmetros, sementes e ramo "
                 "estático; muda só o ramo da sequência", fontsize=12, color=COR_TEXTO_PRIMARIO)
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    _salvar(fig, caminho)


def no_cenario_n_horas(cenario, metricas_v2):
    return metricas_v2.loc[(metricas_v2["cenario"] == cenario)
                           & (metricas_v2["modelo"] == REF_PRINCIPAL), "n_horas"].iloc[0]


def carregar_ablacao():
    caminho_prev = caminho_predicoes("lstm_ablacao")
    caminho_resumo = caminho_saida("modelos", "lstm_ablacao_resumo_v2.json")
    for caminho in (caminho_prev, caminho_resumo):
        if not caminho.exists():
            raise FileNotFoundError(f"{caminho} não existe — rode antes modelos/lstm_ablacao_v2.py")
    prev = pd.read_csv(caminho_prev, parse_dates=["tempo"])
    with open(caminho_resumo, encoding="utf-8") as f:
        resumo = json.load(f)
    return prev, resumo


# ------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Figuras complementares do TG (v2)")
    parser.add_argument("--rapido", action="store_true",
                        help="lê e grava no sandbox do modo rápido (resultados_v2/rapido/)")
    args = parser.parse_args()
    if args.rapido:
        ativar_modo_rapido()

    figuras = {nome: caminho_saida("figuras", f"{nome}_v2.png") for nome in (
        "importancia_grupos", "erro_por_classe_ceu", "diebold_mariano", "cv_por_dobra",
        "ablacao_lstm")}
    garantir_inexistentes(figuras.values())

    # leitura completa antes de gravar qualquer figura (falta de insumo aborta sem saída parcial)
    metricas_v2 = ler_tabela("metricas")
    importancia = ler_tabela("importancia_permutacao")
    classe = ler_tabela("erro_por_classe_ceu")
    dm = ler_tabela("diebold_mariano")
    cv = ler_tabela("cv_por_dobra")
    prev_ablacao, resumo_ablacao = carregar_ablacao()
    tabela_ablacao = rmse_ablacao(prev_ablacao)

    figura_importancia_grupos(importancia, figuras["importancia_grupos"])
    figura_erro_por_classe(classe, figuras["erro_por_classe_ceu"])
    figura_diebold_mariano(dm, figuras["diebold_mariano"])
    figura_cv_por_dobra(cv, figuras["cv_por_dobra"])
    figura_ablacao(tabela_ablacao, resumo_ablacao, metricas_v2, figuras["ablacao_lstm"])

    resumo = (tabela_ablacao[tabela_ablacao["semente"] != "media"]
              .groupby(["cenario", "variante"])["RMSE_kW"].agg(["mean", "std"]))
    print("RMSE da ablação por semente (média ± dp), para conferência com o registro M17:")
    for (cenario, variante), l in resumo.iterrows():
        print(f"  {cenario}  {variante:<26} {l['mean']:.1f} ± {l['std']:.1f} kW")
    print(f"Figuras salvas em {figuras['ablacao_lstm'].parent}: "
          + ", ".join(c.name for c in figuras.values()))


if __name__ == "__main__":
    main()

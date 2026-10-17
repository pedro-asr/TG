#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fluxo comum de treino da v2 — compartilhado pelos scripts das 4 famílias
(`modelos/svm_v2.py`, `mlp_v2.py`, `lstm_v2.py`, `xgboost_model_v2.py`), para que todas
passem exatamente pelas mesmas etapas, com os mesmos dados, filtros e saídas:
  1. ajuste de hiperparâmetros (`ajuste_hp_v2.otimizar`) com UMA semente (42);
  2. retreino final com os melhores hiperparâmetros em todo o treino efetivo (sem as
     linhas de `alvo_imputado`/`evento_operacional`), com 10 sementes (0 a 9) para as
     famílias estocásticas (decisão D5) ou uma única vez para as determinísticas (SVR);
  3. previsões de teste no formato longo comum (`dados_v2.predicoes_formato_longo`);
  4. JSON por modelo (melhores hiperparâmetros, RMSE de CV médio e por dobra com datas,
     avisos de borda) e modelos/scalers salvos, com conferência de recarga.

Linha de comando comum: `--rapido` (N_TRIALS = 2, épocas = 5, 2 sementes; saídas no
sandbox `resultados_v2/rapido/`) e `--retomar` (completa um estudo Optuna interrompido).
"""
import argparse
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True  # não regravar os .pyc versionados da v1 ao importar

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ajuste_hp_v2 import (  # noqa: E402
    SEED as SEMENTE_AJUSTE, imprimir_resumo, n_trials, otimizar, resumo_estudo, salvar_json,
)
from avaliacao import avaliar  # noqa: E402  (v1, inalterado)
from dados_v2 import (  # noqa: E402
    ALVO, CAPACIDADE_KWP, SEMENTE_DETERMINISTICO, caminho_predicoes, carregar_split,
    colunas_features, predicoes_formato_longo,
)
from saidas_v2 import ativar_modo_rapido, caminho_saida, garantir_inexistentes  # noqa: E402

SEMENTES = list(range(10))      # decisão D5: retreino final com as sementes 0 a 9
SEMENTES_RAPIDO = [0, 1]
EPOCAS = 150                    # mesmo teto de épocas da v1 (com parada antecipada)
EPOCAS_RAPIDO = 5
TOLERANCIA_RECARGA_KW = 1e-2    # previsões do modelo recarregado vs. em memória


def argumentos(descricao):
    parser = argparse.ArgumentParser(description=descricao)
    parser.add_argument("--rapido", action="store_true",
                        help="teste de ponta a ponta: 2 trials, 5 épocas, 2 sementes (sandbox)")
    parser.add_argument("--retomar", action="store_true",
                        help="completa um estudo Optuna já existente em vez de abortar")
    return parser.parse_args()


def _metricas(y_obs, y_pred, mascara):
    return avaliar(y_obs[mascara], y_pred[mascara], potencia_instalada_kw=CAPACIDADE_KWP)


def executar(nome, construir_modelo, espaco_busca, *, deterministico, arquivos_modelo,
             salvar_modelo, carregar_modelo, descricao):
    """Roda o fluxo completo para uma família.
    - `construir_modelo(params, semente, epocas)` -> estimator sklearn-like não treinado;
    - `espaco_busca(trial)` -> dict de hiperparâmetros;
    - `arquivos_modelo(semente)` -> {rótulo: Path} dos arquivos do modelo daquela semente;
    - `salvar_modelo(modelo, arquivos)` / `carregar_modelo(arquivos)` -> persistência."""
    args = argumentos(descricao)
    if args.rapido:
        ativar_modo_rapido()
    trials = n_trials(args.rapido)
    epocas = EPOCAS_RAPIDO if args.rapido else EPOCAS
    sementes = ([SEMENTE_DETERMINISTICO] if deterministico
                else (SEMENTES_RAPIDO if args.rapido else SEMENTES))

    saidas = {"predicoes": caminho_predicoes(nome),
              "resumo": caminho_saida("modelos", f"{nome}_resumo_v2.json")}
    arquivos = {s: arquivos_modelo(s) for s in sementes}
    garantir_inexistentes(list(saidas.values())
                          + [c for por_semente in arquivos.values() for c in por_semente.values()])

    treino, teste, m = carregar_split()
    colunas_x = colunas_features()
    excluir = m["treino_excluir"].to_numpy()
    X_treino, y_treino = treino[colunas_x], treino[ALVO].to_numpy()
    X_teste, y_teste = teste[colunas_x], teste[ALVO].to_numpy()
    print(f"Treino: {len(X_treino):,} linhas ({int((~excluir).sum()):,} usadas no ajuste) | "
          f"Teste: {len(X_teste):,} linhas | {len(colunas_x)} features | "
          f"{trials} trials | {'determinístico' if deterministico else f'{len(sementes)} sementes'}"
          + ("" if deterministico else f" | até {epocas} épocas" if nome in ("mlp", "lstm") else ""))

    # 1. Ajuste de hiperparâmetros com uma única semente
    inicio = time.time()
    estudo = otimizar(
        lambda params: construir_modelo(params, semente=SEMENTE_AJUSTE, epocas=epocas),
        espaco_busca, X_treino, y_treino, nome_estudo=f"{nome}_bet01_v2", n_trials=trials,
        mascara_excluir=excluir, retomar=args.retomar)
    tempo_ajuste = time.time() - inicio
    resumo = resumo_estudo(estudo)
    imprimir_resumo(resumo)

    # 2. Retreino final com as sementes, no treino efetivo completo
    inicio = time.time()
    tabelas, previsoes = [], {}
    print("\nRetreino final (melhores hiperparâmetros, treino efetivo completo):")
    for s in sementes:
        semente_fit = SEMENTE_AJUSTE if s == SEMENTE_DETERMINISTICO else s
        modelo = construir_modelo(estudo.best_params, semente=semente_fit, epocas=epocas)
        modelo.fit(X_treino[~excluir], y_treino[~excluir])
        y_pred = np.asarray(modelo.predict(X_teste), dtype=float).ravel()
        tabela = predicoes_formato_longo(teste, y_pred, nome, s)
        tabelas.append(tabela)
        previsoes[s] = tabela["y_pred"].to_numpy()
        salvar_modelo(modelo, arquivos[s])
        if s == sementes[0]:
            recarregado = np.asarray(carregar_modelo(arquivos[s]).predict(X_teste), dtype=float).ravel()
            dif = float(np.max(np.abs(recarregado - y_pred)))
            if dif > TOLERANCIA_RECARGA_KW:
                raise RuntimeError(f"Modelo recarregado diverge do treinado: {dif:.4g} kW")
            print(f"   [recarga] semente {s}: modelo salvo e recarregado, máx |dif| = {dif:.2e} kW -> OK")
        a = _metricas(y_teste, previsoes[s], m["teste_A"].to_numpy())
        b = _metricas(y_teste, previsoes[s], m["teste_B"].to_numpy())
        rotulo = "única" if s == SEMENTE_DETERMINISTICO else f"{s:>5}"
        print(f"   semente {rotulo}: RMSE teste A {a['RMSE_kW']:.2f} kW | B {b['RMSE_kW']:.2f} kW")
    tempo_retreino = time.time() - inicio

    if len(sementes) > 1:
        for cenario in ("teste_A", "teste_B"):
            mask = m[cenario].to_numpy()
            rmses = [_metricas(y_teste, p, mask)["RMSE_kW"] for p in previsoes.values()]
            media = _metricas(y_teste, np.mean(list(previsoes.values()), axis=0), mask)["RMSE_kW"]
            print(f"   cenário {cenario[-1]}: RMSE {np.mean(rmses):.2f} ± {np.std(rmses, ddof=1):.2f} kW "
                  f"entre sementes | previsão média das sementes: {media:.2f} kW")

    # 3. Saídas
    pd.concat(tabelas, ignore_index=True).to_csv(saidas["predicoes"], index=False)
    resumo.update({
        "modelo": nome,
        "modo": "rapido" if args.rapido else "completo",
        "n_trials_alvo": trials,
        "semente_ajuste": SEMENTE_AJUSTE,
        "sementes_retreino": sementes,
        "epocas_max": epocas if nome in ("mlp", "lstm") else None,
        "n_treino_ajuste": int((~excluir).sum()),
        "features": colunas_x,
        "tempo_ajuste_s": round(tempo_ajuste, 1),
        "tempo_retreino_s": round(tempo_retreino, 1),
        "arquivos_modelo": {str(s): {k: v.name for k, v in a.items()} for s, a in arquivos.items()},
    })
    salvar_json(resumo, saidas["resumo"])
    print(f"\nTempo: ajuste {tempo_ajuste:.0f}s ({tempo_ajuste / max(trials, 1):.0f}s/trial) | "
          f"retreino {tempo_retreino:.0f}s")
    print(f"Salvos em {saidas['predicoes'].parents[1]}: {saidas['predicoes'].name}, "
          f"{saidas['resumo'].name} e {sum(len(a) for a in arquivos.values())} arquivo(s) de modelo")
    return estudo, resumo

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regressão linear (StandardScaler + LinearRegression) para previsão day-ahead de P_ac —
baseline linear da v2, sem ajuste de hiperparâmetros.

Usa os mesmos dados, split, dobras de CV e filtros (decisões D2/D3: linhas com
`alvo_imputado` ou `evento_operacional` fora do ajuste) dos 4 modelos de ML, e serve de
referência para os testes de Diebold-Mariano (cada modelo contra a regressão linear) e
para a importância por permutação.

Uso:  python -B modelos/regressao_linear_v2.py            (grava em resultados_v2/)
      python -B modelos/regressao_linear_v2.py --rapido   (grava no sandbox resultados_v2/rapido/)
"""
import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # não regravar os .pyc versionados da v1 ao importar

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ajuste_hp_v2 import salvar_json, tabela_dobras, validar_cv  # noqa: E402
from avaliacao import avaliar  # noqa: E402  (v1, inalterado)
from dados_v2 import (  # noqa: E402
    ALVO, CAPACIDADE_KWP, caminho_predicoes, carregar_split, colunas_features,
    predicoes_formato_longo,
)
from saidas_v2 import ativar_modo_rapido, caminho_saida, garantir_inexistentes  # noqa: E402

NOME = "regressao_linear"


def construir_modelo(params=None):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressao", LinearRegression()),
    ])


def main():
    parser = argparse.ArgumentParser(description="Regressão linear (baseline v2)")
    parser.add_argument("--rapido", action="store_true",
                        help="grava no sandbox do modo rápido (sem trials/sementes: resultado igual)")
    args = parser.parse_args()
    if args.rapido:
        ativar_modo_rapido()
    saidas = {
        "predicoes": caminho_predicoes(NOME),
        "modelo": caminho_saida("modelos", f"{NOME}_bet01_v2.joblib"),
        "resumo": caminho_saida("modelos", f"{NOME}_resumo_v2.json"),
    }
    garantir_inexistentes(saidas.values())

    treino, teste, m = carregar_split()
    colunas_x = colunas_features()
    excluir = m["treino_excluir"].to_numpy()
    X_treino, y_treino = treino[colunas_x], treino[ALVO].to_numpy()

    print(f"Treino: {len(X_treino):,} linhas ({int((~excluir).sum()):,} usadas no ajuste) | "
          f"Teste: {len(teste):,} linhas | {len(colunas_x)} features")

    cv = validar_cv(construir_modelo, {}, X_treino, y_treino, mascara_excluir=excluir)
    print(f"RMSE (CV, treino): {cv['rmse_medio']:.2f} kW")
    for d in tabela_dobras(cv):
        print(f"   dobra {d['dobra']}: {d['inicio_validacao'][:10]} a {d['fim_validacao'][:10]}  "
              f"(treino {d['n_treino']:,}, val {d['n_validacao']:,})  RMSE {d['rmse_kW']:.2f} kW")

    modelo = construir_modelo().fit(X_treino[~excluir], y_treino[~excluir])
    previsoes = predicoes_formato_longo(teste, modelo.predict(teste[colunas_x]), NOME)

    for cenario in ("teste_A", "teste_B"):
        mascara = m[cenario].to_numpy()
        r = avaliar(previsoes.loc[mascara, "y_obs"], previsoes.loc[mascara, "y_pred"],
                    potencia_instalada_kw=CAPACIDADE_KWP, nome=NOME)
        print(f"Teste cenário {cenario[-1]} (n={r['n']}): RMSE {r['RMSE_kW']:.2f} kW | "
              f"MAE {r['MAE_kW']:.2f} | MBE {r['MBE_kW']:+.2f} | nRMSE {r['nRMSE_%']:.2f}% | "
              f"R² {r['R2']:.3f}")

    previsoes.to_csv(saidas["predicoes"], index=False)
    joblib.dump(modelo, saidas["modelo"])
    salvar_json({
        "modelo": NOME,
        "melhores_hiperparametros": {},
        "obs": "sem ajuste de hiperparâmetros (StandardScaler + LinearRegression)",
        "rmse_cv_medio_kW": cv["rmse_medio"],
        "dobras_cv": tabela_dobras(cv),
        "avisos_borda": [],
        "n_treino_ajuste": int((~excluir).sum()),
        "features": colunas_x,
        "coeficientes_padronizados": dict(zip(colunas_x, modelo.named_steps["regressao"].coef_.tolist())),
    }, saidas["resumo"])
    print(f"\nSalvos: {saidas['predicoes'].name}, {saidas['modelo'].name}, {saidas['resumo'].name} "
          f"em {saidas['predicoes'].parents[1]}")
    return modelo, previsoes


if __name__ == "__main__":
    main()

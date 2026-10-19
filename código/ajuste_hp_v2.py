#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ajuste de hiperparâmetros v2 compartilhado pelas 4 famílias (SVR, MLP, LSTM, XGBoost) —
mesmo esquema da v1 (`ajuste_hp.py`, mantido intacto): otimização bayesiana Optuna/TPE
(seed 42) com validação cruzada walk-forward (`TimeSeriesSplit`, 5 dobras). Mudanças:
  - `N_TRIALS` único para todas as famílias (decisão D4), definido aqui;
  - RMSE de cada dobra e as datas inicial/final de cada dobra de validação ficam em
    `trial.user_attrs`;
  - estudos persistidos em SQLite (`resultados_v2/optuna/estudos_v2.db`) e retomáveis
    (`retomar=True` completa os trials que faltam; sem isso, um estudo homônimo aborta);
  - máscara opcional de linhas a excluir do ajuste e do cálculo do RMSE (decisões D2/D3).
    As dobras são definidas sobre a série de treino COMPLETA (as fronteiras temporais não
    dependem da máscara); dentro de cada dobra, as linhas marcadas saem do fit e do RMSE.

Obs.: ao retomar, o TPE relê os trials já gravados, mas o gerador aleatório é recriado.
Para não repetir os sorteios iniciais (os primeiros trials do TPE são aleatórios), a
semente na retomada é `SEED + nº de trials já gravados`; numa execução ininterrupta ela é
exatamente `SEED` (42), como na v1. O estudo retomado não é idêntico, trial a trial, a
uma execução ininterrupta.

Uso:  python -B ajuste_hp_v2.py --teste   (autoteste com Ridge em SQLite temporário)
"""
import json
import math
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True  # não regravar os .pyc versionados da v1 ao importar

import numpy as np
import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ajuste_hp import N_SPLITS_CV, SEED  # noqa: E402  (v1: mesmas 5 dobras e seed 42)
from saidas_v2 import caminho_saida, garantir_inexistentes, modo_rapido  # noqa: E402

N_TRIALS = 50          # decisão D4: mesmo orçamento a priori para todas as famílias
N_TRIALS_RAPIDO = 2    # modo --rapido (teste de ponta a ponta)
LIMIAR_BORDA = 0.05    # aviso se o ótimo ficar a menos de 5% da borda do espaço de busca
NOME_BANCO = "estudos_v2.db"

optuna.logging.set_verbosity(optuna.logging.WARNING)


def n_trials(rapido=False):
    return N_TRIALS_RAPIDO if rapido else N_TRIALS


def dobras_cv(n_linhas, n_splits=N_SPLITS_CV):
    """Pares (idx_treino, idx_val) posicionais do TimeSeriesSplit sobre a série de treino."""
    return list(TimeSeriesSplit(n_splits=n_splits).split(np.arange(n_linhas)))


def validar_cv(construir_modelo, params, X, y, mascara_excluir=None, n_splits=N_SPLITS_CV):
    """RMSE por dobra de walk-forward para um conjunto de hiperparâmetros.
    `construir_modelo(params)` retorna um estimator sklearn-like (fit/predict) ainda não
    treinado. Linhas com `mascara_excluir` = True ficam fora do fit e do RMSE de cada dobra.
    Retorna dict com 'rmse_medio' e listas por dobra ('rmse_dobras', 'inicio_val',
    'fim_val', 'n_treino', 'n_val')."""
    y = np.asarray(y, dtype=float)
    usar = (np.ones(len(X), dtype=bool) if mascara_excluir is None
            else ~np.asarray(mascara_excluir, dtype=bool))
    r = {"rmse_dobras": [], "inicio_val": [], "fim_val": [], "n_treino": [], "n_val": []}
    for idx_treino, idx_val in dobras_cv(len(X), n_splits):
        it, iv = idx_treino[usar[idx_treino]], idx_val[usar[idx_val]]
        modelo = construir_modelo(params)
        modelo.fit(X.iloc[it], y[it])
        pred = np.asarray(modelo.predict(X.iloc[iv]), dtype=float).ravel()
        r["rmse_dobras"].append(float(np.sqrt(np.mean((y[iv] - pred) ** 2))))
        r["inicio_val"].append(str(X.index[idx_val[0]]))
        r["fim_val"].append(str(X.index[idx_val[-1]]))
        r["n_treino"].append(int(len(it)))
        r["n_val"].append(int(len(iv)))
    r["rmse_medio"] = float(np.mean(r["rmse_dobras"]))
    return r


def url_banco(caminho=None):
    caminho = caminho_saida("optuna", NOME_BANCO) if caminho is None else Path(caminho)
    return f"sqlite:///{caminho.resolve().as_posix()}"


def otimizar(construir_modelo, espaco_busca, X, y, *, nome_estudo, n_trials,
             mascara_excluir=None, retomar=False, n_splits=N_SPLITS_CV, banco=None):
    """`espaco_busca(trial)` amostra e retorna um dict de hiperparâmetros a partir de um
    `optuna.Trial`. Roda até o estudo ter `n_trials` trials concluídos e retorna o
    `optuna.Study` (melhor trial em `.best_trial`, dobras em `.best_trial.user_attrs`)."""
    armazenamento = url_banco(banco)
    existe = nome_estudo in optuna.study.get_all_study_names(storage=armazenamento)
    if existe and not retomar:
        if not modo_rapido():
            raise FileExistsError(f"Estudo '{nome_estudo}' já existe em {armazenamento} — use "
                                  "--retomar para completá-lo (ou remova-o manualmente).")
        optuna.delete_study(study_name=nome_estudo, storage=armazenamento)  # sandbox
        existe = False

    ja_gravados = (len(optuna.load_study(study_name=nome_estudo, storage=armazenamento).trials)
                   if existe else 0)
    estudo = optuna.create_study(
        direction="minimize",
        study_name=nome_estudo,
        storage=armazenamento,
        sampler=optuna.samplers.TPESampler(seed=SEED + ja_gravados),
        load_if_exists=True,
    )
    n_excluidas = 0 if mascara_excluir is None else int(np.asarray(mascara_excluir).sum())
    estudo.set_user_attr("n_splits_cv", n_splits)
    estudo.set_user_attr("seed_tpe", SEED)
    estudo.set_user_attr("n_trials_alvo", n_trials)
    estudo.set_user_attr("n_linhas_treino", int(len(X)))
    estudo.set_user_attr("n_linhas_excluidas", n_excluidas)

    concluidos = len(estudo.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]))
    restantes = max(0, n_trials - concluidos)
    if concluidos:
        print(f"   [{nome_estudo}] retomando: {concluidos} trials concluídos, {restantes} a rodar",
              flush=True)

    def objetivo(trial):
        params = espaco_busca(trial)
        r = validar_cv(construir_modelo, params, X, y, mascara_excluir, n_splits)
        for chave in ("rmse_dobras", "inicio_val", "fim_val", "n_treino", "n_val"):
            trial.set_user_attr(chave, r[chave])
        return r["rmse_medio"]

    inicio = time.time()

    def log_trial(estudo, trial):
        decorrido = time.time() - inicio
        print(f"   [{nome_estudo}] trial {trial.number + 1}/{n_trials}  "
              f"RMSE={trial.value:.2f} kW  |  melhor={estudo.best_value:.2f} kW  |  "
              f"{decorrido:.0f}s decorridos  |  params={trial.params}", flush=True)

    if restantes:
        estudo.optimize(objetivo, n_trials=restantes, show_progress_bar=False,
                        callbacks=[log_trial])
    return estudo


def tabela_dobras(attrs):
    """Lista de dicts (uma por dobra) a partir do resultado de `validar_cv` ou dos
    `user_attrs` de um trial."""
    return [{"dobra": i + 1, "inicio_validacao": ini, "fim_validacao": fim,
             "n_treino": nt, "n_validacao": nv, "rmse_kW": rmse}
            for i, (ini, fim, nt, nv, rmse) in enumerate(zip(
                attrs["inicio_val"], attrs["fim_val"], attrs["n_treino"], attrs["n_val"],
                attrs["rmse_dobras"]))]


def avisos_borda(trial, limiar=LIMIAR_BORDA):
    """Avisa quando um hiperparâmetro ótimo fica a menos de `limiar` (fração do intervalo,
    em escala log quando a distribuição é log) de uma borda do espaço de busca. Para
    categóricos numéricos ordenáveis com > 2 opções, avisa se o ótimo é a categoria extrema.
    Inteiros com só 2 valores possíveis (ex.: 1 ou 2 camadas) não geram aviso: qualquer
    valor estaria "na borda"."""
    avisos = []
    for nome, dist in trial.distributions.items():
        valor = trial.params[nome]
        if isinstance(dist, (FloatDistribution, IntDistribution)):
            lo, hi = dist.low, dist.high
            if hi == lo or (isinstance(dist, IntDistribution) and (hi - lo) // dist.step + 1 <= 2):
                continue
            if dist.log:
                pos = (math.log(valor) - math.log(lo)) / (math.log(hi) - math.log(lo))
            else:
                pos = (valor - lo) / (hi - lo)
            if pos < limiar or pos > 1 - limiar:
                borda = "inferior" if pos < 0.5 else "superior"
                avisos.append(f"{nome}={valor:.4g} a {100 * min(pos, 1 - pos):.1f}% da borda {borda}"
                              f" de [{lo:g}, {hi:g}] (escala {'log' if dist.log else 'linear'})")
        elif isinstance(dist, CategoricalDistribution):
            opcoes = dist.choices
            if len(opcoes) > 2 and all(isinstance(o, (int, float)) for o in opcoes):
                if valor in (min(opcoes), max(opcoes)):
                    avisos.append(f"{nome}={valor} é a categoria extrema de {sorted(opcoes)}")
    return avisos


def resumo_estudo(estudo, **extras):
    """Dict serializável com melhores hiperparâmetros, RMSE médio de CV, RMSE por dobra
    (com datas) e avisos de borda — base do JSON salvo por modelo."""
    melhor = estudo.best_trial
    concluidos = estudo.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    return {
        "estudo": estudo.study_name,
        "n_trials_concluidos": len(concluidos),
        "melhor_trial_optuna": melhor.number,  # numeração do Optuna (começa em 0)
        "melhores_hiperparametros": melhor.params,
        "rmse_cv_medio_kW": melhor.value,
        "dobras_cv": tabela_dobras(melhor.user_attrs),
        "avisos_borda": avisos_borda(melhor),
        "obs_dobras": ("RMSE por dobra da configuração escolhida nessas mesmas dobras "
                       "(estimativa otimista)."),
        **extras,
    }


def salvar_json(dados, caminho):
    """Grava um dict em JSON (UTF-8, indentado); aborta se o arquivo já existir."""
    garantir_inexistentes([caminho])
    caminho.write_text(json.dumps(dados, ensure_ascii=False, indent=2, default=_json_padrao),
                       encoding="utf-8")


def _json_padrao(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def imprimir_resumo(resumo):
    print(f"\nMelhores hiperparâmetros (trial {resumo['melhor_trial_optuna'] + 1} de "
          f"{resumo['n_trials_concluidos']}): {resumo['melhores_hiperparametros']}")
    print(f"RMSE (CV, treino): {resumo['rmse_cv_medio_kW']:.2f} kW")
    for d in resumo["dobras_cv"]:
        print(f"   dobra {d['dobra']}: {d['inicio_validacao'][:10]} a {d['fim_validacao'][:10]}  "
              f"(treino {d['n_treino']:,}, val {d['n_validacao']:,})  RMSE {d['rmse_kW']:.2f} kW")
    for aviso in resumo["avisos_borda"]:
        print(f"   AVISO DE BORDA: {aviso}")


def _autoteste():
    """Ridge(alpha) nos dados v2 reais, em SQLite temporário fora do projeto: 2 trials,
    depois retomada até 3; confere user_attrs, datas das dobras e avisos de borda."""
    import tempfile
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from dados_v2 import ALVO, carregar_split, colunas_features

    treino, _, m = carregar_split()
    X, y = treino[colunas_features()], treino[ALVO].to_numpy()
    construir = lambda p: make_pipeline(StandardScaler(), Ridge(alpha=p["alpha"]))  # noqa: E731
    espaco = lambda t: {"alpha": t.suggest_float("alpha", 1e-3, 1e3, log=True)}  # noqa: E731

    # ignore_cleanup_errors: no Windows o SQLite pode seguir aberto pelo engine do SQLAlchemy
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        banco = Path(tmp) / "autoteste_v2.db"
        otimizar(construir, espaco, X, y, nome_estudo="autoteste", n_trials=2,
                 mascara_excluir=m["treino_excluir"].to_numpy(), banco=banco)
        try:
            otimizar(construir, espaco, X, y, nome_estudo="autoteste", n_trials=3, banco=banco)
            print("   FALHOU: estudo homônimo sem --retomar deveria abortar")
        except FileExistsError:
            print("   OK: estudo homônimo sem retomar abortou")
        estudo = otimizar(construir, espaco, X, y, nome_estudo="autoteste", n_trials=3,
                          mascara_excluir=m["treino_excluir"].to_numpy(), retomar=True, banco=banco)
        print(f"   trials concluídos após retomar: {len(estudo.trials)} (esperado 3)")
        imprimir_resumo(resumo_estudo(estudo))
        print(f"   user_attrs do melhor trial: {sorted(estudo.best_trial.user_attrs)}")


if __name__ == "__main__":
    if "--teste" in sys.argv:
        _autoteste()
    else:
        print(__doc__)

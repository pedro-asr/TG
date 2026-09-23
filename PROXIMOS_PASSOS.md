# TG — Previsão day-ahead de P_ac (usina bet01) — Próximos passos

## Estado atual

Pipeline completo e funcional em `D:\tg\código`, usando o dataset regenerado a partir do
bruto mais completo (`dataset/bet01_completo_bruto_concatenado_202505_202607.csv`,
2025-05-02 a 2026-07-09).

Arquivos:
- `dados.py` — carregamento + split treino/teste (**placeholder por fração de 15%**;
  trocar para `dividir_por_data` quando ago/set 2026 forem adicionados — ver pendência abaixo).
- `features.py` — gera `bet01_features_dayahead.csv` (4919 linhas diurnas → 4919 após
  engenharia de lags e `dropna`, 4919 no total, split em 4182 treino / 737 teste). Alvo:
  `P_ac` direto (kW). Features: defasagens causais (D-1, D-2, D-3, D-7 + médias móveis
  3d/7d) de P_ac, temp_ambiente, umidade_relativa, velocidade_vento, irradiações,
  T_inv, mais posição solar/calendário (determinísticos, conhecidos com antecedência).
- `avaliacao.py` — métricas: RMSE, MAE, MBE, nRMSE (normalizado por 2500 kWp), skill
  score vs. persistência.
- `baselines.py` — persistência ingênua e inteligente (via índice de céu claro, pvlib).
- `ajuste_hp.py` — otimização de hiperparâmetros compartilhada (Optuna/TPE +
  `TimeSeriesSplit` walk-forward), usada pelos 4 modelos.
- `modelos/svm.py`, `modelos/mlp.py`, `modelos/lstm.py`, `modelos/xgboost_model.py` —
  os 4 modelos, todos treinados e avaliados no mesmo split.

## Resultados (conjunto de teste definitivo, split por data: treino < 01/08/2026,
## teste = ago-set/2026, 491 linhas / 41 dias)

| Modelo | RMSE (kW) | R² | nRMSE | Skill vs. persist. ingênua |
|---|---|---|---|---|
| **LSTM** | **312,3** | 0,805 | 12,5% | **+26,3%** |
| XGBoost | 321,3 | 0,794 | 12,9% | +24,1% |
| SVM (SVR) | 326,3 | 0,787 | 13,1% | +22,9% |
| MLP | 330,9 | 0,781 | 13,2% | +21,9% |
| Persistência inteligente | 384,2 | 0,705 | 15,4% | +9,3% |
| Persistência ingênua | 423,5 | 0,642 | 16,9% | — |

Os 4 modelos batem as duas persistências. Margem entre eles é pequena (LSTM à frente).
Importância de features do XGBoost: `P_ac` defasado (médias 7d/3d, D-1) domina, seguido
por irradiância horizontal — consistente com a ausência de previsão meteorológica (NWP)
como entrada.

## Pendências / próximos passos

1. ~~**Split por data (prioridade alta)**~~ — **resolvido.** `dados.carregar_split()`
   já usa `dividir_por_data` por padrão (treino < 01/08/2026, teste = ago-set/2026,
   491 linhas). `dividir` (fração) ficou só como utilitário auxiliar, não é mais usado
   por nenhum modelo/baseline. Tabela de resultados acima já reflete esse split.

2. **Consolidar comparação final** — feito em `comparacao_final.py` (carrega os 4
   modelos + baselines, gera tabela + 3 gráficos em `resultados/`, seguindo a skill
   `dataviz` do projeto).

3. **Redação do TG** — usar os resultados acima para preencher "Materiais e métodos"
   (dataset, engenharia de features, modelos, métricas) e "Resultados" no rascunho
   (`rascunhos/estrutura_tg.txt` já tem o esqueleto dessas seções).

4. **Diagnóstico de resíduos (concluído) — conclusão: teto próximo do limite
   informacional day-ahead com os dados atuais.** Script: `diagnostico_residuos.py`
   (reaproveita `comparacao_final.gerar_previsoes`, não re-treina nada). Testes de
   Ljung-Box/Box-Pierce (`statsmodels`, adicionado ao venv) em 3 granularidades +
   correlação entre resíduos dos 4 modelos:
   - **Resíduo diário (41 dias) ≈ ruído branco** para LSTM/MLP/SVM (nenhum lag
     significativo em 1,2,3,7) — as defasagens D-1/D-2/D-3/D-7 já capturam bem a
     dependência entre dias. XGBoost teve 1 rejeição isolada e limítrofe (lag 2,
     p=0,049), não é padrão robusto.
   - **Resíduo horário tem autocorrelação forte e altamente significativa** em
     todos os lags testados (1,2,3,6,12,24; p ~1e-40 a 1e-56) para os 4 modelos.
   - Ao remover o perfil médio de resíduo por hora do dia (viés fixo por hora), as
     estatísticas praticamente não mudam (ex.: SVM lag1: 179,3 → 179,5) — **não é
     viés fixo por hora** (corrigível com features). O perfil por hora
     (`resultados/diagnostico_residuos_perfil_hora.png`) também não mostra padrão
     robusto (barras de erro cruzando zero na maioria das horas).
   - **Interpretação:** a autocorrelação horária que sobra é persistência do
     regime climático *dentro do próprio dia de teste* (se o dia amanhece mais
     nublado que o típico, o "choque" se propaga por todas as horas daquele dia) —
     informação que não existe no momento da previsão *day-ahead*, logo não
     corrigível com engenharia de feature/tuning sobre os dados atuais.
   - **Correlação entre resíduos horários dos 4 modelos: 0,93–0,99**
     (`resultados/diagnostico_residuos_correlacao.png`) — os modelos erram nos
     mesmos momentos, com a mesma magnitude. Consequência prática: **ensemble
     dos 4 modelos tende a ganhar pouco** (não há muito ruído idiossincrático
     independente entre eles pra cancelar na média/stacking).
   - **Conclusão geral:** mais *trials* de Optuna, mais engenharia de feature
     dia/hora, ou ensembling têm retorno esperado baixo — o "buraco" que resta
     não é dessa natureza. Ganhos adicionais relevantes provavelmente exigiriam
     dado novo (NWP, nowcasting/atualização intradiária), fora do escopo atual.
     Saídas completas em `resultados/diagnostico_residuos_*.{csv,png}`.

5. **Item de segurança pendente (não bloqueante)** — `dataset/formando_dataset/config.yaml`
   tem credenciais de banco em texto puro (SSH keys, senhas Postgres/MySQL). Mover para
   variáveis de ambiente / `.gitignore` antes de qualquer commit ou compartilhamento do
   projeto.

## Notas técnicas úteis para retomar

- O console do Windows quebra em caracteres acentuados dos prints (`γ`, `é`, etc.) —
  sempre rodar os scripts com `$env:PYTHONIOENCODING="utf-8"` antes.
- Ambiente Python: `D:\tg\dataset\formando_dataset\.venv` (compartilhado por todo o
  projeto — pandas, numpy, scipy, pvlib, scikit-learn, optuna, tensorflow, xgboost,
  joblib, statsmodels).
- `direcao_vento` tem ~57% de valores faltantes no dataset horário — atualmente não é
  usada nas features causais (só `velocidade_vento` é), então não bloqueia nada, mas vale
  mencionar como limitação de dado na redação.
- Capacidade instalada da usina bet01: **2500 kWp** (usada em `dados.py:CAPACIDADE_KWP`
  para normalizar o nRMSE e como teto físico de clipping das previsões).

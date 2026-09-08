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

## Resultados (conjunto de teste, 737 linhas, últimos ~2 meses da série atual)

| Modelo | RMSE (kW) | nRMSE | Skill vs. persist. ingênua |
|---|---|---|---|
| **LSTM** | **468,8** | 18,8% | **+25,7%** |
| MLP | 483,5 | 19,3% | +23,4% |
| SVM (SVR) | 487,4 | 19,5% | +22,8% |
| XGBoost | 489,1 | 19,6% | +22,5% |
| Persistência inteligente | 537,0 | 21,5% | +14,9% |
| Persistência ingênua | 631,1 | 25,2% | — |

Os 4 modelos batem as duas persistências. Margem entre eles é pequena (LSTM à frente).
Importância de features do XGBoost: `P_ac` defasado (médias 7d/3d, D-1) domina, seguido
por irradiância horizontal — consistente com a ausência de previsão meteorológica (NWP)
como entrada.

## Pendências / próximos passos

1. **Split por data (prioridade alta)** — o split atual em `dados.py` (`dividir`,
   `carregar_split`) é um placeholder por fração (últimos 15% das linhas). O split
   definitivo é por data: treino até 31/07/2026, teste = agosto-setembro/2026 (os meses
   mais recentes da usina). Passos:
   - Extrair/concatenar o bruto de ago-set 2026 (mesmo formato de
     `bet01_completo_bruto_concatenado_202505_202607.csv`).
   - Rodar de novo `variáveis_calculadas.py` → `preprocessamento.py` (apontando `ENTRADA`
     para o bruto atualizado) → `features.py`, para regenerar `bet01_features_dayahead.csv`
     com o período estendido.
   - Trocar `carregar_split()` por `dividir_por_data(df)` em `baselines.py` e em todos os
     `modelos/*.py`, e re-treinar/reavaliar os 4 modelos com o split definitivo.

2. **Consolidar comparação final** — script único que carrega os resultados dos 4
   modelos + baselines e gera uma tabela/gráfico comparativo final (previsto vs.
   observado para dias de teste representativos), seguindo a skill `dataviz` do projeto.

3. **Redação do TG** — usar os resultados acima para preencher "Materiais e métodos"
   (dataset, engenharia de features, modelos, métricas) e "Resultados" no rascunho
   (`rascunhos/estrutura_tg.txt` já tem o esqueleto dessas seções).

4. **Possível refinamento de modelos** — se quiser espremer mais desempenho antes da
   redação final: mais *trials* do Optuna (o usuário já testou `N_TRIALS=1000` no SVM
   manualmente), outras features (ex.: variáveis de vento com mais defasagens), ou
   ensemble dos 4 modelos.

5. **Item de segurança pendente (não bloqueante)** — `dataset/formando_dataset/config.yaml`
   tem credenciais de banco em texto puro (SSH keys, senhas Postgres/MySQL). Mover para
   variáveis de ambiente / `.gitignore` antes de qualquer commit ou compartilhamento do
   projeto.

## Notas técnicas úteis para retomar

- O console do Windows quebra em caracteres acentuados dos prints (`γ`, `é`, etc.) —
  sempre rodar os scripts com `$env:PYTHONIOENCODING="utf-8"` antes.
- Ambiente Python: `D:\tg\dataset\formando_dataset\.venv` (compartilhado por todo o
  projeto — pandas, numpy, scipy, pvlib, scikit-learn, optuna, tensorflow, xgboost, joblib).
- `direcao_vento` tem ~57% de valores faltantes no dataset horário — atualmente não é
  usada nas features causais (só `velocidade_vento` é), então não bloqueia nada, mas vale
  mencionar como limitação de dado na redação.
- Capacidade instalada da usina bet01: **2500 kWp** (usada em `dados.py:CAPACIDADE_KWP`
  para normalizar o nRMSE e como teto físico de clipping das previsões).

# Registro de mudanças v1 → v2 — previsão day-ahead de P_ac (usina bet01)

Base para o capítulo de Materiais e Métodos. Cada entrada traz **o que mudou**, **por quê** e o
**impacto esperado**. Achados citados como "Fase 0 §n" estão em
`diagnostico/fase0_saida_console_v2.txt` (gerado por `diagnostico/diagnostico_fase0_v2.py`).
A v1 permanece intacta: seus arquivos são apenas lidos ou importados.

Decisões D1–D8 aprovadas após a Fase 0 (13/10/2026).

---

## Fase 1 — Engenharia de features (`features_v2.py`)

Saídas: `dados/bet01_features_dayahead_v2.csv` (5.725 linhas; alvo `P_ac`, 39 features e
2 colunas de filtragem) e `dados/dicionario_features_v2.csv` (nome, grupo, descrição, unidade,
tipo, variável de origem e defasagem de cada feature). Contagem de features: v1 = 55, v2 = 39
(−16 em M1, −1 em M4, +1 em M3).

### M1. Remoção das médias móveis (`{var}_media3d` e `{var}_media7d`)
- **O que mudou:** as 16 colunas de média móvel foram removidas (8 de 3 dias e 8 de 7 dias).
  Ficam apenas os lags de 1, 2, 3 e 7 dias de cada variável de `VARS_LAG`.
- **Por quê:**
  - `media3d = (lag1d + lag2d + lag3d) / 3` exatamente (erro máximo 2e-12, Fase 0 §7). A matriz
    de features da v1 tinha colinearidade exata, sem informação nova.
  - `media7d` foi removida por decisão do autor depois da primeira versão da Fase 1. Quatro dos
    seus sete termos (lags 1, 2, 3 e 7) já são features. Com ela, 6 features tinham VIF > 100
    (máximo 482, `irradiacao_horizontal_media7d`). A única informação exclusiva dela, os lags de
    4 a 6 dias, entrava diluída na média.
- **Impacto esperado:**
  - a matriz tem posto completo (39/39), com número de condição 54 (era 156 com a `media7d`) e
    VIF máximo de 63;
  - para SVR, MLP e LSTM, somem direções redundantes;
  - para o XGBoost, somem atalhos pré-computados que as árvores não reproduzem exatamente;
  - perde-se a informação dos lags de 4 a 6 dias, que se espera ser marginal diante dos lags de 1
    a 3 dias;
  - +18 linhas de treino (24 a 26/10/2025, das 11h às 16h), antes descartadas porque a `media7d`
    exigia 7 dias consecutivos de histórico e esses dias apontavam para a lacuna de 20/10/2025.

### M2. Posição solar e calendário calculados a partir do timestamp, no ponto médio da hora
- **O que mudou:** `elevacao_solar`, azimute, `hora_*` e `dia_ano_*` passam a ser calculados com
  pvlib (NREL SPA, `method="nrel_numpy"`; elevação **aparente**, como na v1) no instante h:30 de
  cada hora, localizado em America/Fortaleza, com LAT/LON de `dados.py`. Na v1 essas grandezas
  eram a média horária dos valores calculados em cada leitura de 1 min
  (`variaveis_calculadas.py` + `etapa_reamostragem`).
- **Por quê:**
  1. a média aritmética do azimute na hora do meio-dia solar, quando o sol passa ao norte
     (0°/360°), resulta em ≈180° (sul): 362 linhas afetadas, 41 delas no teste (Fase 0 §3);
  2. horas sem leituras recebiam posição solar interpolada por `etapa_faltantes`, com erro de até
     23,6° na elevação;
  3. uma grandeza determinística não deve depender da cobertura do logger.

  O rótulo horário marca o início do intervalo [h, h+1) e a média horária da elevação já
  equivalia ao ponto médio (RMS 0,42° contra h:30 e 6,27° contra h:00), então o instante de
  referência não muda.
- **Impacto esperado:** azimute fisicamente correto nas 362 linhas; elevação praticamente igual
  à v1 (RMS 0,415°, com diferenças concentradas na passagem do sol perto do zênite e nas horas sem
  dado); `hora_*` passa de h+0,49 (média dos minutos) para h+0,5.

### M3. Azimute em seno e cosseno
- **O que mudou:** `azimute_solar` (graus) foi substituído por `azimute_sin` e `azimute_cos`.
- **Por quê:** o azimute é uma variável circular. Nesta latitude (−3,3°), entre meados de março e
  o início de outubro o sol passa ao norte e o azimute salta de ≈0° para ≈360° no meio-dia solar,
  uma descontinuidade artificial para SVR, MLP e LSTM.
- **Impacto esperado:** representação contínua do azimute; +1 feature.

### M4. Remoção do zênite solar
- **O que mudou:** `zenite_solar` foi removido.
- **Por quê:** `elevacao_solar + zenite_solar = 90°` em todas as linhas (Fase 0 §3), outra
  colinearidade exata.
- **Impacto esperado:** −1 feature, sem perda de informação.

### M5. Recorte diurno pela elevação no ponto médio
- **O que mudou:** as linhas-alvo passam a ser as horas com elevação solar aparente em h:30 > 0.
  Na v1, o critério era a média horária da elevação > 0 (o ramo `elevacao_solar` de
  `etapa_recorte_diurno`: o critério medido de `preprocessamento.main()` nunca foi usado por
  `features.py`, Fase 0 §2).
- **Por quê:** consistência com M2. O critério continua astronômico, sem seleção de linhas por
  dado medido de D+1.
- **Impacto esperado:** +21 linhas (7 às 5h e 14 às 17h, todas no treino; P_ac médio 42 kW,
  máximo 68 kW); nenhuma linha da v1 sai. Somadas às +18 da M1, o total vai de 5.686 para 5.725
  (treino 5.195 → 5.234; teste 491, inalterado). Nas linhas em comum, o alvo e os lags são
  idênticos aos da v1.

### M6. Coluna `alvo_imputado` (decisão D3)
- **O que mudou:** nova coluna booleana, verdadeira quando o `P_ac` da hora-alvo estava ausente
  depois de `etapa_reamostragem` e antes de `etapa_faltantes` (ou seja, foi interpolado). A grade
  horária é reconstruída localmente chamando as etapas importadas de `preprocessamento.py`, sem
  alterá-lo.
- **Por quê:** `etapa_faltantes` interpola `P_ac` e, com `limit=1`, o pandas preenche a primeira
  hora de **qualquer** lacuna interna, não só lacunas de 1 h (Fase 0 §1). Exemplo: 20/10/2025 10h,
  sem nenhuma leitura, recebeu 1.442 kW interpolados até a noite.
- **Impacto esperado:** 2 linhas (ambas no treino; 0 no teste) excluídas do treino e da
  avaliação. Os valores imputados continuam alimentando os lags. Como usa dado de D+1, a coluna
  **nunca** é feature.

### M7. Coluna `evento_operacional` (decisão D2)
- **O que mudou:** nova coluna booleana.
  - **Núcleo:** hora diurna com G_poa > 200 W/m² e PR = P_ac / (2500 · G_poa / 1000) < 0,3.
  - **Expansão:** horas vizinhas (±1 h, no mesmo dia) de uma hora-núcleo com G_poa > 50 W/m² e
    PR < 0,7.

  Os quatro limiares são constantes ajustáveis em `features_v2.py`.
- **Por quê:** há paradas reais da usina, com os 10 inversores reportando 0 kW, comunicação ativa
  e irradiância alta (ex.: 07/09/2026, das 6h às 9h25; Fase 0 §4–5). São falhas operacionais, não
  variabilidade meteorológica, e não são previsíveis com as features disponíveis. A expansão
  captura as horas de início e fim do evento (em 07/09/2026: 6h, com G_poa = 196 W/m², e 9h, com
  PR = 0,48).
- **Impacto esperado:** 377 linhas marcadas (núcleo 303 + expansão 74; treino 370, teste 7),
  excluídas do treino. O treino efetivo fica com 4.862 linhas. O teste é reportado com elas
  (cenário A, 491 linhas) e sem elas (cenário B, 484). Como usa dado de D+1, a coluna **nunca** é feature. Os lags que vêm de horas
  de evento (694 linhas, Fase 0 §5) não são tratados, porque um previsor em operação também
  veria esses valores: limitação documentada.

### Verificações automáticas da Fase 1
- **Posto** da matriz de treino efetivo padronizada (4.862 linhas × 39): 39, número de condição 54.
- **VIF:** 31 features acima de 10 e nenhuma acima de 100. O maior é o de
  `irradiacao_horizontal_lag2d` (63); os lags da irradiância horizontal ficam entre 44 e 63, por
  causa da correlação com as irradiâncias inclinada e refletida. Elevação e `hora_*` ficam entre
  35 e 44, porque ambas descrevem o ciclo diurno.
- **Lags:** 200 linhas sorteadas × 32 colunas = 6.400 comparações com a grade horária em
  t − d dias, sem divergências.
- **Causalidade:** em 30 dias-alvo sorteados, todos os dados medidos a partir de 00h00 de D+1
  foram trocados por ruído e nenhuma feature mudou. O controle positivo (perturbar a partir de
  00h00 de D) alterou 96 valores, o que mostra que o teste detecta vazamento.

---

## Decisões mantidas da v1
- **`VARS_LAG` (D1):** P_ac, temp_ambiente, umidade_relativa, velocidade_vento, as três
  irradiâncias e T_inv, com lags de 1, 2, 3 e 7 dias. Nenhum par tem |r| de Pearson ≥ 0,95 no lag
  de 1 dia (máximo 0,93, temp_ambiente × umidade_relativa). `temp_modulo` não foi incluída
  (r = 0,92 com G_poa).
- **`preprocessamento.py`** é importado sem alteração, inclusive a semântica de `limit=1` em
  `etapa_faltantes` (tratada pela M6) e o clip de P_ac em 2.500 kW em `etapa_outliers` (o máximo
  real de 1 min é 2.552 kW; 1,35% das leituras ficam acima).
- **Lags "mesma hora, d dias antes"** via pivô data × hora. A previsão de D+1 usa dados de D até
  17h59, logo a emissão precisa ocorrer depois de ~18h de D (Fase 0 §7).

---

## Fase 2 — Infraestrutura (`dados_v2.py`, `ajuste_hp_v2.py`, `baselines_v2.py`, `modelos/regressao_linear_v2.py`)

### M8. Céu claro da persistência inteligente corrigido (`baselines_v2.py`, decisão D6)
- **O que mudou:** `clearsky_potencia` passa a localizar os timestamps em America/Fortaleza e a
  avaliar o céu claro no ponto médio da hora (h:30). O modelo continua o mesmo (Ineichen,
  P = 2500 · GHI_cs / 1000), assim como a lógica da persistência inteligente (kt médio de 3 dias
  × céu claro do dia-alvo, com queda para a ingênua onde kt não é definido). A persistência
  ingênua é reutilizada da v1 sem alteração.
- **Por quê:** a v1 passava ao pvlib um índice sem fuso, que o pvlib trata como UTC. A curva
  ficava deslocada em +3 h, com pico às 15h em vez de 12h (Fase 0 §3). Na prática, as horas das
  5h às 8h caíam na persistência ingênua e as demais viravam uma média de 3 dias.
- **Impacto medido no teste** (o cálculo é determinístico):

  | Cenário | Ingênua | Inteligente v1 | Inteligente v2 | Skill da inteligente vs. ingênua |
  |---|---|---|---|---|
  | A (491 horas) | 423,5 kW | 384,2 kW | 346,5 kW | 9,3% → 18,2% |
  | B (484 horas) | 392,9 kW | 357,1 kW | 320,0 kW | 9,1% → 18,5% |

  A referência principal do skill score fica mais exigente, então os skill scores dos modelos
  de ML devem ser menores que os da v1. A versão da v1 (384,2 kW, idêntica à
  `comparacao_final_tabela.csv`) é mantida nas previsões como `persistencia_inteligente_v1`, só
  para rastreabilidade.

### M9. Ajuste de hiperparâmetros (`ajuste_hp_v2.py`, decisão D4)
- **O que mudou:** mesmo esquema da v1 (`TimeSeriesSplit` com 5 dobras, TPE com semente 42),
  mas agora:
  - `N_TRIALS = 50` vale para todas as famílias (na v1: SVR 1000, MLP 30, LSTM 30, XGBoost 40);
  - o RMSE de cada dobra, as datas inicial e final de cada dobra de validação e o número de
    linhas usadas são gravados em `trial.user_attrs`;
  - os estudos ficam em SQLite (`optuna/estudos_v2.db`) e podem ser retomados. Na retomada, a
    semente passa a ser 42 + nº de trials já gravados, para não repetir os sorteios aleatórios
    iniciais; sem interrupção ela é 42;
  - uma máscara exclui linhas do ajuste e do RMSE (D2/D3). As dobras são definidas sobre a
    série de treino completa, então as fronteiras temporais não dependem da máscara; dentro de
    cada dobra, as linhas marcadas saem do fit e do RMSE;
  - há um aviso automático quando o ótimo fica a menos de 5% da borda do espaço de busca, em
    escala log quando é o caso, ou quando é a categoria extrema.
- **Por quê:** orçamento igual entre as famílias, decidido a priori por causa do *winner's
  curse*: com 1000 trials, o SVR da v1 tinha vantagem de seleção sobre as demais famílias. Os
  RMSEs por dobra permitem a análise sazonal e a rastreabilidade.
- **Impacto esperado:** comparação mais justa entre as famílias; o CV do SVR fica menos
  otimista do que na v1.

### M10. Baseline de regressão linear (`modelos/regressao_linear_v2.py`)
- **O que mudou:** novo baseline, `StandardScaler` + `LinearRegression`, sem ajuste de
  hiperparâmetros, com os mesmos dados, dobras e filtros dos modelos de ML.
- **Por quê:** separa o ganho que vem da não linearidade dos modelos de ML do ganho que já vem
  das features. Serve de referência nos testes de Diebold-Mariano e na importância por permutação.
- **Impacto observado:** RMSE de CV de 462,4 kW; no teste, 340,2 kW no cenário A e 304,9 kW no
  cenário B. Já fica abaixo da persistência inteligente corrigida.

### M11. Dados e formato das previsões (`dados_v2.py`)
- **O que mudou:**
  - o carregamento valida o índice (estritamente crescente, porque o `validation_split` do
    Keras usa as últimas linhas), a ausência de NaN e o tipo booleano das flags;
  - `colunas_features()` lê as features do dicionário, nunca incluindo alvo ou flags;
  - máscaras: `treino_excluir` (372 linhas), `teste_A` (491) e `teste_B` (484);
  - as previsões de teste usam um formato longo comum a todos os modelos (`tempo`, `modelo`,
    `semente`, `y_obs`, `y_pred`, `alvo_imputado`, `evento_operacional`), com clip em
    [0, 2500] kW como na v1. Modelos determinísticos recebem `semente = -1`.
- **Por quê:** garante que todos os modelos usem exatamente as mesmas linhas, filtros e formato,
  e que a comparação final leia só as previsões salvas.

---

## Fase 3 — Modelos (`modelos/svm_v2.py`, `mlp_v2.py`, `lstm_v2.py`, `xgboost_model_v2.py`, `treino_v2.py`)

Os quatro scripts mantêm a estrutura da v1 (`construir_modelo`, `espaco_busca`, wrapper Keras
para MLP e LSTM) e delegam a um fluxo comum (`treino_v2.executar`): ajuste, retreino com
sementes, previsões, JSON e modelos salvos. Os espaços de busca de MLP, LSTM e XGBoost e as
arquiteturas são os mesmos da v1; o teto de épocas (150), a paciência (15) e o
`validation_split` (0,15) também.

### M12. SVR com alvo padronizado e novo espaço de busca
- **O que mudou:** o SVR passa a ser envolvido por um `TransformedTargetRegressor` com
  `StandardScaler` no alvo. O espaço de busca, todo em escala log, fica C ∈ [1e-1, 1e3],
  epsilon ∈ [1e-3, 1] e gamma ∈ [1e-4, 1].
- **Por quê:** na v1, C e epsilon atuavam em kW (epsilon ∈ [0,01; 31,6] kW, minúsculo diante de
  um alvo de 0 a 2.500 kW), então o espaço de busca dependia da escala de P_ac. Com o alvo
  padronizado, C e epsilon ficam em unidades de desvio-padrão, comparáveis aos intervalos
  usuais da literatura.
- **Impacto esperado:** busca mais eficiente com 50 trials. O tubo epsilon passa a ter tamanho
  relevante diante da variabilidade do alvo.

### M13. Sementes: ajuste com uma semente, retreino final com 10 (decisão D5)
- **O que mudou:**
  - o ajuste de hiperparâmetros de todas as famílias usa a semente 42;
  - o retreino final de MLP, LSTM e XGBoost usa as sementes 0 a 9, no treino efetivo completo,
    com 10 modelos salvos por família; o SVR é determinístico e é treinado uma única vez;
  - no Keras, cada `fit` começa com `tf.keras.backend.clear_session()` e
    `tf.keras.utils.set_random_seed(semente)`, com `tf.config.experimental.enable_op_determinism()`
    ativado;
  - o XGBoost usa `random_state = semente` e `n_jobs = 1` (na v1, `n_jobs = -1`).
- **Por quê:**
  - na v1, cada família tinha um único treino (sementes 42 fixas), então diferenças entre
    famílias se confundiam com a variância da inicialização;
  - com 10 sementes é possível reportar média ± desvio-padrão e testar a previsão média;
  - `n_jobs = 1` porque o XGBoost (histograma) gera árvores diferentes com a mesma semente
    conforme o número de threads: no teste, a mesma configuração deu 314,9 kW numa hora com 1
    thread e 332,3 kW com 8. Neste conjunto pequeno, 1 thread também não é mais lento.
- **Verificação:** com a mesma semente, os dois treinos de MLP, LSTM e XGBoost geram previsões
  idênticas (máx |dif| = 0); com sementes diferentes, os modelos diferem.

### M14. LSTM com entradas derivadas do dicionário
- **O que mudou:** as colunas da sequência e do ramo estático passam a vir de
  `dicionario_features_v2.csv`, não mais escritas à mão.
  - **Sequência:** 4 passos (lags de 7, 3, 2 e 1 dias, em ordem cronológica) × 8 variáveis
    de `VARS_LAG`.
  - **Ramo estático:** só as 7 determinísticas (elevação, azimute em seno/cosseno, hora e dia
    do ano em seno/cosseno). Na v1 entravam também as médias móveis de 3 e 7 dias (removidas
    na M1), além de zênite e azimute em graus.

  O script aborta se alguma variável tiver lags diferentes ou se alguma feature ficar de fora.
- **Por quê:** garante que a LSTM veja exatamente as mesmas 39 features dos demais modelos.
- **Impacto esperado:** o ramo estático fica só com o "quando" (geometria solar e calendário);
  toda a informação medida passa pela recorrência.

### M15. Treino, previsões e artefatos
- **O que mudou:**
  - o retreino final usa só o treino efetivo (sem `alvo_imputado` e `evento_operacional`; 4.862
    linhas);
  - as previsões cobrem todas as 491 horas de teste, no formato longo comum, com as flags para
    filtrar na comparação;
  - cada família gera um JSON (`modelos/<família>_resumo_v2.json`) com os melhores
    hiperparâmetros, o RMSE de CV médio e por dobra com as datas, os avisos de borda, as
    sementes e os tempos;
  - modelos e scalers são salvos por semente, e o primeiro é recarregado e conferido: a previsão
    do modelo recarregado é idêntica à do modelo em memória;
  - o aviso de borda ignora inteiros com só 2 valores possíveis (ex.: 1 ou 2 camadas de LSTM).
- **Por quê:** a Fase 4 lê só as previsões salvas, e a importância por permutação precisa dos
  modelos recarregáveis.

---

## Fase 4 — Comparação (`comparacao_v2.py`)

### M16. Protocolo de avaliação
- **O que mudou:** na v1, `comparacao_final.py` recarregava os modelos e reportava uma única
  tabela de métricas e três figuras. A v2 lê só as previsões salvas e acrescenta:
  - **Dois cenários de teste:**
    - A: todas as horas sem alvo imputado. São as mesmas 491 horas e observações da v1, o que
      o script confere; é o cenário principal, usado nas figuras e na permutação.
    - B: também sem `evento_operacional` (484 horas).
  - **Métricas:** média ± desvio-padrão entre as 10 sementes. MLP, LSTM e XGBoost também são
    avaliados pela previsão média das sementes.
  - **Skill score** em RMSE contra a persistência inteligente (referência principal) e contra a
    ingênua (secundária).
  - **Teste de Diebold-Mariano com correção HLN** sobre perdas diárias: L_t é a média dos erros
    quadráticos horários do dia, d_t = L1_t − L2_t, h = 1, DM = média(d)/√(γ₀/n), com
    γ₀ = (1/n) Σ(d_t − média)², DM* = DM·√((n−1)/n), e o p-valor bicaudal vem da t de Student
    com n − 1 graus de liberdade. Também se reporta a autocorrelação de lag 1 de d_t. Com h = 1,
    DM* coincide com a estatística t pareada das perdas diárias.
    - **Pares testados:** os modelos de ML entre si; cada modelo (inclusive a regressão linear)
      contra a persistência inteligente; cada modelo de ML contra a regressão linear.
  - **Erro por hora do dia** (RMSE e MBE).
  - **Erro por classe de céu:** tercis do índice diário k_d = Σ P_obs / Σ P_céu claro, com o
    céu claro corrigido de `baselines_v2`. O k_d usa só as horas sem evento operacional, para
    que uma parada não rebaixe a classe de um dia de céu claro. As classes são relativas ao
    teste; k_d pode passar de 1, porque o envelope de céu claro usa a GHI e a usina tem
    rastreadores.
  - **Importância por permutação** no teste: 10 repetições, aumento do RMSE, por feature e por
    grupo de variáveis (grupo permutado de uma vez). As mesmas permutações valem para todos os
    modelos. MLP, LSTM e XGBoost entram pela previsão média das sementes. Os modelos salvos são
    recarregados, e o script confere que a previsão sem permutação reproduz a salva.
  - **Tabela sazonal** com o RMSE por dobra de CV da melhor configuração de cada família,
    marcada como otimista.
  - **Tabela v1 × v2**, que lê `comparacao_final_tabela.csv` só para leitura.
- **Por quê:**
  - com a v1 não era possível dizer se as diferenças entre modelos eram estatisticamente
    significativas nem estáveis entre sementes;
  - a referência principal passa a ser a persistência inteligente, mais exigente que a ingênua;
  - os cenários A e B separam o erro de previsão do efeito de falhas operacionais.
- **Impacto esperado:** conclusões com incerteza explícita (desvio-padrão entre sementes e
  p-valores). Diferenças pequenas entre famílias (± 10 kW) tendem a não ser significativas com
  41 dias de teste.

### Decisão sobre os avisos de borda (15/10/2026): manter
- **Situação:** no ajuste completo, ao menos um hiperparâmetro ótimo de cada família ficou na
  borda do espaço de busca:
  - SVR: `gamma` = 1,08e-4, a 0,9% do mínimo;
  - MLP: `unidades` = 128, a categoria máxima;
  - LSTM: `unidades_densas` = 16 (mínimo) e `batch_size` = 128 (máximo);
  - XGBoost: `max_depth` = 2 (mínimo), `min_child_weight` = 10 (máximo) e
    `colsample_bytree` = 0,99.

  Em SVR e XGBoost, o ótimo ficou no lado do modelo mais simples ou mais regularizado.
- **Decisão do autor:** manter os espaços de busca e o orçamento definidos a priori (D4) e
  reportar os avisos como limitação. Ampliar os espaços depois de conhecer o desempenho no
  teste introduziria uma escolha condicionada ao conjunto de teste.
- **Implicação para o texto:** os ótimos de SVR e XGBoost podem estar truncados pelo espaço de
  busca, com o sinal aproveitável parecendo de baixa complexidade, o que é coerente com a
  regressão linear competitiva. O desempenho dessas famílias pode estar levemente subestimado.

---

## Fase 5 — Ablação da LSTM (`modelos/lstm_ablacao_v2.py`)

### M17. Ablação do ramo sequencial
- **O que é:** um experimento novo, sem equivalente na v1. Três variantes compartilham os
  melhores hiperparâmetros da LSTM, as sementes 0 a 9, o treino efetivo, o ramo estático e a
  cabeça densa; muda só o ramo da sequência:
  - **(a) LSTM original:** passos em ordem cronológica (D-6, D-2, D-1, D). Funciona como
    controle: tem de reproduzir exatamente as previsões salvas na Fase 3.
  - **(b) Sequência invertida:** a mesma rede, lendo os passos do mais recente para o mais
    antigo.
  - **(c) Ramo denso:** a LSTM trocada por Dense(unidades_lstm, relu) sobre a sequência
    achatada (32 valores), sem recorrência.

  Variantes comparadas por métricas (média ± dp entre sementes; cenários A e B) e por
  Diebold-Mariano (HLN) sobre a previsão média das sementes, entre si e contra a persistência
  inteligente.
- **Por quê:** a "sequência" da LSTM tem só 4 passos irregulares (lags de 7, 3, 2 e 1 dias), e a
  importância por permutação mostrou que a LSTM depende mais do ramo estático (calendário e
  posição solar) do que da sequência. A ablação testa se o desempenho da LSTM, empatada com o
  XGBoost, vem da recorrência ou das features e da arquitetura de dois ramos.
- **Ressalva de desenho:** o ramo denso tem menos parâmetros (1.985 contra 6.177 da LSTM com 32
  unidades) e usa os hiperparâmetros otimizados para a LSTM, não para ele. Uma eventual
  desvantagem de (c) é, portanto, um limite superior para o ganho da recorrência.
- **Resultados** (execução completa em 15/10/2026):
  - **Controle:** a variante (a) reproduziu as previsões da LSTM da Fase 3 com máx |dif| de
    2,3e-13 kW em todas as 10 sementes. As comparações são, portanto, nas mesmas condições.
  - **RMSE no teste:**

    | Variante | Cenário A, por semente (kW) | Cenário A, média das sementes (kW) | Cenário B, média das sementes (kW) |
    |---|---|---|---|
    | (a) LSTM original | 319,9 ± 5,5 | 313,8 | 279,7 |
    | (b) Sequência invertida | 317,2 ± 6,9 | 311,7 | 279,3 |
    | (c) Ramo denso | 325,0 ± 7,0 | 315,0 | 282,1 |

  - **Diebold-Mariano (HLN):** nenhuma diferença significativa entre as variantes, nos dois
    cenários (p de 0,37 a 0,89). As três são significativamente melhores que a persistência
    inteligente (p ≤ 0,008).
  - **Interpretação:**
    - inverter a ordem dos passos não piora o modelo; ele não usa a ordem temporal da
      sequência;
    - tirar a recorrência custa no máximo 5 kW por semente, ou 1,2 kW na média das sementes,
      menos que a variação entre sementes (de 5,5 a 7 kW);
    - pela ressalva de desenho acima, isso é um limite superior para o ganho da recorrência.

    O desempenho da LSTM vem das features e da arquitetura de dois ramos, não da recorrência:
    com 4 passos irregulares, a "sequência" funciona como um vetor de features.
  - **Observação:** o ramo denso tem MAE maior (238,6 contra 222,2 kW por semente) e viés menor
    (MBE de +30,9 contra +61,4 kW). A diferença de RMSE não é significativa.

---

## Ambiente de execução da rodada v2 (13 a 15/10/2026)
- **Software:** Python 3.12.10, pandas 3.0.3, NumPy 2.4.6, scikit-learn 1.9.0, XGBoost 3.4.1,
  TensorFlow 2.21 / Keras 3.15, Optuna 4.9, pvlib 0.15.1, SciPy 1.17.1. O venv é
  `dataset/formando_dataset/.venv`.
- **Hardware:** só CPU, AMD Ryzen 5 3500U (4 núcleos), 10 GB de RAM. Não há GPU utilizável: a
  Radeon Vega 8 é integrada, e o TensorFlow não usa GPU no Windows nativo desde a versão 2.11.
- **Tempos do modo completo** (ajuste com 50 trials + retreino):

  | Modelo | Ajuste | Retreino |
  |---|---|---|
  | SVR | 3 min | 0 min |
  | XGBoost | 11 min | 0,2 min |
  | MLP | 64 min | 1,8 min |
  | LSTM | 59 min | 2,2 min |

  A parada antecipada interrompeu a LSTM, na maioria das sementes, entre 16 e 19 épocas.
- **Proveniência:** o `.joblib` da regressão linear foi gravado num ambiente com scikit-learn
  1.6.1. Suas previsões são idênticas, bit a bit, às geradas no venv do projeto, então não há
  efeito nos resultados.

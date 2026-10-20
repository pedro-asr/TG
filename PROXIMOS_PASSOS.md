# TG — Previsão day-ahead de P_ac (usina bet01) — Próximos passos

Atualizado em 19/10/2026, depois da v2 e da leitura do PDF do texto de 19/10/2026 (Fases 0 a 5 concluídas; PR #1 `tg-v2` → `main` mesclado
em 19/10/2026). A versão anterior deste arquivo, com o estado da v1, está no histórico do git
(commit 25fedd6). O detalhamento das mudanças está em `código/resultados_v2/registro_mudancas_v2.md`
(M1–M17).

---

## 1. Onde estamos: o que os resultados da v2 dizem

Teste: 01/08 a 10/09/2026 (41 dias, 491 horas). Cenário A = todas as horas sem alvo imputado;
cenário B = também sem evento operacional (484 horas). MLP, LSTM e XGBoost: média ± dp entre as
10 sementes, com a previsão média das sementes entre parênteses. O p do DM-HLN usa a previsão
média das sementes.

| Modelo | RMSE A (kW) | RMSE B (kW) | Skill vs. intelig. (A) | DM vs. intelig. (A) |
|---|---|---|---|---|
| Persistência ingênua | 423,5 | 392,9 | −22,2% | — |
| Persistência inteligente | 346,5 | 320,0 | 0 | — |
| Regressão linear | 340,2 | 304,9 | +1,8% | p = 0,669 |
| SVR | 329,5 | 297,9 | +4,9% | p = 0,115 |
| MLP | 331,4 ± 9,6 (323,7) | 304,0 ± 11,5 (295,5) | +4,3% (+6,6%) | p = 0,058 |
| LSTM | 319,9 ± 5,5 (313,8) | 286,5 ± 4,5 (279,7) | +7,7% (+9,4%) | **p = 0,008** |
| XGBoost | 317,8 ± 0,3 (317,7) | 284,0 ± 0,3 (283,9) | +8,3% (+8,3%) | **p = 0,015** |

Leituras principais (são a espinha da discussão do TG):

1. **Só XGBoost e LSTM superam com significância a persistência inteligente e a regressão linear**,
   nos dois cenários. LSTM e XGBoost empatam (p = 0,435 em A; 0,406 em B). SVR e MLP não se
   separam da persistência inteligente.
2. **O ganho sobre a referência correta é pequeno:** 8 a 9% em A, 11 a 13% em B. Boa parte do skill
   de 22 a 26% da v1 vinha da referência fraca: o céu claro deslocado em +3 h (M8) levava a
   persistência inteligente de 346,5 para 384,2 kW.
3. **O "1º lugar da LSTM" na v1 era ruído de semente:** os 312,3 kW da v1 (uma semente) caem
   dentro de 319,9 ± 5,5. O XGBoost é o mais estável (dp de 0,3 kW) e o menos enviesado (MBE de
   −2,4 kW, contra +61,4 kW da LSTM e +63,8 kW do SVR).
4. **O erro tem cara de regressão à média:** nos dias nublados todos os modelos superestimam (MBE
   de +157 a +234 kW) e nos claros subestimam (−100 a −171 kW). No nublado, a LSTM empata com a
   persistência inteligente (372,7 contra 373,6 kW). Os maiores erros estão nas horas de 7h a 10h
   (RMSE de 357 a 491 kW para XGBoost e LSTM). É o padrão esperado de quem não sabe o tempo de
   D+1 (não há NWP) e bate com o diagnóstico de resíduos da v1 (limite informacional).
5. **A recorrência não agrega** (ablação M17). O desempenho da LSTM vem das features e da
   arquitetura de dois ramos. Na permutação, a LSTM depende sobretudo do ramo estático (calendário
   +100 kW, posição solar +92 kW) e o XGBoost dos lags da irradiância horizontal (+244 kW).
6. **O teste cobre só a estação seca.** O RMSE por dobra de CV vai de ~300 kW (out–dez/2025) a
   475–553 kW (dez/2025–mai/2026, período chuvoso). O RMSE de teste provavelmente subestima o erro
   anual.
7. **As redes mal treinam antes da parada antecipada:** em 9 das 10 sementes, a LSTM parou entre as
   épocas 16 e 19 com paciência 15 e `restore_best_weights=True`, ou seja, os pesos usados são os
   das épocas 1 a 4. O critério de parada é o bloco de validação do Keras (últimas 730 linhas do
   treino efetivo, 31/05 a 31/07/2026). Isso ajuda a explicar a variância entre sementes e o viés
   da LSTM

---

## 2. Prioridade 1: redação do TG (caminho crítico)

Base: PDF de 19/10/2026 (`Desenvolvimento_de_um_Modelo_de_Previsao_de_Gerac__1_ 3.pdf`, 73 f.).

**Estado do texto:**
- Cap. 1 (Introdução) e Cap. 2 (Referencial teórico) estão escritos.
- O Cap. 3 aparece vazio, com o mesmo título do Cap. 4 ("Materiais e Métodos").
- O Cap. 4 (Materiais e Métodos) está quase completo.
- Não existem os capítulos de Resultados e de Conclusão.
- Estão vazios: §1.2 Objetivo, §1.3 Estrutura do texto e §2.1.5 Abordagem utilizada.
- Resumo, Abstract, Dedicatória e Agradecimentos ainda têm o texto do modelo.

**Problema central:** o Cap. 4 descreve um pipeline anterior, e não o que gerou os resultados da
v2. Ele usa 6 meses de dados, um modelo por horizonte, 20 features e um céu claro POA calibrado.
Como os resultados vêm do código v2, é o texto que precisa ser alinhado ao código, e não o
contrário. Abaixo, cada divergência e a correção (referências Mn em `registro_mudancas_v2.md`).

### 2.1 Cap. 4 (Materiais e Métodos): divergências com o pipeline v2

| § do texto | O texto diz | O pipeline v2 faz | Ação |
|---|---|---|---|
| 4.1.1 | Dados de 06/12/2025 a 06/06/2026 (182 dias, 262.940 registros); "não cobre um ciclo anual" | Dados de 02/05/2025 a 10/09/2026, ~16 meses e mais de um ciclo anual | Atualizar o período e as contagens; refazer a Tab. 4.2 e as Figs. 4.1 a 4.3 com o período todo; trocar a limitação: o ciclo anual está coberto, mas o **teste** é só da estação seca |
| 4.1.4 | 4.383 h → 2.207 h diurnas; interpolação de até 2 h; ffill do vento; piso de 6/12 leituras para as séries de 5 min | 5.725 linhas finais (treino 5.234, efetivo 4.862; teste 491); `preprocessamento.py` usa `LIMITE_INTERPOLACAO_HORAS = 1` | Conferir cada número e regra contra `preprocessamento.py`; descrever a semântica de `limit=1` (M6) |
| 4.1.4 | Não há flags de qualidade | `alvo_imputado` (2 linhas) e `evento_operacional` (377 linhas, critério de PR); cenários A e B | Incluir M6 e M7: critério, contagens e uso só para filtragem, nunca como feature |
| 4.1.3 / Tab. 4.3 | Posição solar por registro; zênite e azimute em graus; `irradiacao_refletica` | Ponto médio h:30, fuso local (M2); azimute em sen/cos (M3); zênite removido (M4); recorte por elevação em h:30 > 0 (M5) | Reescrever; corrigir o nome para `irradiacao_refletida` |
| 4.1.5 "Esquema direto" | Um modelo f_h por horizonte; lags relativos à emissão (t, t−24 h); horizontes maiores recuam os lags | **Um único modelo** para todas as horas de D+1; lags "mesma hora, d dias antes" (P(D,h), P(D−1,h)…); emissão depois de ~18h de D | Reescrever a subseção e a Eq. 4.4; tirar o parágrafo "treinar \|H\| modelos" |
| 4.1.5 FACP | A FACP indica lags de até 2 a 3 dias | Lags de 1, 2, 3 e **7** dias (decisão D1 da v1, mantida) | Justificar o lag de 7 dias (ex.: candidato semanal, mantido da v1) ou dizer que foi um conjunto candidato |
| 4.1.5 cíclicas | Direção do vento em sen/cos; zênite mantido | Direção do vento não é usada (~57% faltante); zênite removido | Corrigir |
| 4.1.6 / Tab. 4.5 | 20 features; das irradiâncias, só Gt; Tmod incluída; vw só em t | **39 features:** 8 variáveis (P_ac, T_amb, UR, vento, Gt, Gh, Gr, T_inv) × lags de 1, 2, 3 e 7 d = 32, mais 7 determinísticas; **sem Tmod** (r = 0,92 com Gt); sem médias móveis (M1) | Refazer a Tab. 4.5 a partir de `dados/dicionario_features_v2.csv`; incluir as verificações de posto, VIF, lags e causalidade |
| 4.2 | "Cerca de seis meses de operação"; o XGBoost fornece importâncias | ~15 meses; a v2 usa importância por permutação, igual para todos os modelos | Atualizar |
| 4.2.2 SVR | O tipo de kernel é ajustado | RBF fixo; alvo padronizado; C ∈ [0,1; 1e3], ε ∈ [1e-3; 1], γ ∈ [1e-4; 1], em log (M12) | Corrigir |
| 4.2.3 MLP | Ajusta função de ativação, penalização dos pesos e nº de épocas | Ajusta camadas (1 a 3), unidades {16, 32, 64, 128}, dropout (0 a 0,5), taxa (1e-4 a 1e-2), batch {32, 64, 128}; até 150 épocas, paciência 15 | Corrigir |
| 4.2.4 LSTM | Ajusta o comprimento da janela e o dropout recorrente; saída nativa de dimensão \|H\| | Sequência fixa de 4 passos (lags de 7, 3, 2 e 1 d) × 8 variáveis + ramo estático com as 7 determinísticas (M14); saída escalar; ajusta camadas (1 a 2), unidades LSTM e densas {16, 32, 64}, dropout, taxa e batch | Reescrever; tirar o parágrafo da saída \|H\| |
| 4.2 (falta) | — | Optuna/TPE, 50 trials por família, semente 42, `TimeSeriesSplit` com 5 dobras, máscaras de exclusão (M9); retreino com 10 sementes e determinismo (M13) | Criar uma subseção "Ajuste de hiperparâmetros e treinamento" |
| 4.3.2 céu claro | GHI de céu claro **no plano dos módulos**, com fator calibrado no envelope dos dias limpos | Ineichen em **GHI**, P_cs = 2.500 · GHI_cs / 1.000, sem calibração, no ponto médio e no fuso local (M8) | Descrever o que foi feito; citar a limitação para usina com rastreador (k_d até 1,28). A versão do texto pode virar trabalho futuro |
| 4.3.2 persistência | Remete à Eq. 2.10, na forma intradiária (Δt) | Day-ahead: P̂(D+1,h) = P_cs(D+1,h) · (1/3) Σ_{d=1..3} P(D+1−d,h) / P_cs(D+1−d,h), caindo para a ingênua onde kt não é definido, com clip em [0; 2.500] kW | Escrever a forma day-ahead; incluir também a persistência ingênua (P̂(D+1,h) = P(D,h)) |
| 4.3.1 | Métricas: RMSE, MAE e skill | Também MBE, nRMSE (2.500 kWp) e R²; skill vs. inteligente (principal) e vs. ingênua | Acrescentar |
| 4.3 (falta) | — | Diebold-Mariano com correção HLN sobre perdas diárias; erro por hora e por classe de céu (tercis de k_d); importância por permutação; ablação da LSTM (M16, M17) | Criar uma subseção "Análises complementares e significância" |
| 4.3.3 | Walk-forward genérico | `TimeSeriesSplit` (5 dobras) só para o ajuste; hold-out final: treino até 31/07/2026 e teste de 01/08 a 10/09/2026 (41 dias, 491 h); cenários A e B | Detalhar as datas e as contagens |

As Figs. 4.1 a 4.4 e a Tab. 4.2 foram calculadas no período antigo. Um script novo (ex.:
`código/estatisticas_texto_v2.py`, com saída em `resultados_v2/figuras/` e nomes novos) pode
regerá-las no período completo. Isso pode mudar números citados no texto, como r = −0,95 entre
T_amb e UR, r = 0,93 entre Tmod e Gt e a FACP de 0,43, 0,20 e 0,14.

### 2.2 Estrutura e consistência geral
- **Numeração dos capítulos:** tirar o Cap. 3 vazio. Fica 3 = Materiais e Métodos,
  4 = Resultados e Discussão, 5 = Conclusão. As remissões "Capítulo 4" do Cap. 2 mudam junto.
- **38 referências "??" no PDF:** chaves de citação e rótulos ausentes ou compilação sem
  BibTeX. Exemplos: §2.2.1 inteira, §4.2 e "Subseção ??" em 4.1.3, 4.2.2 e 4.2.4. Também falta o
  ano em "Taieb et al. ()". Rodar `pdflatex → bibtex → pdflatex ×2` e conferir as chaves no
  `.bib`.
- **Cap. 2:** a Eq. 2.10 fica como definição geral, mas o Cap. 3 precisa da forma day-ahead. Faltam
  no referencial:
  - Diebold-Mariano (Diebold e Mariano, 1995; Harvey, Leybourne e Newbold, 1997);
  - importância por permutação (Breiman, 2001);
  - otimização bayesiana/TPE (Bergstra et al., 2011; Akiba et al., 2019, para o Optuna).
- **§2.1.5 Abordagem utilizada:** previsão direta de potência (sem a etapa irradiância → potência),
  day-ahead horário, endógena (sem NWP), um único modelo para as horas de D+1.
- **Revisão:** separador decimal misto ("28.1 %", "0.006 %", "6553.5" ao lado de "0,78"; uniformizar
  com vírgula pelo `siunitx`); "Alén" → "Além"; "dit" → "dito"; conferir o alinhamento da Tab. 4.2
  no PDF (a extração mostrou colunas deslocadas).
- **Expectativas que os resultados confirmam:** a §4.2.4 antecipa que a LSTM perderia para as
  árvores com poucos dados. Na v2 ela empata com o XGBoost, e a ablação mostra que a recorrência não
  agrega. A §4.3.2 diz que a complexidade só se justifica se superar a regressão linear: só XGBoost e
  LSTM a superam com significância. Retomar as duas na discussão.

### 2.3 Novo Cap. 4: Resultados e Discussão
Ordem sugerida, com a fonte de cada item em `código/resultados_v2/`:
1. Métricas nos cenários A e B: `tabelas/metricas_v2.csv` + `figuras/rmse_barras_v2.png`.
2. Significância: `tabelas/diebold_mariano_v2.csv`.
3. Série prevista × observada e dispersão: `figuras/previsto_vs_observado_v2.png`,
   `figuras/dispersao_v2.png`.
4. Erro por hora (`figuras/erro_por_hora_v2.png`) e por classe de céu
   (`tabelas/erro_por_classe_ceu_v2.csv`): leitura 4 da seção 1.
5. Importância por permutação por grupo: `tabelas/importancia_permutacao_v2.csv`.
6. Ablação da LSTM (M17): `modelos/lstm_ablacao_resumo_v2.json`.
7. Estabilidade entre sementes e v1 × v2 (`tabelas/comparacao_v1_v2.csv`). Só entra se o texto
   mencionar a v1; senão, basta a estabilidade entre sementes.
8. RMSE por dobra de CV (`tabelas/cv_por_dobra_v2.csv`), marcado como otimista: sazonalidade
   (leitura 6).
9. Discussão:
   - limite informacional sem NWP e regressão à média;
   - o ganho vem da arquitetura de dois ramos, não da recorrência;
   - os ótimos de SVR e XGBoost ficaram no lado simples do espaço, coerente com a regressão linear
     competitiva;
   - recomendação prática: o XGBoost empata com a LSTM, mas é mais estável, menos enviesado e mais
     barato (0,2 min de retreino contra 2,2 min).

### 2.4 Novo Cap. 5: Conclusão, limitações e trabalhos futuros
- **Limitações a declarar:**
  - 41 dias de teste, só na estação seca;
  - uma única usina;
  - sem NWP;
  - avisos de borda mantidos;
  - lags vindos de horas de evento não tratados (694 linhas);
  - clip em 2.500 kW;
  - céu claro e classes de céu baseados em GHI numa usina com rastreadores (k_d chega a 1,28).
- Trabalhos futuros: ver a seção 5.

### 2.5 Figuras e tabelas que faltam
**Feito em 19/10/2026:** o script `código/figuras_tg_v2.py` lê só os CSVs e JSON já salvos (sem
retreino), não sobrescreve nada e gerou em `resultados_v2/figuras/`:
- `importancia_grupos_v2.png`: importância por grupo, um painel por modelo;
- `erro_por_classe_ceu_v2.png`: RMSE e MBE por classe de céu;
- `diebold_mariano_v2.png`: matriz de p-valores do DM, cenários A e B;
- `cv_por_dobra_v2.png`: RMSE por dobra de CV;
- `ablacao_lstm_v2.png`: ablação da LSTM, cenários A e B.

Com elas, são 9 figuras finais. Para regerar alguma, apague a figura antiga antes de rodar.

Pendente: gerar gerar as tabelas LaTeX (métricas, DM, espaços de busca, features) direto dos CSVs e
JSON, para não transcrever números à mão.

### 2.6 Pré-textuais e acabamento
- Objetivo geral e objetivos específicos (§1.2).
- Estrutura do texto (§1.3), depois de fechados os Caps. 3 a 5.
- Resumo e abstract (com os números finais).
- Dedicatória e agradecimentos.
- Nome do coordenador, CDU e data da folha de aprovação.
- Nota de rodapé da §4.1.1: definir o que pode ser dito sobre a localização (confidencialidade).

---

## 3. Prioridade 2: experimentos complementares (opcionais, se houver prazo)

Regras: definir cada experimento **antes** de ver o resultado (registro prévio), não sobrescrever
nenhuma saída da v2 e registrar tudo em `registro_mudancas_v2.md` (ex.: como "Fase 6").
Em ordem de custo-benefício:

1. **Teste prospectivo com dados novos (maior valor).** O bruto vai até 10/09/2026; hoje já
   existem cerca de 5 a 6 semanas a mais. Avaliar os modelos v2 **congelados** (sem reajuste nem
   retreino) de 11/09/2026 em diante:
   - é a única avaliação realmente fora da amostra, já que o teste de ago–set foi visto na v1 e
     na Fase 0 (ex.: o evento de 07/09/2026);
   - aproxima o número de dias de 80 e aumenta o poder do DM, talvez o bastante para separar
     LSTM e XGBoost;
   - exige: extrair o bruto novo, gerar as features com a lógica de `features_v2.py` num arquivo
     novo (não sobrescrever `bet01_features_dayahead_v2.csv`), parametrizar o período de teste da
     comparação e conferir as flags no período novo;
   - limitação: set–out ainda é estação seca, então isso não resolve a representatividade sazonal.
2. **Combinação LSTM + XGBoost** (média das previsões médias das sementes), com custo zero de
   treino porque usa as previsões salvas:
   - no teste atual, reportar só como resultado exploratório;
   - registrar antes e avaliar de fato no teste prospectivo (item 1);
   - calcular também a correlação entre os resíduos dos modelos v2 (na v1 foi de 0,93 a 0,99). O
     MLP é o melhor nos dias nublados (333 kW, contra 352 a 373 kW) e o pior nos claros (337 kW,
     contra 271 a 286 kW), o perfil oposto ao de LSTM e XGBoost.
3. **Curvas de aprendizado da LSTM e do MLP** (diagnóstico, sem mudar os modelos):
   - retreinar 1 ou 2 sementes registrando o `val_loss` por época, com saída em
     `resultados_v2/diagnostico/`;
   - o objetivo é explicar o melhor ponto nas épocas 1 a 4 (leitura 7) e ver se o viés de +61 kW
     da LSTM vem do bloco de validação (jun–jul);
   - custa cerca de 15 s por semente.
4. **Avaliação sazonal em origem móvel:**
   - com os hiperparâmetros congelados, retreinar mês a mês em janela expansiva e prever o mês
     seguinte (nov/2025 a jul/2026), para ter o desempenho no período chuvoso;
   - fica levemente otimista, porque os hiperparâmetros vieram dessas dobras;
   - para limitar o custo, rodar só XGBoost, regressão linear e as persistências.

---

## 4. Trabalhos futuros (para o Cap. 5; fora do escopo do TG)

- **Previsão meteorológica numérica (NWP) como entrada.** É a principal alavanca: o erro
  remanescente é falta de informação sobre o tempo de D+1, não falta de modelo. Usar previsões
  arquivadas (sem vazamento), por exemplo de serviços com histórico de rodadas, como Open-Meteo e
  ECMWF open data (verificar a disponibilidade para o período).
- **Alvo normalizado pelo céu claro** (prever o índice kt) e **céu claro próprio para rastreador**
  (pvlib com seguidor de um eixo + transposição para o plano dos módulos). O envelope em GHI não
  representa a usina (k_d de até 1,28).
- **Previsão probabilística:** quantis (XGBoost com perda quantílica) ou intervalos conformais,
  úteis para operação e mercado.
- **Eventos operacionais:** detecção em tempo real e tratamento dos lags contaminados.
- **Atualização intradiária (nowcasting)** para aproveitar a persistência do regime dentro do dia
  (autocorrelação horária dos resíduos).
- **Mais usinas:** generalização e transferência entre plantas.

---

## 5. Manutenção do repositório

- O PR #1 foi mesclado e a branch remota `tg-v2` apagada, mas a cópia local continua em `tg-v2`,
  com a `main` local desatualizada: `git switch main`, `git pull` e `git branch -d tg-v2`.
- `.gitignore`: acrescentar `*.keras`, `*.db`, `*.parquet` e `código/resultados_v2/rapido/`. Hoje
  modelos, banco do Optuna, sandbox e o parquet da raiz (38 MB) aparecem como não rastreados e
  podem ser commitados por engano.
- Decidir se os JSON de resumo (`modelos/*_resumo_v2.json`, pequenos) e os logs de console entram
  no git. Os logs (`comparacao_saida_console_v2.txt`, `diagnostico/fase0_saida_console_v2.txt`)
  estão em UTF-16 (saída do PowerShell 5.1); converter para UTF-8 antes de versionar.
- **Fonte do texto do TG:** o LaTeX em `rascunhos/tg_v2/` é de jun/2026 e está bem atrás do PDF de
  19/10/2026, que não tem fonte no projeto. Além disso, `rascunhos/` está no `.gitignore`, então
  nenhuma versão do texto tem controle de versão. Manter a fonte atual num lugar com histórico
  (Overleaf, repositório privado ou nuvem); não usar o repositório atual, que é público.

## Notas técnicas para retomar

- Ambiente: `D:\tg\dataset\formando_dataset\.venv` (Python 3.12.10). Rodar com
  `$env:PYTHONIOENCODING="utf-8"` (o console quebra os acentos) e `python -B` (os `.pyc` de
  `código/__pycache__` estão versionados e não devem ser regravados).
- Testes com `--rapido` gravam só no sandbox `código/resultados_v2/rapido/`. O modo completo é
  rodado localmente pelo autor.
- A v1 é intocável: só ler/importar `código/*.py`, `código/modelos/*.py`, `preprocessamento.py`
  e os artefatos `*_bet01.*`. Arquivos novos levam o sufixo `_v2`.
- `direcao_vento` tem ~57% de faltantes e não é usada: citar como limitação de dado.
- Capacidade instalada: 2.500 kWp (`dados.py:CAPACIDADE_KWP`), usada no nRMSE e no clip das
  previsões.

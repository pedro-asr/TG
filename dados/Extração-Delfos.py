import pandas as pd
pd.set_option('display.max_columns', None)
import requests as req
import json
import os
from pathlib import Path

from sqlalchemy import create_engine,inspect
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime,timedelta
import yaml
#-----------------------------------------------#
# Configuração avançada de log
def setup_logging():
    """
    Configura o sistema de logging com arquivos mensais e logs rotativos
    """
    # Criar diretório de logs se não existir
    log_dir = Path("Extracao_delfo/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Nome do arquivo baseado no mês atual
    current_month = datetime.now().strftime("%Y_%m")
    log_filename = log_dir / f"delfos_extraction_{current_month}.log"
    
    # Configurar formatação mais detalhada
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configurar handler para arquivo com rotação (máx 10MB, 5 backups)
    file_handler = RotatingFileHandler(
        log_filename, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Configurar handler para console com formatação mais limpa
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Configurar logger principal
    logger = logging.getLogger('delfos_extractor')
    logger.setLevel(logging.DEBUG)
    
    # Remover handlers existentes para evitar duplicação
    logger.handlers.clear()
    
    # Adicionar handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log inicial do sistema
    logger.info("="*80)
    logger.info("SISTEMA DE EXTRAÇÃO DELFOS INICIADO")
    logger.info(f"Arquivo de log: {log_filename}")
    logger.info(f"Timestamp de início: {datetime.now().isoformat()}")
    logger.info("="*80)
    
    return logger

# Inicializar logger
logger = setup_logging()

#-----------------------------------------------#
def load_yaml_file(config_path = 'config.yaml'):
    """Carrega arquivo de configuração YAML com logs detalhados"""
    logger.info(f"Iniciando carregamento do arquivo de configuração: {config_path}")
    
    try:
        if not os.path.exists(config_path):
            logger.error(f"Arquivo de configuração não encontrado no caminho: {config_path}")
            raise FileNotFoundError(f"Arquivo {config_path} não encontrado")
            
        with open(config_path,'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            
        logger.info(f"Arquivo de configuração carregado com sucesso")
        logger.debug(f"Chaves principais encontradas: {list(config.keys()) if config else 'Nenhuma'}")
        
        return config

    except FileNotFoundError as e:
        logger.error(f'Arquivo de configuração não encontrado: {e}')
        raise
    except yaml.YAMLError as e:
        logger.error(f'Erro no processamento do arquivo YAML: {e}')
        logger.debug(f'Detalhes do erro YAML: {str(e)}')
        raise
    except Exception as e:
        logger.error(f'Erro inesperado ao carregar configuração: {e}')
        raise

#-----------------------------------------------#

def get_last_date(db_config, table_name='dados_gerais'):
    """Obtém a última data registrada na tabela com logs detalhados"""
    logger.info(f"Verificando última data registrada na tabela '{table_name}'")
    
    # Criar string de conexão
    connection_string = (
        f"postgresql://{db_config['user']}:{db_config['password']}@"
        f"{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
    )
    logger.debug(f"String de conexão criada para: {db_config['host']}:{db_config['port']}/{db_config['dbname']}")
    
    try:
        engine = create_engine(connection_string)
        logger.debug("Engine de conexão com banco criada")
        
        # Verificar se a tabela existe
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        logger.debug(f"Tabelas encontradas no banco: {existing_tables}")
        
        if table_name not in existing_tables:
            logger.warning(f"Tabela '{table_name}' não encontrada no banco de dados")
            # Retornar data mais recente (30 dias atrás) para evitar muito volume de dados
            default_date = '2025-02-01 00:00:00'
            logger.info(f"Retornando data padrão (30 dias atrás): {default_date}")
            return default_date
        
        # Obter última data
        query = f'SELECT MAX(tempo) as last_date FROM {table_name}'
        logger.debug(f"Executando query: {query}")
        
        last_date = pd.read_sql(query, engine)
        
        if pd.isnull(last_date['last_date'][0]):
            logger.warning(f"Nenhuma data encontrada na tabela '{table_name}'")
            logger.info("Retornando None para indicar tabela vazia")
            return None
            
        last_time = last_date['last_date'][0]
        logger.info(f"Última data encontrada: {last_time}")
        logger.debug(f"Tipo do dado retornado: {type(last_time)}")
        
        return last_time
        
    except Exception as e:
        logger.error(f'Erro ao obter última data da tabela {table_name}: {e}')
        logger.debug(f'Detalhes da conexão: host={db_config["host"]}, port={db_config["port"]}, db={db_config["dbname"]}')
        raise

def extract_data_batch(headers, device_ids, variable_ids, start_time, end_time):
    """Extrai um lote específico de dados da API"""
    print(f"device_ids: {device_ids}")
    print(f"variable_ids: {variable_ids}")
    r_config = {
        "device_ids": device_ids,
        "variable_ids": variable_ids,
        "start_time": start_time,
        "end_time": end_time,
        "aggregate": "1 min"
    }
    
    url = 'https://api.delfos.im/solar-data/data-studio/device-types/2/timeseries'
    
    logger.debug(f"Requisição do lote: {start_time} até {end_time}")
    start_request_time = datetime.now()
    
    r = req.post(url, headers=headers, data=json.dumps(r_config), timeout=120)
    
    request_duration = (datetime.now() - start_request_time).total_seconds()
    logger.debug(f'Lote completado em {request_duration:.2f} segundos - Status: {r.status_code}')
    
    if r.status_code != 200:
        error_msg = r.text[:200] if r.text else "Sem detalhes do erro"
        logger.error(f'Erro no lote ({start_time} - {end_time}): Status {r.status_code} - {error_msg}')
        raise req.exceptions.RequestException(f"Status code: {r.status_code}, Erro: {error_msg}")
    
    response = r.json()
    
    # Verificar estrutura da resposta
    if 'sample_time' not in response or 'variables' not in response:
        raise KeyError("Estrutura de resposta inválida")
    
    # Transformar em DataFrame
    df = pd.DataFrame({'tempo': response['sample_time']})
    for var in response['variables']:
        device_name = var['device']['name']
        df[device_name] = var['values']
    
    return df

def extract_data(headers, device_ids, variable_ids, start_time=None, end_time=None):
    """Extrai dados da API Delfos com logs detalhados e processamento em lotes"""
    logger.info("="*50)
    logger.info("INICIANDO EXTRAÇÃO DE DADOS DA API DELFOS")
    
    # Configurar tempos padrão
    if end_time is None:
        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.debug(f"End time não especificado, usando atual: {end_time}")
    if start_time is None:
        start_time = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d %H:%M:%S')
        logger.debug(f"Start time não especificado, usando 5 dias atrás: {start_time}")
    
    logger.info(f"Período de extração: {start_time} até {end_time}")
    logger.info(f"Número de dispositivos: {len(device_ids)}")
    logger.info(f"Device IDs: {device_ids}")
    logger.info(f"Variable IDs: {variable_ids}")
    
    # Calcular período e verificar se precisa dividir em lotes
    start_dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    end_dt = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
    total_days = (end_dt - start_dt).days
    
    logger.info(f"Período total: {total_days} dias")
    
    # Definir tamanho do lote (máximo 7 dias para evitar erro "Too much data")
    batch_days = 7
    url = 'https://api.delfos.im/solar-data/data-studio/device-types/2/timeseries'
    logger.info(f"Endpoint da API: {url}")
    
    if total_days <= batch_days:
        logger.info("Período pequeno, fazendo extração em lote único")
        try:
            df = extract_data_batch(headers, device_ids, variable_ids, start_time, end_time)
            logger.info(f'DataFrame criado com shape: {df.shape}')
            
            # Verificar dados nulos
            null_counts = df.isnull().sum()
            if null_counts.sum() > 0:
                logger.warning("Valores nulos encontrados nos dados:")
                for col, count in null_counts.items():
                    if count > 0:
                        logger.warning(f"  {col}: {count} valores nulos")
            
            logger.info('EXTRAÇÃO DE DADOS CONCLUÍDA COM SUCESSO')
            logger.info("="*50)
            return df
            
        except Exception as e:
            logger.error(f'Erro na extração em lote único: {e}')
            raise
    
    else:
        logger.info(f"Período longo ({total_days} dias), dividindo em lotes de {batch_days} dias")
        
        # Dividir em lotes
        batch_list = []
        current_start = start_dt
        batch_count = 0
        successful_batches = 0
        failed_batches = 0
        
        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=batch_days), end_dt)
            batch_count += 1
            
            batch_start_str = current_start.strftime('%Y-%m-%d %H:%M:%S')
            batch_end_str = current_end.strftime('%Y-%m-%d %H:%M:%S')
            
            logger.info(f"Processando lote {batch_count}: {batch_start_str} até {batch_end_str}")
            
            try:
                batch_df = extract_data_batch(headers, device_ids, variable_ids, batch_start_str, batch_end_str)
                
                if not batch_df.empty:
                    batch_list.append(batch_df)
                    successful_batches += 1
                    logger.info(f"Lote {batch_count} concluído: {len(batch_df)} registros")
                else:
                    logger.warning(f"Lote {batch_count} retornou vazio")
                
            except Exception as e:
                failed_batches += 1
                logger.error(f"❌ Erro no lote {batch_count}: {e}")
                
                # Se muitos lotes falharem consecutivamente, parar
                if failed_batches > 3 and successful_batches == 0:
                    logger.error("Muitos lotes consecutivos falharam, interrompendo extração")
                    raise
            
            current_start = current_end
        
        # Consolidar lotes
        if batch_list:
            logger.info(f"Consolidando {len(batch_list)} lotes bem-sucedidos...")
            df_final = pd.concat(batch_list, ignore_index=True)
            
            # Remover duplicatas baseadas na coluna 'tempo'
            original_rows = len(df_final)
            df_final = df_final.drop_duplicates(subset=['tempo'], keep='first')
            final_rows = len(df_final)
            
            if original_rows != final_rows:
                logger.info(f"Removidas {original_rows - final_rows} linhas duplicadas")
            
            # Ordenar por tempo
            df_final = df_final.sort_values('tempo').reset_index(drop=True)
            
            logger.info(f"RESUMO DA EXTRAÇÃO EM LOTES:")
            logger.info(f"  • Total de lotes processados: {batch_count}")
            logger.info(f"  • Lotes bem-sucedidos: {successful_batches}")
            logger.info(f"  • Lotes com falha: {failed_batches}")
            logger.info(f"  • Taxa de sucesso: {(successful_batches/batch_count*100):.1f}%")
            logger.info(f"  • DataFrame final: {df_final.shape}")
            
            # Verificar dados nulos
            null_counts = df_final.isnull().sum()
            if null_counts.sum() > 0:
                logger.warning("Valores nulos encontrados nos dados consolidados:")
                for col, count in null_counts.items():
                    if count > 0:
                        logger.warning(f"  {col}: {count} valores nulos")
            
            logger.info('EXTRAÇÃO DE DADOS EM LOTES CONCLUÍDA COM SUCESSO')
            logger.info("="*50)
            return df_final
        
        else:
            logger.error("❌ Nenhum lote foi processado com sucesso")
            raise Exception("Falha em todos os lotes de extração")

def standart_columns(df, usina):
    """Padroniza nomes das colunas por usina com logs detalhados"""
    logger.info(f"Padronizando colunas para usina: {usina}")
    logger.debug(f"Colunas antes da padronização: {list(df.columns)}")
    
    original_columns = df.columns.tolist()
    
    if usina == 'aqz01':
        column_mapping = {
            'Aquiraz - UFV 01 - INV01': 'pot_inversor_01', 
            'Aquiraz - UFV 01 - INV02': 'pot_inversor_02',
            'Aquiraz - UFV 01 - INV03': 'pot_inversor_03', 
            'Aquiraz - UFV 02 - INV01': 'pot_inversor_04',
            'Aquiraz - UFV 02 - INV02': 'pot_inversor_05', 
            'Aquiraz - UFV 02 - INV03': 'pot_inversor_06',
            'Aquiraz - UFV 03 - INV01': 'pot_inversor_07', 
            'Aquiraz - UFV 03 - INV02': 'pot_inversor_08'
        }
        df = df.rename(columns=column_mapping)
        
    elif usina == 'utxt01':
        column_mapping = {
            'Unitêxtil - UFV 09 - INV01':'pot_inversor_01','Unitêxtil - UFV 09 - INV02':'pot_inversor_02',
            'Unitêxtil - UFV 09 - INV03':'pot_inversor_03','Unitêxtil - UFV 10 - INV01':'pot_inversor_04',
            'Unitêxtil - UFV 10 - INV02':'pot_inversor_05','Unitêxtil - UFV 10 - INV03':'pot_inversor_06',
            'Unitêxtil - UFV 11 - INV01':'pot_inversor_07','Unitêxtil - UFV 11 - INV02':'pot_inversor_08',
            'Unitêxtil - UFV 11 - INV03':'pot_inversor_09','Unitêxtil - UFV 12 - INV01':'pot_inversor_10',
            'Unitêxtil - UFV 12 - INV02':'pot_inversor_11','Unitêxtil - UFV 12 - INV03':'pot_inversor_12',
            'Unitêxtil - UFV 13 - INV01':'pot_inversor_13','Unitêxtil - UFV 13 - INV02':'pot_inversor_14',
            'Unitêxtil - UFV 13 - INV03':'pot_inversor_15','Unitêxtil - UFV 14 - INV01':'pot_inversor_16',
            'Unitêxtil - UFV 14 - INV02':'pot_inversor_17'
        }
        df = df.rename(columns=column_mapping)
    
    elif usina == 'msp02':
        column_mapping = {
            'Massapê II - INV 02': 'pot_inversor_02', 'Massapê II - INV 01': 'pot_inversor_01',
            'Massapê II - INV 03': 'pot_inversor_03', 'Massapê II - INV 04': 'pot_inversor_04',
            'Massapê II - INV 05': 'pot_inversor_05', 'Massapê II - INV 06': 'pot_inversor_06',
            'Massapê II - INV 07': 'pot_inversor_07', 'Massapê II - INV 08': 'pot_inversor_08',
            'Massapê II - INV 09': 'pot_inversor_09', 'Massapê II - INV 10': 'pot_inversor_10',
            'Massapê II - INV 11': 'pot_inversor_11', 'Massapê II - INV 12': 'pot_inversor_12',
            'Massapê II - INV 13': 'pot_inversor_13'
        }
        df = df.rename(columns=column_mapping)
    
    elif usina == 'bbr01':
        column_mapping = {
            'Beberibe - INV01': 'pot_inversor_01', 'Beberibe - INV02': 'pot_inversor_02',
            'Beberibe - INV03': 'pot_inversor_03', 'Beberibe - INV04': 'pot_inversor_04',
            'Beberibe - INV05': 'pot_inversor_05', 'Beberibe - INV06': 'pot_inversor_06',
            'Beberibe - INV07': 'pot_inversor_07', 'Beberibe - INV08': 'pot_inversor_08',
            'Beberibe - INV09': 'pot_inversor_09', 'Beberibe - INV10': 'pot_inversor_10'
        }
        df = df.rename(columns=column_mapping)
    else:
        logger.warning(f"Usina '{usina}' não reconhecida, mantendo colunas originais")
        return df
    
    # Log das mudanças realizadas
    renamed_columns = [col for col in original_columns if col in column_mapping]
    logger.info(f"Colunas renomeadas: {len(renamed_columns)}")
    for old_col in renamed_columns:
        logger.debug(f"  '{old_col}' -> '{column_mapping[old_col]}'")
    
    logger.debug(f"Colunas após padronização: {list(df.columns)}")
    return df

#-----------------------------------------------#
def load_and_prepare_data(db_config: dict, query: str) -> pd.DataFrame:
    """Carrega dados do banco com logs de monitoramento"""
    logger.info("Carregando dados do banco de dados...")
    logger.debug(f"Query a ser executada: {query}")
    
    connection_string = (
        f"postgresql://{db_config['user']}:{db_config['password']}@"
        f"{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
    )
    logger.debug(f"Conectando em: {db_config['host']}:{db_config['port']}/{db_config['dbname']}")

    try:
        engine = create_engine(connection_string)
        
        df_list = []
        chunk_count = 0
        total_rows = 0
        
        logger.info("Iniciando leitura em chunks de 10.000 registros...")
        for chunk in pd.read_sql_query(query, engine, chunksize=10000):
            chunk_count += 1
            chunk_rows = len(chunk)
            total_rows += chunk_rows
            df_list.append(chunk)
            logger.debug(f"Chunk {chunk_count}: {chunk_rows} registros carregados")

        final_df = pd.concat(df_list, ignore_index=True)
        logger.info(f"Carregamento concluído: {total_rows} registros em {chunk_count} chunks")
        logger.info(f"DataFrame final shape: {final_df.shape}")
        
        return final_df
        
    except Exception as e:
        logger.error(f"Erro ao carregar dados do banco: {e}")
        logger.debug(f"Detalhes da conexão: {db_config}")
        raise

def upload_data(db_config, df, destino):
    """Faz upload dos dados para o banco com logs detalhados"""
    logger.info("="*50)
    logger.info(f"INICIANDO UPLOAD PARA TABELA: {destino}")
    logger.info(f"Shape do DataFrame: {df.shape}")
    logger.info(f"Colunas: {list(df.columns)}")
    
    try:
        # Estatísticas pré-upload
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        logger.info(f"Uso de memória do DataFrame: {memory_usage:.2f} MB")
        
        # Verificar dados nulos
        null_summary = df.isnull().sum()
        if null_summary.sum() > 0:
            logger.warning("Dados nulos encontrados antes do upload:")
            for col, count in null_summary.items():
                if count > 0:
                    logger.warning(f"  {col}: {count} valores nulos")
        
        logger.info('Estabelecendo conexão com banco de dados...')
        connection_string = (
            f"postgresql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
        )
        logger.debug(f"Conectando em: {db_config['host']}:{db_config['port']}/{db_config['dbname']}")
        
        engine = create_engine(connection_string)
        logger.info('Conexão estabelecida com sucesso')
        
        logger.info(f'Iniciando upload de {len(df)} registros para tabela "{destino}"...')
        start_upload_time = datetime.now()
        
        df.to_sql(destino, con=engine, if_exists='append', index=False)
        
        upload_duration = (datetime.now() - start_upload_time).total_seconds()
        logger.info(f'Upload concluído com sucesso em {upload_duration:.2f} segundos')
        logger.info(f'Taxa de upload: {len(df)/upload_duration:.2f} registros/segundo')
        logger.info("="*50)

    except Exception as e:
        logger.error(f'ERRO DURANTE UPLOAD: {e}')
        logger.error(f'Tabela destino: {destino}')
        logger.error(f'Shape do DataFrame: {df.shape}')
        logger.debug(f'Detalhes da conexão: host={db_config["host"]}, port={db_config["port"]}, db={db_config["dbname"]}')
        logger.debug(f'Tipo do erro: {type(e).__name__}')
        raise

#-----------------------------------------------#
def main():
    """Função principal com logs detalhados de execução"""
    execution_start = datetime.now()
    logger.info("🚀 EXECUÇÃO PRINCIPAL INICIADA")
    
    try:
        # Carregamento arquivo configuração
        config_path = r'C:\Users\usuario\Projeto_1\Extração\config_delfos.yaml'
        logger.info(f"Carregando configurações de: {config_path}")
        
        conf = load_yaml_file(config_path)
        if conf is None:
            logger.error("❌ Falha crítica: arquivo de configuração não pôde ser carregado")
            return
            
        logger.info("✅ Configurações carregadas com sucesso")
        
        # Inicialização de requisição
        api_key = conf['delfos_api']['api_key']
        db_config = conf['db_config']
        logger.info(f"API Key presente: {'Sim' if api_key else 'Não'}")
        logger.info(f"Configuração do banco carregada para host: {db_config.get('host', 'N/A')}")
        
        headers = {
            "API-Key": api_key,
            "Content-Type": "application/json"
        }
        
        usinas_id = {'Aquiraz':'aqz01', 'Unitêxtil':'utxt01', 'Massapê II':'msp02', 'Beberibe':'bbr01'}
        
        logger.info(f"Processando {len(usinas_id)} usinas: {list(usinas_id.keys())}")
        
        successful_extractions = 0
        failed_extractions = 0
        
        for i, (key, usina_code) in enumerate(usinas_id.items(), 1):
            logger.info("="*60)
            logger.info(f"📊 PROCESSANDO USINA {i}/{len(usinas_id)}: {key} ({usina_code})")
            logger.info("="*60)
            
            try:
                usina_start_time = datetime.now()
                
                # Obter os IDs dos inversores
                db_config['dbname'] = 'variaveis_delfos'
                logger.info(f"Conectando ao banco 'variaveis_delfos' para buscar IDs dos inversores")
                
                inv_ids = load_and_prepare_data(db_config,'SELECT * FROM "Inversores_ids"')
                logger.info(f"Total de inversores encontrados no banco: {len(inv_ids)}")
                
                inv_ids['name'] = inv_ids['name'].str.split('-').str[0].str.strip()
                inv_ids = inv_ids[inv_ids['name'] == str(key)]
                device_ids = inv_ids['id'].tolist()
                
                logger.info(f"Inversores filtrados para {key}: {len(device_ids)}")
                if len(device_ids) == 0:
                    logger.warning(f"⚠️  Nenhum inversor encontrado para usina {key}")
                    failed_extractions += 1
                    continue
                
                # Obter dados adicionais da requisição
                db_config['dbname'] = usina_code
                logger.info(f"Mudando para banco da usina: {usina_code}")
                
                start_time = get_last_date(db_config,'dados_gerais')
                logger.info(f"Data de início para extração: {start_time}")
                
                # Extraindo dados
                logger.info("Iniciando extração de dados da API...")
                df_usina = extract_data(headers, device_ids, ['signal-131'], start_time)
                
                if df_usina.empty:
                    logger.warning(f"⚠️  DataFrame vazio retornado para usina {key}")
                    failed_extractions += 1
                    continue
                
                # Padronizando colunas
                logger.info("Padronizando nomes das colunas...")
                df_usina = standart_columns(df_usina, usina_code)
                
                # Fazendo o upload
                logger.info("Iniciando upload dos dados...")
                upload_data(db_config, df_usina, 'dados_gerais')
                
                usina_duration = (datetime.now() - usina_start_time).total_seconds()
                logger.info(f"✅ Usina {key} processada com sucesso em {usina_duration:.2f} segundos")
                successful_extractions += 1
                
            except Exception as e:
                logger.error(f"❌ Erro ao processar usina {key}: {e}")
                logger.debug(f"Detalhes do erro para {key}: {type(e).__name__} - {str(e)}")
                failed_extractions += 1
                continue
        
        # Resumo final
        total_execution_time = (datetime.now() - execution_start).total_seconds()
        logger.info("="*80)
        logger.info("📊 RESUMO DA EXECUÇÃO")
        logger.info(f"Tempo total de execução: {total_execution_time:.2f} segundos")
        logger.info(f"Usinas processadas com sucesso: {successful_extractions}")
        logger.info(f"Usinas com falha: {failed_extractions}")
        logger.info(f"Taxa de sucesso: {(successful_extractions/(successful_extractions+failed_extractions)*100):.1f}%")
        
        if successful_extractions > 0:
            logger.info("🎉 EXECUÇÃO CONCLUÍDA COM SUCESSO")
        else:
            logger.error("💥 EXECUÇÃO FALHOU - Nenhuma usina foi processada com sucesso")
            
        logger.info("="*80)
        
    except Exception as e:
        total_execution_time = (datetime.now() - execution_start).total_seconds()
        logger.error(f"💥 ERRO CRÍTICO NA EXECUÇÃO PRINCIPAL: {e}")
        logger.error(f"Tempo até falha: {total_execution_time:.2f} segundos")
        logger.debug(f"Tipo do erro crítico: {type(e).__name__}")
        raise

if __name__ == "__main__":
    main()
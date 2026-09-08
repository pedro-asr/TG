import pandas as pd
import requests as req
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

from sqlalchemy import create_engine, inspect
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
import yaml

# Configuração do pandas para melhor visualização
pd.set_option('display.max_columns', None)

@dataclass
class ExtractionConfig:
    """Configuração para extração de dados"""
    api_key: str
    db_config: Dict
    usinas_mapping: Dict[str, str]
    variable_ids: List[str] = None
    batch_days: int = 7
    timeout_seconds: int = 120
    max_retries: int = 3

class DelfosExtractor:
    """Classe principal para extração de dados da API Delfos"""
    
    def __init__(self, config_path: str):
        self.logger = self._setup_logging()
        self.config = self._load_configuration(config_path)
        self.headers = {
            "API-Key": self.config.api_key,
            "Content-Type": "application/json"
        }
        self.stats = {
            'successful_extractions': 0,
            'failed_extractions': 0,
            'total_records_processed': 0
        }

    def _setup_logging(self) -> logging.Logger:
        """Configura sistema de logging com arquivos rotativos"""
        log_dir = Path("Extracao_delfo/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        current_month = datetime.now().strftime("%Y_%m")
        log_filename = log_dir / f"delfos_extraction_{current_month}.log"
        
        # Formatter detalhado para arquivo
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Formatter limpo para console
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Handler para arquivo
        file_handler = RotatingFileHandler(
            log_filename, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        
        # Handler para console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Configurar logger
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Log inicial
        logger.info("=" * 80)
        logger.info("SISTEMA DE EXTRAÇÃO DELFOS INICIADO")
        logger.info(f"Arquivo de log: {log_filename}")
        logger.info(f"Timestamp de início: {datetime.now().isoformat()}")
        logger.info("=" * 80)
        
        return logger

    def _load_configuration(self, config_path: str) -> ExtractionConfig:
        """Carrega e valida arquivo de configuração"""
        self.logger.info(f"Carregando configuração: {config_path}")
        
        try:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Arquivo não encontrado: {config_path}")
                
            with open(config_path, 'r', encoding='utf-8') as file:
                raw_config = yaml.safe_load(file)
            
            # Validar estrutura básica
            required_keys = ['delfos_api', 'db_config']
            for key in required_keys:
                if key not in raw_config:
                    raise ValueError(f"Chave obrigatória ausente: {key}")
            
            # Criar configuração estruturada
            config = ExtractionConfig(
                api_key=raw_config['delfos_api']['api_key'],
                db_config=raw_config['db_config'],
                usinas_mapping={
                    'Aquiraz': 'aqz01', 
                    'Unitêxtil': 'utxt01', 
                    'Massapê II': 'msp02', 
                    'Beberibe': 'bbr01'
                },
                variable_ids=['signal-131']
            )
            
            self.logger.info("Configuração carregada e validada com sucesso")
            return config
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar configuração: {e}")
            raise

    @contextmanager
    def _database_connection(self, db_name: str):
        """Context manager para conexões de banco de dados"""
        db_config = self.config.db_config.copy()
        db_config['dbname'] = db_name
        
        connection_string = (
            f"postgresql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
        )
        
        engine = None
        try:
            engine = create_engine(connection_string)
            self.logger.debug(f"Conexão estabelecida: {db_name}")
            yield engine
        except Exception as e:
            self.logger.error(f"Erro na conexão com {db_name}: {e}")
            raise
        finally:
            if engine:
                engine.dispose()

    def _get_last_recorded_date(self, db_name: str, table_name: str = 'dados_gerais') -> Optional[str]:
        """Obtém a última data registrada na tabela"""
        self.logger.info(f"Verificando última data em {db_name}.{table_name}")
        
        with self._database_connection(db_name) as engine:
            # Verificar se tabela existe
            inspector = inspect(engine)
            existing_tables = inspector.get_table_names()
            
            if table_name not in existing_tables:
                self.logger.warning(f"Tabela {table_name} não encontrada")
                default_date = '2025-02-01 00:00:00'
                self.logger.info(f"Usando data padrão: {default_date}")
                return default_date
            
            # Obter última data
            query = f'SELECT MAX(tempo) as last_date FROM {table_name}'
            result = pd.read_sql(query, engine)
            
            if pd.isnull(result['last_date'][0]):
                self.logger.warning("Nenhuma data encontrada na tabela")
                return None
                
            last_time = result['last_date'][0]
            self.logger.info(f"Última data encontrada: {last_time}")
            return last_time

    def _get_device_ids(self, usina_name: str) -> List[int]:
        """Obtém IDs dos dispositivos para uma usina específica"""
        self.logger.info(f"Buscando IDs dos dispositivos para: {usina_name}")
        
        with self._database_connection('variaveis_delfos') as engine:
            query = 'SELECT * FROM "Inversores_ids"'
            inv_ids_df = pd.read_sql(query, engine)
            
            # Filtrar por usina
            inv_ids_df['name'] = inv_ids_df['name'].str.split('-').str[0].str.strip()
            filtered_df = inv_ids_df[inv_ids_df['name'] == usina_name]
            device_ids = filtered_df['id'].tolist()
            
            self.logger.info(f"Encontrados {len(device_ids)} dispositivos para {usina_name}")
            return device_ids

    def _extract_data_batch(self, device_ids: List[int], start_time: str, end_time: str) -> pd.DataFrame:
        """Extrai um lote específico de dados da API"""
        request_config = {
            "device_ids": device_ids,
            "variable_ids": self.config.variable_ids,
            "start_time": start_time,
            "end_time": end_time,
            "aggregate": "1 min"
        }
        
        url = 'https://api.delfos.im/solar-data/data-studio/device-types/2/timeseries'
        
        self.logger.debug(f"Extraindo lote: {start_time} até {end_time}")
        start_request_time = datetime.now()
        
        try:
            response = req.post(
                url, 
                headers=self.headers, 
                data=json.dumps(request_config), 
                timeout=self.config.timeout_seconds
            )
            
            request_duration = (datetime.now() - start_request_time).total_seconds()
            self.logger.debug(f'Lote concluído em {request_duration:.2f}s - Status: {response.status_code}')
            
            if response.status_code != 200:
                error_msg = response.text[:200] if response.text else "Sem detalhes do erro"
                raise req.exceptions.RequestException(f"Status {response.status_code}: {error_msg}")
            
            response_data = response.json()
            
            # Validar estrutura da resposta
            if 'sample_time' not in response_data or 'variables' not in response_data:
                raise ValueError("Estrutura de resposta inválida da API")
            
            # Transformar em DataFrame
            df = pd.DataFrame({'tempo': response_data['sample_time']})
            for var in response_data['variables']:
                device_name = var['device']['name']
                df[device_name] = var['values']
            
            return df
            
        except Exception as e:
            self.logger.error(f'Erro no lote ({start_time} - {end_time}): {e}')
            raise

    def _extract_data_with_batching(self, device_ids: List[int], start_time: str, end_time: str) -> pd.DataFrame:
        """Extrai dados com divisão em lotes quando necessário"""
        self.logger.info("INICIANDO EXTRAÇÃO DE DADOS DA API DELFOS")
        
        # Configurar tempos padrão se necessário
        if end_time is None:
            end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if start_time is None:
            start_time = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d %H:%M:%S')
        
        self.logger.info(f"Período: {start_time} até {end_time}")
        self.logger.info(f"Dispositivos: {len(device_ids)}")
        
        # Calcular período total
        start_dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        end_dt = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
        total_days = (end_dt - start_dt).days
        
        self.logger.info(f"Período total: {total_days} dias")
        
        # Se período pequeno, extrair em lote único
        if total_days <= self.config.batch_days:
            self.logger.info("Extração em lote único")
            df = self._extract_data_batch(device_ids, start_time, end_time)
            self._log_data_quality(df)
            return df
        
        # Período longo - dividir em lotes
        self.logger.info(f"Dividindo em lotes de {self.config.batch_days} dias")
        
        batch_list = []
        current_start = start_dt
        batch_count = 0
        successful_batches = 0
        failed_batches = 0
        
        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=self.config.batch_days), end_dt)
            batch_count += 1
            
            batch_start_str = current_start.strftime('%Y-%m-%d %H:%M:%S')
            batch_end_str = current_end.strftime('%Y-%m-%d %H:%M:%S')
            
            self.logger.info(f"Processando lote {batch_count}: {batch_start_str} até {batch_end_str}")
            
            try:
                batch_df = self._extract_data_batch(device_ids, batch_start_str, batch_end_str)
                
                if not batch_df.empty:
                    batch_list.append(batch_df)
                    successful_batches += 1
                    self.logger.info(f"Lote {batch_count} concluído: {len(batch_df)} registros")
                else:
                    self.logger.warning(f"Lote {batch_count} retornou vazio")
                
            except Exception as e:
                failed_batches += 1
                self.logger.error(f"ERRO no lote {batch_count}: {e}")
                
                # Parar se muitos lotes consecutivos falharem
                if failed_batches > 3 and successful_batches == 0:
                    raise Exception("Muitos lotes consecutivos falharam")
            
            current_start = current_end
        
        # Consolidar lotes
        if not batch_list:
            raise Exception("Nenhum lote foi processado com sucesso")
        
        self.logger.info(f"Consolidando {len(batch_list)} lotes...")
        df_final = pd.concat(batch_list, ignore_index=True)
        
        # Remover duplicatas e ordenar
        original_rows = len(df_final)
        df_final = df_final.drop_duplicates(subset=['tempo'], keep='first')
        df_final = df_final.sort_values('tempo').reset_index(drop=True)
        
        if original_rows != len(df_final):
            self.logger.info(f"Removidas {original_rows - len(df_final)} linhas duplicadas")
        
        # Log do resumo
        self.logger.info("RESUMO DA EXTRAÇÃO EM LOTES:")
        self.logger.info(f"  Total de lotes: {batch_count}")
        self.logger.info(f"  Sucessos: {successful_batches}")
        self.logger.info(f"  Falhas: {failed_batches}")
        self.logger.info(f"  Taxa de sucesso: {(successful_batches/batch_count*100):.1f}%")
        self.logger.info(f"  DataFrame final: {df_final.shape}")
        
        self._log_data_quality(df_final)
        return df_final

    def _log_data_quality(self, df: pd.DataFrame):
        """Registra informações sobre qualidade dos dados"""
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            self.logger.warning("Valores nulos encontrados:")
            for col, count in null_counts.items():
                if count > 0:
                    self.logger.warning(f"  {col}: {count} valores nulos")

    def _standardize_columns(self, df: pd.DataFrame, usina_code: str) -> pd.DataFrame:
        """Padroniza nomes das colunas por usina"""
        self.logger.info(f"Padronizando colunas para: {usina_code}")
        
        column_mappings = {
            'aqz01': {
                'Aquiraz - UFV 01 - INV01': 'pot_inversor_01', 
                'Aquiraz - UFV 01 - INV02': 'pot_inversor_02',
                'Aquiraz - UFV 01 - INV03': 'pot_inversor_03', 
                'Aquiraz - UFV 02 - INV01': 'pot_inversor_04',
                'Aquiraz - UFV 02 - INV02': 'pot_inversor_05', 
                'Aquiraz - UFV 02 - INV03': 'pot_inversor_06',
                'Aquiraz - UFV 03 - INV01': 'pot_inversor_07', 
                'Aquiraz - UFV 03 - INV02': 'pot_inversor_08'
            },
            'utxt01': {
                'Unitêxtil - UFV 09 - INV01': 'pot_inversor_01',
                'Unitêxtil - UFV 09 - INV02': 'pot_inversor_02',
                'Unitêxtil - UFV 09 - INV03': 'pot_inversor_03',
                'Unitêxtil - UFV 10 - INV01': 'pot_inversor_04',
                'Unitêxtil - UFV 10 - INV02': 'pot_inversor_05',
                'Unitêxtil - UFV 10 - INV03': 'pot_inversor_06',
                'Unitêxtil - UFV 11 - INV01': 'pot_inversor_07',
                'Unitêxtil - UFV 11 - INV02': 'pot_inversor_08',
                'Unitêxtil - UFV 11 - INV03': 'pot_inversor_09',
                'Unitêxtil - UFV 12 - INV01': 'pot_inversor_10',
                'Unitêxtil - UFV 12 - INV02': 'pot_inversor_11',
                'Unitêxtil - UFV 12 - INV03': 'pot_inversor_12',
                'Unitêxtil - UFV 13 - INV01': 'pot_inversor_13',
                'Unitêxtil - UFV 13 - INV02': 'pot_inversor_14',
                'Unitêxtil - UFV 13 - INV03': 'pot_inversor_15',
                'Unitêxtil - UFV 14 - INV01': 'pot_inversor_16',
                'Unitêxtil - UFV 14 - INV02': 'pot_inversor_17'
            },
            'msp02': {
                'Massapê II - INV 01': 'pot_inversor_01',
                'Massapê II - INV 02': 'pot_inversor_02',
                'Massapê II - INV 03': 'pot_inversor_03',
                'Massapê II - INV 04': 'pot_inversor_04',
                'Massapê II - INV 05': 'pot_inversor_05',
                'Massapê II - INV 06': 'pot_inversor_06',
                'Massapê II - INV 07': 'pot_inversor_07',
                'Massapê II - INV 08': 'pot_inversor_08',
                'Massapê II - INV 09': 'pot_inversor_09',
                'Massapê II - INV 10': 'pot_inversor_10',
                'Massapê II - INV 11': 'pot_inversor_11',
                'Massapê II - INV 12': 'pot_inversor_12',
                'Massapê II - INV 13': 'pot_inversor_13'
            },
            'bbr01': {
                'Beberibe - INV01': 'pot_inversor_01',
                'Beberibe - INV02': 'pot_inversor_02',
                'Beberibe - INV03': 'pot_inversor_03',
                'Beberibe - INV04': 'pot_inversor_04',
                'Beberibe - INV05': 'pot_inversor_05',
                'Beberibe - INV06': 'pot_inversor_06',
                'Beberibe - INV07': 'pot_inversor_07',
                'Beberibe - INV08': 'pot_inversor_08',
                'Beberibe - INV09': 'pot_inversor_09',
                'Beberibe - INV10': 'pot_inversor_10'
            }
        }
        
        if usina_code not in column_mappings:
            self.logger.warning(f"Mapeamento não encontrado para {usina_code}")
            return df
        
        mapping = column_mappings[usina_code]
        df_renamed = df.rename(columns=mapping)
        
        renamed_count = len([col for col in df.columns if col in mapping])
        self.logger.info(f"Renomeadas {renamed_count} colunas")
        
        return df_renamed

    def _upload_data(self, df: pd.DataFrame, db_name: str, table_name: str):
        """Faz upload dos dados para o banco"""
        self.logger.info(f"INICIANDO UPLOAD PARA {db_name}.{table_name}")
        self.logger.info(f"Shape: {df.shape}")
        
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
        self.logger.info(f"Uso de memória: {memory_usage:.2f} MB")
        
        self._log_data_quality(df)
        
        with self._database_connection(db_name) as engine:
            start_time = datetime.now()
            df.to_sql(table_name, con=engine, if_exists='append', index=False)
            
            duration = (datetime.now() - start_time).total_seconds()
            rate = len(df) / duration if duration > 0 else 0
            
            self.logger.info(f'Upload concluído em {duration:.2f}s')
            self.logger.info(f'Taxa: {rate:.2f} registros/segundo')

    def process_usina(self, usina_name: str, usina_code: str) -> bool:
        """Processa uma usina específica"""
        self.logger.info("=" * 60)
        self.logger.info(f"PROCESSANDO USINA: {usina_name} ({usina_code})")
        self.logger.info("=" * 60)
        
        try:
            usina_start_time = datetime.now()
            
            # Obter IDs dos dispositivos
            device_ids = self._get_device_ids(usina_name)
            if not device_ids:
                self.logger.error(f"Nenhum dispositivo encontrado para {usina_name}")
                return False
            
            # Obter última data registrada
            start_time = self._get_last_recorded_date(usina_code, 'dados_gerais')
            
            # Extrair dados
            df_usina = self._extract_data_with_batching(device_ids, start_time, None)
            
            if df_usina.empty:
                self.logger.warning(f"DataFrame vazio para {usina_name}")
                return False
            
            # Padronizar colunas
            df_usina = self._standardize_columns(df_usina, usina_code)
            
            # Upload dos dados
            self._upload_data(df_usina, usina_code, 'dados_gerais')
            
            # Atualizar estatísticas
            self.stats['total_records_processed'] += len(df_usina)
            
            duration = (datetime.now() - usina_start_time).total_seconds()
            self.logger.info(f"SUCESSO: {usina_name} processada em {duration:.2f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"ERRO ao processar {usina_name}: {e}")
            return False

    def run_extraction(self):
        """Executa o processo completo de extração"""
        execution_start = datetime.now()
        self.logger.info("EXECUÇÃO PRINCIPAL INICIADA")
        
        try:
            total_usinas = len(self.config.usinas_mapping)
            self.logger.info(f"Processando {total_usinas} usinas: {list(self.config.usinas_mapping.keys())}")
            
            for i, (usina_name, usina_code) in enumerate(self.config.usinas_mapping.items(), 1):
                self.logger.info(f"Processando {i}/{total_usinas}: {usina_name}")
                
                if self.process_usina(usina_name, usina_code):
                    self.stats['successful_extractions'] += 1
                else:
                    self.stats['failed_extractions'] += 1
            
            # Resumo final
            total_time = (datetime.now() - execution_start).total_seconds()
            success_rate = (self.stats['successful_extractions'] / total_usinas * 100) if total_usinas > 0 else 0
            
            self.logger.info("=" * 80)
            self.logger.info("RESUMO DA EXECUÇÃO")
            self.logger.info(f"Tempo total: {total_time:.2f} segundos")
            self.logger.info(f"Usinas processadas com sucesso: {self.stats['successful_extractions']}")
            self.logger.info(f"Usinas com falha: {self.stats['failed_extractions']}")
            self.logger.info(f"Taxa de sucesso: {success_rate:.1f}%")
            self.logger.info(f"Total de registros processados: {self.stats['total_records_processed']}")
            
            if self.stats['successful_extractions'] > 0:
                self.logger.info("EXECUÇÃO CONCLUÍDA COM SUCESSO")
            else:
                self.logger.error("EXECUÇÃO FALHOU - Nenhuma usina processada")
            
            self.logger.info("=" * 80)
            
        except Exception as e:
            total_time = (datetime.now() - execution_start).total_seconds()
            self.logger.error(f"ERRO CRÍTICO NA EXECUÇÃO: {e}")
            self.logger.error(f"Tempo até falha: {total_time:.2f} segundos")
            raise


def main():
    """Função principal"""
    config_path = r'C:\Users\usuario\Projeto_1\Extração\config_delfos.yaml'
    
    try:
        extractor = DelfosExtractor(config_path)
        extractor.run_extraction()
        
    except Exception as e:
        print(f"Erro na execução: {e}")
        raise


if __name__ == "__main__":
    main()
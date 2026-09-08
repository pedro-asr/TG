import requests as req
import pandas as pd
import json
import yaml
from datetime import datetime, timedelta
from pathlib import Path

# ── Configurações ─────────────────────────────────────────
CONFIG_PATH = r'C:\Users\usuario\Projeto_1\Extração\config_delfos.yaml'

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    conf = yaml.safe_load(f)

API_KEY = conf['delfos_api']['api_key']

HEADERS = {'API-Key': API_KEY, 'Content-Type': 'application/json'}

# Aquiraz - UFV01: INV01(id=4), INV02(id=5), INV03(id=6)
DEVICE_IDS_UFV01 = [4, 5, 6]
VARIABLE_ID      = 'signal-131'  # Potência ativa

OUTPUT_DIR  = Path(r'C:\Users\usuario\Projeto_1\Extração')
OUTPUT_FILE = OUTPUT_DIR / 'dados_aqz01.xlsx'

START_TIME_DEFAULT = '2025-02-01 00:00:00'

# ── Funções ───────────────────────────────────────────────

def get_last_date(excel_path=OUTPUT_FILE):
    if not excel_path.exists():
        return START_TIME_DEFAULT
    df = pd.read_excel(excel_path, usecols=['tempo'])
    if df.empty or df['tempo'].isna().all():
        return START_TIME_DEFAULT
    return str(df['tempo'].max())


def extract_data_batch(device_ids, start_time, end_time):
    payload = {
        'device_ids':   device_ids,
        'variable_ids': [VARIABLE_ID],
        'start_time':   start_time,
        'end_time':     end_time,
        'aggregate':    '1 min'
    }
    url = 'https://api.delfos.im/solar-data/data-studio/device-types/2/timeseries'
    r = req.post(url, headers=HEADERS, data=json.dumps(payload), timeout=120)
    r.raise_for_status()
    resp = r.json()
    df = pd.DataFrame({'tempo': resp['sample_time']})
    for var in resp['variables']:
        df[var['device']['name']] = var['values']
    return df


def extract_data(device_ids, start_time, end_time=None, batch_days=7):
    if end_time is None:
        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    start_dt = datetime.strptime(str(start_time)[:19], '%Y-%m-%d %H:%M:%S')
    end_dt   = datetime.strptime(str(end_time)[:19],   '%Y-%m-%d %H:%M:%S')
    chunks, cur = [], start_dt
    while cur < end_dt:
        nxt = min(cur + timedelta(days=batch_days), end_dt)
        print(f'  Lote: {cur} -> {nxt}')
        try:
            batch = extract_data_batch(device_ids, cur.strftime('%Y-%m-%d %H:%M:%S'), nxt.strftime('%Y-%m-%d %H:%M:%S'))
            if not batch.empty:
                chunks.append(batch)
        except Exception as e:
            print(f'  Erro no lote {cur}: {e}')
        cur = nxt
    if not chunks:
        raise RuntimeError('Nenhum dado retornado pela API.')
    df = pd.concat(chunks, ignore_index=True)
    return df.drop_duplicates(subset=['tempo']).sort_values('tempo').reset_index(drop=True)


def standart_columns_aqz01(df):
    mapping = {
        'Aquiraz - UFV 01 - INV01': 'pot_inversor_01',
        'Aquiraz - UFV 01 - INV02': 'pot_inversor_02',
        'Aquiraz - UFV 01 - INV03': 'pot_inversor_03',
    }
    return df.rename(columns=mapping)


def save_to_excel(df, excel_path=OUTPUT_FILE):
    if excel_path.exists():
        existing = pd.read_excel(excel_path)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=['tempo']).sort_values('tempo').reset_index(drop=True)
    df.to_excel(excel_path, index=False)
    print(f'Salvo em Excel: {len(df)} registros -> {excel_path}')


# ── Execução ──────────────────────────────────────────────

if __name__ == '__main__':
    start_time = get_last_date()
    print(f'Extraindo a partir de: {start_time}')

    df = extract_data(DEVICE_IDS_UFV01, start_time)
    df = standart_columns_aqz01(df)

    print(f'Shape: {df.shape}')
    print(df.head())

    save_to_excel(df)

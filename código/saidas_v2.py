#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Caminhos de saída da v2 — compartilhado por todos os scripts `*_v2.py`.

Todas as saídas da v2 vão para `código/resultados_v2/` (subpastas abaixo), nunca para
as pastas da v1. Nenhum arquivo é sobrescrito: `caminho_novo`/`garantir_inexistentes`
abortam se o destino já existir, e todo nome de arquivo precisa terminar em `_v2`.

Modo rápido (`--rapido`, teste de ponta a ponta): as saídas vão para o sandbox
`resultados_v2/rapido/<subpasta>/`, onde arquivos de teste anteriores PODEM ser
sobrescritos (saídas descartáveis). Fora do sandbox a regra de não sobrescrever vale sempre.
"""
from pathlib import Path

DIR_RESULTADOS_V2 = Path(__file__).resolve().parent / "resultados_v2"
DIR_SANDBOX_RAPIDO = DIR_RESULTADOS_V2 / "rapido"
SUBPASTAS = ("diagnostico", "dados", "modelos", "optuna", "predicoes", "tabelas", "figuras")
REGISTRO_MUDANCAS = DIR_RESULTADOS_V2 / "registro_mudancas_v2.md"

_modo_rapido = False


def ativar_modo_rapido():
    """Redireciona as saídas seguintes para o sandbox `resultados_v2/rapido/`."""
    global _modo_rapido
    _modo_rapido = True
    print(f"[modo rápido] saídas em {DIR_SANDBOX_RAPIDO} (sandbox; sobrescrita permitida)",
          flush=True)


def modo_rapido():
    return _modo_rapido


def pasta(subpasta):
    """Diretório `resultados_v2/<subpasta>/` (ou do sandbox, no modo rápido), criado se
    ainda não existir."""
    if subpasta not in SUBPASTAS:
        raise ValueError(f"Subpasta desconhecida: {subpasta!r} (válidas: {SUBPASTAS})")
    caminho = (DIR_SANDBOX_RAPIDO if _modo_rapido else DIR_RESULTADOS_V2) / subpasta
    caminho.mkdir(parents=True, exist_ok=True)
    return caminho


def caminho_saida(subpasta, nome):
    """Caminho de um arquivo de saída da v2, SEM checar existência (usar com
    `garantir_inexistentes` antes de gravar)."""
    if not Path(nome).stem.endswith("_v2"):
        raise ValueError(f"Arquivo de saída da v2 sem o sufixo '_v2': {nome}")
    return pasta(subpasta) / nome


def _no_sandbox(caminho):
    return Path(caminho).resolve().is_relative_to(DIR_SANDBOX_RAPIDO.resolve())


def garantir_inexistentes(caminhos):
    """Aborta (sem gravar nada) se qualquer um dos caminhos já existir — exceto arquivos
    do sandbox do modo rápido, que podem ser sobrescritos."""
    existentes = [str(c) for c in caminhos if Path(c).exists() and not _no_sandbox(c)]
    if existentes:
        raise FileExistsError("Arquivo(s) de saída já existem — abortando sem gravar nada "
                              "(remova/renomeie manualmente se quiser regerar):\n  "
                              + "\n  ".join(existentes))


def caminho_novo(subpasta, nome):
    """Caminho de um arquivo NOVO em `resultados_v2/<subpasta>/`; aborta se já existir."""
    caminho = caminho_saida(subpasta, nome)
    garantir_inexistentes([caminho])
    return caminho

#!/bin/bash
# Script para rodar o extractor com o ambiente virtual correto

cd "$(dirname "$0")"

# Ativa o ambiente virtual
source .venv/bin/activate

# Executa o script principal com todos os argumentos passados
python run.py "$@"


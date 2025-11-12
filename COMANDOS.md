# Comandos do Projeto

## Setup Inicial
```bash
source .venv/bin/activate  
pip install -r requirements.txt
```

## Execução Normal
```bash
# Execução padrão (com checkpoint - pula páginas já processadas)
python -m extractor

# Com logs em tempo real
PYTHONUNBUFFERED=1 python -m extractor
```

## Checkpoint de Páginas
O sistema agora verifica automaticamente se uma página já foi processada antes de extrair novamente.

**Como funciona:**
- Se a pasta `page-XXX` existe E contém arquivos `.html` válidos → **página é pulada**
- Se a pasta não existe OU não tem HTMLs → **página é processada**
- Páginas com erro anterior (sem HTML) serão **reprocessadas automaticamente**

### Forçar Reprocessamento
Para reprocessar todas as páginas (ignorando checkpoint):
```bash
FORCE_REPROCESS=1 python -m extractor
```

## Logs e Depuração
```bash
# Logs em tempo real + checkpoint
PYTHONUNBUFFERED=1 python -m extractor

# Ver qual página está sendo processada ou pulada
# Procure por:
# ✅ Página XXX JÁ PROCESSADA (checkpoint) - pulando
# 🔄 Página XXX será REPROCESSADA (force_reprocess=True)
```

## Limitação de Tamanho de Imagens
O sistema agora reduz automaticamente imagens grandes antes de enviar para a API:
- Limite: 15MB (Azure OpenAI aceita até 20MB)
- Downscale automático mantendo legibilidade mínima de 800px
- Logs detalhados do processo de redução

## Melhorias na Extração de Tabelas Múltiplas
O sistema foi aprimorado para detectar e extrair corretamente páginas com **múltiplas tabelas separadas**:
- ✅ Detecta automaticamente quando há 2+ tabelas fisicamente separadas
- ✅ Cria um arquivo separado para cada tabela (table-01.xlsx, table-02.xlsx, etc.)
- ✅ Não mistura tabelas diferentes em um único HTML
- ✅ Evita linhas vazias artificiais (`<td colspan="X"></td>`)

**Como reprocessar páginas com erro de formatação:**
1. Identifique a página com problema (ex: `page-100`)
2. Delete a pasta `output/NOME_PDF/llm_tables/page-100/`
3. Execute novamente: `python -m extractor`
4. Apenas a página deletada será reprocessada (checkpoint automático)

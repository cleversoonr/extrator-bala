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

## Checkpoint Duplo (Rasterização + Extração)
O sistema agora verifica automaticamente em **duas etapas** para evitar reprocessamento desnecessário:

### 1. Checkpoint de Rasterização (PNG)
- Se `pages/page-XXX.png` existe → **pula renderização**
- Se não existe → **renderiza do PDF**
- Economiza tempo significativo em DPI alto (900)

### 2. Checkpoint de Extração (HTML)
- Se `page-XXX/` existe E contém `.html` válidos → **pula extração**
- Se não existe OU sem HTMLs → **processa com LLM**
- Páginas com erro (sem HTML) são **reprocessadas automaticamente**

### Forçar Reprocessamento
Para reprocessar todas as páginas (ignorando checkpoints):
```bash
# Ignora checkpoint de extração (LLM) apenas
FORCE_REPROCESS=1 python -m extractor

# Para forçar re-renderização também, delete as imagens:
rm -rf output/NOME_PDF/pages/
```

## Logs e Depuração
```bash
# Logs em tempo real + checkpoint
PYTHONUNBUFFERED=1 python -m extractor

# Logs de checkpoint que você verá:
# ✅ 95/105 páginas JÁ RASTERIZADAS (checkpoint) - pulando: 1-95
# 🖼️  Rasterizando 10/105 páginas em output/.../pages dpi=900: 96-105
# ✅ Página 100 JÁ PROCESSADA (checkpoint) - pulando
# 🔄 Página 105 será REPROCESSADA (force_reprocess=True)
```

## Limitação de Tamanho de Imagens
O sistema agora reduz automaticamente imagens grandes antes de enviar para a API:
- Limite: 15MB (Azure OpenAI aceita até 20MB)
- Downscale automático mantendo legibilidade mínima de 800px
- Logs detalhados do processo de redução

## Melhorias na Extração de Tabelas Múltiplas (NOVA VERSÃO)

O sistema foi **completamente reformulado** com 4 camadas de proteção:

### 🛡️ 4 Camadas de Proteção Anti-Erro

**1. Conversão P&B Automática**
- Converte imagem para preto e branco antes de enviar
- Melhora contraste e legibilidade de bordas/texto
- Threshold adaptativo para tabelas

**2. Prompt Ultra-Específico**
- Quando detecta 2+ tabelas, adiciona aviso crítico no prompt
- Especifica EXATAMENTE quantos objetos criar no JSON
- Avisa que resposta será rejeitada se errar

**3. Validação Pós-Extração**
- Detecta automaticamente quando tabelas foram mescladas incorretamente
- Conta células vazias (se >30%, é erro de mesclagem)
- Compara quantidade esperada vs extraída

**4. Retry Inteligente**
- Se detecta erro, tenta novamente com prompt ainda mais agressivo
- Usa imagem P&B otimizada
- Só aceita resposta se quantidade bater

### 📊 Resultados Esperados

**Antes:**
- Mesclava 2 tabelas em 1 com células vazias ❌
- 26 colunas onde deveria ter 2 tabelas separadas ❌
- Dados na posição errada ❌

**Depois:**
- 2 objetos separados no JSON ✅
- Cada tabela com suas próprias colunas ✅
- Dados corretos em cada posição ✅

### 🔄 Como Reprocessar Páginas com Erro

```bash
# 1. Delete a pasta da página problemática
rm -rf output/NOME_PDF/llm_tables/page-100/

# 2. Execute novamente (só essa página será reprocessada)
python -m extractor
```

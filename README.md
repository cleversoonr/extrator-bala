# 📊 Extract Bala - Extração de Tabelas e Gráficos de PDFs

Sistema inteligente de extração de dados científicos de PDFs usando **LLMs** (GPT-5) com **fallback automático OCR** para tabelas complexas.

## ✨ Principais Funcionalidades

- 🤖 **Extração com LLM**: GPT-5 para análise precisa
- 🔍 **Pre-check inteligente**: Modelo barato filtra páginas vazias (-90% custo)
- 🔄 **Fallback automático**: Se >30% células vazias → re-tenta com OCR
- 📊 **Gráficos multi-painel**: Extrai TODAS as equações
- 📈 **Tabelas complexas**: OCR célula-a-célula quando necessário
- ⚙️ **Zero configuração**: Sistema decide melhor estratégia automaticamente

---

## 🚀 Uso Rápido

```bash
# 1. Instalar dependências
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configurar .env (copie o exemplo)
cp env.example .env
# Edite .env com suas chaves de API

# 3. Colocar PDFs na pasta docs/
mkdir -p docs
cp seu-artigo.pdf docs/

# 4. Rodar
python run.py
```

---

## 📋 Pré-requisitos

- **Python 3.9+**
- **Chave de API** para LLM:
  - Azure OpenAI (recomendado) OU
  - OpenRouter OU
  - OpenAI direto

**Requisitos opcionais:**
- 🔶 **Tesseract OCR** (recomendado para tabelas complexas): Ver [Docs - Instalação](DOCS.md#instalação-do-tesseract)

---

## ⚙️ Configuração

Crie um arquivo `.env` na raiz do projeto:

```env
# OPÇÃO 1: Azure OpenAI (recomendado)
AZURE_OPENAI_ENDPOINT=https://seu-recurso.openai.azure.com/
AZURE_OPENAI_API_KEY=sua-chave-api
AZURE_OPENAI_API_VERSION=2025-03-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-5

# Pre-check com modelo mais barato (economia de 40-60%)
AZURE_GPT41_ENDPOINT=https://seu-recurso.openai.azure.com/
AZURE_GPT41_API_KEY=sua-chave-api
AZURE_GPT41_DEPLOYMENT=gpt-4o-mini
AZURE_GPT41_API_VERSION=2025-03-01-preview

# OPÇÃO 2: OpenRouter
OPENROUTER_API_KEY=sk-or-v1-...

# OPÇÃO 3: OpenAI direto
OPENAI_API_KEY=sk-...

# Opcional: Paralelização
LLM_MAX_WORKERS=6  # Até 6 páginas processadas em paralelo
```

---

## 🔄 Como Funciona

```
PDF
 ↓
[1] RENDERIZA PÁGINAS (DPI 700)
    → Gera imagens PNG de alta qualidade
 ↓
[2] PRE-CHECK (LLM barata - GPT-4o-mini)
    → Identifica: tipo (tabela/gráfico/texto)
    → Identifica: quantidade (1, 2, 3+)
    → Se não tem conteúdo útil: PULA (economia!)
 ↓
[3] DETECÇÃO DE COMPLEXIDADE (para tabelas)
    → Se >10 linhas horizontais + >5 verticais:
       ↳ Usa ABORDAGEM HÍBRIDA (OCR + LLM)
    → Caso contrário: apenas LLM
 ↓
[4] EXTRAÇÃO
    → **Tabela complexa**: OCR célula-a-célula + LLM combina estrutura
    → **Tabela simples/gráfico**: Apenas LLM (GPT-5)
    → Retorna JSON estruturado
 ↓
[5] SALVA RESULTADOS
    → Excel (.xlsx)
    → JSON (.json)
    → HTML (.html)
    → Summary (summary.html)
    → OCR data (ocr-data.txt) - apenas para tabelas complexas
```

### 🔬 Abordagem Híbrida (Automática!)

Sistema com **fallback inteligente**:

1. ✅ Tenta extração com LLM
2. ✅ Verifica se resultado tem >30% células vazias
3. ✅ Se sim: **Re-tenta automaticamente com OCR + LLM**

**Resultado:** Valores completos mesmo em tabelas hierárquicas!

📖 Ver detalhes: **[DOCS.md - Abordagem Híbrida](DOCS.md#abordagem-híbrida-ocr-llm)**

---

## 📂 Estrutura de Saída

```
output/
└── nome-do-pdf/
    ├── pages/                    # Páginas renderizadas
    │   ├── page-001.png
    │   └── ...
    └── llm_tables/              # Resultados extraídos
        ├── page-001/
        │   ├── page-full.png       # Página processada
        │   ├── page-full.json      # JSON bruto do LLM
        │   ├── table-01.xlsx       # Excel com dados
        │   ├── table-01.html       # Preview HTML
        │   └── table-01-notes.txt  # Notas/legendas
        ├── page-002/
        └── summary.html           # Índice de todas as extrações
```

---

## 📊 Tipos de Conteúdo Suportados

### ✅ Tabelas
- Tabelas com/sem bordas
- Células mescladas
- Células coloridas
- Múltiplas tabelas por página

**Saída Excel:**
| Tratamento | Dose (kg/ha) | Produtividade | % Aumento |
|------------|--------------|---------------|-----------|
| T1 | 0 | 2340,5 | - |
| T2 | 50 | 2890,3 | 23,5 |

### ✅ Gráficos com Equações Quadráticas
- Equações Y = a + bX ± cX²
- Gráficos multi-painel (4, 8, 12 painéis)
- **Cálculos automáticos:** X*, Y_max, X_90%, Y_90%

**Saída Excel:**
| Painel | a | b | c | R² | X* (kg N/ha) | Y_max (kg/ha) | X_90% | Y_90% |
|--------|---|---|---|----|--------------|--------------| ------|-------|
| 1º ANO | 3400,874 | 33,8728 | 0,08110 | 0,9649 | 208,8 | 6938 | 116,3 | 6244 |
| 2º ANO | 3900,876 | 37,4993 | 0,11498 | 0,9174 | 163,1 | 6958 | 85,3 | 6263 |

### ✅ Gráficos de Dados
- Linhas, barras, dispersão
- Múltiplas séries

---

## 💰 Custo Estimado

### Por Página

| Tipo | Pre-check | Extração | Total |
|------|-----------|----------|-------|
| Texto puro | $0.001 | - | $0.001 |
| 1 tabela | $0.001 | $0.05 | $0.051 |
| 4 gráficos | $0.001 | $0.05 | $0.051 |

**Economia:** 40-60% vs fluxo anterior (múltiplas chamadas por crop).

---

## 🐛 Troubleshooting

### "Nenhuma tabela reconhecida"
- Verifique logs: se mostra "sem conteúdo útil", está correto
- Se deveria ter tabela: pode ser qualidade baixa

### "Erro na chamada à LLM"
- Verifique `.env`
- Teste chave API manualmente
- Verifique conexão

### "Tabela extraída incorretamente"
- Revise `page-full.json` para ver resposta bruta
- Imagem de baixa qualidade: pode ser limitação do PDF original

---

## 📚 Documentação

- **[DOCS.md](DOCS.md)** - 📖 Documentação técnica completa
  - Abordagem Híbrida OCR+LLM
  - Correções de Gráficos Multi-Painel
  - Instalação do Tesseract
  - Comandos Úteis
  - Changelog Detalhado

- **[CHANGELOG.md](CHANGELOG.md)** - Histórico de versões
- **[tests/](tests/)** - Scripts de teste e validação
- **[docs-old/](docs-old/)** - Documentação antiga (backup)

---

## 🎯 Filosofia

**Antes (v1.x):**
- Detecção OpenCV → Crop → OCR → LLM fallback
- Complexo, bugs frequentes, tabelas perdidas

**Agora (v2.0):**
- Renderiza → Pre-check → Extração LLM
- Simples, confiável, econômico

**Por quê?**
- ✅ Mais simples (menos código)
- ✅ Mais confiável (GPT-5 vê contexto completo)
- ✅ Mais econômico (1-2 chamadas por página)
- ✅ Mais rápido (sem processamento OpenCV pesado)

---

## 📦 Dependências Principais

```
pymupdf         # Processamento de PDF
opencv-python   # Processamento de imagens
pandas          # Geração de Excel/HTML
openai          # Interface com LLMs
rich            # Interface de terminal
```

**Nota:**
- ✅ `pytesseract` - Reintroduzido para fallback automático em tabelas complexas
- ❌ `python-docx` (geração de DOCX) - Removido
- ❌ `typer` (CLI antiga) - Removido

---

## 🤝 Contribuindo

Este é um projeto simplificado. Foco em:
- Manter o fluxo simples
- Não adicionar complexidade desnecessária
- Confiar no LLM para extração

---

**Última atualização:** 2025-01-08  
**Versão:** 2.0 - Fluxo Simplificado

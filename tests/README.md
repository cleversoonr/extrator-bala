# 🧪 Scripts de Teste

Scripts para validar funcionalidades específicas do extrator.

## Testes Disponíveis

### `test_pagina_7_algodao.py`
Testa abordagem híbrida OCR+LLM em tabelas complexas.

**Arquivo:** MS rec para algodão.pdf (página 7)  
**Testa:** Detecção automática de células vazias + fallback OCR

```bash
python tests/test_pagina_7_algodao.py
```

**Resultado esperado:**
- Detecta >30% células vazias
- Aciona OCR automaticamente
- Gera `ocr-data.txt` e `page-full-hybrid.json`

---

### `test_pagina_418.py`
Testa caso específico de página complexa.

```bash
python tests/test_pagina_418.py
```

---

### `test_ternary.py` e `test_ternary_v3.py`
Testes para gráficos ternários (funcionalidade legada).

```bash
python tests/test_ternary.py
python tests/test_ternary_v3.py
```

---

## Estrutura dos Outputs

Os testes geram outputs em `output/test_*`:

```
output/test_page7_hybrid/
├── pages/
│   └── page-007.png
└── llm_tables/
    ├── page-007/
    │   ├── page-full.json         # Primeira tentativa
    │   ├── page-full-hybrid.json  # Com OCR (se acionado)
    │   ├── ocr-data.txt           # Dados OCR brutos
    │   └── table-*.xlsx           # Resultados finais
    └── summary.html
```

---

## Limpeza

Para limpar outputs de teste:

```bash
rm -rf output/test_*
```

---

## Configuração

Os testes leem configurações do `.env` na raiz do projeto:

```env
# Extração (GPT-5)
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_DEPLOYMENT=gpt-5

# Pre-check (GPT-4.1 - modelo barato)
AZURE_OPENAI_PRECHECK_ENDPOINT=...
AZURE_OPENAI_PRECHECK_API_KEY=...
AZURE_OPENAI_PRECHECK_DEPLOYMENT=gpt-4.1
```

---

## Debug

Para ver logs detalhados:

```bash
python tests/test_pagina_7_algodao.py 2>&1 | tee test.log
```

Para buscar mensagens específicas:

```bash
python tests/test_pagina_7_algodao.py 2>&1 | grep -E "(OCR|null|Re-extração)"
```


"""
Módulo de extração de texto completo de páginas text-only.

Este módulo processa páginas que contêm apenas texto (sem tabelas/gráficos),
convertendo todo o conteúdo em HTML formatado para inclusão no summary.

Uso:
- Ativado por CONVERT_TEXT_ONLY=true no .env
- Processa páginas identificadas como 'text_only' pelo pre-check
- Gera HTML formatado com parágrafos, listas, títulos, etc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
from html import escape

from .logging_utils import get_logger
from .llm_vision import call_openai_vision_json

logger = get_logger(__name__)


# =============================================================================
# PROMPT PARA EXTRAÇÃO DE TEXTO
# =============================================================================

TEXT_EXTRACTION_PROMPT = """Você verá uma página contendo APENAS TEXTO (sem tabelas ou gráficos).

**OBJETIVO:** Extrair TODO o texto da página preservando a estrutura e formatação.

**INSTRUÇÕES:**

1. **Leia TODO o texto** da página, linha por linha
2. **Identifique a estrutura:**
   - Títulos e sub-títulos
   - Parágrafos
   - Listas (numeradas ou com bullets)
   - Notas de rodapé
   - Referências bibliográficas
3. **Preserve a hierarquia:** Mantenha a ordem e organização do texto original

**FORMATO DE SAÍDA:**

```json
{
  "type": "text",
  "title": "Título da seção (se houver, senão null)",
  "sections": [
    {
      "type": "heading",
      "level": 1,
      "text": "Título Principal"
    },
    {
      "type": "paragraph",
      "text": "Texto completo do parágrafo..."
    },
    {
      "type": "list",
      "style": "numbered|bullet",
      "items": ["Item 1", "Item 2", "Item 3"]
    },
    {
      "type": "heading",
      "level": 2,
      "text": "Sub-título"
    }
  ]
}
```

**TIPOS DE SEÇÃO:**
- `heading`: Título (level 1-6, sendo 1 o mais importante)
- `paragraph`: Parágrafo de texto corrido
- `list`: Lista numerada ou com bullets
- `blockquote`: Citação ou nota destacada
- `reference`: Referência bibliográfica

**REGRAS CRÍTICAS:**
- Extraia EXATAMENTE o texto que vê escrito
- NÃO invente ou adicione informações
- NÃO traduza (mantenha idioma original)
- Se não conseguir ler algo, use "[ilegível]"
- Preserve formatações importantes (itálico, negrito) usando marcação:
  - Negrito: **texto**
  - Itálico: *texto*

Retorne APENAS JSON válido."""


# =============================================================================
# FUNÇÕES PRINCIPAIS
# =============================================================================

def extract_text_from_page(
    image_path: Path,
    page_out: Path,
    page_id: str,
    model: str,
    provider: Optional[str],
    api_key: Optional[str],
    azure_endpoint: Optional[str],
    azure_api_version: Optional[str],
    openrouter_api_key: Optional[str],
    locale: str,
) -> tuple[Optional[Path], Optional[Dict[str, str]]]:
    """
    Extrai texto completo de uma página text-only usando GPT-5.
    
    Args:
        image_path: Caminho para imagem da página
        page_out: Diretório de saída
        page_id: ID da página (ex: "095")
        model: Modelo LLM (ex: "gpt-5")
        provider: Provedor (azure, openrouter, openai)
        api_key: Chave de API
        azure_endpoint: Endpoint Azure
        azure_api_version: Versão da API Azure
        openrouter_api_key: Chave OpenRouter
        locale: Idioma (pt-BR, en-US, etc)
    
    Returns:
        Tupla (html_path, summary_entry) ou (None, None) se falhar
    """
    logger.info("📄 Extraindo texto completo da página %s (text-only)", page_id)
    
    try:
        payload = call_openai_vision_json(
            image_path,
            model=model,
            provider=provider,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version,
            openrouter_api_key=openrouter_api_key,
            locale=locale,
            instructions=TEXT_EXTRACTION_PROMPT,
            max_retries=2,
        )
        
        if not payload:
            logger.warning("GPT-5 não retornou dados para página %s (text-only)", page_id)
            return None, None
        
        # Salva JSON bruto
        import json
        (page_out / "page-text.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        
        # Converte para HTML
        html_content = _payload_to_html(payload)
        if not html_content:
            logger.warning("Falha ao converter payload para HTML (página %s)", page_id)
            return None, None
        
        # Salva HTML
        title = payload.get("title") or f"Página {page_id}"
        html_path = _save_text_html(html_content, page_out, title)
        
        if html_path:
            logger.info("✅ Texto extraído e salvo: %s", html_path.name)
            summary_entry = {
                "page": page_id,
                "table": "text",  # Identificador especial para texto
                "html": html_content,
            }
            return html_path, summary_entry
        
        return None, None
        
    except Exception as e:
        logger.error("Erro ao extrair texto da página %s: %s", page_id, e)
        return None, None


def _payload_to_html(payload: Dict[str, Any]) -> Optional[str]:
    """
    Converte payload JSON em HTML formatado.
    
    Args:
        payload: Dicionário com estrutura de texto extraído
    
    Returns:
        String HTML ou None se falhar
    """
    if not isinstance(payload, dict):
        return None
    
    sections = payload.get("sections", [])
    if not sections:
        logger.warning("Payload sem seções de texto")
        return None
    
    html_parts = []
    
    for section in sections:
        section_type = section.get("type")
        
        if section_type == "heading":
            level = section.get("level", 2)
            text = section.get("text", "")
            if text:
                # Converte markdown-style para HTML
                text_html = _format_inline_text(text)
                html_parts.append(f"<h{level}>{text_html}</h{level}>")
        
        elif section_type == "paragraph":
            text = section.get("text", "")
            if text:
                text_html = _format_inline_text(text)
                html_parts.append(f"<p>{text_html}</p>")
        
        elif section_type == "list":
            items = section.get("items", [])
            style = section.get("style", "bullet")
            if items:
                tag = "ol" if style == "numbered" else "ul"
                html_parts.append(f"<{tag}>")
                for item in items:
                    item_html = _format_inline_text(str(item))
                    html_parts.append(f"  <li>{item_html}</li>")
                html_parts.append(f"</{tag}>")
        
        elif section_type == "blockquote":
            text = section.get("text", "")
            if text:
                text_html = _format_inline_text(text)
                html_parts.append(f"<blockquote>{text_html}</blockquote>")
        
        elif section_type == "reference":
            text = section.get("text", "")
            if text:
                text_html = _format_inline_text(text)
                html_parts.append(f"<p class='reference'>{text_html}</p>")
    
    return "\n".join(html_parts)


def _format_inline_text(text: str) -> str:
    """
    Formata texto inline (negrito, itálico) de markdown para HTML.
    
    Args:
        text: Texto com marcação markdown (**negrito**, *itálico*)
    
    Returns:
        HTML escapado com formatação
    """
    import re
    
    # Escapa HTML primeiro
    text = escape(text)
    
    # Converte **negrito** para <strong>
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    
    # Converte *itálico* para <em>
    text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)
    
    return text


def _save_text_html(
    content: str,
    out_dir: Path,
    title: str,
) -> Optional[Path]:
    """
    Salva conteúdo de texto em HTML formatado.
    
    Args:
        content: HTML do conteúdo (já processado)
        out_dir: Diretório de saída
        title: Título da página
    
    Returns:
        Path do arquivo salvo ou None se falhar
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Monta HTML completo com CSS
    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: 'Georgia', 'Times New Roman', serif;
            padding: 40px;
            max-width: 800px;
            margin: 0 auto;
            background: #fafafa;
            line-height: 1.6;
        }}
        .container {{
            background: #fff;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            font-size: 2em;
            margin-bottom: 20px;
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            font-size: 1.5em;
            margin-top: 30px;
            margin-bottom: 15px;
            color: #34495e;
        }}
        h3 {{
            font-size: 1.2em;
            margin-top: 20px;
            margin-bottom: 10px;
            color: #555;
        }}
        p {{
            margin-bottom: 15px;
            text-align: justify;
            color: #333;
        }}
        ul, ol {{
            margin-bottom: 15px;
            padding-left: 30px;
        }}
        li {{
            margin-bottom: 8px;
            color: #333;
        }}
        blockquote {{
            border-left: 4px solid #3498db;
            margin: 20px 0;
            padding: 10px 20px;
            background: #ecf0f1;
            font-style: italic;
        }}
        .reference {{
            font-size: 0.9em;
            color: #666;
            margin-left: 20px;
        }}
        strong {{
            color: #2c3e50;
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>{escape(title)}</h1>
    {content}
</div>
</body>
</html>"""
    
    try:
        html_path = out_dir / "page-text.html"
        html_path.write_text(full_html, encoding="utf-8")
        logger.info("✅ HTML de texto salvo: %s", html_path.name)
        return html_path
    except Exception as e:
        logger.error("Erro ao salvar HTML de texto: %s", e)
        return None


from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv

from .logging_utils import get_logger


logger = get_logger(__name__)


SYSTEM_MSG = (
    "Você é um extrator de dados de gráficos e tabelas. Retorne APENAS JSON válido, sem texto antes/depois. "
    "Se for uma tabela, retorne {type:'table', table:{rows:[...]}}. "
    "Se for um gráfico (linha, barra, dispersão, ternário etc.), retorne {type:'chart', chart:{...}} com as séries numéricas. "
    "Para eixos categóricos (ex. datas), mantenha os rótulos como strings em x.values. "
    "Para valores incertos, use null. Não invente dados além do que é legível."
)

PRECHECK_PROMPT = (
    "Analise esta imagem rapidamente. Retorne JSON: "
    "{'has_content': true/false, 'content_type': 'table'|'chart'|'mixed'|'text_only'|'none', 'count': número}. "
    "\n\n"
    "IMPORTANTE: Campo 'count' = quantas tabelas/gráficos DISTINTOS você vê na página.\n"
    "- Se vê 1 tabela → count=1\n"
    "- Se vê 2 tabelas separadas (ex: Tabela 3 e Tabela 4) → count=2\n"
    "- Se vê 3+ tabelas/gráficos → count=3 (ou mais)\n"
    "- Se não tem conteúdo útil → count=0\n"
    "\n"
    "Regras:\n"
    "- Se tiver APENAS tabela(s), content_type='table'\n"
    "- Se tiver APENAS gráfico(s), content_type='chart'\n"
    "- Se tiver TABELA + GRÁFICO juntos, content_type='mixed'\n"
    "- Se for APENAS texto corrido sem tabelas/gráficos, has_content=false, content_type='text_only', count=0\n"
    "- Se não tiver conteúdo útil, has_content=false, content_type='none', count=0\n"
    "\n"
    "Exemplos:\n"
    "- Página com 'Tabela 3' e 'Tabela 4' → {'has_content': true, 'content_type': 'table', 'count': 2}\n"
    "- Página com 1 gráfico → {'has_content': true, 'content_type': 'chart', 'count': 1}\n"
    "- Página com 1 tabela + 1 gráfico → {'has_content': true, 'content_type': 'mixed', 'count': 2}\n"
    "- Página só com texto → {'has_content': false, 'content_type': 'text_only', 'count': 0}"
)


def _img_to_data_url(path: Path) -> str:
    b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/{path.suffix[1:] or 'png'};base64,{b64}"


def quick_precheck_with_cheap_llm(
    image_path: Path,
    cheap_model: str,
    cheap_provider: Optional[str],
    openrouter_api_key: Optional[str],
    *,
    api_key: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    azure_api_version: Optional[str] = None,
) -> Tuple[bool, str, int]:
    """
    Verificação rápida com LLM barata: retorna se tem conteúdo útil.
    Retorna: (has_content: bool, content_type: str, count: int)
    count = quantas tabelas/gráficos distintos na página
    """
    try:
        payload = call_openai_vision_json(
            image_path,
            model=cheap_model,
            provider=cheap_provider or "openrouter",
            openrouter_api_key=openrouter_api_key,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version,
            instructions=PRECHECK_PROMPT,
            max_retries=0,  # Sem retry no pre-check (só verificação rápida)
        )

        if not payload:
            logger.debug("Pre-check: payload vazio, assumindo sem conteúdo")
            return False, "none", 0

        logger.info(
            "🤖 Pre-check (%s): resposta recebida -> %s",
            cheap_model,
            json.dumps(payload, ensure_ascii=False),
        )

        has_content = payload.get("has_content")
        content_type = payload.get("content_type", "none")
        count = payload.get("count", 1)  # Padrão 1 se não especificado

        logger.info(
            "Pre-check resumo → has_content=%s | type=%s | count=%s",
            has_content,
            content_type,
            count,
        )

        # Se has_content é False ou content_type é text_only/none, não tem conteúdo útil
        if has_content is False or content_type in ("text_only", "none"):
            logger.info("Pre-check LLM barata: SEM conteúdo útil (has_content=%s, type=%s, count=%s)", 
                       has_content, content_type, count)
            return False, str(content_type), 0
        
        # Se has_content é True ou content_type é table/chart/mixed, tem conteúdo
        if has_content is True or content_type in ("table", "chart", "mixed"):
            logger.info("Pre-check LLM barata: TEM conteúdo útil (has_content=%s, type=%s, count=%s)", 
                       has_content, content_type, count)
            return True, str(content_type), int(count) if isinstance(count, (int, float)) else 1
        
        # Caso ambíguo: prossegue (não bloqueia)
        logger.warning("Pre-check LLM barata: resposta ambígua (has_content=%s, type=%s, count=%s), prosseguindo", 
                      has_content, content_type, count)
        return True, str(content_type), int(count) if isinstance(count, (int, float)) else 1
    except Exception as e:
        logger.warning("Erro no pre-check com LLM barata: %s. Prosseguindo com GPT-5.", e)
        return True, "unknown", 1  # Em caso de erro, prossegue (não bloqueia)


def call_openai_vision_json(
    image_path: Path,
    model: str = "gpt-5",
    api_key: Optional[str] = None,
    locale: str = "pt-BR",
    azure_endpoint: Optional[str] = None,
    azure_api_version: Optional[str] = None,
    provider: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    instructions: Optional[str] = None,
    max_retries: int = 2,
) -> Optional[dict]:
    """Chama um modelo de visão com retorno JSON.

    - Por padrão usa OpenAI (public). 
    - Se `azure_endpoint` (ou env AZURE_OPENAI_ENDPOINT) estiver definido, usa Azure OpenAI. 
      Em Azure, o `model` deve ser o NOME DO DEPLOYMENT.
    - Se `provider="openrouter"` (ou env OPENROUTER_API_KEY), usa OpenRouter.
    """
    # Ensure .env is loaded if present
    load_dotenv()
    
    # Determinar provider
    if provider is None:
        # Auto-detectar baseado em variáveis de ambiente ou parâmetros
        if openrouter_api_key or os.getenv("OPENROUTER_API_KEY"):
            provider = "openrouter"
        elif azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"):
            provider = "azure"
        else:
            provider = "openai"
    
    # Limpa temporariamente variáveis de proxy para evitar conflitos com httpx
    old_proxy = os.environ.pop("HTTP_PROXY", None)
    old_https_proxy = os.environ.pop("HTTPS_PROXY", None)
    old_all_proxy = os.environ.pop("ALL_PROXY", None)
    
    try:
        if provider == "openrouter":
            openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
            if not openrouter_api_key:
                raise RuntimeError("Defina OPENROUTER_API_KEY para usar OpenRouter.")
            logger.info("Chamando OpenRouter modelo=%s", model)
            http_client = httpx.Client(timeout=180.0)  # 3 minutos para imagens grandes
            client = OpenAI(
                api_key=openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                http_client=http_client,
            )
        elif provider == "azure":
            azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("Defina AZURE_OPENAI_API_KEY para usar Azure OpenAI.")
            if not azure_endpoint:
                raise RuntimeError("Defina AZURE_OPENAI_ENDPOINT para usar Azure OpenAI.")
            azure_api_version = azure_api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")
            logger.info("Chamando Azure OpenAI deployment=%s endpoint=%s", model, azure_endpoint)
            http_client = httpx.Client(timeout=180.0)  # 3 minutos para imagens grandes
            client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=azure_api_version,
                http_client=http_client,
            )
        else:  # provider == "openai" ou padrão
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("Defina OPENAI_API_KEY, AZURE_OPENAI_API_KEY ou OPENROUTER_API_KEY para usar o fallback LLM.")
            logger.info("Chamando OpenAI público modelo=%s", model)
            http_client = httpx.Client(timeout=180.0)  # 3 minutos para imagens grandes
            client = OpenAI(api_key=api_key, http_client=http_client)
    finally:
        # Restaura variáveis de ambiente
        if old_proxy:
            os.environ["HTTP_PROXY"] = old_proxy
        if old_https_proxy:
            os.environ["HTTPS_PROXY"] = old_https_proxy
        if old_all_proxy:
            os.environ["ALL_PROXY"] = old_all_proxy

    extra = f"\nIdioma dos rótulos de saída: {locale}. \nFormato: JSON puro, sem markdown."
    if instructions:
        extra += f"\nTarefa: {instructions.strip()}"
    
    data_url = _img_to_data_url(image_path)
    
    # Retry logic
    for attempt in range(max_retries + 1):
        try:
            prompt = SYSTEM_MSG + extra
            
            # Se for uma retry, adiciona feedback sobre o erro
            if attempt > 0:
                prompt += f"\n\n⚠️ ATENÇÃO: Tentativa {attempt + 1}. A resposta anterior estava incompleta ou inválida. Por favor, retorne um JSON COMPLETO e VÁLIDO com TODOS os dados visíveis na imagem."
            
            msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
            
            # GPT-5 só aceita temperature=1 (padrão)
            temp = 1 if "gpt-5" in model.lower() else (0.2 if attempt == 0 else 0.3)
            
            resp = client.chat.completions.create(
                model=model,
                temperature=temp,
                messages=[msg],
                response_format={"type": "json_object"},
            )
            
            txt = resp.choices[0].message.content
            if not txt:
                logger.warning("Resposta vazia da LLM na tentativa %s", attempt + 1)
                continue
            
            try:
                payload = json.loads(txt)
                
                # Valida o payload
                valid, msg_error = _validate_payload(payload)
                if valid:
                    logger.info("JSON válido obtido na tentativa %s", attempt + 1)
                    return payload
                else:
                    logger.warning("Validação falhou na tentativa %s: %s", attempt + 1, msg_error)
                    if attempt == max_retries:
                        # Última tentativa, retorna mesmo inválido para logging
                        return payload
            except json.JSONDecodeError as e:
                logger.warning("Erro ao parsear JSON na tentativa %s: %s", attempt + 1, e)
                if attempt == max_retries:
                    return None
        
        except Exception as e:
            logger.exception("Erro na chamada à LLM na tentativa %s", attempt + 1)
            if attempt == max_retries:
                raise
    
    return None


def _validate_precheck_payload(payload: dict) -> Tuple[bool, str]:
    """Valida payload do pre-check (formato diferente de extração)."""
    if not payload:
        return False, "Payload vazio"
    
    # Pre-check tem formato: {has_content, content_type, count}
    if "has_content" in payload and "content_type" in payload:
        return True, "OK"
    
    return False, "Formato de pre-check inválido"


def _validate_payload(payload: dict) -> Tuple[bool, str]:
    """Valida se o payload JSON está completo e bem formado (extração de tabelas/gráficos)."""
    if not payload:
        return False, "Payload vazio"
    
    # Se é pre-check, usa validação específica
    if "has_content" in payload or "content_type" in payload:
        return _validate_precheck_payload(payload)
    
    t = payload.get("type")
    if not t:
        return False, "Campo 'type' ausente"
    
    if t == "table":
        # NOVO: Aceita formato HTML ou formato JSON legado
        if payload.get("format") == "html":
            # Formato HTML: valida campo 'html'
            html = payload.get("html")
            if not html or not isinstance(html, str) or len(html.strip()) < 10:
                return False, "Campo 'html' ausente ou inválido"
            if "<table" not in html.lower():
                return False, "HTML não contém <table>"
            return True, "OK"
        else:
            # Formato JSON legado: valida campo 'table' com 'rows'
            table = payload.get("table", {})
            if not table:
                return False, "Campo 'table' ausente"
            rows = table.get("rows")
            if not rows or not isinstance(rows, list) or len(rows) == 0:
                return False, "Campo 'rows' vazio ou inválido"
            # Verifica se as linhas têm conteúdo
            if not any(any(str(cell).strip() for cell in row) for row in rows):
                return False, "Todas as linhas estão vazias"
            return True, "OK"
    
    elif t == "table_set":
        tables = payload.get("tables")
        if not tables or not isinstance(tables, list):
            return False, "Campo 'tables' ausente ou inválido"
        for idx, entry in enumerate(tables, start=1):
            # NOVO: Aceita formato HTML ou formato JSON legado
            if (entry or {}).get("format") == "html":
                # Formato HTML: valida campo 'html'
                html = (entry or {}).get("html")
                if not html or not isinstance(html, str) or len(html.strip()) < 10:
                    return False, f"Entrada {idx} com 'format': 'html' mas sem 'html' válido"
                # HTML válido se contém <table>
                if "<table" not in html.lower():
                    return False, f"Entrada {idx}: HTML não contém <table>"
            else:
                # Formato JSON legado: valida campo 'table' com 'rows'
                table = (entry or {}).get("table")
                if not table:
                    return False, f"Entrada {idx} sem campo 'table' ou 'html'"
                rows = table.get("rows")
                if not rows or not isinstance(rows, list):
                    return False, f"Tabela {idx} sem 'rows'"
        return True, "OK"
    
    elif t == "chart":
        chart = payload.get("chart", {})
        if not chart:
            return False, "Campo 'chart' ausente"
        
        # Valida estrutura ternária (triângulo de textura, etc)
        ternary = chart.get("ternary")
        if ternary:
            if not isinstance(ternary, dict):
                return False, "Campo 'ternary' com formato inválido"
            
            # Formato 1: ternary.axes + ternary.regions
            axes = ternary.get("axes")
            regions = ternary.get("regions")
            if axes or regions:
                return True, "OK"
            
            # Formato 2: ternary.a/b/c + chart.series ou chart.regions
            has_abc = ternary.get("a") or ternary.get("b") or ternary.get("c")
            series = chart.get("series")
            regions = chart.get("regions") or ternary.get("regions")
            if has_abc and (series is not None or regions):
                return True, "OK"
            
            # Formato 3: só os eixos a/b/c (sem series)
            if has_abc:
                return True, "OK"
            
            return False, "Gráfico ternário sem estrutura reconhecida"
        
        # Valida estrutura x/series
        if chart.get("x"):
            x_vals = chart.get("x", {}).get("values", [])
            series = chart.get("series", [])
            if not x_vals or not series:
                return False, "Gráfico com x ou series vazios"
            if not isinstance(series, list) or len(series) == 0:
                return False, "Series inválido ou vazio"
            # Verifica se pelo menos uma série tem valores
            has_values = any(s.get("values") for s in series if isinstance(s, dict))
            if not has_values:
                return False, "Nenhuma série contém valores"
            return True, "OK"
        
        # Valida estrutura labels/series alternativa
        labels = chart.get("labels")
        series_as_rows = chart.get("series")
        if labels and series_as_rows:
            if not isinstance(labels, list) or not isinstance(series_as_rows, list):
                return False, "Labels ou series com formato inválido"
            if len(series_as_rows) == 0:
                return False, "Series vazio"
            return True, "OK"
        
        return False, "Estrutura de gráfico não reconhecida"
    
    return False, f"Tipo '{t}' não reconhecido"


def to_table_from_llm_payload(payload: dict) -> Optional[List[List[str]]]:
    if not payload:
        logger.warning("Payload vazio")
        return None
    
    # Valida antes de processar
    valid, msg = _validate_payload(payload)
    if not valid:
        logger.warning("Validação falhou: %s", msg)
        return None
    
    t = payload.get("type")
    if t == "table":
        rows = payload.get("table", {}).get("rows") or []
        headers = payload.get("table", {}).get("headers")
        if headers:
            result = [list(map(str, headers))] + [[str(c) for c in r] for r in rows]
        else:
            result = [[str(c) for c in r] for r in rows]
        
        # Remove linhas completamente vazias
        result = [row for row in result if any(cell.strip() for cell in row)]
        
        if not result or len(result) < 1:
            logger.warning("Tabela resultante vazia após limpeza")
            return None
        
        return result
    
    if t == "chart":
        chart = payload.get("chart", {})
        
        # Case 0: Ternary diagram (triângulo de textura, etc)
        ternary = chart.get("ternary")
        if ternary and isinstance(ternary, dict):
            # Formato 1: ternary.axes (dict de eixos) + ternary.regions (lista)
            axes = ternary.get("axes", {})
            regions = ternary.get("regions", [])
            
            if axes and isinstance(axes, dict):
                # Cria cabeçalho com os 3 eixos
                axis_names = list(axes.keys())
                header = ["Região/Classe"] + [axes[ax].get("label", ax) for ax in axis_names]
                
                table: List[List[str]] = [header]
                
                # Adiciona regiões
                for region in regions:
                    if isinstance(region, dict):
                        name = region.get("name", "")
                        # Se a região tiver valores dos eixos, adiciona
                        row = [name]
                        for ax in axis_names:
                            val = region.get(ax, "")
                            row.append(str(val) if val else "")
                        table.append(row)
                
                # Se não tem regiões, pelo menos mostra os eixos e seus ticks
                if len(table) == 1:
                    for ax_name in axis_names:
                        ax_data = axes.get(ax_name, {})
                        label = ax_data.get("label", ax_name)
                        ticks = ax_data.get("ticks", [])
                        if ticks:
                            tick_str = f"{min(ticks)}-{max(ticks)}"
                        else:
                            tick_str = ""
                        table.append([label, tick_str, "", ""])
                
                return table if len(table) > 1 else None
            
            # Formato 2: ternary.a/b/c + chart.series (estrutura alternativa do GPT)
            a_axis = ternary.get("a")
            b_axis = ternary.get("b")
            c_axis = ternary.get("c")
            
            if a_axis or b_axis or c_axis:
                # Extrai labels dos eixos
                labels = []
                for axis in [a_axis, b_axis, c_axis]:
                    if axis and isinstance(axis, dict):
                        label = axis.get("label", "")
                        labels.append(label)
                
                # Extrai ranges dos eixos (dos ticks/values)
                axis_ranges = []
                for axis_name in ['a', 'b', 'c']:
                    axis_obj = ternary.get(axis_name)
                    if axis_obj and isinstance(axis_obj, dict):
                        ticks = axis_obj.get("ticks", []) or axis_obj.get("values", [])
                        if ticks:
                            try:
                                nums = [float(v) for v in ticks if str(v).replace('.','').replace('-','').isdigit()]
                                if nums:
                                    range_str = f"{min(nums):.0f}-{max(nums):.0f}%"
                                else:
                                    range_str = f"{ticks[0]}-{ticks[-1]}%"
                            except:
                                range_str = f"{ticks[0]}-{ticks[-1]}%"
                        else:
                            range_str = "0-100%"
                        axis_ranges.append(range_str)
                    else:
                        axis_ranges.append("-")
                
                # Cabeçalho com os ranges dos eixos
                header_with_ranges = [f"{label}\n({range_val})" if range_val != "-" else label 
                                     for label, range_val in zip(labels, axis_ranges)]
                header = ["Região/Classe"] + header_with_ranges
                table: List[List[str]] = [header]
                
                # Pega series do chart (fora do ternary)
                series = chart.get("series", [])
                
                # Pega regions do chart (pode estar em chart.regions ou ternary.regions)
                regions = chart.get("regions", []) or ternary.get("regions", [])
                
                # Prioriza regions (classes/regiões do diagrama)
                if regions and isinstance(regions, list) and len(regions) > 0:
                    for region in regions:
                        if isinstance(region, dict):
                            name = region.get("name", "")
                            # Tenta pegar valores específicos da região (se existirem)
                            vals = []
                            for axis_name in ['a', 'b', 'c']:
                                val = region.get(axis_name, "")
                                if not val:
                                    # Tenta com o nome completo do label
                                    axis_obj = ternary.get(axis_name)
                                    if axis_obj:
                                        axis_label = axis_obj.get("label", "")
                                        val = region.get(axis_label, "")
                                vals.append(str(val) if val else "Varia")
                            
                            row = [name] + vals
                            table.append(row)
                # Se não tem regions, tenta series
                elif series and isinstance(series, list) and len(series) > 0:
                    for s in series:
                        if isinstance(s, dict):
                            name = s.get("name", "")
                            values = s.get("values", [])
                            # Se não tem values, deixa vazio
                            row = [name] + ["Varia" for _ in labels]
                            table.append(row)
                
                # Se não tem nem regions nem series, mostra apenas os eixos e ranges
                if len(table) == 1:
                    for axis, label in zip([a_axis, b_axis, c_axis], labels):
                        if axis and isinstance(axis, dict):
                            # Tenta 'values' primeiro, depois 'ticks'
                            vals = axis.get("values", []) or axis.get("ticks", [])
                            if vals:
                                try:
                                    nums = [float(v) for v in vals if str(v).replace('.','').replace('-','').isdigit()]
                                    if nums:
                                        range_str = f"{min(nums):.0f}-{max(nums):.0f}"
                                    else:
                                        range_str = ", ".join(str(v) for v in vals[:3])
                                except:
                                    range_str = ", ".join(str(v) for v in vals[:3])
                            else:
                                range_str = ""
                            table.append([label, range_str, "", ""])
                
                return table if len(table) > 1 else None
        
        # Case 1: structured x/series arrays (OpenAI default)
        if chart.get("x") and isinstance(chart.get("series"), list) and chart.get("series"):
            x_vals = chart.get("x", {}).get("values", [])
            series = chart.get("series", [])
            x_label = chart.get("x", {}).get("label") or "x"
            x_unit = chart.get("x", {}).get("unit", "")
            if x_unit:
                x_label = f"{x_label} ({x_unit})"
            
            header = [x_label] + [s.get("name") or f"serie_{i+1}" for i, s in enumerate(series)]
            table: List[List[str]] = [header]
            max_len = max(len(x_vals), *(len(s.get("values", [])) for s in series if isinstance(s, dict))) if series else len(x_vals)
            
            for i in range(max_len):
                row = []
                row.append(str(x_vals[i]) if i < len(x_vals) else "")
                for s in series:
                    if isinstance(s, dict):
                        vals = s.get("values", [])
                        v = vals[i] if i < len(vals) else None
                        row.append("" if v is None else str(v))
                    else:
                        row.append("")
                table.append(row)
            
            # Remove linhas vazias
            table = [row for row in table if any(cell.strip() for cell in row)]
            return table if len(table) > 1 else None
        
        # Case 2: LLM returned "labels" + list of dicts per row
        labels = chart.get("labels")
        series_as_rows = chart.get("series")
        if isinstance(labels, list) and isinstance(series_as_rows, list) and series_as_rows:
            header = [str(h) for h in labels]
            table = [header]
            for row in series_as_rows:
                if isinstance(row, dict):
                    ordered = []
                    for h in header:
                        ordered.append(str(row.get(h, "")))
                    table.append(ordered)
            return table if len(table) > 1 else None
        
        return None
    
    return None

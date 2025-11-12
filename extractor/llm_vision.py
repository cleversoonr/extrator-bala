from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
import cv2
import numpy as np
from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv

from .logging_utils import get_logger


logger = get_logger(__name__)


SYSTEM_MSG = (
    "Você é um extrator de dados de gráficos e tabelas. Retorne APENAS JSON válido, sem texto antes/depois. "
    "Se for uma tabela, retorne {type:'table', table:{rows:[...]}}. "
    "Se for um gráfico (linha, barra, dispersão, ternário etc.), retorne {type:'chart', chart:{...}} com as séries numéricas. "
    "Para eixos categóricos (ex. datas), mantenha os rótulos como strings em x.values. "
    "\n"
    "⚠️ PRINCÍPIO FUNDAMENTAL: Extraia EXATAMENTE como está na imagem. NÃO force padrões. "
    "Sua única fonte de verdade é a IMAGEM + LEGENDA que você está vendo. "
    "\n"
    "🔄 IMAGEM GIRADA: Se a imagem estiver rotacionada/girada, LEIA NORMALMENTE ajustando mentalmente a orientação. "
    "Identifique headers/eixos primeiro para determinar a orientação correta, então extraia célula por célula. "
    "NÃO mencione que está girada - apenas retorne os dados corretos. "
    "\n"
    "Para GRÁFICOS: "
    "- Leia TODOS os valores numéricos visíveis dos pontos. "
    "- Use null apenas se um ponto específico realmente não existe para aquela coordenada X. "
    "\n"
    "Para TABELAS: "
    "\n"
    "🔴 **PROCEDIMENTO OBRIGATÓRIO - LEIA CÉLULA POR CÉLULA**: "
    "\n"
    "1️⃣ Vá para a PRIMEIRA célula da linha "
    "2️⃣ Amplie zoom mental MÁXIMO naquela célula "
    "3️⃣ LEIA o texto/símbolo/número ESCRITO na célula "
    "4️⃣ Transcreva EXATAMENTE o que você VIU escrito "
    "5️⃣ Vá para a PRÓXIMA célula → REPITA desde passo 2 "
    "6️⃣ Complete TODA a linha antes de passar para a próxima "
    "\n"
    "🔄 **IMPORTANTE - PRESERVE A ESTRUTURA ORIGINAL**: "
    "- Se a coluna com nomes das linhas está à DIREITA na imagem → mantenha à DIREITA no HTML "
    "- Se a coluna com nomes das linhas está à ESQUERDA na imagem → mantenha à ESQUERDA no HTML "
    "- NÃO reorganize a tabela - transcreva EXATAMENTE na ordem que você vê "
    "- O sistema vai ajustar a ordem depois se necessário "
    "\n"
    "📝 **FOCO NO CONTEÚDO, NÃO NA APARÊNCIA**: "
    "- Ignore cores, ignore design - LEIA O TEXTO "
    "- Se vê 'C' escrito → `<td>C</td>` "
    "- Se vê 'CL' escrito → `<td>CL</td>` (NÃO confunda com 'C' ou 'I'!) "
    "- Se vê 'I' escrito → `<td>I</td>` "
    "- Se vê número → transcreva o número "
    "- Se vê texto → transcreva o texto "
    "- Se célula vazia (sem nada escrito) → `<td></td>` "
    "\n"
    "⚠️  **ATENÇÃO ESPECIAL - CÉLULAS COM 'CL'**: "
    "- 'CL' são DUAS letras juntas: 'C' + 'L' "
    "- Se ver só 'C' (uma letra sozinha) → escreva 'C' "
    "- Se ver 'CL' (duas letras) → escreva 'CL' "
    "- Se ver 'I' (uma letra) → escreva 'I' "
    "- AMPLIE o zoom ao MÁXIMO para ver se é 'C' ou 'CL' "
    "- NÃO confunda 'CL' com 'C' nem com 'I' "
    "\n"
    "⚠️  **ATENÇÃO CRÍTICA**: "
    "- Texto pode ser EXTREMAMENTE pequeno (2-3 pixels) "
    "- Amplie mentalmente SEMPRE antes de ler "
    "- Se não consegue ver, AMPLIE MAIS e tente novamente "
    "- NUNCA assuma conteúdo - SEMPRE leia "
    "- Trabalhe DEVAGAR, uma célula de cada vez "
    "\n"
    "🚫 **PROIBIDO**: "
    "- Assumir conteúdo baseado em cores/padrões "
    "- Assumir simetria (célula [i,j] ≠ célula [j,i]) "
    "- Copiar linha/coluna inteira "
    "- Deixar tabela vazia sem ler todas as células "
    "- Criar linhas vazias com `<td colspan=\"X\"></td>` para separar seções "
    "- Juntar tabelas FISICAMENTE SEPARADAS em um único HTML "
    "\n"
    "📊 **TABELAS MÚLTIPLAS**: "
    "Se a página tem VÁRIAS tabelas fisicamente SEPARADAS (com espaço/borda entre elas): "
    "- Identifique quantas tabelas distintas existem "
    "- Crie uma entrada SEPARADA para cada tabela no JSON "
    "- NÃO junte tabelas diferentes com linhas vazias "
    "- Sinais: espaço vertical, bordas completas, headers totalmente diferentes "
    "\n"
    "🏗️ **ESTRUTURA HTML CORRETA**: "
    "- Use `<thead>` para cabeçalhos (pode ter múltiplas linhas `<tr>`) "
    "- Use `<tbody>` para dados "
    "- Use `colspan=\"N\"` quando célula ocupa N colunas "
    "- Use `rowspan=\"N\"` quando célula ocupa N linhas "
    "- NÃO invente linhas vazias - cada `<tr>` deve ter conteúdo real "
    "\n"
    "✅ **PROCESSO CORRETO**: "
    "Célula 1 → amplio → vejo 'C' → `<td>C</td>` "
    "Célula 2 → amplio → vejo '123' → `<td>123</td>` "
    "Célula 3 → amplio → vejo 'CL' → `<td>CL</td>` "
    "Célula 4 → amplio → não vejo nada → `<td></td>` "
    "Célula 5 → amplio → vejo 'texto' → `<td>texto</td>`"
)

PRECHECK_PROMPT = (
    "Analise esta imagem DETALHADAMENTE e identifique as características do conteúdo.\n"
    "Seja ESPECÍFICO e PRECISO - essas informações serão usadas para gerar instruções de extração.\n"
    "\n"
    "Retorne JSON:\n"
    "{\n"
    "  'has_content': true/false,\n"
    "  'content_type': 'table'|'chart'|'mixed'|'text_only'|'none',\n"
    "  'count': número,\n"
    "  'rotation': 0|90|180|270,\n"
    "  'elements': [\n"
    "    {\n"
    "      'type': 'table'|'chart',\n"
    "      'description': 'Descrição específica do elemento',\n"
    "      'structure': {\n"
    "        // Para TABELAS:\n"
    "        'table_structure': 'compatibility_matrix'|'data_table'|'list'|'comparison'|'other',\n"
    "        'rows': número aproximado,\n"
    "        'columns': número aproximado,\n"
    "        'has_header': true/false,\n"
    "        'has_colors': true/false,\n"
    "        'color_meaning': 'descrição do que as cores representam (se aplicável)',\n"
    "        'has_merged_cells': true/false,  ⚠️ Olhe: células que ocupam MAIS de 1 coluna/linha\n"
    "        'merged_cells_location': 'header'|'body'|'both'|'none',\n"
    "        'diagonal_empty': true/false,  ⚠️ CRÍTICO para matrizes: se células da diagonal (onde linha = coluna) estão VAZIAS/CINZAS sem texto\n"
    "        'cell_content_type': 'symbols'|'numbers'|'text'|'mixed',\n"
    "        'cell_content_description': 'o que está escrito nas células',\n"
    "        'has_legend': true/false,\n"
    "        'legend_content': 'conteúdo da legenda (se tiver)',\n"
    "        // Para GRÁFICOS:\n"
    "        'chart_type': 'bar'|'line'|'scatter'|'pie'|'ternary'|'heatmap'|'other',\n"
    "        'has_multiple_series': true/false,\n"
    "        'axis_types': 'numeric'|'categorical'|'date'|'mixed',\n"
    "        'has_grid': true/false,\n"
    "        'data_points_visible': true/false\n"
    "      }\n"
    "    }\n"
    "  ]\n"
    "}\n"
    "\n"
    "**INSTRUÇÕES CRÍTICAS:**\n"
    "\n"
    "1. **count**: Número EXATO de elementos (tabelas + gráficos)\n"
    "   ⚠️ ATENÇÃO: Se vê MÚLTIPLAS tabelas FISICAMENTE SEPARADAS (com espaço/borda entre elas):\n"
    "   - Cada tabela separada = 1 elemento no count\n"
    "   - Exemplo: 2 tabelas separadas verticalmente = count: 2\n"
    "   - Sinais de separação: espaço vertical significativo, bordas completas, headers diferentes\n"
    "\n"
    "2. **rotation**: Olhe o TÍTULO PRINCIPAL da página (ex: 'Anexo 21', 'Compatibilidade de fertilizantes').\n"
    "   NÃO olhe headers de tabela/colunas (podem estar na vertical por design).\n"
    "   \n"
    "   Em qual ÂNGULO está o TÍTULO PRINCIPAL ATUALMENTE?\n"
    "   - Título horizontal (normal, legível)? → rotation = 0\n"
    "   - Título virado 90° (à direita)? → rotation = 90\n"
    "   - Título de cabeça pra baixo? → rotation = 180\n"
    "   - Título virado 270° (à esquerda)? → rotation = 270\n"
    "   \n"
    "   ⚠️ Informe a POSIÇÃO ATUAL do título (onde está agora), não a correção necessária.\n"
    "\n"
    "3. **elements**: Array com CADA elemento detectado\n"
    "   - Se tem 2 tabelas SEPARADAS → 2 objetos no array (mesmo que compartilhem primeira coluna)\n"
    "   - Se tem 1 tabela + 1 gráfico → 2 objetos no array\n"
    "   - Cada objeto deve ter descrição ESPECÍFICA daquele elemento\n"
    "\n"
    "4. **description**: Descreva o que VÊ na imagem (ex: 'Matriz 21x21 com células coloridas verde/amarelo/vermelho')\n"
    "\n"
    "5. **color_meaning**: Se células têm cores, descreva o que representam baseado na legenda ou contexto visual\n"
    "\n"
    "6. **cell_content_description**: Descreva o que está ESCRITO nas células (ex: 'Letras C, CL e I', 'Números decimais', 'Nomes de fertilizantes')\n"
    "\n"
    "7. **legend_content**: Se tem legenda, transcreva o conteúdo (ex: 'C = Compatível, CL = Compatibilidade Limitada, I = Incompatível')\n"
    "\n"
    "8. **diagonal_empty** (CRÍTICO para MATRIZES): Em matrizes onde linhas e colunas têm os MESMOS nomes (matriz de compatibilidade):\n"
    "   - Olhe as células onde linha = coluna (diagonal principal)\n"
    "   - Essas células estão VAZIAS/CINZAS sem nenhum texto/símbolo? → diagonal_empty = true\n"
    "   - Têm texto/símbolo (mesmo que seja '-' ou outro)? → diagonal_empty = false\n"
    "\n"
    "**EXEMPLO 1 (tabela única):**\n"
    "{\n"
    "  'has_content': true,\n"
    "  'content_type': 'table',\n"
    "  'count': 1,\n"
    "  'rotation': 0,\n"
    "  'elements': [{\n"
    "    'type': 'table',\n"
    "    'description': 'Matriz 21x21 simétrica com células coloridas em verde, amarelo e vermelho',\n"
    "    'structure': {\n"
    "      'table_structure': 'compatibility_matrix',\n"
    "      'rows': 21,\n"
    "      'columns': 21,\n"
    "      'has_header': true,\n"
    "      'has_colors': true,\n"
    "      'color_meaning': 'Verde = compatível, Amarelo = compatibilidade limitada, Vermelho = incompatível',\n"
    "      'has_merged_cells': false,\n"
    "      'diagonal_empty': true,\n"
    "      'cell_content_type': 'symbols',\n"
    "      'cell_content_description': 'Letras C (células verdes), CL (células amarelas), I (células vermelhas)',\n"
    "      'has_legend': true,\n"
    "      'legend_content': '[C] Compatíveis, [CL] Compatibilidade limitada, [I] Incompatíveis'\n"
    "    }\n"
    "  }]\n"
    "}\n"
    "\n"
    "**EXEMPLO 2 (múltiplas tabelas separadas):**\n"
    "{\n"
    "  'has_content': true,\n"
    "  'content_type': 'table',\n"
    "  'count': 2,\n"
    "  'rotation': 0,\n"
    "  'elements': [\n"
    "    {\n"
    "      'type': 'table',\n"
    "      'description': 'Tabela superior com dados de pH, MO, P, K, Ca, Mg, Al, etc.',\n"
    "      'structure': {\n"
    "        'table_structure': 'data_table',\n"
    "        'rows': 2,\n"
    "        'columns': 14,\n"
    "        'has_header': true,\n"
    "        'has_colors': true,\n"
    "        'cell_content_type': 'numbers',\n"
    "        'cell_content_description': 'Valores numéricos de análise de solo'\n"
    "      }\n"
    "    },\n"
    "    {\n"
    "      'type': 'table',\n"
    "      'description': 'Tabela inferior com micronutrientes S, Zn, B, Cu, Mn, Fe e relações',\n"
    "      'structure': {\n"
    "        'table_structure': 'data_table',\n"
    "        'rows': 2,\n"
    "        'columns': 12,\n"
    "        'has_header': true,\n"
    "        'cell_content_type': 'numbers',\n"
    "        'cell_content_description': 'Valores numéricos de micronutrientes'\n"
    "      }\n"
    "    }\n"
    "  ]\n"
    "}"
)


def _img_to_data_url(path: Path, max_size_mb: float = 15.0) -> str:
    """
    Converte imagem para data URL com verificação de tamanho.
    Se a imagem em base64 exceder max_size_mb, faz downscale automático.
    
    Args:
        path: Caminho para a imagem
        max_size_mb: Tamanho máximo em MB (padrão 15MB, Azure OpenAI aceita até 20MB)
    
    Returns:
        Data URL da imagem (possivelmente redimensionada)
    """
    max_size_bytes = int(max_size_mb * 1024 * 1024)
    
    # Tenta converter diretamente primeiro
    img_bytes = path.read_bytes()
    b64 = base64.b64encode(img_bytes).decode("ascii")
    
    # Base64 adiciona ~33% de overhead, então o tamanho final é maior que o arquivo original
    b64_size = len(b64)
    
    if b64_size <= max_size_bytes:
        # Imagem OK, retorna direto
        return f"data:image/{path.suffix[1:] or 'png'};base64,{b64}"
    
    # Imagem muito grande, precisa fazer downscale
    logger.warning(
        "⚠️  Imagem muito grande: %.1f MB em base64 (limite %.1f MB). Fazendo downscale...",
        b64_size / (1024 * 1024),
        max_size_mb
    )
    
    # Carrega imagem com OpenCV
    img = cv2.imread(str(path))
    if img is None:
        logger.error("Falha ao carregar imagem para downscale, usando original")
        return f"data:image/{path.suffix[1:] or 'png'};base64,{b64}"
    
    h, w = img.shape[:2]
    original_size = (w, h)
    
    # Calcula fator de redução necessário
    # Como base64 tem overhead, precisamos reduzir mais que a proporção direta
    reduction_factor = (max_size_bytes / b64_size) ** 0.5  # Raiz quadrada porque área é w*h
    
    # Aplica redução iterativa até ficar abaixo do limite
    quality = 85
    max_attempts = 5
    
    for attempt in range(max_attempts):
        # Calcula novo tamanho
        new_w = int(w * reduction_factor)
        new_h = int(h * reduction_factor)
        
        # Garante mínimo de 800px no lado menor para manter legibilidade
        min_side = min(new_w, new_h)
        if min_side < 800:
            scale = 800 / min_side
            new_w = int(new_w * scale)
            new_h = int(new_h * scale)
        
        # Redimensiona
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Codifica como JPEG com qualidade ajustável
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        success, buffer = cv2.imencode('.jpg', resized, encode_params)
        
        if not success:
            logger.error("Falha ao encodar imagem redimensionada")
            break
        
        # Converte para base64
        b64_new = base64.b64encode(buffer.tobytes()).decode("ascii")
        b64_new_size = len(b64_new)
        
        logger.info(
            "📐 Tentativa %d: %dx%d → %dx%d | %.1f MB → %.1f MB (qualidade %d%%)",
            attempt + 1,
            w, h, new_w, new_h,
            b64_size / (1024 * 1024),
            b64_new_size / (1024 * 1024),
            quality
        )
        
        if b64_new_size <= max_size_bytes:
            # Sucesso!
            logger.info("✅ Downscale concluído: %dx%d → %dx%d", w, h, new_w, new_h)
            return f"data:image/jpeg;base64,{b64_new}"
        
        # Ainda muito grande, reduz mais
        reduction_factor *= 0.9  # Reduz mais 10%
        quality = max(60, quality - 10)  # Reduz qualidade
    
    # Se chegou aqui, não conseguiu reduzir o suficiente
    # Retorna a última versão reduzida mesmo que ainda grande
    logger.error(
        "❌ Não foi possível reduzir imagem para %.1f MB após %d tentativas. Usando última versão (%.1f MB)",
        max_size_mb,
        max_attempts,
        b64_new_size / (1024 * 1024)
    )
    return f"data:image/jpeg;base64,{b64_new}"


def quick_precheck_with_cheap_llm(
    image_path: Path,
    cheap_model: str,
    cheap_provider: Optional[str],
    openrouter_api_key: Optional[str],
    *,
    api_key: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    azure_api_version: Optional[str] = None,
) -> Tuple[bool, str, int, int, dict]:
    """
    Verificação rápida com LLM barata: retorna se tem conteúdo útil + características.
    Retorna: (has_content: bool, content_type: str, count: int, rotation: int, characteristics: dict)
    count = quantas tabelas/gráficos distintos na página
    rotation = graus de rotação detectados (0, 90, 180, 270)
    characteristics = dict com tipo de tabela, complexidade, etc
    """
    try:
        logger.info(
            "⚡ Chamando pre-check LLM (%s via %s) para %s",
            cheap_model,
            cheap_provider or "default",
            image_path.name,
        )
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
            return False, "none", 0, 0, {}

        logger.info(
            "🤖 Pre-check (%s): resposta recebida",
            cheap_model,
        )

        has_content = payload.get("has_content")
        content_type = payload.get("content_type", "none")
        count = payload.get("count", 1)
        rotation = payload.get("rotation", 0)
        elements = payload.get("elements", [])
        
        # Log detalhado de cada elemento detectado
        if elements:
            for idx, elem in enumerate(elements, 1):
                elem_type = elem.get("type")
                description = elem.get("description", "")
                structure = elem.get("structure", {})
                
                logger.info(
                    "📊 Elemento %d/%d: %s - %s",
                    idx,
                    len(elements),
                    elem_type,
                    description[:80] + "..." if len(description) > 80 else description
                )
                
                if elem_type == "table":
                    logger.info(
                        "   └─ Estrutura: %s | %dx%d | Cores: %s | Legenda: %s",
                        structure.get("table_structure", "?"),
                        structure.get("rows", 0),
                        structure.get("columns", 0),
                        structure.get("has_colors", False),
                        structure.get("has_legend", False)
                    )
                elif elem_type == "chart":
                    logger.info(
                        "   └─ Tipo: %s | Séries múltiplas: %s",
                        structure.get("chart_type", "?"),
                        structure.get("has_multiple_series", False)
                    )
        
        # Compatibilidade com código antigo: cria dict 'characteristics' com primeiro elemento
        characteristics = {}
        if elements:
            first_elem = elements[0]
            characteristics = {
                "elements": elements,  # Array completo
                "description": first_elem.get("description", ""),
                **first_elem.get("structure", {})
            }
        
        logger.info(
            "Pre-check resumo → has_content=%s | type=%s | count=%s | rotation=%s°",
            has_content,
            content_type,
            count,
            rotation,
        )

        # Se has_content é False ou content_type é text_only/none, não tem conteúdo útil
        if has_content is False or content_type in ("text_only", "none"):
            logger.info("Pre-check: SEM conteúdo útil")
            return False, str(content_type), 0, int(rotation) if isinstance(rotation, (int, float)) else 0, {}
        
        # Se has_content é True ou content_type é table/chart/mixed, tem conteúdo
        if has_content is True or content_type in ("table", "chart", "mixed"):
            logger.info("Pre-check: TEM conteúdo útil")
            return (
                True, 
                str(content_type), 
                int(count) if isinstance(count, (int, float)) else 1, 
                int(rotation) if isinstance(rotation, (int, float)) else 0,
                characteristics
            )
        
        # Caso ambíguo: prossegue (não bloqueia)
        logger.warning("Pre-check: resposta ambígua, prosseguindo")
        return (
            True, 
            str(content_type), 
            int(count) if isinstance(count, (int, float)) else 1, 
            int(rotation) if isinstance(rotation, (int, float)) else 0,
            characteristics
        )
    except Exception as e:
        logger.warning("Erro no pre-check: %s. Prosseguindo.", e)
        return True, "unknown", 1, 0, {}  # Em caso de erro, prossegue


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
            http_client = httpx.Client(timeout=300.0)  # 5 minutos para páginas complexas
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
            http_client = httpx.Client(timeout=300.0)  # 5 minutos para páginas complexas
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
            http_client = httpx.Client(timeout=300.0)  # 5 minutos para páginas complexas
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
                
                # 🐛 DEBUG: Salva resposta RAW do LLM para debug
                try:
                    debug_path = image_path.parent / f"{image_path.stem}-llm-response.json"
                    debug_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
                    logger.info("💾 Resposta LLM salva em: %s", debug_path.name)
                except Exception as e:
                    logger.debug("Falha ao salvar debug JSON: %s", e)
                
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
                # Salva texto bruto se não for JSON válido
                try:
                    debug_path = image_path.parent / f"{image_path.stem}-llm-response-raw.txt"
                    debug_path.write_text(txt, encoding="utf-8")
                    logger.warning("💾 Texto bruto salvo em: %s", debug_path.name)
                except Exception:
                    pass
                if attempt == max_retries:
                    return None
        
        except Exception as e:
            # Tratamento especial para timeout
            if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                logger.error("⏱️  Timeout na tentativa %s/%s (página muito complexa)", attempt + 1, max_retries + 1)
                if attempt < max_retries:
                    logger.info("⏭️  Tentando novamente em 5 segundos...")
                    import time
                    time.sleep(5)  # Espera 5s antes de retry
                else:
                    logger.error("❌ Timeout após %s tentativas. Página será pulada.", max_retries + 1)
                    raise
            else:
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

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List

from rich import print
from rich.prompt import Confirm, Prompt
from rich.table import Table
from dotenv import load_dotenv

from .image_tables import ImageProcessingConfig, process_pdf_images
from .logging_utils import get_logger


logger = get_logger(__name__)


def _env_any(*names: str, default: str | None = None) -> str | None:
    for name in names:
        val = os.getenv(name)
        if val:
            return val
    return default


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _list_pdfs(docs_dir: Path) -> List[Path]:
    return sorted([p for p in docs_dir.glob("*.pdf") if p.is_file()])


def _choose_pdfs(pdfs: List[Path]) -> List[Path]:
    table = Table(title="PDFs encontrados em docs/")
    table.add_column("#", justify="right")
    table.add_column("Arquivo")
    for i, p in enumerate(pdfs, 1):
        table.add_row(str(i), p.name)
    print(table)
    choice = Prompt.ask("Quais arquivos processar? (enter=todos; ex: 1,3-5)", default="")
    if not choice.strip():
        return pdfs

    # parse ranges
    idxs: List[int] = []
    for part in choice.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            try:
                a_i = int(a)
                b_i = int(b)
            except ValueError:
                continue
            if a_i > b_i:
                a_i, b_i = b_i, a_i
            idxs.extend(list(range(a_i, b_i + 1)))
        else:
            try:
                idxs.append(int(part))
            except ValueError:
                pass
    sel = []
    for i in idxs:
        if 1 <= i <= len(pdfs):
            sel.append(pdfs[i - 1])
    return sel or pdfs


def _ask_pages() -> str | None:
    full = Confirm.ask("Processar o PDF inteiro?", default=True)
    if full:
        return None
    pages = Prompt.ask("Informe páginas (ex.: 34, 1-5, 10,12-15)")
    return pages.strip() or None


def _auto_detect_precheck_model(llm_cfg: dict) -> tuple:
    """
    Detecta automaticamente o melhor modelo para pre-check.
    PRE-CHECK É OBRIGATÓRIO e SEMPRE usa GPT-4.1 (Azure) quando disponível.
    
    Retorna: (model, provider, endpoint, api_version, api_key)
    """
    # Prioridade 1: GPT-4.1 (Azure) - SEMPRE preferido para pre-check
    if llm_cfg.get("provider") == "azure":
        az_pre_ep = _env_any("AZURE_OPENAI_PRECHECK_ENDPOINT", "AZURE_GPT41_ENDPOINT")
        az_pre_key = _env_any("AZURE_OPENAI_PRECHECK_API_KEY", "AZURE_GPT41_API_KEY")
        az_pre_dep = _env_any("AZURE_OPENAI_PRECHECK_DEPLOYMENT", "AZURE_GPT41_DEPLOYMENT", default="gpt-4o")
        az_pre_ver = _env_any("AZURE_OPENAI_PRECHECK_API_VERSION", "AZURE_GPT41_API_VERSION", default="2025-03-01-preview")
        
        if az_pre_ep and az_pre_key:
            logger.info("✅ Pre-check: GPT-4.1 (Azure) detectado automaticamente")
            return az_pre_dep, "azure", az_pre_ep, az_pre_ver, az_pre_key
    
    # Prioridade 2: OpenRouter com modelo barato
    if llm_cfg.get("provider") == "openrouter":
        logger.info("✅ Pre-check: GPT-4o-mini (OpenRouter) como fallback")
        return "openai/gpt-4o-mini", "openrouter", None, None, llm_cfg.get("api_key")
    
    # Prioridade 3: OpenAI com modelo barato
    if llm_cfg.get("provider") == "openai":
        logger.info("✅ Pre-check: GPT-4o-mini (OpenAI) como fallback")
        return "gpt-4o-mini", "openai", None, None, llm_cfg.get("api_key")
    
    # Fallback: Usa o mesmo modelo da extração (não ideal, mas funciona)
    logger.warning("⚠️  Pre-check: Usando mesmo modelo da extração (não otimizado)")
    return (
        llm_cfg.get("model", "gpt-5"),
        llm_cfg.get("provider"),
        llm_cfg.get("endpoint"),
        llm_cfg.get("api_version"),
        llm_cfg.get("api_key"),
    )


def _ask_extraction_model(llm_cfg: dict) -> tuple:
    """
    Pergunta qual modelo usar para extração (tabelas/gráficos).
    Retorna: (model, provider, endpoint, api_version, api_key)
    """
    print("\n[bold cyan]═══ Modelo para Extração ═══[/bold cyan]")
    print("O modelo de extração lê tabelas/gráficos e converte para JSON/HTML.")
    print("GPT-5 é mais preciso, GPT-4.1 é mais rápido/barato.")
    
    options = []
    
    # Opção 1: GPT-5 (padrão)
    options.append((
        "GPT-5 (mais preciso - recomendado)",
        llm_cfg.get("model", "gpt-5"),
        llm_cfg.get("provider"),
        llm_cfg.get("endpoint"),
        llm_cfg.get("api_version"),
        llm_cfg.get("api_key"),
    ))
    
    # Opção 2: GPT-4.1 (se disponível)
    if llm_cfg.get("provider") == "azure":
        az_pre_ep = _env_any("AZURE_OPENAI_PRECHECK_ENDPOINT", "AZURE_GPT41_ENDPOINT")
        az_pre_key = _env_any("AZURE_OPENAI_PRECHECK_API_KEY", "AZURE_GPT41_API_KEY")
        az_pre_dep = _env_any("AZURE_OPENAI_PRECHECK_DEPLOYMENT", "AZURE_GPT41_DEPLOYMENT", default="gpt-4.1")
        az_pre_ver = _env_any("AZURE_OPENAI_PRECHECK_API_VERSION", "AZURE_GPT41_API_VERSION", default="2025-03-01-preview")
        
        if az_pre_ep and az_pre_key:
            options.append(("GPT-4.1 (mais rápido/barato)", az_pre_dep, "azure", az_pre_ep, az_pre_ver, az_pre_key))
    
    for i, opt in enumerate(options, 1):
        print(f"{i}. {opt[0]}")
    
    choice = Prompt.ask("\n🧠 Escolha o modelo de extração", choices=[str(i) for i in range(1, len(options) + 1)], default="1")
    selected = options[int(choice) - 1]
    
    return selected[1], selected[2], selected[3], selected[4], selected[5]


def _detect_llm_config() -> dict:
    """Auto-detecta configuração LLM das variáveis de ambiente e permite escolha explícita."""
    az_ep = _env_any(
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_ENDPOINT_MAIN",
        "AZURE_GPT5_ENDPOINT",
    )
    az_key = _env_any(
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY_MAIN",
        "AZURE_GPT5_API_KEY",
    )
    az_ver = _env_any(
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_API_VERSION_MAIN",
        "AZURE_GPT5_API_VERSION",
        default="2025-03-01-preview",
    )
    az_dep = _env_any(
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_DEPLOYMENT_MAIN",
        "AZURE_GPT5_DEPLOYMENT",
        default="gpt-5",
    )
    az_pre_ep = _env_any(
        "AZURE_OPENAI_PRECHECK_ENDPOINT",
        "AZURE_OPENAI_ENDPOINT_PRECHECK",
        "AZURE_GPT41_ENDPOINT",
    )
    az_pre_key = _env_any(
        "AZURE_OPENAI_PRECHECK_API_KEY",
        "AZURE_OPENAI_API_KEY_PRECHECK",
        "AZURE_GPT41_API_KEY",
    )
    az_pre_ver = _env_any(
        "AZURE_OPENAI_PRECHECK_API_VERSION",
        "AZURE_OPENAI_API_VERSION_PRECHECK",
        "AZURE_GPT41_API_VERSION",
        default=az_ver,
    )
    az_pre_dep = _env_any(
        "AZURE_OPENAI_PRECHECK_DEPLOYMENT",
        "AZURE_OPENAI_DEPLOYMENT_PRECHECK",
        "AZURE_GPT41_DEPLOYMENT",
    )
    oi_key = os.getenv("OPENAI_API_KEY")
    oi_model = os.getenv("OPENAI_MODEL", "gpt-5")
    or_key = os.getenv("OPENROUTER_API_KEY")
    or_model = os.getenv("OPENROUTER_MODEL", "gpt-5")

    options = []
    if or_key:
        options.append({
            "label": "OpenRouter",
            "details": f"Modelo: {or_model}",
            "config": {
                "use": True,
                "provider": "openrouter",
                "model": or_model,
                "openrouter_api_key": or_key,
            },
        })
    if az_ep and az_key:
        details = f"Endpoint: {az_ep}"
        if az_pre_ep:
            details += f"\nPre-check: {az_pre_ep}"
        azure_cfg = {
            "label": "Azure OpenAI",
            "details": details,
            "config": {
                "use": True,
                "provider": "azure",
                "endpoint": az_ep,
                "api_version": az_ver,
                "model": az_dep,
                "api_key": az_key,
            },
        }
        if az_pre_ep and az_pre_key and az_pre_dep:
            azure_cfg["config"].update({
                "cheap_model": az_pre_dep,
                "cheap_provider": "azure",
                "cheap_endpoint": az_pre_ep,
                "cheap_api_version": az_pre_ver,
                "cheap_api_key": az_pre_key,
            })
        options.append(azure_cfg)
    if oi_key:
        options.append({
            "label": "OpenAI",
            "details": f"Modelo: {oi_model}",
            "config": {
                "use": True,
                "provider": "openai",
                "model": oi_model,
                "api_key": oi_key,
            },
        })

    if not options:
        return {"use": False}

    if len(options) == 1:
        return options[0]["config"]

    table = Table(title="Selecione o provedor LLM para esta execução")
    table.add_column("#", justify="right")
    table.add_column("Provedor")
    table.add_column("Detalhes")
    for idx, opt in enumerate(options, 1):
        table.add_row(str(idx), opt["label"], opt["details"])
    print(table)
    choice = Prompt.ask(
        "Informe o número do provedor desejado",
        choices=[str(i) for i in range(1, len(options) + 1)],
        default="2",
    )
    return options[int(choice) - 1]["config"]


def main() -> None:
    # Carrega .env se existir
    load_dotenv()
    docs_dir = Path("docs")
    out_dir = Path("output")
    done_dir = Path("docs-gerados")
    done_dir.mkdir(parents=True, exist_ok=True)

    pdfs = _list_pdfs(docs_dir)
    if not pdfs:
        print("[red]Nenhum PDF encontrado em docs/[/red]")
        return

    sel = _choose_pdfs(pdfs)
    pages = _ask_pages()
    
    # Auto-detecta LLM das variáveis de ambiente
    llm_cfg = _detect_llm_config()
    if not llm_cfg.get("use"):
        print("[red]É necessário configurar uma LLM via variáveis de ambiente.[/red]")
        print("[yellow]Defina: OPENROUTER_API_KEY, AZURE_OPENAI_ENDPOINT ou OPENAI_API_KEY[/yellow]")
        return
    
    # ═══════════════════════════════════════════════════════════════
    # MENU DE CONFIGURAÇÃO INTERATIVA
    # ═══════════════════════════════════════════════════════════════
    
    # 1. Escolher modelo de extração (GPT-5 ou GPT-4.1)
    extraction_model, extraction_provider, extraction_endpoint, extraction_api_version, extraction_api_key = _ask_extraction_model(llm_cfg)
    
    # 2. Pre-check: Detecção AUTOMÁTICA (sempre GPT-4.1 quando disponível)
    print("\n[bold cyan]═══ Pre-Check (Automático) ═══[/bold cyan]")
    print("Pre-check é OBRIGATÓRIO no sistema inteligente.")
    print("Detectando melhor modelo automaticamente...")
    cheap_model, cheap_provider, cheap_endpoint, cheap_api_version, cheap_api_key = _auto_detect_precheck_model(llm_cfg)
    
    print("\n[bold green]✅ Configuração concluída![/bold green]")
    print(f"   🧠 Extração: {extraction_model} via {extraction_provider}")
    print(f"   🤖 Pre-check: {cheap_model} via {cheap_provider} [dim](automático)[/dim]")
    print(f"   🔧 OCR: [yellow]Decisão AUTOMÁTICA[/yellow] (baseado em quantidade de elementos)")
    
    # ═══════════════════════════════════════════════════════════════
    
    # Atualiza llm_cfg com escolhas do usuário
    llm_cfg["model"] = extraction_model
    llm_cfg["provider"] = extraction_provider
    llm_cfg["endpoint"] = extraction_endpoint
    llm_cfg["api_version"] = extraction_api_version
    llm_cfg["api_key"] = extraction_api_key
    
    # Valores padrão otimizados
    render_dpi_val = 900  # DPI alto para letras muito pequenas serem legíveis
    try:
        llm_max_workers = int(os.getenv("LLM_MAX_WORKERS", "3"))
    except ValueError:
        llm_max_workers = 6
    llm_max_workers = max(1, llm_max_workers)
    
    # PRE-CHECK: Sempre ativo (já detectado automaticamente)
    use_precheck = True
    
    # OCR: Inicializa como True (será decidido automaticamente pelo sistema baseado em content_count)
    use_layout_ocr = True
    
    logger.info(
        "Fluxo: Página → %s (pre-check via %s) → Decisão OCR → %s (extração)",
        cheap_model,
        cheap_provider,
        llm_cfg.get("model", "gpt-5"),
    )
    logger.info(
        "Pre-check configurado: deployment=%s | endpoint=%s | api_version=%s",
        cheap_model,
        cheap_endpoint or "default",
        cheap_api_version or "default",
    )

    if llm_max_workers > 1:
        logger.info("Processando até %s páginas em paralelo", llm_max_workers)
    
    # SKIP_OCR_PAGES não é mais necessário - OCR é decidido automaticamente
    # Mantido como lista vazia por compatibilidade com código existente
    skip_ocr_pages = []
    
    # CHECKPOINT: Controla se deve reprocessar páginas já extraídas
    force_reprocess = _env_flag("FORCE_REPROCESS", default=False)
    if force_reprocess:
        logger.warning("⚠️  FORCE_REPROCESS ativado - todas as páginas serão reprocessadas")
    else:
        logger.info("✅ Checkpoint ativado - páginas já processadas serão puladas")

    for pdf in sel:
        logger.info("Processando %s", pdf)
        try:
            base_output = out_dir / pdf.stem
            base_output.mkdir(parents=True, exist_ok=True)
            config = ImageProcessingConfig(
                model=extraction_model,
                provider=extraction_provider,
                azure_endpoint=extraction_endpoint or llm_cfg.get("endpoint") or os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_api_version=extraction_api_version or llm_cfg.get("api_version") or os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
                api_key=extraction_api_key or llm_cfg.get("api_key") or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
                openrouter_api_key=llm_cfg.get("openrouter_api_key") or os.getenv("OPENROUTER_API_KEY"),
                cheap_model=cheap_model,
                cheap_provider=cheap_provider,
                cheap_api_key=cheap_api_key,
                cheap_azure_endpoint=cheap_endpoint,
                cheap_azure_api_version=cheap_api_version,
                render_dpi=render_dpi_val,
                use_cheap_precheck=use_precheck,
                llm_max_workers=llm_max_workers,
                use_layout_ocr=use_layout_ocr,
                skip_ocr_pages=skip_ocr_pages,
                force_reprocess=force_reprocess,
            )
            results = process_pdf_images(
                pdf,
                base_output,
                pages,
                img_dir_name="images",
                tables_dir_name="llm_tables",
                config=config,
            )
            if results:
                print(f"[cyan]{len(results)} tabelas/séries salvas em {base_output / 'llm_tables'}[/cyan]")
            else:
                print(f"[yellow]Nenhuma tabela reconhecida para {pdf.name}[/yellow]")

            # 🚧 MODO TESTE: NÃO move PDF (para facilitar testes)
            # Quando finalizar testes, descomente o código abaixo
            
            # move PDF para docs-gerados
            target = done_dir / pdf.name
            # evitar overwrite
            if target.exists():
                base = target.stem
                ext = target.suffix
                i = 1
                while True:
                    alt = done_dir / f"{base}-{i}{ext}"
                    if not alt.exists():
                        target = alt
                        break
                    i += 1
            shutil.move(str(pdf), str(target))
            logger.info("PDF original movido para %s", target)
            
            logger.info("✅ PDF mantido em docs/ (modo teste)")
        except Exception as e:
            logger.exception("Falha ao processar %s", pdf)

    print("\n[bold]Concluído.[/bold]")


if __name__ == "__main__":
    main()

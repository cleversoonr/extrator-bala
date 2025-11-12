from __future__ import annotations

import logging
from typing import Optional

from rich.logging import RichHandler


_LOGGER_CONFIGURED = False


def get_logger(name: Optional[str] = None) -> logging.Logger:
    global _LOGGER_CONFIGURED
    logger_name = name or "extractor"
    logger = logging.getLogger(logger_name)

    if not _LOGGER_CONFIGURED:
        handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
        )
        handler.setLevel(logging.INFO)

        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.handlers.clear()
        root.addHandler(handler)

        _LOGGER_CONFIGURED = True

    logger.setLevel(logging.INFO)
    return logger

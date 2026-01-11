from typing import Callable

from . import config

class logging: # torq logging class
    info: Callable = lambda x: (
        print(f"[torq] INFO: {x}") if config.verbose else None
    )
    warning: Callable = lambda x: (
        print(f"[torq] WARNING: {x}") if config.verbose else None
    )
    error: Callable = lambda x: (
        print(f"[torq] ERROR: {x}") if config.verbose else None
    )

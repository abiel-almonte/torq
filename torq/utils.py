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


class StreamRoundRobin:
    def __init__(self, n_streams) -> None:
        self.n_streams = n_streams
        self._idx = 0

    def __next__(self) -> int:
        self._idx = (self._idx + 1) % self.n_streams
        return self._idx

_stream_pool = StreamRoundRobin(config.max_streams)

class resources:
    get_next_streamid: Callable = lambda: next(_stream_pool)

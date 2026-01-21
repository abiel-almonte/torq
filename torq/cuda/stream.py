from typing import Optional

from .handlers import CUDAStream
from .launcher import GraphLauncher


class Stream:
    def __init__(self, stream_id: int) -> None:
        self.stream_id = stream_id
        self._stream = CUDAStream()

    def synchronize(self) -> None:
        self._stream.synchronize()

    def capture(self) -> Optional[GraphLauncher]:
        return GraphLauncher(self._stream)

    def __repr__(self) -> str:
        return f"Stream(id={self.stream_id}, stream={self._stream})"

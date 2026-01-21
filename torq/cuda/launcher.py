from typing import Optional, Any

from .handlers import CUDAStream, CUDAGraphExec, CUDAGraph


class GraphLauncher:
    def __init__(self, stream: CUDAStream):
        self._stream = stream
        self._graph = CUDAGraph()
        self._exec: Optional[CUDAGraphExec] = None

    def _launch_no_op(self) -> None:
        raise RuntimeError("Cannot launch uncaptured CUDA Graph")

    def _launch(self) -> None:
        self._graph.launch(self._exec, self._stream)

    def launch(self) -> None:
        self._launch_no_op()

    def __enter__(self) -> "GraphLauncher":
        self._graph.begin_capture(self._stream)
        return self

    def __exit__(self, *args: Any) -> None:
        self._graph.end_capture(self._stream)
        self._exec = CUDAGraphExec(self._graph)
        self._stream.synchronize()

        self.launch = self._launch

    def __repr__(self) -> str:
        return f"GraphLauncher(status={'captured' if self._exec else 'uncaptured'})"

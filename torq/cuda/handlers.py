from typing import Optional

from .types import *
from . import _torq as _C


class Handler:
    _name = "handler"

    def __init__(self, handler: handler_t) -> None:
        self._handler = handler

    @property
    def ptr(self) -> ptr_t:
        get_ptr_fn = getattr(_C, f"get_{self._name}_ptr")
        return get_ptr_fn(self._handler)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ptr={self.ptr:#x})"


class CUDAStream(Handler):
    _name = "stream"

    def __init__(self) -> None:
        stream: cudaStream_t = _C.create_stream()
        super().__init__(stream)

    def synchronize(self) -> None:
        _C.sync_stream(self._handler)


class CUDAGraph(Handler):
    _name = "graph"

    def __init__(self, graph: Optional[cudaGraph_t] = None) -> None:
        super().__init__(graph)

    def begin_capture(self, stream: CUDAStream) -> None:
        _C.inject_stream(stream._handler)
        _C.begin_capture(stream._handler)

    def end_capture(self, stream: CUDAStream) -> None:
        self._handler: cudaGraph_t = _C.end_capture(stream._handler)
        _C.clear_injection()

    def launch(self, exec: "CUDAGraphExec", stream: CUDAStream) -> None:
        _C.launch_graph(exec._handler, stream._handler)


class CUDAGraphExec(Handler):
    _name = "executor"

    def __init__(self, graph: CUDAGraph) -> None:
        executor: cudaGraph_t = _C.create_executor(graph._handler)
        super().__init__(executor)

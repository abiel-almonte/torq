from . import _C

class TraceMemoryCtx:
    def __init__(self, node_id) -> None:
        self._node_id = node_id

    def __enter__(self) -> "TraceMemoryCtx":
        _C.trace_memory(self._node_id)
        return self

    def __exit__(self, *args) -> None:
        _C.end_trace(self._node_id)


class ApplyTraceCtx:
    def __init__(self, node_id) -> None:
        self._node_id = node_id

    def __enter__(self) -> "ApplyTraceCtx":
        _C.set_tracer(self._node_id)
        return self

    def __exit__(self, *args) -> None:
        _C.unset_tracer()


class MemoryTracer:
    def __init__(self, node_id: str) -> None:
        self._node_id = node_id

    def trace(self) -> TraceMemoryCtx:
        return TraceMemoryCtx(self._node_id)

    def apply(self) -> ApplyTraceCtx:
        return ApplyTraceCtx(self._node_id)

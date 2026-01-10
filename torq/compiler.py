from typing import Any, Union, Protocol, Optional, runtime_checkable

from .runnable import Runnable
from .pipeline import System
from .dag import DAG


@runtime_checkable
class CompilerBackend(Protocol):
    def __call__(self, graph: DAG) -> DAG: ...


_registered_backend: Optional[CompilerBackend] = None


def register_backend(fn: CompilerBackend):
    import inspect

    sig = inspect.signature(fn)
    n_params = len(list(sig.parameters.values()))

    if n_params != 1 or not isinstance(fn, CompilerBackend):
        raise TypeError(f"Object {fn} does not implement CompilerBackend protocol")

    global _registered_backend
    _registered_backend = fn
    return fn


def compile_graph(graph: DAG) -> DAG:
    if _registered_backend is None:
        raise RuntimeError("No compiler backend registered")
    g = _registered_backend(graph)
    g.semantic_lint() # structural lints like cycles/ orphans will surface naturally
    return g


@register_backend
def torq_backend(graph: DAG) -> DAG:
    return graph


class CompiledSystem(Runnable):
    def __init__(self, system: System) -> None:
        self._inner: Union[System, DAG] = system
        self.caller = self._compile_and_run
        self.compiled = False

    def _compile_and_run(self, *args):
        outs = self._inner(*args)  # run if already materialized
        self._inner = compile_graph(graph=DAG.from_system(self._inner))

        self.caller = self._inner
        self.compiled = True
        return outs

    def __call__(self, *args: Any) -> Any:
        return self.caller(*args)

    def __repr__(self) -> str:
        return repr(self._inner)


def compile(system: System) -> CompiledSystem:  # lazy
    return CompiledSystem(system=system)

from typing import Any, Union

from ..core import Runnable, System

from . import registry
from .dag import DAG


def compile_graph(graph: DAG) -> DAG:
    if registry._registered_backend is None:
        raise RuntimeError("No compiler backend registered")
    g = registry._registered_backend(graph)
    g.semantic_lint()  # structural lints like cycles/ orphans will surface naturally
    return g


class CompiledSystem(Runnable):
    def __init__(self, system: System) -> None:
        self._inner: Union[System, DAG] = system
        self.caller = self._compile_and_run
        self.compiled = False

    def _compile_and_run(self, *args):
        outs = self._inner(*args)  # run if already materialized
        self._inner = compile_graph(graph=DAG.from_system(self._inner, *args))

        self.caller = self._inner
        self.compiled = True
        return outs

    def __call__(self, *args: Any) -> Any:
        return self.caller(*args)

    def __repr__(self) -> str:
        return repr(self._inner)


def compile(system: System) -> CompiledSystem:  # lazy
    return CompiledSystem(system=system)

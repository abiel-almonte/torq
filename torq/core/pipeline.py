from typing import Any, TypeAlias
from abc import ABC, abstractmethod

from .runnable import Runnable
from .pipes import Pipe


class Opaque:
    def __init__(self, inner, is_pipeline, callback):
        self._inner = inner
        self.is_pipeline = is_pipeline
        self.materialization_callback = callback

    def __call__(self, *args):
        if self.is_pipeline:
            pipe, outs = self._inner, self._inner(*args)  # materialize next pipeline
        else:
            pipe, outs = Pipe.from_opaque(self._inner, args)
        self.materialization_callback(pipe)
        return outs

    def __repr__(self) -> str:
        if self.is_pipeline:
            return repr(self._inner)
        return f"Opaque(inner={self._inner})"


class Pipeline(ABC, Runnable):
    def __init__(self, *args: Any) -> None:  # stages
        self._opaques = tuple()
        self._pipes = tuple()  # will be filled out lazily

        def register_pipe(pipe):
            self._pipes += (pipe,)

        for arg in args:
            self._opaques += (
                Opaque(
                    inner=arg,
                    is_pipeline=isinstance(arg, Pipeline),
                    callback=register_pipe,
                ),
            )

        self._materialized = False

    @property
    def container(self):
        return self._pipes if self._materialized else self._opaques

    def __iter__(self):
        for x in self.container:
            yield x

    @abstractmethod
    def _call_impl(self, *args: Any) -> Any: ...

    def _step(self, *args: Any) -> Any:
        return self._call_impl(*args)

    def _materialization_step(self, *args: Any) -> Any:
        outs = self._call_impl(*args)
        self._materialized = True
        return outs

    def __call__(self, *args: Any) -> Any:
        if self._materialized:
            return self._step(*args)
        return self._materialization_step(*args)

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__}("]
        for idx, pipe in enumerate(self.container):
            comma = "," if idx < len(self.container) - 1 else ""

            for line in repr(pipe).splitlines():
                lines.append(f"\t{line}")

            lines[-1] += comma

        lines.append(")")
        return "\n".join(lines)


class Sequential(Pipeline):

    def _call_impl(self, *args: Any) -> Any:

        x = args
        for fn in self.container:
            x = fn(*x)

            if not isinstance(x, tuple):
                x = (x,)

        return x


class Concurrent(Pipeline):

    def _call_impl(self, *args: Any) -> Any:

        outs = tuple()
        for fn in self.container:
            out = fn(*args)

            if not isinstance(out, tuple):
                out = (out,)

            outs += out

        return outs


System: TypeAlias = Pipeline

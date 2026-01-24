from types import FunctionType
from typing import Tuple, Callable, Any

import inspect
from ..utils import _as_tuple
from .registry import get_registered_types, get_adapter
from .runnable import Runnable


class Pipe(Runnable):
    _name = "inner"

    def __init__(
        self,
        inner: Any,
        caller: Callable[..., Any] = None,
    ) -> None:
        self._inner = inner

        if caller is not None:
            self.caller = caller
        else:
            adapter = get_adapter(inner)
            if callable(adapter(inner)):
                self.caller = lambda *args: adapter(inner)(*args)
            else:
                self.caller = lambda *_: adapter(inner)

    # duck type pipe
    @staticmethod
    def from_opaque(opaque: Any, ins: Any) -> Tuple["Pipe", Any]:
        if not isinstance(opaque, get_registered_types()):
            if not callable(opaque) or getattr(type(opaque), "__call__", None) is object.__call__:
                raise TypeError(f"Pipeline contains an unknown opaque type {type(opaque)} (no registered adapter and not explicitly callable)")

        pipe = Pipe(opaque)
        outs = pipe(*ins)
        outs = _as_tuple(outs)

        has_input = ins is not None and len(ins) != 0
        has_output = outs is not None and len(outs) != 0

        if not has_input:
            pipe.__class__ = Input
        elif not has_output:
            pipe.__class__ = Output
        elif has_input and has_output:
            if isinstance(opaque, FunctionType):
                pipe.__class__ = Functional
            else:
                pipe.__class__ = Model
        else:
            raise TypeError(f"Unable to reconcile {opaque}")

        return pipe, outs

    def __call__(self, *args: Any) -> Any:
        return self.caller(*args)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._name}={self._inner})"


class Input(Pipe):
    _name = "source"


class Model(Pipe):
    _name = "model"


class Output(Pipe):
    _name = "sink"


class Functional(Pipe):
    def __repr__(self) -> str:
        fn_name = getattr(self._inner, "__name__", "function")
        sig = inspect.signature(self._inner)
        return f"Functional(fn={fn_name}{sig})"

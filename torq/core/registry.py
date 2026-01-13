from typing import Type, Callable, Any, Tuple
from types import FunctionType

from ..utils import logging

_pipe_registry: Tuple[Tuple[Type, Callable[..., Any]], ...] = (
    (FunctionType, lambda f: f),  # general function, same as lambda type
)


def register_pipe(cls: Type, adapter: Callable[..., Any]) -> None:
    if not callable(adapter):
        raise TypeError(f"Unable to use adapter of type {type(adapter)}")

    global _pipe_registry
    _pipe_registry += ((cls, adapter),)


def register_pipe_decorator(adapter: Callable[..., Any]) -> Callable:
    def decorator(cls):
        register_pipe(cls, adapter)
        return cls

    return decorator


try:
    import cv2  # opencv preloaded

    register_pipe(cv2.VideoCapture, lambda x: x.read()[-1])
except ImportError:
    pass

try:
    import torch.nn as nn  # pytorch preloaded

    register_pipe(nn.Module, lambda m: m.forward)
except ImportError:
    pass


def get_registered_types() -> Tuple[Type, ...]:
    return tuple(cls for cls, _ in _pipe_registry)


def get_adapter(obj: object) -> Callable[..., Any]:
    for cls, adapter in _pipe_registry:
        if isinstance(obj, cls):
            return adapter

    logging.warning(
        f"Unable to retrieve adapter for object of type {type(obj)}. Assuming object is callable."
    )
    return lambda x: x

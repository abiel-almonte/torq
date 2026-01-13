from .core import Pipe, Sequential, Concurrent, register_pipe, register_pipe_decorator
from .compiler import register_backend, compile

HAS_CUDA = False

try:
    from . import cuda

    HAS_CUDA = True
except ImportError:
    pass

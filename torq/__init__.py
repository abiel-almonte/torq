from .core import (
    Pipe as Pipe,
    Sequential as Sequential,
    Concurrent as Concurrent,
    register_pipe as register_pipe,
    register_pipe_decorator as register_pipe_decorator,
)
from .compiler import (
    register_backend as register_backend,
    compile as compile,
    _fulqrum as _fulqrum,
)


HAS_CUDA = False

try:
    from . import cuda as cuda

    HAS_CUDA = True
except ImportError as e:
    print(e)

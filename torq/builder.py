from typing import Union, Tuple
from collections import defaultdict

from .nodes import DAGNode
from .pipeline import System, Pipeline, Sequential, Concurrent, Pipe

from .utils import logging
from . import config


class Namer:
    def __init__(self):
        self._global = defaultdict(int)
        self._local = defaultdict(int)

    def pipeline_name(self, cls_name: str, parent: str) -> str:
        idx = self._global[cls_name]
        self._global[cls_name] += 1
        self._local.clear()  # each pipeline gets its own local counter

        base = f"{parent}.{cls_name.lower()}" if parent else cls_name.lower()
        return f"{base}{idx}"

    def node_name(self, cls_name: str, parent: str) -> str:
        idx = self._local[cls_name]
        self._local[cls_name] += 1

        base = f"{parent}.{cls_name.lower()}" if parent else cls_name.lower()
        return f"{base}{idx}"


class StreamAssigner:
    def __init__(self, n_streams: int) -> None:
        self._n = n_streams
        self._idx = -1

    def get_next_streamid(self) -> int:
        self._idx = (self._idx + 1) % self._n
        return self._idx


def _build_dag_from_system(system: System):
    nodes = tuple()
    namer = Namer()
    assigner = StreamAssigner(config.max_streams)

    def walk(
        pipe: Union[Pipeline, Pipe],
        prev: Union[Tuple[DAGNode, ...], DAGNode, None] = None,
        name: str = "",
        stream_id: int = 0,
    ) -> Union[DAGNode, Tuple[DAGNode, ...]]:

        cls_name = str(pipe.__class__.__name__)
        if isinstance(pipe, Pipeline):
            pipeline = pipe
            name = namer.pipeline_name(cls_name, name)

            if isinstance(pipeline, Sequential):
                curr = prev
                for pipe in pipeline._pipes:
                    curr = walk(pipe, curr, name, stream_id)

                if curr is None:
                    raise RuntimeError(f"Invalid pipeline. Sequential is empty")

                return curr

            elif isinstance(pipeline, Concurrent):
                outs = tuple()

                for pipe in pipeline._pipes:
                    out = walk(
                        pipe,
                        prev,
                        name,
                        stream_id=(
                            stream_id
                            if isinstance(pipe, Concurrent)
                            else assigner.get_next_streamid()
                        ),
                    )

                    if not isinstance(out, tuple):
                        out = (out,)

                    outs += out

                if len(outs) == 0:
                    raise RuntimeError(f"Invalid pipeline. Concurrent is empty")

                return outs

            else:
                raise TypeError(f"Unknown pipeline type in system: {type(pipeline)}")

        elif isinstance(pipe, Pipe):
            args = tuple()

            if prev:
                if not isinstance(prev, tuple):
                    prev = (prev,)

                args += prev

            name = namer.node_name(cls_name, name)
            node = DAGNode(name, stream_id, pipe, args)

            nonlocal nodes
            nodes += (node,)

            return node

        else:
            raise TypeError(f"Unknown type in system: {type(pipe)}")

    leaves = walk(system)

    if not isinstance(leaves, tuple):
        leaves = (leaves,)  # pack single leaf

    logging.info(f"Graph built with {len(nodes)} nodes and {len(leaves)} trees")

    return nodes, leaves

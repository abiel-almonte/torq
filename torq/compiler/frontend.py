from typing import Optional, Union, Tuple
from collections import defaultdict

from .. import config
from ..core import System, Pipeline, Sequential, Concurrent, Pipe
from ..utils import logging

from .nodes import DAGNode


class StreamAssigner:
    def __init__(self, n_streams: int) -> None:
        self._n = n_streams
        self._idx = -1

    def get_next_streamid(self) -> int:
        self._idx = (self._idx + 1) % self._n
        return self._idx


def lower(system: System):
    nodes = tuple()
    assigner = StreamAssigner(config.max_streams)
    global_cnt = defaultdict(int)

    def walk(
        pipe: Union[Pipeline, Pipe],
        prev: Union[Tuple[DAGNode, ...], DAGNode, None] = None,
        name: str = "",
        stream_id: int = 0,
        local_cnt: Optional[defaultdict] = None,
    ) -> Union[DAGNode, Tuple[DAGNode, ...]]:

        if local_cnt is None:
            local_cnt = defaultdict(int)

        cls_name = str(pipe.__class__.__name__)
        base = f"{name}.{cls_name.lower()}" if name else cls_name.lower()

        if isinstance(pipe, Pipeline):
            pipeline = pipe
            name = base + str(global_cnt[cls_name])

            global_cnt[cls_name] += 1
            local_cnt.clear()
            if isinstance(pipeline, Sequential):
                curr = prev
                for pipe in pipeline._pipes:
                    curr = walk(pipe, curr, name, stream_id, local_cnt)

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
                        local_cnt=local_cnt,
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
            name = base + str(local_cnt[cls_name])
            local_cnt[cls_name] += 1

            args = tuple()

            if prev:
                if not isinstance(prev, tuple):
                    prev = (prev,)

                args += prev

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

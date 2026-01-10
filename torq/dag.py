from typing import Tuple, Union
from collections import defaultdict

from .runnable import Runnable
from .pipes import Pipe
from .pipeline import System, Pipeline, Sequential, Concurrent
from .utils import logging


class StreamRoundRobin:
    def __init__(self, n_streams) -> None:
        self.n_streams = n_streams
        self._streams = [x for x in range(n_streams)]
        self._idx = 0

    def __next__(self) -> int:
        stream = self._streams[self._idx]
        self._idx = (self._idx + 1) % self.n_streams
        return stream


stream_roundrobin = StreamRoundRobin(10)
get_next_streamid = lambda: next(stream_roundrobin)


class DAGNode:
    def __init__(
        self, node_id: str, stream_id: int, pipe: Pipe, args: Tuple["DAGNode", ...]
    ):
        self.node_id = node_id
        self.stream_id = stream_id
        self.pipe = pipe
        self.args = args

    def __call__(self, *args):
        return self.pipe(*args)

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __repr__(self) -> str:
        def _repr(node: DAGNode, level=0):
            space = "\t"
            indent = space * level

            s = f"{indent}DAGNode(\n"
            s += f"{indent}{space}node_id={node.node_id},\n"
            s += f"{indent}{space}stream={node.stream_id},\n"
            s += f"{indent}{space}pipe={node.pipe.__class__.__name__},\n"

            if node.args:
                s += f"{indent}{space}args=(\n"
                for arg in node.args:
                    s += _repr(arg, level + 2) + ",\n"
                s += f"{indent}{space})\n"
            else:
                s += f"{indent}{space}args=()\n"

            s += f"{indent})"
            return s

        return _repr(self)


class DAG(Runnable):
    def __init__(self) -> None:
        self.nodes: Tuple[DAGNode, ...] = tuple()
        self.leaves: Tuple[DAGNode, ...] = tuple()

    @staticmethod
    def from_system(system: System) -> "DAG":

        dag = DAG()
        system_cnt = defaultdict(int)

        def walk(
            pipe: Union[Pipeline, Pipe],
            prev: Union[Tuple[DAGNode, ...], DAGNode, None] = None,
            name: str = "",
            stream_id: int = 0,
            cnt=None,  # every pipeline gets its own counter
        ) -> Union[DAGNode, Tuple[DAGNode, ...]]:

            if cnt is None:
                cnt = defaultdict(int)

            cls_name = f"{pipe.__class__.__name__}"
            base_name = name + ("." if name else "") + cls_name.lower()

            if isinstance(pipe, Pipeline):

                pipeline = pipe
                pipeline_name = base_name + str(system_cnt[cls_name])
                system_cnt[cls_name] += 1

                if isinstance(pipeline, Sequential):
                    curr = prev
                    for pipe in pipeline._pipes:
                        curr = walk(pipe, curr, name=pipeline_name, stream_id=stream_id)

                    if curr is None:
                        raise RuntimeError(f"Invalid pipeline. {cls_name} is empty")

                    return curr

                elif isinstance(pipeline, Concurrent):
                    outs = tuple()

                    for pipe in pipeline._pipes:
                        out = walk(
                            pipe,
                            prev,
                            name=pipeline_name,
                            stream_id=(
                                stream_id
                                if isinstance(pipe, Concurrent)
                                else get_next_streamid()
                            ),
                        )

                        if not isinstance(out, tuple):
                            out = (out,)

                        outs += out

                    if len(outs) == 0:
                        raise RuntimeError(f"Invalid pipeline. {cls_name} is empty")

                    return outs

                else:
                    raise TypeError(
                        f"Unknown pipeline type in system: {type(pipeline)}"
                    )

            elif isinstance(pipe, Pipe):
                node_name = base_name + str(cnt[cls_name])
                cnt[cls_name] += 1

                args = tuple()

                if prev:
                    if not isinstance(prev, tuple):
                        prev = (prev,)

                    args += prev

                node = DAGNode(node_name, stream_id, pipe, args)
                dag.nodes += (node,)

                return node

            else:
                raise TypeError(f"Unknown type in system: {type(pipe)}")

        leaves = walk(system)

        if not isinstance(leaves, tuple):
            leaves = (leaves,)  # pack single leaf

        dag.leaves = leaves

        logging.info(
            f"Graph built with {len(dag.nodes)} nodes and {len(dag.leaves)} trees"
        )
        return dag

    @staticmethod
    def _execute(leaf_nodes, *args):
        cache = {}

        def visit(node: DAGNode, *ins):
            if node.node_id in cache:
                return cache[node.node_id]

            ins = tuple(visit(arg, *ins) for arg in node.args)

            outs = node(*ins)
            cache[node.node_id] = outs

            return outs

        outs = tuple(visit(leaf, *args) for leaf in leaf_nodes)

        if len(outs) == 1:
            outs = outs[0]

        return outs

    def __call__(self, *args):
        return self._execute(self.leaves, *args)

    def __iter__(self):
        visited = set()

        def visit(node: DAGNode):
            if node in visited:
                return

            # visit up the tree through the args
            if len(node.args) > 0:
                for arg in node.args:
                    yield from visit(arg)

            yield node
            visited.add(node)

        for leaf in self.leaves:
            yield from visit(leaf)

    def __repr__(self) -> str:
        line = "=" * 80
        return f"\n".join(
            [
                f"Tree {i}:\n{line}\n{repr(leaf)}\n{line}"
                for i, leaf in enumerate(self.leaves)
            ]
        )

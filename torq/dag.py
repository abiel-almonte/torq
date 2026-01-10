from typing import Tuple, Union
from collections import defaultdict

from .runnable import Runnable
from .pipes import Pipe
from .pipeline import System, Pipeline, Sequential, Concurrent
from .nodes import DAGNode

from .utils import resources, logging

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
                                else resources.get_next_streamid()
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

    def __call__(self, *args):
        cache = {}

        def visit(node: DAGNode, *ins):
            if node.node_id in cache:
                return cache[node.node_id]

            ins = tuple(visit(arg, *ins) for arg in node.args)

            outs = node(*ins)
            cache[node.node_id] = outs

            return outs

        outs = tuple(visit(leaf, *args) for leaf in self.leaves)

        if len(outs) == 1:
            outs = outs[0]

        return outs

    def __iter__(self):
        visited = set()

        def visit(node: DAGNode):
            if node in visited:
                return

            # visit up the tree through the args
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

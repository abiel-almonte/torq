from typing import Tuple, Union
from collections import defaultdict

from .runnable import Runnable
from .pipes import Pipe, Input, Output
from .pipeline import System, Pipeline, Sequential, Concurrent
from .nodes import DAGNode

from .utils import resources, logging


class DAG(Runnable):
    def __init__(self) -> None:
        self.nodes: Tuple[DAGNode, ...] = tuple()
        self.leaves: Tuple[DAGNode, ...] = tuple()

    @staticmethod
    def from_system(system: System) -> "DAG":
        return _build_dag_from_system(system)

    def semantic_lint(self):
        for node in self:
            if isinstance(node.pipe, Input):
                if len(node.args) > 0:
                    raise RuntimeError(
                        f"Input node {node.node_id} cannot have incoming edges"
                    )

            for arg in node.args:
                if isinstance(arg.pipe, Output):
                    raise RuntimeError(
                        f"Output node {arg.node_id} cannot have outgoing edges"
                    )

    def __call__(self, *args):
        cache = {}

        for node in self:
            ins = tuple(cache[arg] for arg in node.args)
            cache[node] = node(*ins)

        outs = tuple(cache[leaf] for leaf in self.leaves)

        return outs[0] if len(outs) == 1 else outs

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


def _build_dag_from_system(system: System) -> "DAG":

    dag = DAG()
    system_cnt = defaultdict(int)

    def walk(
        pipe: Union[Pipeline, Pipe],
        prev: Union[Tuple[DAGNode, ...], DAGNode, None] = None,
        stream_id: int = 0,
        name: str = "",
        cnt=None,
    ) -> Union[DAGNode, Tuple[DAGNode, ...]]:

        if cnt is None:
            cnt = defaultdict(int)

        cls_name = f"{pipe.__class__.__name__}"
        base_name = name + ("." if name else "") + cls_name.lower()

        if isinstance(pipe, Pipeline):
            cnt.clear()  # every pipeline gets its own counter

            pipeline = pipe
            pipeline_name = base_name + str(system_cnt[cls_name])
            system_cnt[cls_name] += 1

            if isinstance(pipeline, Sequential):
                curr = prev
                for pipe in pipeline._pipes:
                    curr = walk(
                        pipe, curr, stream_id=stream_id, name=pipeline_name, cnt=cnt
                    )

                if curr is None:
                    raise RuntimeError(f"Invalid pipeline. {cls_name} is empty")

                return curr

            elif isinstance(pipeline, Concurrent):
                outs = tuple()

                for pipe in pipeline._pipes:
                    out = walk(
                        pipe,
                        prev,
                        stream_id=(
                            stream_id
                            if isinstance(pipe, Concurrent)
                            else resources.get_next_streamid()
                        ),
                        name=pipeline_name,
                        cnt=cnt,
                    )

                    if not isinstance(out, tuple):
                        out = (out,)

                    outs += out

                if len(outs) == 0:
                    raise RuntimeError(f"Invalid pipeline. {cls_name} is empty")

                return outs

            else:
                raise TypeError(f"Unknown pipeline type in system: {type(pipeline)}")

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
    dag.semantic_lint()  # API make cycles and orphans impossible to occur.

    logging.info(f"Graph built with {len(dag.nodes)} nodes and {len(dag.leaves)} trees")

    return dag

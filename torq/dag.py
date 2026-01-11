from typing import Tuple

from .runnable import Runnable
from .pipes import Input, Output
from .pipeline import System
from .nodes import DAGNode

from .builder import _build_dag_from_system

class DAG(Runnable):
    def __init__(self, nodes: Tuple[DAGNode, ...], leaves: Tuple[DAGNode, ...]) -> None:
        self.nodes = nodes
        self.leaves = leaves

    @staticmethod
    def from_system(system: System) -> "DAG":
        dag = DAG(*_build_dag_from_system(system))
        dag.semantic_lint() # API make cycles and orphans impossible to occur.
        return dag

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

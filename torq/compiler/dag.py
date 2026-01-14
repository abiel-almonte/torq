from typing import Any


from ..core import Runnable, Input, Output, System

from .types import Node, Nodes
from .frontend import build_graph


class DAG(Runnable):
    def __init__(self, nodes: Nodes, leaves: Nodes) -> None:
        self.nodes = nodes
        self.leaves = leaves

    @staticmethod
    def from_system(system: System, *args) -> "DAG":
        dag = DAG(*build_graph(system, *args))
        dag.semantic_lint()  # API make cycles and orphans impossible to occur.
        return dag

    def semantic_lint(self) -> None:
        for node in self:
            if isinstance(node.pipe, Input):
                if len(node.args) > 0:
                    
                    raise RuntimeError(
                        f"Input node {node.id} cannot have incoming edges"
                    )

            for arg in node.args:
                if isinstance(arg.pipe, Output):
                    raise RuntimeError(
                        f"Output node {arg.id} cannot have outgoing edges"
                    )

    def __call__(self, *args: Any) -> Any:
        cache = {}
        args_iter = iter(args)

        for node in self:
            if not node.args and args:
                cache[node] = node(*next(args_iter))
            else:
                ins = tuple(cache[arg] for arg in node.args)
                cache[node] = node(*ins)

        outs = tuple(cache[leaf] for leaf in self.leaves)

        return outs[0] if len(outs) == 1 else outs

    def __iter__(self):
        visited = set()

        def visit(node: Node):
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

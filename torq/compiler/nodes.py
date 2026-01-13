from typing import Tuple

from ..core import Runnable, Pipe


class DAGNode(Runnable):
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

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DAGNode) and self.node_id == other.node_id

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

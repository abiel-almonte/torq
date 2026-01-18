from ..core import Runnable, Pipe
from .types import Node, Nodes


class DAGNode(Node, Runnable):
    def __init__(self, node_id: str, branch: int, pipe: Pipe, args: Nodes):
        self.id = node_id
        self.branch = branch
        self.pipe = pipe
        self.args = args

    def __call__(self, *args):
        return self.pipe(*args)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Node) and self.id == other.id

    def __repr__(self) -> str:
        def _repr(node: DAGNode, level=0):
            space = "\t"
            indent = space * level

            s = f"{indent}{node.__class__.__name__}(\n"
            s += f"{indent}{space}id={node.id},\n"
            s += f"{indent}{space}branch={node.branch},\n"
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


class HostNode(DAGNode):
    pass


class DeviceNode(DAGNode):
    pass  # TODO lazy capture graph


class SyncNode(DAGNode):
    pass

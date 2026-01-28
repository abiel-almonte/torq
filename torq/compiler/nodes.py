from ..core import Runnable, Pipe
from ..cuda import Stream, MemoryTracer
from .types import Node, Nodes


class DAGNode(Node, Runnable):
    def __init__(self, node_id: str, branch: int, pipe: Pipe, args: Nodes):
        self.id = node_id
        self.branch = branch
        self.pipe = pipe
        self.args = args
        self._tracer = MemoryTracer(node_id)
        self._stabilized = False

    def _call_apply(self, *args):
        with self._tracer.apply():
            return self.pipe(*args)

    def _call_trace(self, *args):
        with self._tracer.trace():
            outs = self.pipe(*args)
        
        self._call = self._call_apply
        return outs

    def _call(self, *args):
        return self._call_trace(*args)

    def __call__(self, *args):
        return self._call(*args)

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
    def __init__(self, node_id: str, branch: int, pipe: Pipe, args: Nodes):
        super().__init__(node_id, branch, pipe, args)

        self._launcher = None
        self._captured_outs = None
    
    def __call__(self, *args):
        if self._launcher is None:
            stream = Stream(self.branch) # TODO create a resource assigner to get stream from branch

            with stream.capture() as launcher:
                outs = super().__call__(*args)

            self._launcher = launcher 
            self._captured_outs = outs
            return outs

        self._launcher.launch()
        return self._captured_outs


class SyncNode(DAGNode):
    pass

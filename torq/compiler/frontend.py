from typing import Union, Tuple, List, Any
from collections import defaultdict
from dataclasses import dataclass, replace

from ..core import System, Pipeline, Sequential, Concurrent, Pipe
from ..utils import logging
from .nodes import DAGNode
from .types import Node, Nodes, NodeOrNodes


class _BranchCounter:
    __slots__ = ("_local", "_global")

    def __init__(self, branch: int) -> None:
        self._local = branch
        self._global = branch

    def get_branch(self) -> int:
        return self._local

    def get_next(self) -> "_BranchCounter":
        branch = self._global
        self._global += 1
        return _BranchCounter(branch)


class _NameBuilder:
    __slots__ = ("stack", "_global", "_local")

    def __init__(
        self,
        stack: Tuple[str, ...],
        global_cnt: defaultdict,
    ) -> None:

        self.stack = stack
        self._global = global_cnt
        self._local = defaultdict(int)  # every pipeline gets their own local counter

    def enter_pipeline(self, obj: object) -> "_NameBuilder":
        cls_name = obj.__class__.__name__.lower()

        idx = self._global[cls_name]
        self._global[cls_name] += 1

        return _NameBuilder(
            stack=self.stack + (f"{cls_name}{idx}",),
            global_cnt=self._global,  # maintain the global reference for unique node ids
        )

    def get_name(self, obj: object) -> str:
        cls_name = obj.__class__.__name__.lower()

        idx = self._local[cls_name]
        self._local[cls_name] += 1

        return ".".join(self.stack + (f"{cls_name}{idx}",))


@dataclass
class _TraversalContext:
    name: _NameBuilder
    branch: _BranchCounter
    nodes: List[Node]

    def new_name(self, pipeline: Pipeline) -> "_TraversalContext":
        return replace(self, name=self.name.enter_pipeline(pipeline))

    def get_name(self, pipe: Pipe) -> str:
        return self.name.get_name(pipe)

    def new_branch(self) -> "_TraversalContext":
        return replace(self, branch=self.branch.get_next())

    def get_branch(self) -> str:
        return self.branch.get_branch()

    def push_node(self, node: Node) -> None:
        self.nodes.append(node)


def lazy_build_graph(system: System, *args) -> Tuple[Nodes, Nodes, Any]:
    ctx = _TraversalContext(
        name=_NameBuilder(tuple(), defaultdict(int)),
        branch=_BranchCounter(0),
        nodes=[],
    )

    leaves, _ = _lower_system(system, prev=None, prev_outs=args, ctx=ctx)
    leaves = _as_tuple(leaves)

    logging.info(f"Graph built with {len(ctx.nodes)} nodes and {len(leaves)} trees")
    return tuple(ctx.nodes), leaves


def _as_tuple(x) -> tuple:

    if not isinstance(x, tuple):
        x = (x,)

    return x


def _lower_system(
    pipe: Union[Pipeline, Pipe],
    prev: NodeOrNodes,
    prev_outs: Any,
    ctx: _TraversalContext,
) -> Tuple[NodeOrNodes, Any]:

    if isinstance(pipe, Pipeline):
        pipeline = pipe
        ctx = ctx.new_name(pipeline)

        if isinstance(pipeline, Sequential):
            curr, curr_outs = prev, prev_outs
            for pipe in pipeline:  # shadow pipe
                curr, curr_outs = _lower_system(
                    pipe, prev=curr, prev_outs=curr_outs, ctx=ctx
                )

            if curr is prev:
                raise RuntimeError("Sequential pipeline is empty")

            return curr, _as_tuple(curr_outs)

        elif isinstance(pipeline, Concurrent):
            out_nodes = tuple()
            outs = tuple()

            for pipe in pipeline:
                branch_ctx = ctx if isinstance(pipe, Concurrent) else ctx.new_branch()

                out_node, out = _lower_system(
                    pipe, prev=prev, prev_outs=prev_outs, ctx=branch_ctx
                )

                out_nodes += _as_tuple(out_node)
                outs += _as_tuple(out)

            if not out_nodes:
                raise RuntimeError("Concurrent pipeline is empty")

            return out_nodes, outs

        else:
            raise TypeError(f"Unknown pipeline type in system: {type(pipeline)}")

    elif isinstance(pipe, Pipe):
        return _lower_pipe(pipe, prev, prev_outs, ctx)

    else:
        raise TypeError(f"Unknown type in system: {type(pipe)}")


def _lower_pipe(
    pipe: Pipe, prev: NodeOrNodes, prev_outs: Any, ctx: _TraversalContext
) -> Tuple[Node, Any]:  # TODO lower to GPU Nodes

    outs = pipe(*prev_outs)
    node = DAGNode(
        node_id=ctx.get_name(pipe),
        branch=ctx.get_branch(),
        pipe=pipe,
        args=_as_tuple(prev) if prev else (),
    )

    ctx.push_node(node)
    return node, _as_tuple(outs)

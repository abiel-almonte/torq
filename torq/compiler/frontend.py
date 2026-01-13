from typing import Union, Tuple, List
from collections import defaultdict
from dataclasses import dataclass, replace

from ..core import System, Pipeline, Sequential, Concurrent, Pipe
from ..utils import logging
from .nodes import DAGNode

Nodes = Tuple[DAGNode, ...]
NodeOrNodes = Union[DAGNode, Nodes]


class _BranchCounter:
    __slots__ = ('_local', '_global')

    def __init__(self, branch: int) -> None:
        self._local = branch
        self._global = branch

    def get_branch(self):
        return self._local

    def get_next(self):
        branch = self._global
        self._global += 1
        return _BranchCounter(branch)


class _NameBuilder:
    __slots__ = ('stack', '_global', '_local')

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
    nodes: List[DAGNode]

    def new_name(self, pipeline: Pipeline):
        return replace(self, name=self.name.enter_pipeline(pipeline))
    
    def get_name(self, pipe: Pipe):
        return self.name.get_name(pipe)

    def new_branch(self):
        return replace(self, branch=self.branch.get_next())

    def get_branch(self):
        return self.branch.get_branch()

    def push_node(self, node: DAGNode):
        self.nodes.append(node)

def build_graph(system: System):
    ctx = _TraversalContext(
        name=_NameBuilder(tuple(), defaultdict(int)),
        branch=_BranchCounter(0),
        nodes=[],
    )

    leaves = _parse_system(system, prev=None, ctx=ctx)
    leaves = _as_tuple(leaves)

    logging.info(f"Graph built with {len(ctx.nodes)} nodes and {len(leaves)} trees")
    return tuple(ctx.nodes), leaves


def _as_tuple(x) -> tuple:

    if not isinstance(x, tuple):
        x = (x,)

    return x


def _parse_system(
    pipe: Union[Pipeline, Pipe],
    prev: NodeOrNodes,
    ctx: _TraversalContext,
) -> NodeOrNodes:

    if isinstance(pipe, Pipeline):
        pipeline = pipe
        ctx = ctx.new_name(pipeline)

        if isinstance(pipeline, Sequential):
            curr = prev
            for pipe in pipeline:
                curr = _parse_system(pipe, prev=curr, ctx=ctx)

            if curr is prev:
                raise RuntimeError("Sequential pipeline is empty")

            return curr

        elif isinstance(pipeline, Concurrent):
            outs = tuple()

            for pipe in pipeline: # shadow pipe
                branch_ctx = (
                    ctx
                    if isinstance(pipe, Concurrent)
                    else ctx.new_branch()
                )

                out = _parse_system(pipe, prev=prev, ctx=branch_ctx)
                outs += _as_tuple(out)

            if not outs:
                raise RuntimeError("Concurrent pipeline is empty")

            return outs

        else:
            raise TypeError(f"Unknown pipeline type in system: {type(pipeline)}")

    elif isinstance(pipe, Pipe):
        return _lower_pipe(pipe, prev, ctx)

    else:
        raise TypeError(f"Unknown type in system: {type(pipe)}")


def _lower_pipe(
    pipe: Pipe, prev: NodeOrNodes, ctx: _TraversalContext
) -> DAGNode:  # TODO lower to GPU Nodes

    node = DAGNode(
        node_id=ctx.get_name(pipe),
        branch=ctx.get_branch(),
        pipe=pipe,
        args=_as_tuple(prev) if prev else (),
    )

    ctx.push_node(node)
    return node

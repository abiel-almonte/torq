"""Thin wrapper over the CUDA runtime for torq."""

from .types import *

def sync_device() -> None:
    """Synchronize CUDA GPU"""
    ...

def create_stream() -> cudaStream_t:
    """Create CUDA Stream"""
    ...

def get_stream_ptr(stream: cudaStream_t) -> ptr_t:
    """Get CUDA Stream pointer"""
    ...

def sync_stream(stream: cudaStream_t) -> None:
    """Synchronize CUDA Stream"""
    ...

def begin_capture(stream: cudaStream_t) -> None:
    """Begin CUDA Graph capture on the given stream"""
    ...

def end_capture(stream: cudaStream_t) -> cudaGraph_t:
    """End CUDA Graph capture on the given stream"""
    ...

def get_graph_ptr(graph: cudaGraph_t) -> ptr_t:
    """Get CUDA Graph pointer"""
    ...

def create_executor(graph: cudaGraph_t) -> cudaGraphExec_t:
    """Get CUDA GraphExec"""
    ...

def get_executor_ptr(executor: cudaGraphExec_t) -> ptr_t:
    """Get CUDA GraphExec pointer"""
    ...

def launch_graph(executor: cudaGraphExec_t, stream: cudaStream_t) -> None:
    """Execute CUDA Graph"""
    ...

def detect_synchronization(enable: bool) -> None:
    """Toggle cudaStreamSynchronize or cudaDeviceSynchronize detection"""
    ...

def get_synchronization_detected() -> bool:
    """Check if cudaStreamSynchronize or cudaDeviceSynchronize detected"""
    ...

def clear_synchronization_detected() -> None:
    """Clear synchronization detected flag"""
    ...

def detect_kernel_launch(enable: bool) -> None:
    """Toggle cuLaunchKernel detection"""
    ...

def get_kernel_launch_detected() -> bool:
    """Check if cuLaunchKernel detected"""
    ...

def clear_kernel_launch_detected() -> None:
    """Clear kernel launch detected flag"""
    ...

def inject_stream(stream: cudaStream_t) -> None:
    """Inject CUDA Stream into kernel launch"""
    ...

def clear_injection() -> None:
    """Remove injected CUDA Stream from kernel launch"""
    ...

def trace_memory(node_id: str) -> None:
    """Trace device pointers"""
    ...

def end_trace(node_id: str) -> None:
    """Stop tracing device pointers"""
    ...

def set_tracer(node_id: str) -> None:
    """Use the traced device pointers"""
    ...

def unset_tracer() -> None:
    """Stop using the traced device pointers"""
    ...

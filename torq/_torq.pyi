"""Thin wrapper over the CUDA runtime for torq."""

from typing import NewType
from types import CapsuleType

cudaStream_t = NewType("cudaStream_t", CapsuleType) 
"""CUDA Stream Handle"""

def device_sync() -> None: 
    """Synchronize CUDA GPU"""
    ...

def stream_create() -> cudaStream_t:
    """Create CUDA Stream"""
    ...

def stream_sync(stream: cudaStream_t) -> None:
    """Synchronize CUDA Stream"""
    ...
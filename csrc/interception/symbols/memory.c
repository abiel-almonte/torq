#include "memory.h"
#include "../state.h"

CUresult cuMemcpyAsync (
    CUdeviceptr dst,
    CUdeviceptr src, 
    size_t ByteCount,
    CUstream hStream
) {
    TRACE();
    static cuMemcpyAsync_fn real_memcpy = NULL;

    _LOAD_CUDA_SYMBOL(cuMemcpyAsync, real_memcpy);

    return real_memcpy(
        dst,
        src,
        ByteCount,
        (injected_stream)? (CUstream)(*injected_stream) : hStream
    );
}

CUresult cuMemsetD8Async (
    CUdeviceptr dst,
    unsigned char uc,
    size_t N,
    CUstream hStream
) {
    TRACE();
    static cuMemsetD8Async_fn real_memset = NULL;

    _LOAD_CUDA_SYMBOL(cuMemsetD8Async, real_memset);

    return real_memset(
        dst,
        uc,
        N,
        (injected_stream)? (CUstream)(*injected_stream) : hStream
    );
}

CUresult cuMemsetD32Async (
    CUdeviceptr dst,
    unsigned int ui,
    size_t N,
    CUstream hStream
) {
    TRACE();
    static cuMemsetD32Async_fn real_memset = NULL;

    _LOAD_CUDA_SYMBOL(cuMemsetD32Async, real_memset);

    return real_memset(
        dst,
        ui,
        N,
        (injected_stream)? (CUstream)(*injected_stream) : hStream
    );
}

CUresult cuMemcpy2DAsync_v2(
    const CUDA_MEMCPY2D* pCopy,
    CUstream hStream
){
    TRACE();
    static cuMemcpy2DAsync_v2_fn real_memcpy = NULL;

    _LOAD_CUDA_SYMBOL(cuMemcpy2DAsync_v2, real_memcpy);

    return real_memcpy(
        pCopy,
        (injected_stream)? (CUstream)(*injected_stream) : hStream
    );
}

CUresult cuMemcpy3DAsync_v2(
    const CUDA_MEMCPY3D* pCopy,
    CUstream hStream
){
    TRACE();
    static cuMemcpy3DAsync_v2_fn real_memcpy = NULL;

    _LOAD_CUDA_SYMBOL(cuMemcpy3DAsync_v2, real_memcpy);

    return real_memcpy(
        pCopy,
        (injected_stream)? (CUstream)(*injected_stream) : hStream
    );
}

CUresult cuMemcpyHtoDAsync_v2 (
    CUdeviceptr dst,
    const void* src, 
    size_t ByteCount,
    CUstream hStream
) {
    TRACE();
    static cuMemcpyHtoDAsync_v2_fn real_memcpy = NULL;

    _LOAD_CUDA_SYMBOL(cuMemcpyHtoDAsync_v2, real_memcpy);

    return real_memcpy(
        dst,
        src,
        ByteCount,
        (injected_stream)? (CUstream)(*injected_stream) : hStream
    );
}

CUresult cuMemcpyDtoHAsync_v2 (
    void* dstHost, 
    CUdeviceptr srcDevice,
    size_t ByteCount,
    CUstream hStream
) {
    TRACE();
    static cuMemcpyDtoHAsync_v2_fn real_memcpy = NULL;

    _LOAD_CUDA_SYMBOL(cuMemcpyDtoHAsync_v2, real_memcpy);

    return real_memcpy(
        dstHost,
        srcDevice,
        ByteCount,
        (injected_stream)? (CUstream)(*injected_stream) : hStream
    );
}

CUresult cuMemcpyDtoDAsync_v2 (
    CUdeviceptr dstDevice, 
    CUdeviceptr srcDevice,
    size_t ByteCount,
    CUstream hStream
) {
    TRACE();
    static cuMemcpyDtoDAsync_v2_fn real_memcpy = NULL;

    _LOAD_CUDA_SYMBOL(cuMemcpyDtoDAsync_v2, real_memcpy);

    return real_memcpy(
        dstDevice,
        srcDevice,
        ByteCount,
        (injected_stream)? (CUstream)(*injected_stream) : hStream
    );
}


cudaError_t cudaMemcpyAsync (
    void* dst,
    const void* src,
    size_t count,
    enum cudaMemcpyKind kind,
    cudaStream_t stream
) { 
    TRACE();
    static cudaMemcpyAsync_fn real_memcpy = NULL;

    _LOAD_CUDART_SYMBOL(cudaMemcpyAsync, real_memcpy);

    return real_memcpy(
        dst,
        src,
        count,
        kind,
        (injected_stream)? *injected_stream : stream
    );
}

cudaError_t cudaMemcpy2DAsync (
    void* dst,
    size_t dpitch,
    const void* src,
    size_t spitch,
    size_t width,
    size_t height,
    enum cudaMemcpyKind kind,
    cudaStream_t stream
) { 
    TRACE();
    static cudaMemcpy2DAsync_fn real_memcpy = NULL;

    _LOAD_CUDART_SYMBOL(cudaMemcpy2DAsync, real_memcpy);

    return real_memcpy(
        dst,
        dpitch,
        src,
        spitch,
        width,
        height,
        kind,
        (injected_stream)? *injected_stream : stream
    );
}

cudaError_t cudaMemcpy3DAsync(
    const struct cudaMemcpy3DParms* p,
    cudaStream_t stream
){
    TRACE();
    static cudaMemcpy3DAsync_fn real_memcpy = NULL;

    _LOAD_CUDART_SYMBOL(cudaMemcpy3DAsync, real_memcpy);

    return real_memcpy(
        p,
        (injected_stream)? *injected_stream : stream
    );
} 

cudaError_t cudaMemsetAsync(
    void* devPtr,
    int  value,
    size_t count,
    cudaStream_t stream
) {
    TRACE();
    static cudaMemsetAsync_fn real_memcpy = NULL;

    _LOAD_CUDART_SYMBOL(cudaMemsetAsync, real_memcpy);

    return real_memcpy( 
        devPtr,
        value,
        count,
        (injected_stream)? (*injected_stream) : stream
    );
}

cudaError_t cudaMemset2DAsync  (
    void* devPtr,
    size_t pitch,
    int value,
    size_t width,
    size_t height,
    cudaStream_t stream
) {
    TRACE();
    static cudaMemset2DAsync_fn real_memcpy = NULL;

    _LOAD_CUDART_SYMBOL(cudaMemset2DAsync, real_memcpy);

    return real_memcpy( 
        devPtr,
        pitch,
        value,
        width,
        height,
        (injected_stream)? (*injected_stream) : stream
    );
}

cudaError_t cudaMemset3DAsync  (
    struct cudaPitchedPtr pitchedDevPtr,
    int value,
    struct cudaExtent extent,
    cudaStream_t stream
) {
    TRACE();
    static cudaMemset3DAsync_fn real_memcpy = NULL;

    _LOAD_CUDART_SYMBOL(cudaMemset3DAsync, real_memcpy);

    return real_memcpy( 
        pitchedDevPtr,
        value,
        extent,
        (injected_stream)? (*injected_stream) : stream
    );
}


cudaError_t cudaStreamAttachMemAsync (
    cudaStream_t stream,
    void* devPtr,
    size_t length,
    unsigned int flag
) {
    TRACE();
    static cudaStreamAttachMemAsync_fn real_attach = NULL;

    _LOAD_CUDART_SYMBOL(cudaStreamAttachMemAsync, real_attach);

    return real_attach(
        (injected_stream)? *injected_stream : stream,
        devPtr,
        length,
        flag
    );
}
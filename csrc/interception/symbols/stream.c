#include "stream.h"
#include "../state.h"
#include <stdio.h>


cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t stream){
    TRACE();
    static cudnnSetStream_fn real_setter = NULL;

    _LOAD_CUDNN_SYMBOL(cudnnSetStream, real_setter);

    return real_setter(
        handle,
        (injected_stream) ? *injected_stream : stream
    );
}

CUresult cuStreamIsCapturing(
    CUstream hStream,
    CUstreamCaptureStatus* captureStatus
){
    TRACE();
    static cuStreamIsCapturing_fn real_iscapturing = NULL;

    _LOAD_CUDA_SYMBOL(cuStreamIsCapturing, real_iscapturing);

    return real_iscapturing(
        (injected_stream) ? (CUstream)*injected_stream : hStream, 
        captureStatus
    );
}

CUresult torq_cuStreamGetCaptureInfo_v2(
    CUstream hStream,
    CUstreamCaptureStatus* captureStatus_out,
    cuuint64_t* id_out,
    CUgraph* graph_out,
    const CUgraphNode** dependencies_out,
    const CUgraphEdgeData** edgeData_out,
    size_t* numDependencies_out
){
    TRACE();
    static torq_cuStreamGetCaptureInfo_v2_fn real_fn = NULL;

    if (!real_fn) { // must do this manually b/c cuStreamGetCaptureInfo_v2 is a macro
        void* cuda_handler = dlopen("libcuda.so", RTLD_LAZY);
        void* (*real_dlsym)(void*, const char*) = _get_real_dlsym();
        real_fn = (torq_cuStreamGetCaptureInfo_v2_fn)real_dlsym(cuda_handler, "cuStreamGetCaptureInfo_v2");
        if(!real_fn){
            printf("Unable to load torq_cuStreamGetCaptureInfo_v2_fn");
            abort();
        }
    }

    return real_fn(
        (injected_stream) ? (CUstream)*injected_stream : hStream,
        captureStatus_out,
        id_out,
        graph_out,
        dependencies_out,
        edgeData_out,
        numDependencies_out
    );
}

cudaError_t cudaStreamIsCapturing(
    cudaStream_t stream, 
    enum cudaStreamCaptureStatus* pCaptureStatus
){
    TRACE();
    static cudaStreamIsCapturing_fn real_iscapturing = NULL;

    _LOAD_CUDART_SYMBOL(cudaStreamIsCapturing, real_iscapturing);
    
    return real_iscapturing(
        (injected_stream) ? *injected_stream : stream, 
        pCaptureStatus
    );
}


cudaError_t cudaStreamGetCaptureInfo_v2(
    cudaStream_t stream,
    enum cudaStreamCaptureStatus* captureStatus_out,
    unsigned long long* id_out,
    cudaGraph_t* graph_out,
    const cudaGraphNode_t** dependencies_out,
    size_t* numDependencies_out
){
    TRACE();
    static cudaStreamGetCaptureInfo_v2_fn real_fn = NULL;

    _LOAD_CUDART_SYMBOL(cudaStreamGetCaptureInfo_v2, real_fn);
    
    return real_fn(
        (injected_stream) ? *injected_stream : stream,
        captureStatus_out,
        id_out,
        graph_out, 
        dependencies_out, 
        numDependencies_out
    );
}

cudaError_t cudaStreamGetPriority(
    cudaStream_t stream,
    int* prioirty
){
    TRACE();
    static cudaStreamGetPriority_fn real_getprio = NULL;

    _LOAD_CUDART_SYMBOL(cudaStreamGetPriority, real_getprio);

    return real_getprio(
        (injected_stream) ? *injected_stream : stream,
        prioirty
    );
}

cudaError_t cudaStreamDestroy (cudaStream_t stream){
    TRACE();
    static cudaStreamDestroy_fn real_destory = NULL;

    _LOAD_CUDART_SYMBOL(cudaStreamDestroy, real_destory);

    return real_destory(
        (injected_stream) ? *injected_stream : stream
    );
}

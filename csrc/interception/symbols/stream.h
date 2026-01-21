#ifndef _TORQ_STREAM_H
#define _TORQ_STREAM_H

#include "../utils.h"

typedef void* cudnnHandle_t;
typedef int cudnnStatus_t;
cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t stream);
typedef cudnnStatus_t (*cudnnSetStream_fn)(cudnnHandle_t, cudaStream_t);

// cuStreamIsCapturing (Driver API)
CUresult cuStreamIsCapturing(CUstream, CUstreamCaptureStatus*);
typedef CUresult (*cuStreamIsCapturing_fn)(CUstream, CUstreamCaptureStatus*);

// cuStreamGetCaptureInfo_v2 (Driver API)
CUresult torq_cuStreamGetCaptureInfo_v2(CUstream, CUstreamCaptureStatus*, cuuint64_t*, CUgraph*, const CUgraphNode**, const CUgraphEdgeData**, size_t*);
typedef CUresult (*torq_cuStreamGetCaptureInfo_v2_fn)(CUstream, CUstreamCaptureStatus*, cuuint64_t*, CUgraph*, const CUgraphNode**, const CUgraphEdgeData**, size_t*);

// cudaStreamIsCapturing (Runtime API)
cudaError_t cudaStreamIsCapturing(cudaStream_t, enum cudaStreamCaptureStatus*); 
typedef cudaError_t (*cudaStreamIsCapturing_fn)(cudaStream_t, enum cudaStreamCaptureStatus*); 

// cudaStreamGetCaptureInfo_v2 (Runtime API)
cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t, enum cudaStreamCaptureStatus*, unsigned long long*, cudaGraph_t*, const cudaGraphNode_t**, size_t*);
typedef cudaError_t (*cudaStreamGetCaptureInfo_v2_fn)(cudaStream_t, enum cudaStreamCaptureStatus*, unsigned long long*, cudaGraph_t*, const cudaGraphNode_t**, size_t*);

// cudaStreamGetPriority (Runtime API)
cudaError_t cudaStreamGetPriority(cudaStream_t stream, int* priority);
typedef cudaError_t (*cudaStreamGetPriority_fn)(cudaStream_t stream, int* priority);

// cudaStreamDestroy (Runtime API)
cudaError_t cudaStreamDestroy (cudaStream_t);
typedef cudaError_t (*cudaStreamDestroy_fn) (cudaStream_t);

#endif

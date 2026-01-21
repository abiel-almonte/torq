#ifndef _TORQ_MEMORY_H
#define _TORQ_MEMORY_H

#include "../utils.h"


// cuMemcpyAsync (Driver API)
CUresult cuMemcpyAsync(CUdeviceptr, CUdeviceptr, size_t, CUstream);
typedef CUresult (*cuMemcpyAsync_fn)(CUdeviceptr, CUdeviceptr, size_t, CUstream);

CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D*, CUstream);
typedef CUresult (*cuMemcpy2DAsync_v2_fn)(const CUDA_MEMCPY2D*, CUstream);

CUresult cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D*, CUstream);
typedef CUresult (*cuMemcpy3DAsync_v2_fn)(const CUDA_MEMCPY3D*, CUstream);

// cuMemcpyHtoDAsync_v2 (Driver API)
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr, const void*, size_t, CUstream);
typedef CUresult (*cuMemcpyHtoDAsync_v2_fn)(CUdeviceptr, const void*, size_t, CUstream);

// cuMemcpyDtoHAsync_v2 (Driver API)
CUresult cuMemcpyDtoHAsync_v2(void*, CUdeviceptr, size_t, CUstream);
typedef CUresult (*cuMemcpyDtoHAsync_v2_fn)(void*, CUdeviceptr, size_t, CUstream);

// cuMemcpyDtoDAsync_v2 (Driver API)
CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr, CUdeviceptr, size_t, CUstream);
typedef CUresult (*cuMemcpyDtoDAsync_v2_fn)(CUdeviceptr, CUdeviceptr, size_t, CUstream);

// cudaMemsetD8Async (Driver API)
CUresult cuMemsetD8Async(CUdeviceptr, unsigned char, size_t, CUstream);
typedef CUresult (*cuMemsetD8Async_fn)(CUdeviceptr, unsigned char, size_t, CUstream);

// cudaMemsetD32Async (Driver API)
CUresult cuMemsetD32Async(CUdeviceptr, unsigned int, size_t, CUstream);
typedef CUresult (*cuMemsetD32Async_fn)(CUdeviceptr, unsigned int, size_t, CUstream);

// cudaMemcpyAsync (Runtime API)*
cudaError_t cudaMemcpyAsync(void*, const void*, size_t, enum cudaMemcpyKind, cudaStream_t);
typedef cudaError_t (*cudaMemcpyAsync_fn)(void*, const void*, size_t, enum cudaMemcpyKind, cudaStream_t);

// cudaMemcpy2DAsync (Runtime API)
cudaError_t cudaMemcpy2DAsync(void*, size_t, const void*, size_t, size_t, size_t, enum cudaMemcpyKind, cudaStream_t);
typedef cudaError_t (*cudaMemcpy2DAsync_fn)(void*, size_t, const void*, size_t, size_t, size_t, enum cudaMemcpyKind, cudaStream_t);

// cudaMemcpy3DAsync (Runtime API)
cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms*, cudaStream_t); 
typedef cudaError_t (*cudaMemcpy3DAsync_fn)(const struct cudaMemcpy3DParms*, cudaStream_t); 

// cudaMemsetAsync (Runtime API)
cudaError_t cudaMemsetAsync(void*, int, size_t, cudaStream_t);
typedef cudaError_t (*cudaMemsetAsync_fn)(void*, int, size_t, cudaStream_t);

// cudaMemset2DAsync (Runtime API)
cudaError_t cudaMemset2DAsync(void*, size_t, int, size_t, size_t, cudaStream_t);
typedef cudaError_t (*cudaMemset2DAsync_fn)(void*, size_t, int, size_t, size_t, cudaStream_t);

// cudaMemset3DAsync (Runtime API)
cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr, int, struct cudaExtent, cudaStream_t);
typedef cudaError_t (*cudaMemset3DAsync_fn)(struct cudaPitchedPtr, int, struct cudaExtent, cudaStream_t);

// cudaStreamAttachMemAsync (Runtime API)
cudaError_t cudaStreamAttachMemAsync(cudaStream_t, void*, size_t, unsigned int);
typedef cudaError_t (*cudaStreamAttachMemAsync_fn)(cudaStream_t, void*, size_t, unsigned int);

#endif

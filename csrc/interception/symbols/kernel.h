#ifndef _TORQ_KERNEL_H
#define _TORQ_KERNEL_H

#include "../utils.h"


// cuLaunchKernel (Driver API)
CUresult cuLaunchKernel(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**);
typedef CUresult (*cuLaunchKernel_fn)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**);

// cuLaunchKernelEx (Driver API)
CUresult cuLaunchKernelEx(const CUlaunchConfig*, CUfunction, void**, void**);
typedef CUresult (*cuLaunchKernelEx_fn)(const CUlaunchConfig*, CUfunction, void**, void**);

// cuLaunchCooperativeKernel (Driver API)
CUresult cuLaunchCooperativeKernel(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**);
typedef CUresult (*cuLaunchCooperativeKernel_fn)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**);

// cudaLaunchKernel (Runtime API)
cudaError_t cudaLaunchKernel(const void*, dim3, dim3, void**, size_t, cudaStream_t);
typedef cudaError_t (*cudaLaunchKernel_fn)(const void*, dim3, dim3, void**, size_t, cudaStream_t);

// __cudaLaunchKernel (internal)
cudaError_t __cudaLaunchKernel(const void*, dim3, dim3, void**, size_t, cudaStream_t);
typedef cudaError_t (*__cudaLaunchKernel_fn)(const void*, dim3, dim3, void**, size_t, cudaStream_t);

// cudaLaunchKernelExC (Runtime API)
cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t*, const void*, void**);
typedef cudaError_t (*cudaLaunchKernelExC_fn)(const cudaLaunchConfig_t*, const void*, void**);

// cudaLaunchCooperativeKernel (Runtime API)
cudaError_t cudaLaunchCooperativeKernel(const void*, dim3, dim3, void**, size_t, cudaStream_t);
typedef cudaError_t (*cudaLaunchCooperativeKernel_fn)(const void*, dim3, dim3, void**, size_t, cudaStream_t);

#endif

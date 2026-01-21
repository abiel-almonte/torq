#include "kernel.h"
#include "../state.h"
#include <stdio.h>

CUresult cuLaunchKernel (
    CUfunction f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void** kernelParams,
    void** extra
) {
    TRACE();
    static cuLaunchKernel_fn real_kernel_launch = NULL;

    _LOAD_CUDA_SYMBOL(cuLaunchKernel, real_kernel_launch);

    if (kernel_detection_enabled) {
        kernel_detected = true;
    }

    return real_kernel_launch(
        f, 
        gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        sharedMemBytes,
        (injected_stream)? (CUstream)*injected_stream : hStream,
        kernelParams,
        extra
    );
}

CUresult cuLaunchKernelEx (
    const CUlaunchConfig* config,
    CUfunction f,
    void** kernelParams,
    void** extra
) {
    TRACE();
    static cuLaunchKernelEx_fn real_kernel_launch = NULL;

    _LOAD_CUDA_SYMBOL(cuLaunchKernelEx, real_kernel_launch);

    if (kernel_detection_enabled) {
        kernel_detected = true;
    }

    if (injected_stream) {
        const CUlaunchConfig injected_config = {
            config->gridDimX,
            config->gridDimY,
            config->gridDimZ,
            config->blockDimX,
            config->blockDimY,
            config->blockDimZ,
            config->sharedMemBytes,
            (CUstream)*injected_stream,
            config->attrs,
            config->numAttrs
        };

        return real_kernel_launch(
            &injected_config,
            f,
            kernelParams,
            extra
        );
    }

    return real_kernel_launch(
        config,
        f,
        kernelParams,
        extra
    );
}

cudaError_t cudaLaunchKernel ( 
    const void* func,
    dim3 gridDim,
    dim3 blockDim,
    void** args,
    size_t sharedMem,
    cudaStream_t stream
) {
    TRACE();
    static cudaLaunchKernel_fn real_kernel_launch = NULL;

    _LOAD_CUDART_SYMBOL(cudaLaunchKernel, real_kernel_launch);

    if (kernel_detection_enabled) {
        kernel_detected = true;
    }
    
    return real_kernel_launch(
        func,
        gridDim, blockDim,
        args,
        sharedMem,
        (injected_stream)? *injected_stream : stream
    );
}

cudaError_t __cudaLaunchKernel ( 
    const void* func,
    dim3 gridDim,
    dim3 blockDim,
    void** args,
    size_t sharedMem,
    cudaStream_t stream
) {
    TRACE();
    static __cudaLaunchKernel_fn real_kernel_launch = NULL;

    _LOAD_CUDART_SYMBOL(__cudaLaunchKernel, real_kernel_launch);

    if (kernel_detection_enabled) {
        kernel_detected = true;
    }
    
    return real_kernel_launch(
        func,
        gridDim, blockDim,
        args,
        sharedMem,
        (injected_stream)? *injected_stream : stream
    );
}


cudaError_t cudaLaunchKernelExC (
    const cudaLaunchConfig_t* config,
    const void* func,
    void** args
) {
    TRACE();
    static cudaLaunchKernelExC_fn real_kernel_launch = NULL;

    _LOAD_CUDART_SYMBOL(cudaLaunchKernelExC, real_kernel_launch);

    if (kernel_detection_enabled) {
        kernel_detected = true;
    }

    if (injected_stream) {
        const cudaLaunchConfig_t injected_config = {
            config->gridDim,
            config->blockDim,
            config->dynamicSmemBytes,
            *injected_stream,
            config->attrs,
            config->numAttrs
        };

        return real_kernel_launch(
            &injected_config,
            func,
            args
        );
    }

    return real_kernel_launch(
        config,
        func,
        args
    );
}

cudaError_t cudaLaunchCooperativeKernel(
    const void *func,
    dim3 gridDim,
    dim3 blockDim,
    void **args,
    size_t sharedMem,
    cudaStream_t stream
) {
    TRACE();
    static cudaLaunchCooperativeKernel_fn real_kernel_launch = NULL;

    _LOAD_CUDART_SYMBOL(cudaLaunchCooperativeKernel, real_kernel_launch);

    if (kernel_detection_enabled) {
        kernel_detected = true;
    }
    
    return real_kernel_launch(
        func,
        gridDim,
        blockDim,
        args,
        sharedMem,
        (injected_stream)? *injected_stream : stream
    );
}

CUresult cuLaunchCooperativeKernel(
    CUfunction f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void** kernelParams
) {
    TRACE();
    static cuLaunchCooperativeKernel_fn real_kernel_launch = NULL;

    _LOAD_CUDA_SYMBOL(cuLaunchCooperativeKernel, real_kernel_launch);

    if (kernel_detection_enabled) {
        kernel_detected = true;
    }

    return real_kernel_launch(
        f,
        gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        sharedMemBytes,
        (injected_stream) ? (CUstream)*injected_stream : hStream,
        kernelParams
    );
}

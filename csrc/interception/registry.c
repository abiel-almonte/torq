#include "registry.h"
#include "symbols/kernel.h"
#include "symbols/memory.h"
#include "symbols/sync.h"
#include "symbols/stream.h"
#include <string.h>


const symbol_entry_t torq_registry[] = {
    // cuGetProcAddress alias
    SYMBOL_ALIAS("cuGetProcAddress", cuGetProcAddress_v2),
    SYMBOL_ENTRY(cuGetProcAddress_v2),

    // Event - Runtime API
    SYMBOL_ENTRY(cudaEventRecord),
    
    // Kernel launch - Driver API
    SYMBOL_ENTRY(cuLaunchKernel),
    SYMBOL_ENTRY(cuLaunchKernelEx),
    SYMBOL_ENTRY(cuLaunchCooperativeKernel),
    
    // Kernel launch - Runtime API
    SYMBOL_ENTRY(cudaLaunchKernel),
    SYMBOL_ENTRY(__cudaLaunchKernel),
    SYMBOL_ENTRY(cudaLaunchKernelExC),
    SYMBOL_ALIAS("cudaLaunchKernelExC_v11060", cudaLaunchKernelExC),
    SYMBOL_ENTRY(cudaLaunchCooperativeKernel),
    
    // Memory - Driver API
    SYMBOL_ENTRY(cuMemAlloc_v2),
    SYMBOL_ENTRY(cuMemcpyAsync),
    SYMBOL_ENTRY(cuMemcpy2DAsync_v2),
    SYMBOL_ENTRY(cuMemcpy3DAsync_v2),
    SYMBOL_ENTRY(cuMemcpyHtoDAsync_v2),
    SYMBOL_ENTRY(cuMemcpyDtoHAsync_v2),
    SYMBOL_ENTRY(cuMemcpyDtoDAsync_v2),
    SYMBOL_ENTRY(cuMemsetD32Async),
    
    // Memory - Runtime API
    SYMBOL_ENTRY(cudaMalloc),
    SYMBOL_ENTRY(cudaMemcpyAsync),
    SYMBOL_ENTRY(cudaMemcpy2DAsync),
    SYMBOL_ENTRY(cudaMemcpy3DAsync),
    SYMBOL_ENTRY(cudaMemsetAsync),
    SYMBOL_ENTRY(cudaMemset2DAsync),
    SYMBOL_ENTRY(cudaMemset3DAsync),
    SYMBOL_ENTRY(cudaStreamAttachMemAsync),
    
    // Sync - Runtime API
    SYMBOL_ENTRY(cudaStreamSynchronize),
    SYMBOL_ENTRY(cudaDeviceSynchronize),
    SYMBOL_ENTRY(cudaGetDevice),

    // Stream - Driver API
    SYMBOL_ENTRY(cuStreamIsCapturing),
    SYMBOL_ALIAS("cuStreamGetCaptureInfo", torq_cuStreamGetCaptureInfo_v2),
    SYMBOL_ALIAS("cuStreamGetCaptureInfo_v2", torq_cuStreamGetCaptureInfo_v2),

    // Stream - Runtime API
    SYMBOL_ENTRY(cudaStreamIsCapturing),
    SYMBOL_ALIAS("cudaStreamIsCapturing_v10000", cudaStreamIsCapturing),
    SYMBOL_ENTRY(cudaStreamGetCaptureInfo_v2),
    SYMBOL_ALIAS("cudaStreamGetCaptureInfo_v2_v11030", cudaStreamGetCaptureInfo_v2),
    SYMBOL_ENTRY(cudaStreamGetPriority),
    SYMBOL_ENTRY(cudaStreamDestroy),

    {NULL, NULL} // terminal entry
};

void* torq_lookup_fn(const char* symbol){
    if (!symbol) {
        return NULL;
    }

    for (const symbol_entry_t* entry = &torq_registry[0]; entry->fn && entry->name; entry++) {
        if (strcmp(entry->name, symbol) == 0){
            return entry->fn;
        }
    }

    return NULL;
}

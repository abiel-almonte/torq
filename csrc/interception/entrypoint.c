#include "registry.h"
#include "state.h"
#include <stdio.h>

// all dynamic symbol lookups
void* dlsym(void* handle, const char* symbol) {
    void* (*real_dlsym)(void*, const char*) = _get_real_dlsym();

    fprintf(stderr, "[torq] dlsym lookup: %s\n", symbol);

    void* fn = torq_lookup_fn(symbol);
    return fn ? fn : real_dlsym(handle, symbol);
}

// NVIDIA symbol lookups
CUresult cuGetProcAddress_v2(
    const char* symbol,
    void** pfn,
    int cudaVersion,
    cuuint64_t flags,
    CUdriverProcAddressQueryResult* symbolStatus
) {
    static cuGetProcAddress_v2_fn real_fn = NULL;
    
    _LOAD_CUDA_SYMBOL(cuGetProcAddress_v2, real_fn);

    fprintf(stderr, "[torq] cuGetProcAddress_v2 lookup: %s\n", symbol);


    CUresult result = real_fn(symbol, pfn, cudaVersion, flags, symbolStatus);
    
    if (result == CUDA_SUCCESS && pfn && *pfn) {
        void* fn = torq_lookup_fn(symbol);
        if (fn) {
            fprintf(stderr, "[torq] cuGetProcAddress_v2 INTERCEPTED: %s (our=%p real=%p)\n", symbol, fn, *pfn);
            *pfn = fn;
        }
    }
    
    return result;
}

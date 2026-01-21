#define _GNU_SOURCE
#ifndef _TORQ_INTERCEPTION_UTILS_H
#define _TORQ_INTERCEPTION_UTILS_H

#include <stdio.h>
#include <dlfcn.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "state.h"


static inline void* _get_real_dlsym(void){
    static void* (*real_dlsym)(void*, const char*) = NULL;
    
    if (!real_dlsym) {
        real_dlsym = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
        if (!real_dlsym) {
            fprintf(stderr, "Unable to load real dlsym");
            abort();
        }
    }
    
    return real_dlsym;
}

#define TRACE() fprintf(stderr, "[torq] %s\n", __func__)

// get the real symbol
#define _LOAD_SYMBOL(handler, lib, real_fn_symbol, real_fn) { \
    if (!(real_fn)) { \
        void* (*_real_dlsym)(void*, const char*) = _get_real_dlsym(); \
        if (!(handler)) { \
            handler = dlopen(#lib, RTLD_LAZY); \
            if (!(handler)) { \
                fprintf(stderr, "dlopen failed: %s\n", dlerror()); \
                abort(); \
            } \
        } \
        real_fn = (real_fn_symbol##_fn)_real_dlsym(handler, #real_fn_symbol); \
        if (!real_fn) { \
            fprintf(stderr, "Unable to fetch real " #real_fn_symbol); \
            abort(); \
        } \
    } \
}

// get the real symbol from cuda runtime
#define _LOAD_CUDART_SYMBOL(real_fn_symbol, real_fn) { \
    _LOAD_SYMBOL(cudart_handler, libcudart.so, real_fn_symbol, real_fn) \
}

// get the real symbol from cuda driver
#define _LOAD_CUDA_SYMBOL(real_fn_symbol, real_fn) { \
    _LOAD_SYMBOL(cuda_handler, libcuda.so, real_fn_symbol, real_fn) \
}

// get the real symbol from cudnn
// needs verion number for some reason
#define _LOAD_CUDNN_SYMBOL(real_fn_symbol, real_fn) { \
    _LOAD_SYMBOL(cudnn_handler, libcudnn.so.9, real_fn_symbol, real_fn) \
}


#endif
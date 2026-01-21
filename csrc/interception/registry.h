#ifndef _TORQ_REGISTRY_H
#define _TORQ_REGISTRY_H

#include "utils.h"


typedef struct {
    const char* name;
    void* fn;
} symbol_entry_t;

#define SYMBOL_ENTRY(symbol) {#symbol, (void*)symbol}
#define SYMBOL_ALIAS(name, fn) {name, (void*)fn}

CUresult cuGetProcAddress_v2(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);
typedef CUresult (*cuGetProcAddress_v2_fn)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);

extern const symbol_entry_t torq_registry[];
void* torq_lookup_fn(const char* symbol);

#endif

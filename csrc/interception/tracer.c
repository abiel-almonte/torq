#include "tracer.h"
#include <stdio.h>
#include <string.h>

tracer_entry_t* tracer_registry = NULL;
size_t tracer_registry_size = 0;
size_t tracer_registry_capacity = 0;
tracer_t* global_tracer = NULL;

void tracer_registry_init(void) {
    tracer_registry = (tracer_entry_t*)malloc(INIT_BUFFERS * sizeof(tracer_entry_t));
    if (!tracer_registry) {
        fprintf(stderr, "Failed to init tracer_registry\n");
        abort();
    }
    tracer_registry->node_id = NULL;
    tracer_registry->tracer = NULL;
    tracer_registry_size = 0;
    tracer_registry_capacity = INIT_BUFFERS;
}

void tracer_registry_resize(void) {
    size_t new_capacity = 2 * tracer_registry_capacity; // doubling policy
    tracer_entry_t* new_tracer_registry = (tracer_entry_t*)malloc(new_capacity * sizeof(tracer_entry_t));
    if (!new_tracer_registry) {
        fprintf(stderr, "Failed to resize tracer_registry");
        abort();
    }

    memcpy((void*)new_tracer_registry, (void*)tracer_registry, tracer_registry_size * sizeof(tracer_entry_t));
    free(tracer_registry);
    tracer_registry = new_tracer_registry; // assign to global
    tracer_registry_capacity = new_capacity;
}

void tracer_registry_add(const char* node_id) {
    if (!tracer_registry) {
        tracer_registry_init();
    }

    
    if (tracer_registry_size == tracer_registry_capacity) {
        tracer_registry_resize();
    }
    
    tracer_registry[tracer_registry_size].node_id = node_id;
    tracer_registry[tracer_registry_size].tracer = tracer_init();
    tracer_registry_size += 1;
}

tracer_t* tracer_lookup(const char* node_id) {
    if (!tracer_registry) {
        tracer_registry_init();
    }

    for (size_t i = 0; i < tracer_registry_size; i++) { // linear search
        if (strcmp(tracer_registry[i].node_id, node_id) == 0) {
            return tracer_registry[i].tracer;
        }
    }
    return NULL;
}

tracer_t* tracer_init(void) {
    tracer_t* tracer = (tracer_t*)malloc(sizeof(tracer_t)); // init single tracer
    if (!tracer){
        fprintf(stderr, "Failed to init a tracer\n");
        abort();
    }

    void** device_ptrs = (void**)malloc(INIT_BUFFERS * sizeof(void*));
    if (!device_ptrs) {
        fprintf(stderr, "Failed to init a tracer's buffers\n");
        abort();
    }

    tracer->device_ptrs = device_ptrs;
    tracer->next = 0;
    tracer->size = 0;
    tracer->capacity = INIT_BUFFERS;
    tracer->status.done = false;
    tracer->status.in_progress = false;
    return tracer;
}

void tracer_resize(tracer_t* tracer) {
    size_t new_capacity = 2 * tracer->capacity;
    void** new_device_ptrs = (void**)malloc(new_capacity * sizeof(void*));
    if (!new_device_ptrs) {
        fprintf(stderr, "Failed to resize a tracer\n");
        abort();
    }

    free(tracer->device_ptrs);
    tracer->device_ptrs = new_device_ptrs;
    tracer->capacity = new_capacity; 
}

void tracer_push(tracer_t* tracer, void* device_ptr) {
    if (tracer->size == tracer->capacity) {
        tracer_resize(tracer);
    }

    tracer->device_ptrs[tracer->size] = device_ptr;
    tracer->size += 1;
}

void* next_device_ptr(tracer_t* tracer) {
    void* device_ptr = tracer->device_ptrs[tracer->next];
    tracer->next = (tracer->next + 1) % tracer->size; // round robin policy
    return device_ptr;
}

#ifndef _TORQ_TRACER_H
#define _TORQ_TRACER_H

#include <stdlib.h>
#include <stdbool.h>

#define INIT_BUFFERS 10

typedef struct {
    bool in_progress;
    bool done;
} status_t;

typedef struct {
    void** device_ptrs;
    size_t next;
    size_t size;
    size_t capacity;
    status_t status;
} tracer_t;

typedef struct {
    const char* node_id;
    tracer_t* tracer;
} tracer_entry_t;

extern tracer_entry_t* tracer_registry;
extern size_t tracer_registry_size;
extern size_t tracer_registry_capacity;
extern tracer_t* global_tracer;

void tracer_registry_init(void);
void tracer_registry_resize(void);
void tracer_registry_add(const char*);
tracer_t* tracer_lookup(const char*);

tracer_t* tracer_init(void);
void tracer_push(tracer_t*, void*);
void tracer_resize(tracer_t*);

void* next_device_ptr(tracer_t*);


#endif

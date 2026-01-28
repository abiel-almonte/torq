#ifndef _TORQ_STATE_H
#define _TORQ_STATE_H

#include <cuda_runtime.h>
#include <stdbool.h>

extern bool torq_debug;

// Shared state for interception
extern bool sync_detection_enabled;
extern bool sync_detected;

extern bool kernel_detection_enabled;
extern bool kernel_detected;

extern cudaStream_t* injected_stream;

#endif

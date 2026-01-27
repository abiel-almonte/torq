#ifndef _TORQ_INTERCEPTION_H
#define _TORQ_INTERCEPTION_H

#include <Python.h>
#include <cuda_runtime.h>


// sync detection
PyObject* detect_sync(PyObject* self, PyObject* args);
PyObject* clear_sync_detected(PyObject* self, PyObject* args);
PyObject* get_sync_detected(PyObject* self, PyObject* args);

// kernel launch detection
PyObject* detect_kernel(PyObject* self, PyObject* args);
PyObject* clear_kernel_detected(PyObject* self, PyObject* args);
PyObject* get_kernel_detected(PyObject* self, PyObject* args);

// stream injection
PyObject* inject_stream(PyObject* self, PyObject* args);
PyObject* clear_injection(PyObject* self, PyObject* args);

// trace device pointers
PyObject* trace_memory(PyObject* self, PyObject* args);
PyObject* end_trace(PyObject* self, PyObject* args);
PyObject* set_tracer(PyObject* self, PyObject* args);
PyObject* unset_tracer(PyObject* self, PyObject* args);

#endif
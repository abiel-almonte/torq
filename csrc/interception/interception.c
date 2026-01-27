#include "interception.h"
#include "tracer.h"
#include "state.h"
#include <stdio.h>

bool torq_debug = false;

void* cudart_handler = NULL;
void* cuda_handler = NULL;
void* cudnn_handler = NULL;

bool sync_detection_enabled = false;
bool sync_detected = false;

bool kernel_detection_enabled = false;
bool kernel_detected = false;

cudaStream_t* injected_stream = NULL;

bool memory_tracing_enabled = false;

PyObject* get_sync_detected(PyObject* self, PyObject* args){
    if (sync_detected) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

PyObject* clear_sync_detected(PyObject* self, PyObject* args){
    sync_detected = false;
    Py_RETURN_NONE;
}

PyObject* detect_sync(PyObject* self, PyObject* args){
    int value;

    if (!PyArg_ParseTuple(args, "p", &value)){
        return NULL;
    }

    sync_detection_enabled = (bool)value;

    Py_RETURN_NONE;
}

PyObject* get_kernel_detected(PyObject* self, PyObject* args){
    if (kernel_detected) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

PyObject* clear_kernel_detected(PyObject* self, PyObject* args){
    kernel_detected = false;
    Py_RETURN_NONE;
}

PyObject* detect_kernel(PyObject* self, PyObject* args){
    int value;

    if (!PyArg_ParseTuple(args, "p", &value)){
        return NULL;
    }

    kernel_detection_enabled = (bool)value;

    Py_RETURN_NONE;
}


PyObject* inject_stream(PyObject* self, PyObject* args) {
    PyObject* capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)){
        return NULL;
    }

    injected_stream = (cudaStream_t*)PyCapsule_GetPointer(capsule, "cudaStream_t");

    Py_RETURN_NONE;
}

PyObject* clear_injection(PyObject* self, PyObject* args) {
    injected_stream = NULL;
    Py_RETURN_NONE;
}

PyObject* trace_memory(PyObject* self, PyObject* args) {
    const char* node_id;
    if (!PyArg_ParseTuple(args, "s", &node_id)) {
        return NULL;
    }

    tracer_t* tracer = tracer_lookup(node_id);
    
    if (tracer){
        if (tracer->status.done) {
            PyErr_SetString(PyExc_RuntimeError, "This trace is already finalized. Call clear_trace(node_id) before starting a new session.");
            return NULL;
        } // if it is not in progress then make it in progress. if if is in_progres, perfect, assign true anyways. 
    } else {
        tracer_registry_add(strdup(node_id));
        tracer = tracer_lookup(node_id);
    }

    tracer->status.in_progress = true;
    global_tracer = tracer;
    memory_tracing_enabled = true;

    Py_RETURN_NONE;
}

PyObject* end_trace(PyObject* self, PyObject* args) {
    const char* node_id;
    if(!PyArg_ParseTuple(args, "s", &node_id)){
        return NULL;
    }

    tracer_t* tracer = tracer_lookup(node_id);
    if (!tracer){
        PyErr_SetString(PyExc_RuntimeError, "This trace has not began. Call trace_memory(node_id) to begin tracing session.");
        return NULL;
    }

    tracer->status.in_progress = false;
    tracer->status.done = true;
    Py_RETURN_NONE;
}

PyObject* set_tracer(PyObject* self, PyObject* args) {
    const char* node_id;
    if(!PyArg_ParseTuple(args, "s", &node_id)){
        return NULL;
    }

    tracer_t* tracer = tracer_lookup(node_id);
    if (!tracer){
        PyErr_SetString(PyExc_RuntimeError, "Tracer for this node does not exist.");
        return NULL;
    }

    tracer = tracer;
    Py_RETURN_NONE;
}

PyObject* unset_tracer(PyObject* self, PyObject* args) {
    global_tracer = NULL;
    Py_RETURN_NONE;
}

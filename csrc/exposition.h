#ifndef _TORQ_EXPOSITION_H
#define _TORQ_EXPOSITION_H
#include "utils.h"


// synchronization
PyObject* device_sync(PyObject* self, PyObject *args);
PyObject* stream_sync(PyObject* self, PyObject* args);

// stream handler
void stream_destroy(PyObject *capsule);
PyObject* stream_create(PyObject* self, PyObject *args);
PyObject* get_stream_ptr(PyObject* self, PyObject *args);

// graph handler
void graph_destroy(PyObject* capsule);
PyObject* capture_begin(PyObject* self, PyObject* args); // graph_create
PyObject* capture_end(PyObject* self, PyObject* args); // graph_create
PyObject* get_graph_ptr(PyObject* self, PyObject *args);

// graph executor handler
void executor_destroy(PyObject* capsule);
PyObject* executor_create(PyObject* self, PyObject *args);
PyObject* get_executor_ptr(PyObject* self, PyObject *args);

// fn to launch graph
PyObject* graph_launch(PyObject* self, PyObject *args);



#endif
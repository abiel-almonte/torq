#include "exposition.h"


PyObject* device_sync(PyObject* self, PyObject *args){
    _CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize CUDA GPU");
    Py_RETURN_NONE;
}


void stream_destroy(PyObject *capsule){ 
    cudaStream_t* stream = (cudaStream_t*)PyCapsule_GetPointer(capsule, "cudaStream_t");

    if (stream) { 
        cudaStreamDestroy(*stream);
        free(stream);
    }
}

PyObject* stream_create(PyObject* self, PyObject *args){ 
    cudaStream_t* stream = (cudaStream_t*)malloc(sizeof(cudaStream_t));

    if (!stream) { 
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate CUDA stream");
        return NULL;
    }

    _CUDA_CHECK(cudaStreamCreate(stream), "Failed to create CUDA stream");
    return PyCapsule_New(stream, "cudaStream_t", stream_destroy);
}

PyObject* get_stream_ptr(PyObject* self, PyObject *args){
    PyObject* capsule;

    if (!PyArg_ParseTuple(args, "O", &capsule)){
        return NULL;
    }

    cudaStream_t* stream = (cudaStream_t*)PyCapsule_GetPointer(capsule, "cudaStream_t");
    if (!stream){
        return NULL;
    }

    return PyLong_FromVoidPtr((void*)(*stream));
}

PyObject* stream_sync(PyObject* self, PyObject* args){
    PyObject* capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)){
        return NULL;
    }
    
    cudaStream_t* stream = (cudaStream_t*)PyCapsule_GetPointer(capsule, "cudaStream_t"); 
    if (!stream){
        return NULL;
    }
    
    _CUDA_CHECK(cudaStreamSynchronize(*stream), "Failed to synchronize CUDA stream");
    Py_RETURN_NONE;
}


PyObject* capture_begin(PyObject* self, PyObject* args) {
    PyObject* capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return NULL;
    }
    
    cudaStream_t* stream = (cudaStream_t*)PyCapsule_GetPointer(capsule, "cudaStream_t");
    if (!stream){
        return NULL;
    }
    
    _CUDA_CHECK(cudaStreamBeginCapture(*stream, cudaStreamCaptureModeGlobal), "Failed to begin graph capture");
    Py_RETURN_NONE;
}

void graph_destroy(PyObject* capsule) {
    cudaGraph_t* graph = (cudaGraph_t*)PyCapsule_GetPointer(capsule, "cudaGraph_t");
    
    if (graph) {
        cudaGraphDestroy(*graph);
        free(graph);
    }
}

PyObject* capture_end(PyObject* self, PyObject* args) {
    PyObject* capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return NULL;
    }
    
    cudaStream_t* stream = (cudaStream_t*)PyCapsule_GetPointer(capsule, "cudaStream_t");
    if (!stream){
        return NULL;
    }
    
    cudaGraph_t* graph = (cudaGraph_t*)malloc(sizeof(cudaGraph_t));
    if (!graph){
        PyErr_SetString(PyExc_MemoryError,"Failed to allocate CUDA graph");
        return NULL;
    }
    
    _CUDA_CHECK(cudaStreamEndCapture(*stream, graph), "Failed to end graph capture");
    return PyCapsule_New(graph, "cudaGraph_t", graph_destroy);
}

PyObject* get_graph_ptr(PyObject* self, PyObject *args){
    PyObject* capsule;
    
    if (!PyArg_ParseTuple(args, "O", &capsule)){
        return NULL;
    }
    
    cudaGraph_t* graph = (cudaGraph_t*)PyCapsule_GetPointer(capsule, "cudaGraph_t");
    if (!graph){
        return NULL;
    }

    return PyLong_FromVoidPtr((void*)(*graph));
}


void executor_destroy(PyObject* capsule){
    cudaGraphExec_t* executor = (cudaGraphExec_t*)PyCapsule_GetPointer(capsule, "cudaGraphExec_t");

    if (executor) {
        cudaGraphExecDestroy(*executor);
        free(executor);
    }
}

PyObject* executor_create(PyObject* self, PyObject *args){
    PyObject* capsule;
    
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return NULL;
    }

    cudaGraph_t* graph = (cudaGraph_t*)PyCapsule_GetPointer(capsule, "cudaGraph_t");
    if (!graph){
        return NULL;
    }

    cudaGraphExec_t* executor = (cudaGraphExec_t*)malloc(sizeof(cudaGraphExec_t));

    _CUDA_CHECK(cudaGraphInstantiate(executor, *graph, NULL), "Failed to instantiate executor");

    return PyCapsule_New(executor, "cudaGraphExec_t", executor_destroy);
}


PyObject* get_executor_ptr(PyObject* self, PyObject *args){
    PyObject* capsule;
    
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return NULL;
    }
    
    cudaGraphExec_t* exec = (cudaGraphExec_t*)PyCapsule_GetPointer(capsule, "cudaGraphExec_t");
    if (!exec) {
        return NULL;
    }

    return PyLong_FromVoidPtr((void*)(*exec));
}


PyObject* graph_launch(PyObject* self, PyObject *args){
    PyObject* executor_capsule;
    PyObject* stream_capsule;
    
    if (!PyArg_ParseTuple(args, "OO", &executor_capsule, &stream_capsule)) {
        return NULL;
    }
    
    cudaGraphExec_t* exec = (cudaGraphExec_t*)PyCapsule_GetPointer(executor_capsule, "cudaGraphExec_t");
    if (!exec) {
        return NULL;
    }

    cudaStream_t* stream = (cudaStream_t*)PyCapsule_GetPointer(stream_capsule, "cudaStream_t");
    if (!stream) {
        return NULL;
    }

    _CUDA_CHECK(cudaGraphLaunch(*exec, *stream), "Failed to launch graph");
    Py_RETURN_NONE;
}

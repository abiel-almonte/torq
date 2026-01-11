#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <cuda_runtime.h>

#define _CUDA_CHECK(expr, message) { \
    cudaError_t _err = expr; \
    if (_err != cudaSuccess) { \
        PyErr_SetString(PyExc_RuntimeError, message); \
        return NULL; \
    } \
} \

static PyObject* device_sync(PyObject* self, PyObject *args){
    _CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize CUDA GPU");
    Py_RETURN_NONE;
}

static void stream_destroy(PyObject *capsule){ 
    cudaStream_t* stream = (cudaStream_t*)PyCapsule_GetPointer(capsule, "cudaStream_t");

    if (stream) { 
        cudaStreamDestroy(*stream);
        free(stream);
    }
}

static PyObject* stream_create(PyObject* self, PyObject *args){ 
    cudaStream_t* stream = (cudaStream_t*)malloc(sizeof(cudaStream_t));

    if (!stream) { 
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate CUDA stream");
        return NULL;
    }

    _CUDA_CHECK(cudaStreamCreate(stream), "Failed to create CUDA stream");
    return PyCapsule_New(stream, "cudaStream_t", stream_destroy);
}

static PyObject* stream_sync(PyObject* self, PyObject* args){
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

static PyMethodDef _torq_methods [] = {
    {"stream_create", &stream_create, METH_NOARGS, "Create CUDA Stream"},
    {"stream_sync", &stream_sync, METH_VARARGS, "Synchronize CUDA Stream"},
    {"device_sync", &device_sync, METH_NOARGS, "Synchronize CUDA GPU"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef _torq_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_torq",
    .m_doc = "Thin wrapper over the CUDA runtime for torq.",
    .m_methods = _torq_methods,

    .m_size = -1,
    .m_slots = NULL,
    .m_traverse = NULL,
    .m_clear = NULL,
    .m_free = NULL
};

PyMODINIT_FUNC PyInit__torq(void){
    return PyModule_Create(&_torq_module);
}

/* TODO:
capture_begin(stream)
capture_end(stream, graph)

graph_instantiate(executor, graph)
graph_launch(executor, stream)
graph_destroy(executor, graph)
*/

#define PY_SSIZE_T_CLEAN
#include "interception.h"
#include "exposition.h"


static PyMethodDef _torq_methods [] = {
    {"clear_kernel_launch_detected", &clear_kernel_detected, METH_NOARGS, "Clear kernel_deteced flag"},
    {"get_kernel_launch_detected", &get_kernel_detected, METH_NOARGS, "Check if cuLaunchKernel detected"},
    {"detect_kernel_launch", &detect_kernel, METH_VARARGS, "Toggle cuLaunchKernel detection"},

    {"clear_synchronization_detected", &clear_sync_detected, METH_NOARGS, "Clear sync_detected flag"},
    {"get_synchronization_detected", &get_sync_detected, METH_NOARGS, "Check if cudaStreamSynchronize detected"},
    {"detect_synchronization", &detect_sync, METH_VARARGS, "Toggle cudaStreamSynchronize detection"},

    {"launch_graph", &graph_launch, METH_VARARGS, "Launch CUDA Graph"},

    {"create_executor", &executor_create, METH_VARARGS, "Create CUDA GraphExec"},
    {"get_executor_ptr", &get_executor_ptr, METH_VARARGS, "Get CUDA GraphExec pointer"},

    {"begin_capture", &capture_begin, METH_VARARGS, "Begin CUDA Graph capture"},
    {"end_capture", &capture_end, METH_VARARGS, "End CUDA Graph capture"},
    {"get_graph_ptr", &get_graph_ptr, METH_VARARGS, "Get CUDA Graph pointer"},

    {"create_stream", &stream_create, METH_NOARGS, "Create CUDA Stream"},
    {"get_stream_ptr", &get_stream_ptr, METH_VARARGS, "Get CUDA Stream pointer"},
    {"sync_stream", &stream_sync, METH_VARARGS, "Synchronize CUDA Stream"},
    
    {"sync_device", &device_sync, METH_NOARGS, "Synchronize CUDA GPU"},

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

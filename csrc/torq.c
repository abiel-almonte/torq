#define PY_SSIZE_T_CLEAN
#include "interception/interception.h"
#include "exposition.h"


static PyMethodDef _torq_methods [] = {
    {"unset_tracer", &unset_tracer, METH_NOARGS,  "Stop using the traced device pointers"},
    {"set_tracer", &set_tracer, METH_VARARGS,  "Use the traced device pointers"},
    {"end_trace", &end_trace, METH_VARARGS, "Stop tracing device pointers"},
    {"trace_memory", &trace_memory, METH_VARARGS,  "Trace device pointers"},

    {"clear_injection", &clear_injection, METH_NOARGS, "Remove injected CUDA Stream from kernel launch"},
    {"inject_stream", &inject_stream, METH_VARARGS, "Inject CUDA Stream into kernel launch"},

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

PyMODINIT_FUNC PyInit__C(void){
    return PyModule_Create(&_torq_module);
}

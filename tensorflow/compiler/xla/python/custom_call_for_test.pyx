# cython: language_level=2
# distutils: language = c++

# Test case for defining a XLA custom call target in Cython, and registering
# it via the xla_client SWIG API.

from cpython.pycapsule cimport PyCapsule_New

cdef void test_subtract_f32(void* out_ptr, void** data_ptr) nogil:
  cdef float a = (<float*>(data_ptr[0]))[0]
  cdef float b = (<float*>(data_ptr[1]))[0]
  cdef float* out = <float*>(out_ptr)
  out[0] = a - b


cpu_custom_call_targets = {}

cdef register_custom_call_target(fn_name, void* fn):
  cdef const char* name = "xla._CUSTOM_CALL_TARGET"
  cpu_custom_call_targets[fn_name] = PyCapsule_New(fn, name, NULL)

register_custom_call_target(b"test_subtract_f32", <void*>(test_subtract_f32))

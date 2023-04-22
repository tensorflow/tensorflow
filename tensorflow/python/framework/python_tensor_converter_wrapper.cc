/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Note: This library is only used by python_tensor_converter_test.  It is
// not meant to be used in other circumstances.

#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "tensorflow/python/eager/pywrap_tfe.h"
#include "tensorflow/python/framework/python_tensor_converter.h"

#if PY_MAJOR_VERSION < 3
// Python 2.x:
#define PY_STRING_INTERN_FROM_STRING(x) (PyString_InternFromString(x))
#define PY_INT_AS_LONG(x) (PyInt_AsLong(x))
#define PY_INT_FROM_LONG(x) (PyInt_FromLong(x))
#else
// Python 3.x:
#define PY_INT_AS_LONG(x) (PyLong_AsLong(x))
#define PY_STRING_INTERN_FROM_STRING(x) (PyUnicode_InternFromString(x))
#define PY_INT_FROM_LONG(x) (PyLong_FromLong(x))
#endif

namespace py = pybind11;

namespace tensorflow {
namespace {

Safe_PyObjectPtr GetAttr_ThreadLocalData(PyObject* eager_context) {
  static PyObject* attr = PY_STRING_INTERN_FROM_STRING("_thread_local_data");
  return Safe_PyObjectPtr(PyObject_GetAttr(eager_context, attr));
}

Safe_PyObjectPtr GetAttr_ContextHandle(PyObject* eager_context) {
  static PyObject* attr = PY_STRING_INTERN_FROM_STRING("_context_handle");
  return Safe_PyObjectPtr(PyObject_GetAttr(eager_context, attr));
}

Safe_PyObjectPtr GetAttr_IsEager(PyObject* tld) {
  static PyObject* attr = PY_STRING_INTERN_FROM_STRING("is_eager");
  return Safe_PyObjectPtr(PyObject_GetAttr(tld, attr));
}

Safe_PyObjectPtr GetAttr_DeviceName(PyObject* tld) {
  static PyObject* attr = PY_STRING_INTERN_FROM_STRING("device_name");
  return Safe_PyObjectPtr(PyObject_GetAttr(tld, attr));
}

Safe_PyObjectPtr GetAttr_TypeEnum(PyObject* dtype) {
  static PyObject* attr = PY_STRING_INTERN_FROM_STRING("_type_enum");
  return Safe_PyObjectPtr(PyObject_GetAttr(dtype, attr));
}

PythonTensorConverter MakePythonTensorConverter(py::handle py_eager_context) {
  Safe_PyObjectPtr tld = GetAttr_ThreadLocalData(py_eager_context.ptr());
  if (!tld) throw py::error_already_set();

  Safe_PyObjectPtr py_is_eager = GetAttr_IsEager(tld.get());
  if (!py_is_eager) throw py::error_already_set();
  bool is_eager = PyObject_IsTrue(py_is_eager.get());

  // Initialize the eager context, if necessary.
  TFE_Context* ctx = nullptr;
  const char* device_name = nullptr;
  if (is_eager) {
    Safe_PyObjectPtr context_handle =
        GetAttr_ContextHandle(py_eager_context.ptr());
    if (!context_handle) throw py::error_already_set();
    if (context_handle.get() == Py_None) {
      throw std::runtime_error("Error retrieving context handle.");
    }
    Safe_PyObjectPtr py_device_name = GetAttr_DeviceName(tld.get());
    if (!py_device_name) {
      throw std::runtime_error("Error retrieving device name.");
    }
    device_name = TFE_GetPythonString(py_device_name.get());
    ctx = reinterpret_cast<TFE_Context*>(
        PyCapsule_GetPointer(context_handle.get(), nullptr));
  }

  return PythonTensorConverter(py_eager_context.ptr(), ctx, device_name);
}

py::handle Convert(tensorflow::PythonTensorConverter* self, py::handle obj,
                   py::handle dtype) {
  DataType dtype_enum = static_cast<DataType>(PY_INT_AS_LONG(dtype.ptr()));
  bool used_fallback = false;
  Safe_PyObjectPtr converted =
      self->Convert(obj.ptr(), dtype_enum, &used_fallback);
  if (!converted) throw py::error_already_set();

  PyObject* result = PyTuple_New(3);
  PyTuple_SET_ITEM(result, 0, converted.release());
  PyTuple_SET_ITEM(result, 1, PY_INT_FROM_LONG(dtype_enum));
  PyTuple_SET_ITEM(result, 2, used_fallback ? Py_True : Py_False);
  Py_INCREF(PyTuple_GET_ITEM(result, 1));
  Py_INCREF(PyTuple_GET_ITEM(result, 2));
  return result;
}

}  // namespace
}  // namespace tensorflow

PYBIND11_MODULE(_pywrap_python_tensor_converter, m) {
  py::class_<tensorflow::PythonTensorConverter>(m, "PythonTensorConverter")
      .def(py::init(&tensorflow::MakePythonTensorConverter))
      .def("Convert", tensorflow::Convert);
}

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#define TFLITE_IMPORT_NUMPY  // See numpy.h for explanation.
#include "tensorflow/lite/python/interpreter_wrapper/numpy.h"

#include <memory>

namespace tflite {
namespace python {

void ImportNumpy() { import_array1(); }

}  // namespace python

namespace python_utils {

struct PyObjectDereferencer {
  void operator()(PyObject* py_object) const { Py_DECREF(py_object); }
};
using UniquePyObjectRef = std::unique_ptr<PyObject, PyObjectDereferencer>;

int TfLiteTypeToPyArrayType(TfLiteType tf_lite_type) {
  switch (tf_lite_type) {
    case kTfLiteFloat32:
      return NPY_FLOAT32;
    case kTfLiteFloat16:
      return NPY_FLOAT16;
    case kTfLiteFloat64:
      return NPY_FLOAT64;
    case kTfLiteInt32:
      return NPY_INT32;
    case kTfLiteUInt32:
      return NPY_UINT32;
    case kTfLiteUInt16:
      return NPY_UINT16;
    case kTfLiteInt16:
      return NPY_INT16;
    case kTfLiteUInt8:
      return NPY_UINT8;
    case kTfLiteInt8:
      return NPY_INT8;
    case kTfLiteInt64:
      return NPY_INT64;
    case kTfLiteUInt64:
      return NPY_UINT64;
    case kTfLiteString:
      return NPY_STRING;
    case kTfLiteBool:
      return NPY_BOOL;
    case kTfLiteComplex64:
      return NPY_COMPLEX64;
    case kTfLiteComplex128:
      return NPY_COMPLEX128;
    case kTfLiteResource:
    case kTfLiteVariant:
      return NPY_OBJECT;
    case kTfLiteNoType:
      return NPY_NOTYPE;
      // Avoid default so compiler errors created when new types are made.
  }
  return NPY_NOTYPE;
}

TfLiteType TfLiteTypeFromPyType(int py_type) {
  switch (py_type) {
    case NPY_FLOAT32:
      return kTfLiteFloat32;
    case NPY_FLOAT16:
      return kTfLiteFloat16;
    case NPY_FLOAT64:
      return kTfLiteFloat64;
    case NPY_INT32:
      return kTfLiteInt32;
    case NPY_UINT32:
      return kTfLiteUInt32;
    case NPY_INT16:
      return kTfLiteInt16;
    case NPY_UINT8:
      return kTfLiteUInt8;
    case NPY_INT8:
      return kTfLiteInt8;
    case NPY_INT64:
      return kTfLiteInt64;
    case NPY_UINT64:
      return kTfLiteUInt64;
    case NPY_BOOL:
      return kTfLiteBool;
    case NPY_OBJECT:
    case NPY_STRING:
    case NPY_UNICODE:
      return kTfLiteString;
    case NPY_COMPLEX64:
      return kTfLiteComplex64;
    case NPY_COMPLEX128:
      return kTfLiteComplex128;
  }
  return kTfLiteNoType;
}

TfLiteType TfLiteTypeFromPyArray(PyArrayObject* array) {
  int pyarray_type = PyArray_TYPE(array);
  return TfLiteTypeFromPyType(pyarray_type);
}

#if PY_VERSION_HEX >= 0x03030000
bool FillStringBufferFromPyUnicode(PyObject* value,
                                   DynamicBuffer* dynamic_buffer) {
  Py_ssize_t len = -1;
  const char* buf = PyUnicode_AsUTF8AndSize(value, &len);
  if (buf == nullptr) {
    PyErr_SetString(PyExc_ValueError, "PyUnicode_AsUTF8AndSize() failed.");
    return false;
  }
  dynamic_buffer->AddString(buf, len);
  return true;
}
#else
bool FillStringBufferFromPyUnicode(PyObject* value,
                                   DynamicBuffer* dynamic_buffer) {
  UniquePyObjectRef utemp(PyUnicode_AsUTF8String(value));
  if (!utemp) {
    PyErr_SetString(PyExc_ValueError, "PyUnicode_AsUTF8String() failed.");
    return false;
  }
  char* buf = nullptr;
  Py_ssize_t len = -1;
  if (PyBytes_AsStringAndSize(utemp.get(), &buf, &len) == -1) {
    PyErr_SetString(PyExc_ValueError, "PyBytes_AsStringAndSize() failed.");
    return false;
  }
  dynamic_buffer->AddString(buf, len);
  return true;
}
#endif

bool FillStringBufferFromPyString(PyObject* value,
                                  DynamicBuffer* dynamic_buffer) {
  if (PyUnicode_Check(value)) {
    return FillStringBufferFromPyUnicode(value, dynamic_buffer);
  }

  char* buf = nullptr;
  Py_ssize_t len = -1;
  if (PyBytes_AsStringAndSize(value, &buf, &len) == -1) {
    PyErr_SetString(PyExc_ValueError, "PyBytes_AsStringAndSize() failed.");
    return false;
  }
  dynamic_buffer->AddString(buf, len);
  return true;
}

bool FillStringBufferWithPyArray(PyObject* value,
                                 DynamicBuffer* dynamic_buffer) {
  if (!PyArray_Check(value)) {
    PyErr_Format(PyExc_ValueError,
                 "Passed in value type is not a numpy array, got type %s.",
                 value->ob_type->tp_name);
    return false;
  }

  PyArrayObject* array = reinterpret_cast<PyArrayObject*>(value);
  switch (PyArray_TYPE(array)) {
    case NPY_OBJECT:
    case NPY_STRING:
    case NPY_UNICODE: {
      if (PyArray_NDIM(array) == 0) {
        dynamic_buffer->AddString(static_cast<char*>(PyArray_DATA(array)),
                                  PyArray_NBYTES(array));
        return true;
      }
      UniquePyObjectRef iter(PyArray_IterNew(value));
      while (PyArray_ITER_NOTDONE(iter.get())) {
        UniquePyObjectRef item(PyArray_GETITEM(
            array, reinterpret_cast<char*>(PyArray_ITER_DATA(iter.get()))));

        if (!FillStringBufferFromPyString(item.get(), dynamic_buffer)) {
          return false;
        }

        PyArray_ITER_NEXT(iter.get());
      }
      return true;
    }
    default:
      break;
  }

  PyErr_Format(PyExc_ValueError,
               "Cannot use numpy array of type %d for string tensor.",
               PyArray_TYPE(array));
  return false;
}

}  // namespace python_utils
}  // namespace tflite

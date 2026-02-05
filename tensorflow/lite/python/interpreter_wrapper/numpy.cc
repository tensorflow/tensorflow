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

#include <cstddef>
#include <cstdint>
#include <memory>

#define TFLITE_IMPORT_NUMPY  // See numpy.h for explanation.
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/python/interpreter_wrapper/numpy.h"

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
    case kTfLiteBFloat16:
      // TODO(b/329491949): Supports other ml_dtypes user-defined types.
      return NPY_USERDEF;
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
    case kTfLiteInt4:
      // TODO(b/246806634): NPY_INT4 currently doesn't exist
      return NPY_BYTE;
    case kTfLiteUInt4:
      return NPY_BYTE;
    case kTfLiteInt2:
      // TODO(b/246806634): NPY_INT2 currently doesn't exist
      return NPY_BYTE;
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
}  // NOLINT(direct import ndarraytypes.h cannot be used here)

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
    case NPY_UINT16:
      return kTfLiteUInt16;
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
    case NPY_USERDEF:
      // User-defined types are defined in ml_dtypes. (bfloat16, float8, etc.)
      // For now, we only support bfloat16.
      return kTfLiteBFloat16;
  }
  return kTfLiteNoType;
}  // NOLINT(direct import ndarraytypes.h cannot be used here)

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

// Helper function to pack int8/uint8 numpy array data into an INT4/UINT4
// tensor.
PyObject* Set4BitTensor(TfLiteTensor* tensor, PyArrayObject* array,
                        int tensor_index) {
  TfLiteType incoming_type = TfLiteTypeFromPyArray(array);
  if (tensor->type == kTfLiteInt4) {
    if (incoming_type != kTfLiteInt8) {
      PyErr_Format(PyExc_ValueError,
                   "Cannot set tensor:"
                   " Expected a numpy array of int8 for INT4 input "
                   "%d, name: %s, but got %s",
                   tensor_index, tensor->name,
                   TfLiteTypeGetName(incoming_type));
      return nullptr;
    }
  } else if (tensor->type == kTfLiteUInt4) {
    if (incoming_type != kTfLiteUInt8) {
      PyErr_Format(PyExc_ValueError,
                   "Cannot set tensor:"
                   " Expected a numpy array of uint8 for UINT4 input "
                   "%d, name: %s, but got %s",
                   tensor_index, tensor->name,
                   TfLiteTypeGetName(incoming_type));
      return nullptr;
    }
  }

  size_t num_elements = 1;
  for (int i = 0; i < tensor->dims->size; ++i) {
    num_elements *= tensor->dims->data[i];
  }
  size_t expected_packed_bytes = (num_elements + 1) / 2;
  size_t actual_numpy_bytes = PyArray_NBYTES(array);

  const char* tensor_type_name = TfLiteTypeGetName(tensor->type);

  if (actual_numpy_bytes != num_elements) {
    PyErr_Format(PyExc_ValueError,
                 "Cannot set tensor:"
                 " Numpy array for %s input %d, name: %s, has %zu bytes, "
                 "but expected %zu bytes for %zu elements",
                 tensor_type_name, tensor_index, tensor->name,
                 actual_numpy_bytes, num_elements, num_elements);
    return nullptr;
  }

  if (tensor->data.raw == nullptr && tensor->bytes) {
    PyErr_Format(PyExc_ValueError,
                 "Cannot set tensor:"
                 " Tensor is unallocated. Try calling allocate_tensors()"
                 " first for input %d, name: %s",
                 tensor_index, tensor->name);
    return nullptr;
  }

  // Pack the int8/uint8 array into int4/uint4
  uint8_t* packed_data = reinterpret_cast<uint8_t*>(tensor->data.raw);

  if (tensor->type == kTfLiteInt4) {
    int8_t* numpy_data = reinterpret_cast<int8_t*>(PyArray_DATA(array));
    for (size_t i = 0; i < expected_packed_bytes; ++i) {
      int8_t first_nibble = numpy_data[2 * i];
      int8_t second_nibble =
          (2 * i + 1 < num_elements) ? numpy_data[2 * i + 1] : 0;
      if ((first_nibble < -8 || first_nibble > 7) ||
          (second_nibble < -8 || second_nibble > 7)) {
        PyErr_Format(PyExc_ValueError,
                     "Cannot set tensor:"
                     " Values for INT4 input must be between -8 and 7.");
        return nullptr;
      }
      // Pack the two int8 values into a single byte. The first nibble
      // occupies the lower 4 bits and the second nibble occupies the upper 4
      // bits. We mask the first nibble with 0x0F to ensure only the lower 4
      // bits are used, handling potential sign extension in the int8 value.
      packed_data[i] = (first_nibble & 0x0F) | (second_nibble << 4);
    }
  } else {  // kTfLiteUInt4
    uint8_t* numpy_data = reinterpret_cast<uint8_t*>(PyArray_DATA(array));
    for (size_t i = 0; i < expected_packed_bytes; ++i) {
      uint8_t first_nibble = numpy_data[2 * i];
      uint8_t second_nibble =
          (2 * i + 1 < num_elements) ? numpy_data[2 * i + 1] : 0;
      if (first_nibble > 15 || second_nibble > 15) {
        PyErr_Format(PyExc_ValueError,
                     "Cannot set tensor:"
                     " Values for UINT4 input must be between 0 and 15.");
        return nullptr;
      }
      packed_data[i] = (first_nibble & 0x0F) | (second_nibble << 4);
    }
  }
  Py_RETURN_NONE;
}

}  // namespace python_utils
}  // namespace tflite

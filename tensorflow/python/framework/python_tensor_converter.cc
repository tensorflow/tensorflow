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
#include "tensorflow/python/framework/python_tensor_converter.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/python/eager/pywrap_tensor.h"
#include "tensorflow/python/eager/pywrap_tfe.h"
#include "tensorflow/python/util/util.h"

#if PY_MAJOR_VERSION < 3
// Python 2.x:
#define PY_INT_AS_LONG(x) (PyInt_AsLong(x))
#define PY_STRING_INTERN_FROM_STRING(x) (PyString_InternFromString(x))
#else
// Python 3.x:
#define PY_INT_AS_LONG(x) (PyLong_AsLong(x))
#define PY_STRING_INTERN_FROM_STRING(x) (PyUnicode_InternFromString(x))
#endif

namespace tensorflow {
namespace {

// Returns `tensor.dtype._type_enum` as a DataType enum.  Assumes that `tensor`
// is a python `Tensor` object.
//
// On error: sets a python AttributeError exception and returns DT_INVALID.
DataType DataTypeForTensor(PyObject* tensor) {
  static PyObject* dtype_attr = PY_STRING_INTERN_FROM_STRING("dtype");
  static PyObject* type_enum_attr = PY_STRING_INTERN_FROM_STRING("_type_enum");

  Safe_PyObjectPtr py_dtype(PyObject_GetAttr(tensor, dtype_attr));
  if (!py_dtype) return DT_INVALID;

  Safe_PyObjectPtr enum_field(PyObject_GetAttr(py_dtype.get(), type_enum_attr));
  if (!enum_field) return DT_INVALID;

  DataType result = static_cast<DataType>(PY_INT_AS_LONG(enum_field.get()));
  return result;
}

// Check that actual_dtype == expected_dtype.  If not, set an exception and
// return false.  (If expected_dtype is DT_INVALID, then instead simply update
// its value to `actual_dtype` and return true.)
bool CheckDType(DataType actual_dtype, DataType& expected_dtype) {
  if (expected_dtype == DT_INVALID) {
    expected_dtype = actual_dtype;  // set output parameter.
  } else if (expected_dtype != actual_dtype) {
    PyErr_SetString(PyExc_TypeError,
                    absl::StrCat("Expected ", DataType_Name(expected_dtype),
                                 " but got ", DataType_Name(actual_dtype))
                        .c_str());
    return false;
  }
  return true;
}

}  // namespace

Safe_PyObjectPtr PythonTensorConverter::Convert(PyObject* src, DataType& dtype,
                                                bool* used_fallback) const {
  // First, try converting `src` to a Tensor without calling back into Python.
  if (ctx_) {  // Eager mode
    // TODO(b/164980194): Handle resource variables as well.  (See
    // ConvertToTensor function in pywrap_tfe_src.cc).
    if (EagerTensor_CheckExact(src)) {
      // `src` is already an eager tensor; check its type, and return it as-is.
      if (!CheckDType(PyEagerTensor_Dtype(src), dtype)) return nullptr;
      Py_INCREF(src);
      return Safe_PyObjectPtr(src);
    } else {
      TFE_TensorHandle* handle =
          tensorflow::ConvertToEagerTensor(ctx_, src, dtype, device_name_);
      if (handle) {
        Safe_PyObjectPtr result(EagerTensorFromHandle(handle));
        if (!CheckDType(PyEagerTensor_Dtype(result.get()), dtype)) {
          return nullptr;
        }
        return result;
      } else {
        PyErr_Clear();
      }
    }
  } else {  // Graph mode
    if (swig::IsTensor(src)) {
      DataType src_dtype = DataTypeForTensor(src);
      if (src_dtype == DT_INVALID) return nullptr;
      if (!CheckDType(src_dtype, dtype)) return nullptr;
      Py_INCREF(src);
      return Safe_PyObjectPtr(src);
    }
  }

  // Fallback: use the Python tf.convert_to_tensor function.
  // Currently this is used:
  //
  // * In Eager mode: for anything that's not already an Eager tensor, or
  //   handled by `tensorflow::ConvertToEagerTensor`.  (At time of writing
  //   for this comment, ConvertToEagerTensor handles simple values like ints,
  //   nested lists of simple values, and numpy arrays.)
  // * In graph mode: for anything that's not already a tensor.
  //
  // TODO(b/164980194) Reduce/eliminate cases where fallback is used.
  if (used_fallback) *used_fallback = true;
  static PyObject* convert_to_tensor =
      swig::GetRegisteredPyObject("tf.convert_to_tensor");
  if (!convert_to_tensor) return nullptr;

  Safe_PyObjectPtr args(PyTuple_New(dtype == DT_INVALID ? 1 : 2));
  Safe_PyObjectPtr kwargs(PyDict_New());
  Py_INCREF(src);
  PyTuple_SetItem(args.get(), 0, src);
  if (dtype != DT_INVALID) {
    PyTuple_SetItem(args.get(), 1, PyLong_FromLong(dtype));
  }
  PyDict_SetItemString(kwargs.get(), "ctx", py_eager_context_);
  Safe_PyObjectPtr result(
      PyObject_Call(convert_to_tensor, args.get(), kwargs.get()));
  if (!result) return nullptr;
  dtype = DataTypeForTensor(result.get());  // set output parameter.
  if (dtype == DT_INVALID) return nullptr;
  return result;
}

}  // namespace tensorflow

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

#include "tensorflow/python/lib/core/ndarray_tensor_types.h"

#include <Python.h>

// Must be included first.
// clang-format: off
#include "tensorflow/python/lib/core/numpy.h"
// clang-format: on

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/python/lib/core/bfloat16.h"

namespace tensorflow {

PyArray_Descr* BFLOAT16_DESCR = nullptr;
PyArray_Descr* QINT8_DESCR = nullptr;
PyArray_Descr* QINT16_DESCR = nullptr;
PyArray_Descr* QINT32_DESCR = nullptr;
PyArray_Descr* QUINT8_DESCR = nullptr;
PyArray_Descr* QUINT16_DESCR = nullptr;
PyArray_Descr* RESOURCE_DESCR = nullptr;

// Define a struct array data type `[(tag, type_num)]`.
PyArray_Descr* DefineStructTypeAlias(const char* tag, int type_num) {
#if PY_MAJOR_VERSION < 3
  auto* py_tag = PyBytes_FromString(tag);
#else
  auto* py_tag = PyUnicode_FromString(tag);
#endif
  auto* descr = PyArray_DescrFromType(type_num);
  auto* py_tag_and_descr = PyTuple_Pack(2, py_tag, descr);
  auto* obj = PyList_New(1);
  PyList_SetItem(obj, 0, py_tag_and_descr);
  PyArray_Descr* alias_descr = nullptr;
  // TODO(slebedev): Switch to PyArray_DescrNewFromType because struct
  // array dtypes could not be used for scalars. Note that this will
  // require registering type conversions and UFunc specializations.
  // See b/144230631.
  CHECK_EQ(PyArray_DescrConverter(obj, &alias_descr), NPY_SUCCEED);
  Py_DECREF(obj);
  Py_DECREF(py_tag_and_descr);
  Py_DECREF(py_tag);
  Py_DECREF(descr);
  CHECK_NE(alias_descr, nullptr);
  return alias_descr;
}

void MaybeRegisterCustomNumPyTypes() {
  static bool registered = false;
  if (registered) return;
  ImportNumpy();  // Ensure NumPy is loaded.
  // Make sure the tags are consistent with DataTypeToPyArray_Descr.
  QINT8_DESCR = DefineStructTypeAlias("qint8", NPY_INT8);
  QINT16_DESCR = DefineStructTypeAlias("qint16", NPY_INT16);
  QINT32_DESCR = DefineStructTypeAlias("qint32", NPY_INT32);
  QUINT8_DESCR = DefineStructTypeAlias("quint8", NPY_UINT8);
  QUINT16_DESCR = DefineStructTypeAlias("quint16", NPY_UINT16);
  RESOURCE_DESCR = DefineStructTypeAlias("resource", NPY_UBYTE);
  RegisterNumpyBfloat16();
  BFLOAT16_DESCR = PyArray_DescrFromType(Bfloat16NumpyType());
  registered = true;
}

const char* PyArray_DescrReprAsString(PyArray_Descr* descr) {
  auto* descr_repr = PyObject_Repr(reinterpret_cast<PyObject*>(descr));
  const char* result;
#if PY_MAJOR_VERSION < 3
  result = PyBytes_AsString(descr_repr);
#else
  auto* tmp = PyUnicode_AsASCIIString(descr_repr);
  result = PyBytes_AsString(tmp);
  Py_DECREF(tmp);
#endif

  Py_DECREF(descr_repr);
  return result;
}

Status DataTypeToPyArray_Descr(DataType dt, PyArray_Descr** out_descr) {
  switch (dt) {
    case DT_HALF:
      *out_descr = PyArray_DescrFromType(NPY_FLOAT16);
      break;
    case DT_FLOAT:
      *out_descr = PyArray_DescrFromType(NPY_FLOAT32);
      break;
    case DT_DOUBLE:
      *out_descr = PyArray_DescrFromType(NPY_FLOAT64);
      break;
    case DT_INT32:
      *out_descr = PyArray_DescrFromType(NPY_INT32);
      break;
    case DT_UINT32:
      *out_descr = PyArray_DescrFromType(NPY_UINT32);
      break;
    case DT_UINT8:
      *out_descr = PyArray_DescrFromType(NPY_UINT8);
      break;
    case DT_UINT16:
      *out_descr = PyArray_DescrFromType(NPY_UINT16);
      break;
    case DT_INT8:
      *out_descr = PyArray_DescrFromType(NPY_INT8);
      break;
    case DT_INT16:
      *out_descr = PyArray_DescrFromType(NPY_INT16);
      break;
    case DT_INT64:
      *out_descr = PyArray_DescrFromType(NPY_INT64);
      break;
    case DT_UINT64:
      *out_descr = PyArray_DescrFromType(NPY_UINT64);
      break;
    case DT_BOOL:
      *out_descr = PyArray_DescrFromType(NPY_BOOL);
      break;
    case DT_COMPLEX64:
      *out_descr = PyArray_DescrFromType(NPY_COMPLEX64);
      break;
    case DT_COMPLEX128:
      *out_descr = PyArray_DescrFromType(NPY_COMPLEX128);
      break;
    case DT_STRING:
      *out_descr = PyArray_DescrFromType(NPY_OBJECT);
      break;
    case DT_QINT8:
      *out_descr = PyArray_DescrFromType(NPY_INT8);
      break;
    case DT_QINT16:
      *out_descr = PyArray_DescrFromType(NPY_INT16);
      break;
    case DT_QINT32:
      *out_descr = PyArray_DescrFromType(NPY_INT32);
      break;
    case DT_QUINT8:
      *out_descr = PyArray_DescrFromType(NPY_UINT8);
      break;
    case DT_QUINT16:
      *out_descr = PyArray_DescrFromType(NPY_UINT16);
      break;
    case DT_RESOURCE:
      *out_descr = PyArray_DescrFromType(NPY_UBYTE);
      break;
    case DT_BFLOAT16:
      Py_INCREF(BFLOAT16_DESCR);
      *out_descr = BFLOAT16_DESCR;
      break;
    default:
      return errors::Internal("TensorFlow data type ", DataType_Name(dt),
                              " cannot be converted to a NumPy data type.");
  }

  return Status::OK();
}

// NumPy defines fixed-width aliases for platform integer types. However,
// some types do not have a fixed-width alias. Specifically
//
//  * on a LLP64 system NPY_INT32 == NPY_LONG therefore NPY_INT is not aliased;
//  * on a LP64 system NPY_INT64 == NPY_LONG and NPY_LONGLONG is not aliased.
//
int MaybeResolveNumPyPlatformType(int type_num) {
  switch (type_num) {
#if NPY_BITS_OF_INT == 32 && NPY_BITS_OF_LONGLONG == 32
    case NPY_INT:
      return NPY_INT32;
    case NPY_UINT:
      return NPY_UINT32;
#endif
#if NPY_BITSOF_INT == 32 && NPY_BITSOF_LONGLONG == 64
    case NPY_LONGLONG:
      return NPY_INT64;
    case NPY_ULONGLONG:
      return NPY_UINT64;
#endif
    default:
      return type_num;
  }
}

Status PyArray_DescrToDataType(PyArray_Descr* descr, DataType* out_dt) {
  const int type_num = MaybeResolveNumPyPlatformType(descr->type_num);
  switch (type_num) {
    case NPY_FLOAT16:
      *out_dt = DT_HALF;
      break;
    case NPY_FLOAT32:
      *out_dt = DT_FLOAT;
      break;
    case NPY_FLOAT64:
      *out_dt = DT_DOUBLE;
      break;
    case NPY_INT8:
      *out_dt = DT_INT8;
      break;
    case NPY_INT16:
      *out_dt = DT_INT16;
      break;
    case NPY_INT32:
      *out_dt = DT_INT32;
      break;
    case NPY_INT64:
      *out_dt = DT_INT64;
      break;
    case NPY_UINT8:
      *out_dt = DT_UINT8;
      break;
    case NPY_UINT16:
      *out_dt = DT_UINT16;
      break;
    case NPY_UINT32:
      *out_dt = DT_UINT32;
      break;
    case NPY_UINT64:
      *out_dt = DT_UINT64;
      break;
    case NPY_BOOL:
      *out_dt = DT_BOOL;
      break;
    case NPY_COMPLEX64:
      *out_dt = DT_COMPLEX64;
      break;
    case NPY_COMPLEX128:
      *out_dt = DT_COMPLEX128;
      break;
    case NPY_OBJECT:
    case NPY_STRING:
    case NPY_UNICODE:
      *out_dt = DT_STRING;
      break;
    case NPY_VOID: {
      if (descr == QINT8_DESCR) {
        *out_dt = DT_QINT8;
        break;
      } else if (descr == QINT16_DESCR) {
        *out_dt = DT_QINT16;
        break;
      } else if (descr == QINT32_DESCR) {
        *out_dt = DT_QINT32;
        break;
      } else if (descr == QUINT8_DESCR) {
        *out_dt = DT_QUINT8;
        break;
      } else if (descr == QUINT16_DESCR) {
        *out_dt = DT_QUINT16;
        break;
      } else if (descr == RESOURCE_DESCR) {
        *out_dt = DT_RESOURCE;
        break;
      }

      return errors::Internal("Unsupported NumPy struct data type: ",
                              PyArray_DescrReprAsString(descr));
    }
    default:
      if (type_num == Bfloat16NumpyType()) {
        *out_dt = DT_BFLOAT16;
        break;
      }

      return errors::Internal("Unregistered NumPy data type: ",
                              PyArray_DescrReprAsString(descr));
  }
  return Status::OK();
}

}  // namespace tensorflow

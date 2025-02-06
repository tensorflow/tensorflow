/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// clang-format off
// Must be included first.
#include "tensorflow/c/tf_datatype.h"
#include "xla/tsl/python/lib/core/numpy.h"
// clang-format on

#include "tensorflow/python/lib/core/ndarray_tensor_bridge.h"

#include <vector>

#include "tensorflow/c/c_api.h"
#include "xla/tsl/python/lib/core/ml_dtypes.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/python/lib/core/py_util.h"

namespace tensorflow {

// Mutex used to serialize accesses to cached vector of pointers to python
// arrays to be dereferenced.
static mutex* DelayedDecrefLock() {
  static mutex* decref_lock = new mutex;
  return decref_lock;
}

// Caches pointers to numpy arrays which need to be dereferenced.
static std::vector<void*>* DecrefCache() {
  static std::vector<void*>* decref_cache = new std::vector<void*>;
  return decref_cache;
}

// Destructor passed to TF_NewTensor when it reuses a numpy buffer. Stores a
// pointer to the pyobj in a buffer to be dereferenced later when we're actually
// holding the GIL.
void DelayedNumpyDecref(void* data, size_t len, void* obj) {
  mutex_lock ml(*DelayedDecrefLock());
  DecrefCache()->push_back(obj);
}

// Actually dereferences cached numpy arrays. REQUIRES being called while
// holding the GIL.
void ClearDecrefCache() {
  std::vector<void*> cache_copy;
  {
    mutex_lock ml(*DelayedDecrefLock());
    cache_copy.swap(*DecrefCache());
  }
  for (void* obj : cache_copy) {
    Py_DECREF(reinterpret_cast<PyObject*>(obj));
  }
}

// Structure which keeps a reference to a Tensor alive while numpy has a pointer
// to it.
struct TensorReleaser {
  // Python macro to include standard members.
  PyObject_HEAD

      // Destructor responsible for releasing the memory.
      std::function<void()>* destructor;
};

extern PyTypeObject TensorReleaserType;

static void TensorReleaser_dealloc(PyObject* pself) {
  TensorReleaser* self = reinterpret_cast<TensorReleaser*>(pself);
  (*self->destructor)();
  delete self->destructor;
  TensorReleaserType.tp_free(pself);
}

// clang-format off
PyTypeObject TensorReleaserType = {
    PyVarObject_HEAD_INIT(nullptr, 0) /* head init */
    "tensorflow_wrapper",             /* tp_name */
    sizeof(TensorReleaser),           /* tp_basicsize */
    0,                                /* tp_itemsize */
    /* methods */
    TensorReleaser_dealloc,      /* tp_dealloc */
#if PY_VERSION_HEX < 0x03080000
    nullptr,                     /* tp_print */
#else
    0,                           /* tp_vectorcall_offset */
#endif
    nullptr,                     /* tp_getattr */
    nullptr,                     /* tp_setattr */
    nullptr,                     /* tp_compare */
    nullptr,                     /* tp_repr */
    nullptr,                     /* tp_as_number */
    nullptr,                     /* tp_as_sequence */
    nullptr,                     /* tp_as_mapping */
    nullptr,                     /* tp_hash */
    nullptr,                     /* tp_call */
    nullptr,                     /* tp_str */
    nullptr,                     /* tp_getattro */
    nullptr,                     /* tp_setattro */
    nullptr,                     /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,          /* tp_flags */
    "Wrapped TensorFlow Tensor", /* tp_doc */
    nullptr,                     /* tp_traverse */
    nullptr,                     /* tp_clear */
    nullptr,                     /* tp_richcompare */
};
// clang-format on

absl::Status TF_DataType_to_PyArray_TYPE(TF_DataType tf_datatype,
                                         int* out_pyarray_type) {
  const tsl::ml_dtypes::NumpyDtypes& custom_dtypes =
      tsl::ml_dtypes::GetNumpyDtypes();
  switch (tf_datatype) {
    case TF_HALF:
      *out_pyarray_type = NPY_FLOAT16;
      break;
    case TF_FLOAT:
      *out_pyarray_type = NPY_FLOAT32;
      break;
    case TF_DOUBLE:
      *out_pyarray_type = NPY_FLOAT64;
      break;
    case TF_INT32:
      *out_pyarray_type = NPY_INT32;
      break;
    case TF_UINT32:
      *out_pyarray_type = NPY_UINT32;
      break;
    case TF_UINT8:
      *out_pyarray_type = NPY_UINT8;
      break;
    case TF_UINT16:
      *out_pyarray_type = NPY_UINT16;
      break;
    case TF_INT8:
      *out_pyarray_type = NPY_INT8;
      break;
    case TF_INT16:
      *out_pyarray_type = NPY_INT16;
      break;
    case TF_INT64:
      *out_pyarray_type = NPY_INT64;
      break;
    case TF_UINT64:
      *out_pyarray_type = NPY_UINT64;
      break;
    case TF_BOOL:
      *out_pyarray_type = NPY_BOOL;
      break;
    case TF_COMPLEX64:
      *out_pyarray_type = NPY_COMPLEX64;
      break;
    case TF_COMPLEX128:
      *out_pyarray_type = NPY_COMPLEX128;
      break;
    case TF_STRING:
      *out_pyarray_type = NPY_OBJECT;
      break;
    case TF_RESOURCE:
      *out_pyarray_type = NPY_VOID;
      break;
    // TODO(keveman): These should be changed to NPY_VOID, and the type used for
    // the resulting numpy array should be the custom struct types that we
    // expect for quantized types.
    case TF_QINT8:
      *out_pyarray_type = NPY_INT8;
      break;
    case TF_QUINT8:
      *out_pyarray_type = NPY_UINT8;
      break;
    case TF_QINT16:
      *out_pyarray_type = NPY_INT16;
      break;
    case TF_QUINT16:
      *out_pyarray_type = NPY_UINT16;
      break;
    case TF_QINT32:
      *out_pyarray_type = NPY_INT32;
      break;
    case TF_BFLOAT16:
      *out_pyarray_type = custom_dtypes.bfloat16;
      break;
    case TF_FLOAT8_E5M2:
      *out_pyarray_type = custom_dtypes.float8_e5m2;
      break;
    case TF_FLOAT8_E4M3FN:
      *out_pyarray_type = custom_dtypes.float8_e4m3fn;
      break;
    case TF_FLOAT8_E4M3FNUZ:
      *out_pyarray_type = custom_dtypes.float8_e4m3fnuz;
      break;
    case TF_FLOAT8_E4M3B11FNUZ:
      *out_pyarray_type = custom_dtypes.float8_e4m3b11fnuz;
      break;
    case TF_FLOAT8_E5M2FNUZ:
      *out_pyarray_type = custom_dtypes.float8_e5m2fnuz;
      break;
    case TF_INT4:
      *out_pyarray_type = custom_dtypes.int4;
      break;
    case TF_UINT4:
      *out_pyarray_type = custom_dtypes.uint4;
      break;
    default:
      return errors::Internal("Tensorflow type ", tf_datatype,
                              " not convertible to numpy dtype.");
  }
  return absl::OkStatus();
}

absl::Status ArrayFromMemory(int dim_size, npy_intp* dims, void* data,
                             DataType dtype, std::function<void()> destructor,
                             PyObject** result) {
  if (dtype == DT_STRING || dtype == DT_RESOURCE) {
    return errors::FailedPrecondition(
        "Cannot convert string or resource Tensors.");
  }

  int type_num = -1;
  absl::Status s =
      TF_DataType_to_PyArray_TYPE(static_cast<TF_DataType>(dtype), &type_num);
  if (!s.ok()) {
    return s;
  }

  if (dim_size > NPY_MAXDIMS) {
    return errors::InvalidArgument(
        "Cannot convert tensor with ", dim_size,
        " dimensions to NumPy array. NumPy arrays can have at most ",
        NPY_MAXDIMS, " dimensions");
  }
  auto* np_array = reinterpret_cast<PyArrayObject*>(
      PyArray_SimpleNewFromData(dim_size, dims, type_num, data));
  if (np_array == nullptr) {
    string shape_str = absl::StrJoin(
        absl::Span<npy_intp>{dims, static_cast<size_t>(dim_size)}, ", ");
    if (PyErr_Occurred()) {
      string exception_str = PyExceptionFetch();
      PyErr_Clear();
      return errors::InvalidArgument(
          "Failed to create numpy array from tensor of shape [", shape_str,
          "]. Numpy error: ", exception_str);
    }
    return errors::Internal(
        "Failed to create numpy array from tensor of shape [", shape_str, "]");
  }

  PyArray_CLEARFLAGS(np_array, NPY_ARRAY_OWNDATA);
  if (PyType_Ready(&TensorReleaserType) == -1) {
    return errors::Unknown("Python type initialization failed.");
  }
  auto* releaser = reinterpret_cast<TensorReleaser*>(
      TensorReleaserType.tp_alloc(&TensorReleaserType, 0));
  releaser->destructor = new std::function<void()>(std::move(destructor));
  if (PyArray_SetBaseObject(np_array, reinterpret_cast<PyObject*>(releaser)) ==
      -1) {
    Py_DECREF(releaser);
    return errors::Unknown("Python array refused to use memory.");
  }
  *result = reinterpret_cast<PyObject*>(np_array);
  return absl::OkStatus();
}

}  // namespace tensorflow

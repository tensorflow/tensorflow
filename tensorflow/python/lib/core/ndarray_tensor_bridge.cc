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

#include "tensorflow/python/lib/core/ndarray_tensor_bridge.h"

#include <vector>

// Must be included first.
// clang-format: off
#include "tensorflow/python/lib/core/numpy.h"
// clang-format: on

#include "tensorflow/c/c_api.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/python/lib/core/ndarray_tensor_types.h"

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

PyTypeObject TensorReleaserType = {
    PyVarObject_HEAD_INIT(nullptr, 0) /* head init */
    "tensorflow_wrapper",             /* tp_name */
    sizeof(TensorReleaser),           /* tp_basicsize */
    0,                                /* tp_itemsize */
    /* methods */
    TensorReleaser_dealloc,      /* tp_dealloc */
    0,                           /* tp_print */
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

Status ArrayFromMemory(int dim_size, npy_intp* dims, void* data, DataType dtype,
                       std::function<void()> destructor, PyObject** result) {
  if (dtype == DT_STRING || dtype == DT_RESOURCE) {
    return errors::FailedPrecondition(
        "Cannot convert string or resource Tensors.");
  }

  PyArray_Descr* descr = nullptr;
  TF_RETURN_IF_ERROR(DataTypeToPyArray_Descr(dtype, &descr));
  auto* np_array = reinterpret_cast<PyArrayObject*>(
      PyArray_SimpleNewFromData(dim_size, dims, descr->type_num, data));
  CHECK_NE(np_array, nullptr);
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
  return Status::OK();
}

}  // namespace tensorflow

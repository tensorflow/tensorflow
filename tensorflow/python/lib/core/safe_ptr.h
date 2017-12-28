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

#ifndef THIRD_PARTY_TENSORFLOW_PYTHON_LIB_CORE_SAFE_PTR_H_
#define THIRD_PARTY_TENSORFLOW_PYTHON_LIB_CORE_SAFE_PTR_H_

#include <memory>

#include <Python.h>
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"

namespace tensorflow {
namespace detail {

struct PyDecrefDeleter {
  void operator()(PyObject* p) const { Py_DECREF(p); }
};

struct TFTensorDeleter {
  void operator()(TF_Tensor* p) const { TF_DeleteTensor(p); }
};

struct TFETensorHandleDeleter {
  void operator()(TFE_TensorHandle* p) const { TFE_DeleteTensorHandle(p); }
};

struct TFStatusDeleter {
  void operator()(TF_Status* p) const { TF_DeleteStatus(p); }
};

}  // namespace detail

// Safe container for an owned PyObject. On destruction, the reference count of
// the contained object will be decremented.
using Safe_PyObjectPtr = std::unique_ptr<PyObject, detail::PyDecrefDeleter>;
Safe_PyObjectPtr make_safe(PyObject* o);

// Safe containers for an owned TF_Tensor. On destruction, the tensor will be
// deleted by TF_DeleteTensor.
using Safe_TF_TensorPtr = std::unique_ptr<TF_Tensor, detail::TFTensorDeleter>;
Safe_TF_TensorPtr make_safe(TF_Tensor* tensor);

// Safe containers for an owned TFE_TensorHandle. On destruction, the handle
// will be deleted by TFE_DeleteTensorHandle.
using Safe_TFE_TensorHandlePtr =
    std::unique_ptr<TFE_TensorHandle, detail::TFETensorHandleDeleter>;
Safe_TFE_TensorHandlePtr make_safe(TFE_TensorHandle* handle);

// Safe containers for an owned TF_Status. On destruction, the handle
// will be deleted by TF_DeleteStatus.
using Safe_TF_StatusPtr = std::unique_ptr<TF_Status, detail::TFStatusDeleter>;
Safe_TF_StatusPtr make_safe(TF_Status* status);

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_PYTHON_LIB_CORE_SAFE_PTR_H_

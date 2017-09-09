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

namespace tensorflow {

// Safe container for an owned PyObject. On destruction, the reference count of
// the contained object will be decremented.
typedef void (*Py_DECREF_wrapper_type)(PyObject*);
typedef std::unique_ptr<PyObject, Py_DECREF_wrapper_type> Safe_PyObjectPtr;
Safe_PyObjectPtr make_safe(PyObject* o);

// Safe containers for an owned TF_Tensor. On destruction, the tensor will be
// deleted by TF_DeleteTensor.
// Note: can't use decltype(&TF_DeleteTensor) due to SWIG
typedef void (*TF_DeleteTensor_type)(TF_Tensor*);
typedef std::unique_ptr<TF_Tensor, TF_DeleteTensor_type> Safe_TF_TensorPtr;
Safe_TF_TensorPtr make_safe(TF_Tensor* tensor);

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_PYTHON_LIB_CORE_SAFE_PTR_H_

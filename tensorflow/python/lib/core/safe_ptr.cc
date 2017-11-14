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

#include "tensorflow/python/lib/core/safe_ptr.h"

namespace tensorflow {
namespace {

inline void Py_DECREF_wrapper(PyObject* o) { Py_DECREF(o); }

}  // namespace

Safe_PyObjectPtr make_safe(PyObject* o) {
  return Safe_PyObjectPtr(o, Py_DECREF_wrapper);
}

Safe_TF_TensorPtr make_safe(TF_Tensor* tensor) {
  return Safe_TF_TensorPtr(tensor, TF_DeleteTensor);
}

Safe_TFE_TensorHandlePtr make_safe(TFE_TensorHandle* handle) {
  return Safe_TFE_TensorHandlePtr(handle, TFE_DeleteTensorHandle);
}

Safe_TF_StatusPtr make_safe(TF_Status* status) {
  return Safe_TF_StatusPtr(status, TF_DeleteStatus);
}
}  // namespace tensorflow

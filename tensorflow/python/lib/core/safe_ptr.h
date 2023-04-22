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

#ifndef TENSORFLOW_PYTHON_LIB_CORE_SAFE_PTR_H_
#define TENSORFLOW_PYTHON_LIB_CORE_SAFE_PTR_H_

#include <Python.h>

#include <memory>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"

namespace tensorflow {
namespace detail {

struct TFTensorDeleter {
  void operator()(TF_Tensor* p) const { TF_DeleteTensor(p); }
};

struct TFETensorHandleDeleter {
  void operator()(TFE_TensorHandle* p) const { TFE_DeleteTensorHandle(p); }
};

struct TFStatusDeleter {
  void operator()(TF_Status* p) const { TF_DeleteStatus(p); }
};

struct TFBufferDeleter {
  void operator()(TF_Buffer* p) const { TF_DeleteBuffer(p); }
};

}  // namespace detail

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

// Safe containers for an owned TF_Buffer. On destruction, the handle
// will be deleted by TF_DeleteBuffer.
using Safe_TF_BufferPtr = std::unique_ptr<TF_Buffer, detail::TFBufferDeleter>;
Safe_TF_BufferPtr make_safe(TF_Buffer* buffer);

}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_LIB_CORE_SAFE_PTR_H_

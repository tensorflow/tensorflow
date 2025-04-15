/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_C_TF_TENSOR_HELPER_H_
#define TENSORFLOW_C_TF_TENSOR_HELPER_H_

#include <memory>

#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

class Tensor;

absl::Status TF_TensorToTensor(const TF_Tensor* src, Tensor* dst);

TF_Tensor* TF_TensorFromTensor(const Tensor& src, absl::Status* status);

TF_Tensor* TF_TensorFromTensorShallow(const Tensor& src, absl::Status* status);

namespace internal {

struct TFTensorDeleter {
  void operator()(TF_Tensor* tf_tensor) const { TF_DeleteTensor(tf_tensor); }
};

}  // namespace internal

// Struct that wraps TF_Tensor to delete once out of scope.
using TF_TensorPtr = std::unique_ptr<TF_Tensor, internal::TFTensorDeleter>;

}  // namespace tensorflow

#endif  // TENSORFLOW_C_TF_TENSOR_HELPER_H_

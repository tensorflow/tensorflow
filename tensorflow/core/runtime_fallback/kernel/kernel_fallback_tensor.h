/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// This file declares TF kernel fallback tensor.

#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_TENSOR_H_
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_TENSOR_H_

#include <utility>

#include "tensorflow/core/framework/tensor.h"
#include "tfrt/dtype/dtype.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/tensor/tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_shape.h"  // from @tf_runtime

namespace tensorflow {

class BaseKernelFallbackTensor : public tfrt::Tensor {
 public:
  explicit BaseKernelFallbackTensor(::tensorflow::Tensor tensor);
  BaseKernelFallbackTensor(const tfrt::TensorShape& shape, tfrt::DType dtype,
                           ::tensorflow::Tensor tensor);

  void Print(tfrt::raw_ostream& os) const override;

  const ::tensorflow::Tensor* GetTensor() const { return &tensor_; }

 private:
  ::tensorflow::Tensor tensor_;
  bool is_valid_type_;
};

class KernelFallbackTensor final
    : public BaseKernelFallbackTensor,
      public tfrt::TensorTraits<KernelFallbackTensor> {
 public:
  explicit KernelFallbackTensor(::tensorflow::Tensor tensor)
      : BaseKernelFallbackTensor(std::move(tensor)) {}
  KernelFallbackTensor(const tfrt::TensorShape& shape, tfrt::DType dtype,
                       ::tensorflow::Tensor tensor)
      : BaseKernelFallbackTensor(shape, dtype, std::move(tensor)) {}

  static KernelFallbackTensor Create(const tensorflow::Tensor& tensor) {
    return KernelFallbackTensor(tensor);
  }

  // Tensor type name for KernelFallbackTensor.
  static const char* name() { return "KernelFallback"; }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_TENSOR_H_

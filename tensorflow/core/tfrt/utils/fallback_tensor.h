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
#ifndef TENSORFLOW_CORE_TFRT_UTILS_FALLBACK_TENSOR_H_
#define TENSORFLOW_CORE_TFRT_UTILS_FALLBACK_TENSOR_H_

#include <utility>

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/tensor.h"
#include "tsl/profiler/lib/traceme.h"

namespace tensorflow {
namespace tfrt_stub {

// A special tensor wrapper for immutable tensors that live a long time and are
// reused across steps in a program, eg. weights.
class ImmutableTensor {
 public:
  ImmutableTensor() = default;
  // Create an ImmutableTensor by copying the content in `tensor`.
  static ImmutableTensor Create(tensorflow::Tensor tensor);

  // Accessors for this underlying tensor. Users must not modify its content. It
  // is guaranteed that RefCountIsOne() always return false for the tensor.
  tensorflow::Tensor& tensor() { return tensor_; }
  const tensorflow::Tensor& tensor() const { return tensor_; }

 private:
  explicit ImmutableTensor(tensorflow::Tensor tensor)
      : tensor_(std::move(tensor)) {
    DCHECK(!tensor_.RefCountIsOne())
        << "Immutable tensors' buffers cannot be forwarded.";
  }

  tensorflow::Tensor tensor_;
};

// A wrapper class over normal tensors and immutable tensors. This class is used
// as the currency type in TFRT fallback execution. Note that this class does
// not own the underlying tensor if it is an immutable tensor.
class FallbackTensor {
 public:
  FallbackTensor() = default;

  explicit FallbackTensor(const tensorflow::Tensor& tensor) : tensor_(tensor) {}
  explicit FallbackTensor(tensorflow::Tensor&& tensor)
      : tensor_(std::move(tensor)) {}

  explicit FallbackTensor(ImmutableTensor* immutable_tensor)
      : tensor_(immutable_tensor->tensor()), is_immutable_(true) {}

  FallbackTensor(const FallbackTensor& other) { *this = other; }
  FallbackTensor& operator=(const FallbackTensor& other) {
    tsl::profiler::TraceMe trace_me("FallbackTensor::Copy");
    if (!other.is_immutable() && other.buffer() != nullptr) {
      // Create a new TensorBuffer which contains a new atomic counter for each
      // result, to avoid downstream threads contending the original atomic
      // counter.
      tensor_ = std::move(
          tensorflow::tfrt_stub::ImmutableTensor::Create(other.tensor())
              .tensor());
    } else {
      // For immutable tensors or empty tensors, we just need to copy the
      // pointer as they don't incur atomic operations when they are referenced.
      tensor_ = other.tensor();
    }
    is_immutable_ = true;
    return *this;
  }

  FallbackTensor(FallbackTensor&&) noexcept = default;
  FallbackTensor& operator=(FallbackTensor&&) noexcept = default;

  const TensorBuffer* buffer() const {
    return tensorflow::DMAHelper::buffer(&tensor());
  }
  TensorBuffer* buffer() { return tensorflow::DMAHelper::buffer(&tensor()); }

  bool is_immutable() const { return is_immutable_; }

  tensorflow::Tensor& tensor() { return tensor_; }
  const tensorflow::Tensor& tensor() const { return tensor_; }

 private:
  tensorflow::Tensor tensor_;
  bool is_immutable_ = false;
};

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_UTILS_FALLBACK_TENSOR_H_

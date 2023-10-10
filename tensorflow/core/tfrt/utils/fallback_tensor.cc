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
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"

#include <utility>

#include "tensorflow/core/common_runtime/dma_helper.h"

namespace tensorflow {
namespace tfrt_stub {
namespace {

// An immutable buffer for tensors that can use memcpy. The content cannot be
// changed and users can safely read its content without reference counting.
class ImmutableTensorBuffer final : public tensorflow::TensorBuffer {
 public:
  static tensorflow::core::RefCountPtr<ImmutableTensorBuffer> Create(
      tensorflow::Tensor tensor);

  explicit ImmutableTensorBuffer(tensorflow::Tensor tensor)
      : tensorflow::TensorBuffer(tensor.data()), tensor_(std::move(tensor)) {
    if (auto* buf = tensorflow::DMAHelper::buffer(&tensor_)) {
      root_buffer_ = buf->root_buffer();
    } else {
      root_buffer_ = this;
    }
  }
  ~ImmutableTensorBuffer() override = default;

  size_t size() const override {
    // Instead of using tensorflow::Tensor::TotalBytes(),
    // tensorflow::TensorBuffer::size() should be used, because for cases like
    // tstring they don't match.
    return tensorflow::DMAHelper::buffer(&tensor_)->size();
  }

  // Force OwnsMemory() to return false so that it can never be
  // buffer-forwarded.
  bool OwnsMemory() const override { return false; }

  tensorflow::TensorBuffer* root_buffer() override { return root_buffer_; }
  void FillAllocationDescription(AllocationDescription* proto) const override {}
  bool GetAllocatedBytes(size_t*) const override { return false; }

 private:
  tensorflow::Tensor tensor_;
  tensorflow::TensorBuffer* root_buffer_ = nullptr;
};

tensorflow::core::RefCountPtr<ImmutableTensorBuffer>
ImmutableTensorBuffer::Create(tensorflow::Tensor tensor) {
  return tensorflow::core::RefCountPtr<ImmutableTensorBuffer>(
      new ImmutableTensorBuffer(std::move(tensor)));
}

}  // namespace

ImmutableTensor ImmutableTensor::Create(tensorflow::Tensor tensor) {
  auto dtype = tensor.dtype();
  auto shape = tensor.shape();
  auto immutable_buffer = ImmutableTensorBuffer::Create(std::move(tensor));
  return ImmutableTensor(
      tensorflow::Tensor(dtype, shape, std::move(immutable_buffer)));
}

}  // namespace tfrt_stub
}  // namespace tensorflow

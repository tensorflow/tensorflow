/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TFRT_COMMON_ASYNC_VALUE_TENSOR_H_
#define TENSORFLOW_CORE_TFRT_COMMON_ASYNC_VALUE_TENSOR_H_

#include <memory>

#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/core/framework/tensor.h"
#include "tfrt/support/ref_count.h"  // from @tf_runtime

namespace tfrt {
class AsyncValue;
}  // namespace tfrt
namespace tensorflow {

// The implementation of a Tensor for an AsyncValue and PjRtBuffer. We used it
// to integrate TF with TFRT.
// TODO(b/243983834) After the migration of using PjRt for data transfer is
// completed, GetAsyncRef and SetAsyncRef will be removed and this class will be
// renamed to PjRtBufferTensor.
class AsyncValueTensor {
 public:
  // Downcast from a Tensor to an AsyncValueTensor. Return nullptr if the
  // downcast fails.
  static AsyncValueTensor* FromTensor(const Tensor* tensor);

  const tfrt::RCReference<tfrt::AsyncValue>& GetAsyncRef();

  void SetAsyncRef(tfrt::RCReference<tfrt::AsyncValue> av_ref);

  std::shared_ptr<xla::PjRtBuffer> GetBuffer();

  void SetBuffer(std::shared_ptr<xla::PjRtBuffer> buffer);

  // Convert from a raw pointer to an AsyncValueTensor, removing the pointer
  // tag.
  static AsyncValueTensor* FromOpaquePointer(void* ptr);

  // Convert to a raw pointer from an AsyncValueTensor, adding the pointer tag.
  static void* ToOpaquePointer(AsyncValueTensor* tensor);

 private:
  tfrt::RCReference<tfrt::AsyncValue> av_ref_;
  std::shared_ptr<xla::PjRtBuffer> buffer_;
};

class AsyncValueAllocator : public Allocator {
 public:
  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void DeallocateRaw(void* ptr) override;

  bool AllocatesOpaqueHandle() const override { return true; }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_COMMON_ASYNC_VALUE_TENSOR_H_

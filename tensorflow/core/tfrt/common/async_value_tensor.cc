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
#include "tensorflow/core/tfrt/common/async_value_tensor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "xla/pjrt/pjrt_client.h"
#include "tensorflow/core/framework/tensor.h"
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime

namespace tensorflow {

namespace {
constexpr uintptr_t kTag = 0x1ULL;
}

/*static*/ AsyncValueTensor* AsyncValueTensor::FromTensor(
    const Tensor* tensor) {
  AsyncValueTensor* av_tensor =
      FromOpaquePointer(const_cast<char*>(tensor->tensor_data().data()));
  return av_tensor;
}

const tfrt::RCReference<tfrt::AsyncValue>& AsyncValueTensor::GetAsyncRef() {
  return av_ref_;
}

void AsyncValueTensor::SetAsyncRef(tfrt::RCReference<tfrt::AsyncValue> av_ref) {
  av_ref_ = std::move(av_ref);
}

std::shared_ptr<xla::PjRtBuffer> AsyncValueTensor::GetBuffer() {
  return buffer_;
}

void AsyncValueTensor::SetBuffer(std::shared_ptr<xla::PjRtBuffer> buffer) {
  buffer_ = std::move(buffer);
}

/*static*/ AsyncValueTensor* AsyncValueTensor::FromOpaquePointer(void* ptr) {
  uintptr_t value = reinterpret_cast<uintptr_t>(ptr);
  if (value & kTag) {
    return reinterpret_cast<AsyncValueTensor*>(value & ~kTag);
  } else {
    return nullptr;
  }
}

/*static*/ void* AsyncValueTensor::ToOpaquePointer(AsyncValueTensor* tensor) {
  uintptr_t value = reinterpret_cast<uintptr_t>(tensor);
  CHECK_EQ(value & kTag, 0);  // Crash OK
  value |= kTag;
  return reinterpret_cast<AsyncValueTensor*>(value);
}

void* AsyncValueAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  return AsyncValueTensor::ToOpaquePointer(new AsyncValueTensor);
}

void AsyncValueAllocator::DeallocateRaw(void* ptr) {
  delete AsyncValueTensor::FromOpaquePointer(ptr);
}

}  // namespace tensorflow

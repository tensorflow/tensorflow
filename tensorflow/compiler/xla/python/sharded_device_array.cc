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

#include "tensorflow/compiler/xla/python/sharded_device_array.h"

#include <optional>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/py_buffer.h"
#include "tensorflow/core/platform/statusor.h"

namespace jax {

void ShardedDeviceArray::Delete() {
  // If already deleted, do nothing.
  if (is_deleted_) {
    return;
  }
  for (xla::PjRtBuffer* pjrt_buffer : GetPjRtBuffers().ConsumeValueOrDie()) {
    pjrt_buffer->Delete();
  }
  device_buffers_ = absl::nullopt;
  cpp_device_buffers_ = absl::nullopt;
  npy_value_ = absl::nullopt;
  is_deleted_ = true;
}

xla::StatusOr<absl::Span<xla::PjRtBuffer* const>>
ShardedDeviceArray::GetPjRtBuffers() {
  if (cpp_device_buffers_.has_value()) {
    return absl::MakeConstSpan(cpp_device_buffers_.value());
  }

  if (!device_buffers_.has_value()) {
    return xla::InvalidArgument("ShardedDeviceArray has been deleted.");
  }
  const int num_devices = device_buffers_->size();
  std::vector<xla::PjRtBuffer*> cpp_device_buffers;
  cpp_device_buffers.reserve(num_devices);
  int i = 0;
  for (auto& handle : device_buffers_.value()) {
    // Note that invariants guarantee the cast should never fail.
    TF_ASSIGN_OR_RETURN(xla::PyBuffer * pybuffer,
                        xla::PyBuffer::AsPyBuffer(handle));
    cpp_device_buffers.push_back(pybuffer->buffer());
    i += 1;
  }
  cpp_device_buffers_ = std::move(cpp_device_buffers);
  return absl::MakeConstSpan(cpp_device_buffers_.value());
}

}  // namespace jax

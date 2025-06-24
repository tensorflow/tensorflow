/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/pjrt/gpu/raw_buffer.h"

#include <cstdint>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {

PjRtFuture<> PjRtStreamExecutorRawBuffer::CopyRawHostToDevice(
    const void* src, int64_t offset, int64_t transfer_size) {
  return client_->CopyRawHostToDevice(local_device_, device_buffer_, src,
                                      offset, transfer_size);
}

PjRtFuture<> PjRtStreamExecutorRawBuffer::CopyRawDeviceToHost(
    void* dst, int64_t offset, int64_t transfer_size) {
  return client_->CopyRawDeviceToHost(local_device_, device_buffer_, dst,
                                      offset, transfer_size);
}

std::optional<absl::StatusOr<tsl::RCReference<PjRtRawBuffer>>>
CreateGPURawBuffer(PjRtBuffer* buffer) {
  if (auto* se_buffer = dynamic_cast<PjRtStreamExecutorBuffer*>(buffer)) {
    auto* se_client = dynamic_cast<PjRtStreamExecutorClient*>(buffer->client());
    if (se_client == nullptr) {
      return absl::InvalidArgumentError("invalid se-client");
    }
    PjRtStreamExecutorBuffer::ScopedHold hold(
        se_buffer->GetBufferWithUsageHold());
    if (!hold.ok()) {
      return hold.status();
    }
    if (!hold->device_memory()) {
      return absl::InvalidArgumentError(
          "Create raw buffer called on an invalid buffer");
    }
    return tsl::MakeRef<PjRtStreamExecutorRawBuffer>(
        se_client, se_buffer->memory_space(),
        se_buffer->device()->local_device_state(), hold->device_memory());
  }
  return std::nullopt;
}

REGISTER_PJRT_RAW_BUFFER_FACTORY(CreateGPURawBuffer);

}  // namespace xla

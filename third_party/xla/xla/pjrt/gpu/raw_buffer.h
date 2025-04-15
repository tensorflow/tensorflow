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

#ifndef XLA_PJRT_GPU_RAW_BUFFER_H_
#define XLA_PJRT_GPU_RAW_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <utility>

#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/tracked_device_buffer.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla {

class PjRtStreamExecutorRawBuffer : public PjRtRawBuffer {
 public:
  PjRtStreamExecutorRawBuffer(PjRtStreamExecutorClient* client,
                              PjRtMemorySpace* memory_space,
                              LocalDeviceState* local_device,
                              tsl::RCReference<RawSEDeviceMemory> device_buffer)
      : client_(client),
        memory_space_(memory_space),
        local_device_(local_device),
        device_buffer_(device_buffer) {}
  PjRtMemorySpace* memory_space() const override { return memory_space_; }

  absl::StatusOr<size_t> GetOnDeviceSizeInBytes() const override {
    return device_buffer_->mem().size();
  }

  PjRtFuture<> CopyRawHostToDevice(const void* src, int64_t offset,
                                   int64_t transfer_size) override;

  PjRtFuture<> CopyRawDeviceToHost(void* dst, int64_t offset,
                                   int64_t transfer_size) override;

 private:
  PjRtStreamExecutorClient* client_;
  PjRtMemorySpace* memory_space_;
  LocalDeviceState* local_device_;
  tsl::RCReference<RawSEDeviceMemory> device_buffer_;
};

}  // namespace xla

#endif  // XLA_PJRT_GPU_RAW_BUFFER_H_

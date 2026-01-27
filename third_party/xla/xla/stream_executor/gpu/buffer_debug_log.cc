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

#include "xla/stream_executor/gpu/buffer_debug_log.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/backends/gpu/runtime/buffer_debug_log_structs.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"

namespace stream_executor::gpu {

using ::xla::gpu::BufferDebugLogHeader;

absl::StatusOr<DeviceAddress<uint8_t>> BufferDebugLogBase::CreateOnDevice(
    Stream& stream, DeviceAddress<uint8_t> memory, size_t entry_size) {
  if (memory.is_null()) {
    return absl::InvalidArgumentError("Log buffer must be non-null");
  }

  size_t kMinBufferSize = sizeof(BufferDebugLogHeader) + entry_size;
  if (memory.size() < kMinBufferSize) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Log buffer size %u is too small to hold any log "
                        "entries (required: %u bytes)",
                        memory.size(), kMinBufferSize));
  }

  const uint32_t max_entries =
      (memory.size() - sizeof(BufferDebugLogHeader)) / entry_size;
  const BufferDebugLogHeader empty_header{
      /*write_idx=*/0,
      /*capacity=*/max_entries,
  };
  TF_RETURN_IF_ERROR(
      stream.Memcpy(&memory, &empty_header, sizeof(empty_header)));
  return memory;
}

absl::StatusOr<BufferDebugLogHeader> BufferDebugLogBase::ReadHeaderFromDevice(
    Stream& stream, DeviceAddress<uint8_t> memory) const {
  BufferDebugLogHeader header;
  TF_RETURN_IF_ERROR(stream.Memcpy(&header, memory, sizeof(header)));
  TF_RETURN_IF_ERROR(stream.BlockHostUntilDone());
  return header;
}

absl::StatusOr<size_t> BufferDebugLogBase::ReadFromDevice(
    Stream& stream, DeviceAddress<uint8_t> memory, size_t entry_size,
    void* entries_data) const {
  std::vector<uint8_t> buffer(memory.size());
  TF_RETURN_IF_ERROR(stream.Memcpy(buffer.data(), memory, memory.size()));
  TF_RETURN_IF_ERROR(stream.BlockHostUntilDone());

  BufferDebugLogHeader header;
  memcpy(&header, buffer.data(), sizeof(header));

  const uint32_t max_entries =
      (memory.size() - sizeof(BufferDebugLogHeader)) / entry_size;
  const size_t initialized_entries =
      std::min(max_entries, std::min(header.capacity, header.write_idx));
  memcpy(entries_data, buffer.data() + sizeof(header),
         initialized_entries * entry_size);

  return initialized_entries;
}

}  // namespace stream_executor::gpu

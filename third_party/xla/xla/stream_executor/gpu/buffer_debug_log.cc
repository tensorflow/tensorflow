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
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"

namespace stream_executor::gpu {

using ::xla::gpu::BufferDebugLogEntry;
using ::xla::gpu::BufferDebugLogHeader;

absl::StatusOr<BufferDebugLog> BufferDebugLog::CreateOnDevice(
    Stream& stream, DeviceMemory<uint8_t> log_buffer) {
  if (log_buffer.is_null()) {
    return absl::InvalidArgumentError("Log buffer must be non-null");
  }

  static constexpr size_t kMinBufferSize =
      sizeof(BufferDebugLogHeader) + sizeof(BufferDebugLogEntry);
  if (log_buffer.size() < kMinBufferSize) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Log buffer size %u is too small to hold any log "
                        "entries (required: %u bytes)",
                        log_buffer.size(), kMinBufferSize));
  }

  const uint32_t max_entries =
      (log_buffer.size() - sizeof(BufferDebugLogHeader)) /
      sizeof(BufferDebugLogEntry);
  const BufferDebugLogHeader empty_header{
      /*write_idx=*/0,
      /*capacity=*/max_entries,
  };
  TF_RETURN_IF_ERROR(
      stream.Memcpy(&log_buffer, &empty_header, sizeof(empty_header)));
  return BufferDebugLog(log_buffer);
}

absl::StatusOr<BufferDebugLogHeader> BufferDebugLog::ReadHeaderFromDevice(
    Stream& stream) const {
  BufferDebugLogHeader header;
  TF_RETURN_IF_ERROR(stream.Memcpy(&header, memory_, sizeof(header)));
  TF_RETURN_IF_ERROR(stream.BlockHostUntilDone());
  return header;
}

absl::StatusOr<std::vector<BufferDebugLogEntry>> BufferDebugLog::ReadFromDevice(
    Stream& stream) const {
  std::vector<uint8_t> buffer(memory_.size());
  TF_RETURN_IF_ERROR(stream.Memcpy(buffer.data(), memory_, memory_.size()));
  TF_RETURN_IF_ERROR(stream.BlockHostUntilDone());

  BufferDebugLogHeader header;
  memcpy(&header, buffer.data(), sizeof(header));

  const uint32_t max_entries = (memory_.size() - sizeof(BufferDebugLogHeader)) /
                               sizeof(BufferDebugLogEntry);
  const size_t initialized_entries =
      std::min(max_entries, std::min(header.capacity, header.write_idx));
  std::vector<BufferDebugLogEntry> entries(initialized_entries);
  memcpy(entries.data(), buffer.data() + sizeof(header),
         initialized_entries * sizeof(BufferDebugLogEntry));

  return entries;
}

}  // namespace stream_executor::gpu

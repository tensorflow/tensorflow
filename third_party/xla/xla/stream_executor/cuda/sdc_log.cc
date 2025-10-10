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

#include "xla/stream_executor/cuda/sdc_log.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/backends/gpu/runtime/sdc_log_structs.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::cuda {

using ::xla::gpu::SdcLogEntry;
using ::xla::gpu::SdcLogHeader;

absl::StatusOr<SdcLog> SdcLog::CreateOnDevice(
    Stream& stream, DeviceMemory<uint8_t> log_buffer) {
  if (log_buffer.is_null()) {
    return absl::InvalidArgumentError("Log buffer must be non-null");
  }

  static constexpr size_t kMinBufferSize =
      sizeof(SdcLogHeader) + sizeof(SdcLogEntry);
  if (log_buffer.size() < kMinBufferSize) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Log buffer size %u is too small to hold any log "
                        "entries (required: %u bytes)",
                        log_buffer.size(), kMinBufferSize));
  }

  const uint32_t max_entries =
      (log_buffer.size() - sizeof(SdcLogHeader)) / sizeof(SdcLogEntry);
  const SdcLogHeader empty_header{
      /*write_idx=*/0,
      /*capacity=*/max_entries,
  };
  TF_RETURN_IF_ERROR(
      stream.Memcpy(&log_buffer, &empty_header, sizeof(empty_header)));
  return SdcLog(log_buffer);
}

absl::StatusOr<SdcLogHeader> SdcLog::ReadHeaderFromDevice(
    Stream& stream) const {
  SdcLogHeader header;
  TF_RETURN_IF_ERROR(stream.Memcpy(&header, memory_, sizeof(header)));
  TF_RETURN_IF_ERROR(stream.BlockHostUntilDone());
  return header;
}

absl::StatusOr<std::vector<SdcLogEntry>> SdcLog::ReadFromDevice(
    Stream& stream) const {
  std::vector<uint8_t> buffer(memory_.size());
  TF_RETURN_IF_ERROR(stream.Memcpy(buffer.data(), memory_, memory_.size()));
  TF_RETURN_IF_ERROR(stream.BlockHostUntilDone());

  SdcLogHeader header;
  memcpy(&header, buffer.data(), sizeof(header));

  const uint32_t max_entries =
      (memory_.size() - sizeof(SdcLogHeader)) / sizeof(SdcLogEntry);
  const size_t initialized_entries =
      std::min(max_entries, std::min(header.capacity, header.write_idx));
  std::vector<SdcLogEntry> entries(initialized_entries);
  memcpy(entries.data(), buffer.data() + sizeof(header),
         initialized_entries * sizeof(SdcLogEntry));

  return entries;
}

absl::StatusOr<xla::gpu::SdcLogProto> SdcLog::ReadProto(Stream& stream) const {
  TF_ASSIGN_OR_RETURN(std::vector<SdcLogEntry> entries, ReadFromDevice(stream));

  xla::gpu::SdcLogProto sdc_log_proto;
  sdc_log_proto.mutable_entries()->Reserve(entries.size());
  for (const auto& entry : entries) {
    xla::gpu::SdcLogEntryProto* entry_proto = sdc_log_proto.add_entries();
    entry_proto->set_thunk_id(entry.entry_id.thunk_id().value());
    entry_proto->set_buffer_idx(entry.entry_id.buffer_idx());
    entry_proto->set_checksum(entry.checksum);
  }

  return sdc_log_proto;
}

}  // namespace stream_executor::cuda

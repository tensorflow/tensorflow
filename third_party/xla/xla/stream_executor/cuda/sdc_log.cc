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
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::cuda {

absl::StatusOr<SdcLog> SdcLog::CreateOnDevice(
    uint32_t max_entries, Stream& stream, DeviceMemoryAllocator& allocator) {
  TF_ASSIGN_OR_RETURN(
      OwningDeviceMemory memory,
      allocator.Allocate(
          stream.parent()->device_ordinal(),
          sizeof(SdcLogHeader) + max_entries * sizeof(SdcLogEntry)));
  if (memory.is_null()) {
    return absl::InternalError(absl::StrFormat(
        "Failed to allocate SdcLog for %u entries", max_entries));
  }

  const SdcLogHeader empty_header{
      /*write_idx=*/0,
      /*capacity=*/max_entries,
  };
  TF_RETURN_IF_ERROR(
      stream.Memcpy(memory.ptr(), &empty_header, sizeof(empty_header)));
  return SdcLog(std::move(memory));
}

absl::StatusOr<SdcLogHeader> SdcLog::ReadHeaderFromDevice(
    Stream& stream) const {
  SdcLogHeader header;
  TF_RETURN_IF_ERROR(stream.Memcpy(&header, memory_.cref(), sizeof(header)));
  TF_RETURN_IF_ERROR(stream.BlockHostUntilDone());
  return header;
}

absl::StatusOr<std::vector<SdcLogEntry>> SdcLog::ReadFromDevice(
    Stream& stream) const {
  std::vector<uint8_t> buffer(memory_->size());
  TF_RETURN_IF_ERROR(
      stream.Memcpy(buffer.data(), memory_.cref(), memory_->size()));
  TF_RETURN_IF_ERROR(stream.BlockHostUntilDone());

  SdcLogHeader header;
  memcpy(&header, buffer.data(), sizeof(header));

  const size_t initialized_entries =
      std::min(header.capacity, header.write_idx);
  std::vector<SdcLogEntry> entries(initialized_entries);
  memcpy(entries.data(), buffer.data() + sizeof(header),
         initialized_entries * sizeof(SdcLogEntry));

  return entries;
}

}  // namespace stream_executor::cuda

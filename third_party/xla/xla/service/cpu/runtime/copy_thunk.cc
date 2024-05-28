/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/cpu/runtime/copy_thunk.h"

#include <cstdint>
#include <cstring>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/stream_executor/device_memory.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/numbers.h"
#include "tsl/platform/statusor.h"

namespace xla::cpu {

using ::tsl::strings::HumanReadableNumBytes;

CopyThunk::CopyThunk(BufferAllocation::Slice source_buffer,
                     BufferAllocation::Slice destination_buffer,
                     uint64_t size_in_bytes)
    : Thunk(Kind::kCopy),
      source_buffer_(source_buffer),
      destination_buffer_(destination_buffer),
      size_in_bytes_(size_in_bytes) {}

absl::Status CopyThunk::Execute(const ExecuteParams& params) {
  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase source_data,
      params.buffer_allocations->GetDeviceAddress(source_buffer_));

  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase destination_data,
      params.buffer_allocations->GetDeviceAddress(destination_buffer_));

  VLOG(3) << absl::StrFormat(
      "Copy buffer of size %s from slice %s (%p) to slice %s (%p)",
      HumanReadableNumBytes(size_in_bytes_), source_buffer_.ToString(),
      source_data.opaque(), destination_buffer_.ToString(),
      destination_data.opaque());

  // TODO(ezhulenev): Add benchmarks for copy thunk and add support for
  // running it on multiple threads.
  // TODO(ezhulenev): Use StreamExecutor API instead of std::memcpy? This also
  // requires a benchmark and a multi-threaded implementation.
  // TODO(ezhulenev): Add extra checks for buffer overlap.
  std::memcpy(destination_data.opaque(), source_data.opaque(), size_in_bytes_);

  return absl::OkStatus();
}

}  // namespace xla::cpu

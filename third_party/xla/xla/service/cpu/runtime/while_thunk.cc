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

#include "xla/service/cpu/runtime/while_thunk.h"

#include <utility>

#include "absl/status/status.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/stream_executor/device_memory.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::cpu {

WhileThunk::WhileThunk(Info info, BufferAllocation::Slice cond_buffer,
                       ThunkSequence cond_sequence, ThunkSequence body_sequence)
    : Thunk(Kind::kWhile, std::move(info)),
      cond_buffer_(cond_buffer),
      cond_sequence_(std::move(cond_sequence)),
      body_sequence_(std::move(body_sequence)) {}

absl::Status WhileThunk::Execute(const ExecuteParams& params) {
  tsl::profiler::TraceMe trace([&] { return TraceMeEncode(); });

  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase cond_data,
      params.buffer_allocations->GetDeviceAddress(cond_buffer_));

  bool* condition = reinterpret_cast<bool*>(cond_data.opaque());

  TF_RETURN_IF_ERROR(cond_sequence_.Execute(params));
  while (*condition) {
    TF_RETURN_IF_ERROR(body_sequence_.Execute(params));
    TF_RETURN_IF_ERROR(cond_sequence_.Execute(params));
  }

  return absl::OkStatus();
}

}  // namespace xla::cpu

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

#include "xla/service/cpu/runtime/conditional_thunk.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/util.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla::cpu {

ConditionalThunk::ConditionalThunk(Info info,
                                   BufferAllocation::Slice branch_index_buffer,
                                   std::vector<ThunkSequence> branch_sequences)
    : Thunk(Kind::kConditional, std::move(info)),
      branch_index_buffer_(branch_index_buffer),
      branch_sequences_(std::move(branch_sequences)) {}

absl::Status ConditionalThunk::Execute(const ExecuteParams& params) {
  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase branch_index_data,
      params.buffer_allocations->GetDeviceAddress(branch_index_buffer_));

  VLOG(3) << absl::StreamFormat("Conditional: #branches=%d",
                                branch_sequences_.size());
  VLOG(3) << absl::StreamFormat("  branch index: %s (%p)",
                                branch_index_buffer_.ToString(),
                                branch_index_data.opaque());

  // Default case for out-of-bounds branch index is to execute last branch.
  auto clamp = [&](int32_t branch_index) -> int32_t {
    return (branch_index < 0 || branch_index >= branch_sequences_.size())
               ? branch_sequences_.size() - 1
               : branch_index;
  };

  // See operation semantics documentation:
  // https://openxla.org/xla/operation_semantics#conditional

  // Branch index is pred[].
  if (branch_index_buffer_.size() == sizeof(bool)) {
    bool* pred = reinterpret_cast<bool*>(branch_index_data.opaque());
    VLOG(3) << "  loaded pred[] branch index: " << *pred;
    return branch_sequences_.at(*pred ? 0 : 1).Execute(params);
  }

  // Branch index is s32[].
  if (branch_index_buffer_.size() == sizeof(int32_t)) {
    int32_t* index = reinterpret_cast<int32_t*>(branch_index_data.opaque());
    VLOG(3) << "  loaded s32[] branch index: " << *index;
    return branch_sequences_.at(clamp(*index)).Execute(params);
  }

  return Internal("Unsupported branch index buffer size %d",
                  branch_index_buffer_.size());
}

ConditionalThunk::BufferUses ConditionalThunk::buffer_uses() const {
  BufferUses buffer_uses;
  for (const auto& branch_sequence : branch_sequences_) {
    for (const auto& thunk : branch_sequence) {
      BufferUses uses = thunk->buffer_uses();
      buffer_uses.insert(buffer_uses.end(), uses.begin(), uses.end());
    }
  }
  return buffer_uses;
}

}  // namespace xla::cpu

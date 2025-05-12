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

#include "xla/backends/cpu/runtime/conditional_thunk.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_executor.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::cpu {

absl::StatusOr<std::unique_ptr<ConditionalThunk>> ConditionalThunk::Create(
    Info info, BufferAllocation::Slice branch_index_buffer,
    std::vector<ThunkSequence> branch_sequences) {
  std::vector<ThunkExecutor> branch_executors;
  branch_executors.reserve(branch_sequences.size());
  for (auto& branch_sequence : branch_sequences) {
    TF_ASSIGN_OR_RETURN(auto branch_executor,
                        ThunkExecutor::Create(std::move(branch_sequence)));
    branch_executors.push_back(std::move(branch_executor));
  }
  return absl::WrapUnique(new ConditionalThunk(std::move(info),
                                               std::move(branch_index_buffer),
                                               std::move(branch_executors)));
}

ConditionalThunk::ConditionalThunk(Info info,
                                   BufferAllocation::Slice branch_index_buffer,
                                   std::vector<ThunkExecutor> branch_executors)
    : Thunk(Kind::kConditional, std::move(info)),
      branch_index_buffer_(branch_index_buffer),
      branch_executors_(std::move(branch_executors)) {}

tsl::AsyncValueRef<Thunk::ExecuteEvent> ConditionalThunk::Execute(
    const ExecuteParams& params) {
  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase branch_index_data,
      params.buffer_allocations->GetDeviceAddress(branch_index_buffer_));

  VLOG(3) << absl::StreamFormat("Conditional: #branches=%d",
                                branch_executors_.size());
  VLOG(3) << absl::StreamFormat("  branch index: %s (%p)",
                                branch_index_buffer_.ToString(),
                                branch_index_data.opaque());

  // Default case for out-of-bounds branch index is to execute last branch.
  auto clamp = [&](int32_t branch_index) -> int32_t {
    return (branch_index < 0 || branch_index >= branch_executors_.size())
               ? branch_executors_.size() - 1
               : branch_index;
  };

  // See operation semantics documentation:
  // https://openxla.org/xla/operation_semantics#conditional

  // Branch index is pred[].
  if (branch_index_buffer_.size() == sizeof(bool)) {
    bool* pred = reinterpret_cast<bool*>(branch_index_data.opaque());
    VLOG(3) << "  loaded pred[] branch index: " << *pred;
    return branch_executors_.at(*pred ? 0 : 1).Execute(params);
  }

  // Branch index is s32[].
  if (branch_index_buffer_.size() == sizeof(int32_t)) {
    int32_t* index = reinterpret_cast<int32_t*>(branch_index_data.opaque());
    VLOG(3) << "  loaded s32[] branch index: " << *index;
    return branch_executors_.at(clamp(*index)).Execute(params);
  }

  return Internal("Unsupported branch index buffer size %d",
                  branch_index_buffer_.size());
}

ConditionalThunk::BufferUses ConditionalThunk::buffer_uses() const {
  BufferUses buffer_uses = {BufferUse::Read(branch_index_buffer_)};
  for (const auto& branch_executor : branch_executors_) {
    BufferUses uses = branch_executor.buffer_uses();
    buffer_uses.insert(buffer_uses.end(), uses.begin(), uses.end());
  }
  return buffer_uses;
}

ConditionalThunk::ResourceUses ConditionalThunk::resource_uses() const {
  ResourceUses resource_uses;
  for (const auto& branch_executor : branch_executors_) {
    ResourceUses uses = branch_executor.resource_uses();
    resource_uses.insert(resource_uses.end(), uses.begin(), uses.end());
  }
  return resource_uses;
}

absl::flat_hash_map<std::string, const ThunkSequence*>
ConditionalThunk::get_named_nested_thunks() const {
  absl::flat_hash_map<std::string, const ThunkSequence*> result;
  size_t branch_idx = 0;
  for (const auto& branch_executor : branch_executors_) {
    result[absl::StrCat(info().op_name, "-branch_", branch_idx++)] =
        &branch_executor.thunk_sequence();
  }
  return result;
}

}  // namespace xla::cpu

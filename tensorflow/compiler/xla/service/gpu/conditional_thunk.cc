/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/conditional_thunk.h"

#include <memory>

#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/tsl/platform/errors.h"

namespace xla {
namespace gpu {

ConditionalThunk::ConditionalThunk(
    ThunkInfo thunk_info, ConditionalThunkConfig config,
    const BufferAllocation::Slice& branch_index_buffer_index)
    : Thunk(Kind::kConditional, thunk_info),
      config_(std::move(config)),
      branch_index_buffer_index_(branch_index_buffer_index) {}

Status ConditionalThunk::Initialize(const GpuExecutable& executable,
                                    se::StreamExecutor* executor) {
  if (config_.branch_index_is_bool) {
    TF_RET_CHECK(config_.branch_thunks.size() == 2);
  } else {
    TF_RET_CHECK(!config_.branch_thunks.empty());
  }
  for (auto& branch_thunk : config_.branch_thunks) {
    TF_RETURN_IF_ERROR(branch_thunk->Initialize(executable, executor));
  }
  return OkStatus();
}

Status ConditionalThunk::ExecuteOnStream(const ExecuteParams& params) {
  auto& stream = *params.stream;

  // Copy the predicate value from device.
  int32_t branch_index = -1;
  bool pred = false;
  se::DeviceMemoryBase branch_index_address =
      params.buffer_allocations->GetDeviceAddress(branch_index_buffer_index_);
  if (config_.branch_index_is_bool) {
    stream.ThenMemcpy(&pred, branch_index_address, sizeof(bool));
  } else {
    stream.ThenMemcpy(&branch_index, branch_index_address, sizeof(int32_t));
  }

  Status block_status = stream.BlockHostUntilDone();
  if (!block_status.ok()) {
    return InternalError(
        "Failed to retrieve branch_index value on stream %p: %s.", &stream,
        block_status.message());
  }
  if (config_.branch_index_is_bool) {
    branch_index = pred ? 0 : 1;
  } else {
    // Handle default scenario for branch_index not in [0, num_branches).
    if (branch_index < 0 || branch_index >= config_.branch_count) {
      branch_index = config_.branch_count - 1;
    }
  }

  // Execute the branch computation corresponding to the value of branch_index.
  TF_RETURN_IF_ERROR(
      config_.branch_thunks[branch_index]->ExecuteOnStream(params));

  return OkStatus();
}

}  // namespace gpu
}  // namespace xla

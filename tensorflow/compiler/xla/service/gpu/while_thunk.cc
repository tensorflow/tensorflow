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

#include "tensorflow/compiler/xla/service/gpu/while_thunk.h"

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace gpu {

WhileThunk::WhileThunk(
    ThunkInfo thunk_info,
    const BufferAllocation::Slice& condition_result_buffer_index,
    std::unique_ptr<ThunkSequence> condition_thunk_sequence,
    std::unique_ptr<ThunkSequence> body_thunk_sequence)
    : Thunk(Kind::kWhile, thunk_info),
      condition_result_buffer_index_(condition_result_buffer_index),
      condition_thunk_sequence_(absl::make_unique<SequentialThunk>(
          ThunkInfo(), std::move(*condition_thunk_sequence))),
      body_thunk_sequence_(absl::make_unique<SequentialThunk>(
          ThunkInfo(), std::move(*body_thunk_sequence))) {}

Status WhileThunk::Initialize(const GpuExecutable& executable,
                              se::StreamExecutor* executor) {
  TF_RETURN_IF_ERROR(
      condition_thunk_sequence_->Initialize(executable, executor));
  TF_RETURN_IF_ERROR(body_thunk_sequence_->Initialize(executable, executor));
  return OkStatus();
}

Status WhileThunk::ExecuteOnStream(const ExecuteParams& params) {
  auto& stream = *params.stream;

  se::DeviceMemoryBase condition_result_data =
      params.buffer_allocations->GetDeviceAddress(
          condition_result_buffer_index_);

  while (true) {
    // Invoke thunk sequence for while 'condition' computation.
    VLOG(3) << "Executing condition computation";
    TF_RETURN_IF_ERROR(condition_thunk_sequence_->ExecuteOnStream(params));

    // Copy the result of condition computation and break the loop if 'false'.
    bool condition_result;
    stream.ThenMemcpy(&condition_result, condition_result_data, sizeof(bool));
    VLOG(3) << "condition_result = " << condition_result;
    Status block_status = stream.BlockHostUntilDone();
    if (!block_status.ok()) {
      return InternalError(
          "Failed to complete all kernels launched on stream %p: %s", &stream,
          block_status.error_message());
    }

    if (!condition_result) {
      break;
    }

    VLOG(3) << "Executing body computation";
    // Invoke thunk sequence for while 'body' computation.
    TF_RETURN_IF_ERROR(body_thunk_sequence_->ExecuteOnStream(params));
  }
  return OkStatus();
}

}  // namespace gpu
}  // namespace xla

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
#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace gpu {

WhileThunk::WhileThunk(
    ThunkInfo thunk_info,
    const BufferAllocation::Slice& condition_result_buffer_index,
    std::unique_ptr<ThunkSequence> condition_thunk_sequence,
    std::unique_ptr<ThunkSequence> body_thunk_sequence,
    absl::optional<size_t> condition_profile_index,
    absl::optional<size_t> body_profile_index)
    : Thunk(Kind::kWhile, thunk_info),
      condition_result_buffer_index_(condition_result_buffer_index),
      // Pass nullptr as the HloInstruction* to the condition_thunk_sequence_
      // and body_thunk_sequence_ constructors because these SequentialThunks
      // are logically "part of" this WhileThunk, and shouldn't be profiled
      // separately from it.
      condition_thunk_sequence_(absl::make_unique<SequentialThunk>(
          ThunkInfo(), std::move(*condition_thunk_sequence))),
      body_thunk_sequence_(absl::make_unique<SequentialThunk>(
          ThunkInfo(), std::move(*body_thunk_sequence))),
      condition_profile_index_(condition_profile_index),
      body_profile_index_(body_profile_index) {}

Status WhileThunk::Initialize(const GpuExecutable& executable,
                              se::StreamExecutor* executor) {
  TF_RETURN_IF_ERROR(
      condition_thunk_sequence_->Initialize(executable, executor));
  TF_RETURN_IF_ERROR(body_thunk_sequence_->Initialize(executable, executor));
  return Status::OK();
}

Status WhileThunk::ExecuteOnStream(const ExecuteParams& params) {
  auto& profiler = *params.profiler;
  auto& stream = *params.stream;

  se::DeviceMemoryBase condition_result_data =
      params.buffer_allocations->GetDeviceAddress(
          condition_result_buffer_index_);

  auto op_profiler = profiler.MakeScopedInstructionProfiler(profile_index());
  while (true) {
    // Invoke thunk sequence for while 'condition' computation.
    profiler.StartHloComputation();
    VLOG(3) << "Executing condition computation";
    TF_RETURN_IF_ERROR(condition_thunk_sequence_->ExecuteOnStream(params));
    profiler.FinishHloComputation(condition_profile_index_);

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

    // We measure the time of one execution of the while body computation. The
    // while body may be executed more than once, the last measurement "wins".
    profiler.StartHloComputation();
    VLOG(3) << "Executing body computation";
    // Invoke thunk sequence for while 'body' computation, and pass on
    // 'profiler' to measure the timing of the thunks in 'body_thunk_sequence_'.
    TF_RETURN_IF_ERROR(body_thunk_sequence_->ExecuteOnStream(params));
    profiler.FinishHloComputation(body_profile_index_);
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla

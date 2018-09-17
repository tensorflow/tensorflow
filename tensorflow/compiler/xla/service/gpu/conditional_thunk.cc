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

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace gpu {

ConditionalThunk::ConditionalThunk(
    const BufferAllocation::Slice& predicate_buffer_index,
    const BufferAllocation::Slice& true_operand_buffer_index,
    const BufferAllocation::Slice& false_operand_buffer_index,
    ThunkSequence true_thunk_sequence, ThunkSequence false_thunk_sequence,
    const HloInstruction* hlo)
    : Thunk(Kind::kConditional, hlo),
      predicate_buffer_index_(predicate_buffer_index),
      true_operand_buffer_index_(true_operand_buffer_index),
      false_operand_buffer_index_(false_operand_buffer_index),
      // Pass nullptr as the HloInstruction* to the true_thunk_ and false_thunk_
      // constructors because these SequentialThunks are logically "part of"
      // this ConditionalThunk, and shouldn't be profiled separately from it.
      true_thunk_(std::move(true_thunk_sequence), nullptr),
      false_thunk_(std::move(false_thunk_sequence), nullptr) {}

Status ConditionalThunk::Initialize(const GpuExecutable& executable,
                                    se::StreamExecutor* executor) {
  TF_RETURN_IF_ERROR(true_thunk_.Initialize(executable, executor));
  TF_RETURN_IF_ERROR(false_thunk_.Initialize(executable, executor));
  return Status::OK();
}

Status ConditionalThunk::ExecuteOnStream(
    const BufferAllocations& buffer_allocations, se::Stream* stream,
    HloExecutionProfiler* profiler) {
  auto op_profiler = profiler->MakeScopedInstructionProfiler(hlo_instruction());
  // Copy the predicate value from device.
  bool predicate;
  se::DeviceMemoryBase predicate_address =
      buffer_allocations.GetDeviceAddress(predicate_buffer_index_);
  stream->ThenMemcpy(&predicate, predicate_address, sizeof(bool));

  Status block_status = stream->BlockHostUntilDone();
  if (!block_status.ok()) {
    return InternalError("Failed to retrieve predicate value on stream %p: %s.",
                         stream, block_status.error_message());
  }

  // Execute the true or the false computation depending on the value of the
  // predicate.
  if (predicate) {
    profiler->StartHloComputation();
    TF_RETURN_IF_ERROR(
        true_thunk_.ExecuteOnStream(buffer_allocations, stream, profiler));
    profiler->FinishHloComputation(hlo_instruction()->true_computation());
  } else {
    profiler->StartHloComputation();
    TF_RETURN_IF_ERROR(
        false_thunk_.ExecuteOnStream(buffer_allocations, stream, profiler));
    profiler->FinishHloComputation(hlo_instruction()->false_computation());
  }

  return Status::OK();
}

}  // namespace gpu
}  // namespace xla

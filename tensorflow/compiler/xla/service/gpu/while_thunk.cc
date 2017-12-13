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

#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace gpu {

WhileThunk::WhileThunk(
    const BufferAllocation::Slice& condition_result_buffer_index,
    std::unique_ptr<ThunkSequence> condition_thunk_sequence,
    std::unique_ptr<ThunkSequence> body_thunk_sequence,
    const HloInstruction* hlo)
    : Thunk(Kind::kWhile, hlo),
      condition_result_buffer_index_(condition_result_buffer_index),
      condition_thunk_sequence_(MakeUnique<SequentialThunk>(
          std::move(*condition_thunk_sequence), hlo)),
      body_thunk_sequence_(
          MakeUnique<SequentialThunk>(std::move(*body_thunk_sequence), hlo)) {}

Status WhileThunk::Initialize(const GpuExecutable& executable) {
  TF_RETURN_IF_ERROR(condition_thunk_sequence_->Initialize(executable));
  TF_RETURN_IF_ERROR(body_thunk_sequence_->Initialize(executable));
  return Status::OK();
}

Status WhileThunk::ExecuteOnStream(const BufferAllocations& buffer_allocations,
                                   perftools::gputools::Stream* stream) {
  perftools::gputools::DeviceMemoryBase condition_result_data =
      buffer_allocations.GetDeviceAddress(condition_result_buffer_index_);

  while (true) {
    // Invoke thunk sequence for while 'condition' computation.
    TF_RETURN_IF_ERROR(
        condition_thunk_sequence_->ExecuteOnStream(buffer_allocations, stream));

    // Copy the result of condition computation and break the loop if 'false'.
    bool condition_result;
    stream->ThenMemcpy(&condition_result, condition_result_data, sizeof(bool));
    Status block_status = stream->BlockHostUntilDone();
    if (!block_status.ok()) {
      return InternalError(
          "Failed to complete all kernels launched on stream %p: %s", stream,
          block_status.error_message().c_str());
    }

    if (!condition_result) {
      break;
    }

    // Invoke thunk sequence for while 'body' computation.
    TF_RETURN_IF_ERROR(
        body_thunk_sequence_->ExecuteOnStream(buffer_allocations, stream));
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla

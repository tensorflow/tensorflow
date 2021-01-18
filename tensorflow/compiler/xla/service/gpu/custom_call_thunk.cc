/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/custom_call_thunk.h"

#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/stream_executor/gpu/gpu_stream.h"

namespace xla {
namespace gpu {

CustomCallThunk::CustomCallThunk(ThunkInfo thunk_info, void* call_target,
                                 std::vector<BufferAllocation::Slice> operands,
                                 std::vector<BufferAllocation::Slice> results,
                                 const std::string& opaque)
    : Thunk(Thunk::kCustomCall, thunk_info),
      call_target_(call_target),
      operands_(std::move(operands)),
      results_(std::move(results)),
      opaque_(opaque) {}

Status CustomCallThunk::ExecuteOnStream(const ExecuteParams& params) {
  // gpu_stream is CUstream or e.g. the equivalent type in ROCm.
  auto gpu_stream = se::gpu::AsGpuStreamValue(params.stream);
  using call_type = void (*)(decltype(gpu_stream), void** /*buffers*/,
                             const char* /*opaque*/, size_t /*opaque_len*/);
  auto typed_call_target = reinterpret_cast<call_type>(call_target_);

  std::vector<void*> buffers;
  buffers.reserve(operands_.size() + results_.size());
  for (const std::vector<BufferAllocation::Slice>& slices :
       {operands_, results_}) {
    for (const BufferAllocation::Slice& slice : slices) {
      if (!slice.allocation())
        return InternalError("custom call input missing buffer allocation");
      buffers.push_back(
          params.buffer_allocations->GetDeviceAddress(slice).opaque());
    }
  }

  typed_call_target(gpu_stream, buffers.data(), opaque_.data(), opaque_.size());
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla

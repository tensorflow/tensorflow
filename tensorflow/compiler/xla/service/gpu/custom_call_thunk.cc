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
#include "tensorflow/stream_executor/cuda/cuda_stream.h"
#include "tensorflow/stream_executor/gpu/gpu_stream.h"

namespace xla {
namespace gpu {

CustomCallThunk::CustomCallThunk(
    void* call_target,
    std::vector<ShapeTree<BufferAllocation::Slice>> operand_slices,
    ShapeTree<BufferAllocation::Slice> result_slices, std::string opaque,
    const HloInstruction* instr)
    : Thunk(Thunk::kCustomCall, instr),
      call_target_(call_target),
      operand_slices_(std::move(operand_slices)),
      result_slices_(std::move(result_slices)),
      opaque_(std::move(opaque)) {
  CHECK_EQ(instr->operand_count(), operand_slices_.size());
  for (int64 i = 0; i < instr->operand_count(); ++i) {
    const auto& s1 = operand_slices_[i].shape();
    const auto& s2 = instr->operand(i)->shape();
    CHECK(ShapeUtil::Equal(s1, s2)) << absl::StreamFormat(
        "Shape mismatch between instr->operand(%d) and "
        "operand_slices[%d].shape(): %s vs %s",
        i, i, s1.ToString(), s2.ToString());
  }
  CHECK(ShapeUtil::Equal(instr->shape(), result_slices.shape()))
      << absl::StreamFormat(
             "Shape mismatch between instr->shape() and result_slices.shape(): "
             "%s vs %s.",
             instr->shape().ToString(), result_slices.shape().ToString());
}

Status CustomCallThunk::ExecuteOnStream(const ExecuteParams& params) {
  // gpu_stream is CUstream or e.g. the equivalent type in ROCm.
  auto gpu_stream = se::gpu::AsGpuStreamValue(params.stream);
  auto typed_call_target =
      reinterpret_cast<void (*)(decltype(gpu_stream), void** /*buffers*/,
                                const char* /*opaque*/, size_t /*opaque_len*/)>(
          call_target_);

  std::vector<void*> buffers;
  auto append_buffers = [&](const ShapeTree<BufferAllocation::Slice>& slices) {
    slices.ForEachElement([&](const ShapeIndex& /*index*/,
                              const BufferAllocation::Slice& slice) {
      if (slice.allocation() == nullptr) {
        buffers.push_back(nullptr);
      }
      buffers.push_back(
          params.buffer_allocations->GetDeviceAddress(slice).opaque());
    });
  };
  for (const auto& slices : operand_slices_) {
    append_buffers(slices);
  }
  append_buffers(result_slices_);

  typed_call_target(gpu_stream, buffers.data(), opaque_.data(), opaque_.size());
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla

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

#include "tensorflow/compiler/xla/service/gpu/gemm_thunk.h"

#include <utility>

#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/stream_executor/device_memory.h"

namespace xla {
namespace gpu {

GemmThunk::GemmThunk(ThunkInfo thunk_info, GemmConfig config,
                     const BufferAllocation::Slice &lhs_buffer,
                     const BufferAllocation::Slice &rhs_buffer,
                     const BufferAllocation::Slice &output_buffer)
    : Thunk(Kind::kGemm, thunk_info),
      config_(std::move(config)),
      lhs_buffer_(lhs_buffer),
      rhs_buffer_(rhs_buffer),
      output_buffer_(output_buffer) {}

Status GemmThunk::ExecuteOnStream(const ExecuteParams &params) {
  auto get_device_address = [&](const BufferAllocation::Slice &slice) {
    return params.buffer_allocations->GetDeviceAddress(slice);
  };

  se::DeviceMemoryBase lhs_data = get_device_address(lhs_buffer_);
  se::DeviceMemoryBase rhs_data = get_device_address(rhs_buffer_);
  se::DeviceMemoryBase output_data = get_device_address(output_buffer_);

  VLOG(3) << "Running GEMM thunk";
  if (config_.use_cublaslt && params.stream->parent()->SupportsBlasPlans()) {
    auto &buffer_allocations = *params.buffer_allocations;
    se::OwningScratchAllocator<> scratch_allocator(
        buffer_allocations.device_ordinal(),
        buffer_allocations.memory_allocator());
    return RunBlasLtMatmul(config_, lhs_data, rhs_data, output_data,
                           params.stream, scratch_allocator);
  } else {
    return RunGemm(config_, lhs_data, rhs_data, output_data, params.stream);
  }
}

}  // namespace gpu
}  // namespace xla

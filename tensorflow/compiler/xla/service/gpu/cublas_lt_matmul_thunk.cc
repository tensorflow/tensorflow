/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/cublas_lt_matmul_thunk.h"

#include <utility>

#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/stream_executor/cuda/cuda_blas_lt.h"
#include "tensorflow/stream_executor/device_memory.h"

namespace xla {
namespace gpu {

CublasLtMatmulThunk::CublasLtMatmulThunk(
    ThunkInfo thunk_info, cublas_lt::MatmulPlan plan, int64_t algorithm_idx,
    const BufferAllocation::Slice& lhs_buffer,
    const BufferAllocation::Slice& rhs_buffer,
    const BufferAllocation::Slice& output_buffer)
    : Thunk(Kind::kCublasLtMatmul, thunk_info),
      plan_(std::move(plan)),
      algorithm_idx_(algorithm_idx),
      lhs_buffer_(lhs_buffer),
      rhs_buffer_(rhs_buffer),
      output_buffer_(output_buffer) {}

Status CublasLtMatmulThunk::ExecuteOnStream(const ExecuteParams& params) {
  if (!algorithm_) {
    TF_ASSIGN_OR_RETURN(
        std::vector<se::cuda::BlasLt::MatmulAlgorithm> algorithms,
        plan_.GetAlgorithms(params.stream));
    TF_RET_CHECK(algorithm_idx_ >= 0 && algorithm_idx_ < algorithms.size());
    algorithm_ = algorithms[algorithm_idx_];
  }

  VLOG(3) << "Running cublas_lt matmul thunk";
  const BufferAllocations& allocs = *params.buffer_allocations;
  se::OwningScratchAllocator<> scratch_allocator(allocs.device_ordinal(),
                                                 allocs.memory_allocator());
  return plan_.ExecuteOnStream(
      params.stream, allocs.GetDeviceAddress(lhs_buffer_),
      allocs.GetDeviceAddress(rhs_buffer_),
      allocs.GetDeviceAddress(output_buffer_), *algorithm_, scratch_allocator);
}

}  // namespace gpu
}  // namespace xla

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

#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/status.h"
#if GOOGLE_CUDA
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_blas_lt.h"
#else
#include "tensorflow/compiler/xla/stream_executor/rocm/hip_blas_lt.h"
#endif  // GOOGLE_CUDA
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"
#include "tensorflow/tsl/platform/logging.h"

namespace xla {
namespace gpu {

CublasLtMatmulThunk::CublasLtMatmulThunk(
    ThunkInfo thunk_info, GemmConfig gemm_config,
    se::gpu::BlasLt::Epilogue epilogue, int64_t algorithm_idx,
    BufferAllocation::Slice a_buffer, BufferAllocation::Slice b_buffer,
    BufferAllocation::Slice c_buffer, BufferAllocation::Slice d_buffer,
    BufferAllocation::Slice bias_buffer, BufferAllocation::Slice aux_buffer,
    BufferAllocation::Slice a_scale, BufferAllocation::Slice b_scale,
    BufferAllocation::Slice c_scale, BufferAllocation::Slice d_scale,
    BufferAllocation::Slice d_amax)
    : Thunk(Kind::kCublasLtMatmul, thunk_info),
      gemm_config_(std::move(gemm_config)),
      epilogue_(epilogue),
      algorithm_idx_(algorithm_idx),
      a_buffer_(a_buffer),
      b_buffer_(b_buffer),
      c_buffer_(c_buffer),
      d_buffer_(d_buffer),
      bias_buffer_(bias_buffer),
      aux_buffer_(aux_buffer),
      a_scale_buffer_(a_scale),
      b_scale_buffer_(b_scale),
      c_scale_buffer_(c_scale),
      d_scale_buffer_(d_scale),
      d_amax_buffer_(d_amax) {}

Status CublasLtMatmulThunk::ExecuteOnStream(const ExecuteParams& params) {
  TF_ASSIGN_OR_RETURN(cublas_lt::MatmulPlan * plan,
                      GetMatmulPlan(params.stream));
  if (!algorithm_) {
    TF_ASSIGN_OR_RETURN(
        std::vector<se::gpu::BlasLt::MatmulAlgorithm> algorithms,
        plan->GetAlgorithms(params.stream));
    TF_RET_CHECK(algorithm_idx_ >= 0 && algorithm_idx_ < algorithms.size());
    algorithm_ = algorithms[algorithm_idx_];
  }

  VLOG(3) << "Running cublas_lt matmul thunk";
  const BufferAllocations& allocs = *params.buffer_allocations;

  se::DeviceMemoryBase bias, a_scale, b_scale, c_scale, d_scale, d_amax;
  if (bias_buffer_.allocation() != nullptr) {
    bias = allocs.GetDeviceAddress(bias_buffer_);
  }
  if (a_scale_buffer_.allocation() != nullptr) {
    a_scale = allocs.GetDeviceAddress(a_scale_buffer_);
  }
  if (b_scale_buffer_.allocation() != nullptr) {
    b_scale = allocs.GetDeviceAddress(b_scale_buffer_);
  }
  if (c_scale_buffer_.allocation() != nullptr) {
    c_scale = allocs.GetDeviceAddress(c_scale_buffer_);
  }
  if (d_scale_buffer_.allocation() != nullptr) {
    d_scale = allocs.GetDeviceAddress(d_scale_buffer_);
  }
  if (d_amax_buffer_.allocation() != nullptr) {
    d_amax = allocs.GetDeviceAddress(d_amax_buffer_);
  }

  se::DeviceMemoryBase aux;
  if (aux_buffer_.allocation() != nullptr) {
    aux = allocs.GetDeviceAddress(aux_buffer_);
  }

  se::OwningScratchAllocator<> scratch_allocator(allocs.device_ordinal(),
                                                 allocs.memory_allocator());
  return plan->ExecuteOnStream(
      params.stream, allocs.GetDeviceAddress(a_buffer_),
      allocs.GetDeviceAddress(b_buffer_), allocs.GetDeviceAddress(c_buffer_),
      allocs.GetDeviceAddress(d_buffer_), bias, aux, a_scale, b_scale, c_scale,
      d_scale, d_amax, *algorithm_, scratch_allocator);
}

StatusOr<cublas_lt::MatmulPlan*> CublasLtMatmulThunk::GetMatmulPlan(
    const stream_executor::Stream* stream) {
  absl::MutexLock lock(&matmul_plans_cache_mutex_);
  auto it = matmul_plans_cache_.find(stream);
  if (it == matmul_plans_cache_.end()) {
    TF_ASSIGN_OR_RETURN(cublas_lt::MatmulPlan plan,
                        cublas_lt::MatmulPlan::From(gemm_config_, epilogue_));
    it = matmul_plans_cache_
             .insert({stream,
                      std::make_unique<cublas_lt::MatmulPlan>(std::move(plan))})
             .first;
  }
  return it->second.get();
}

}  // namespace gpu
}  // namespace xla

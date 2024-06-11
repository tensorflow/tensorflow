/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/gpu/runtime/gpublas_lt_matmul_thunk.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/stream.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

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
    BufferAllocation::Slice d_amax,
    std::optional<const BufferAllocation::Slice> workspace_buffer)
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
      d_amax_buffer_(d_amax),
      workspace_buffer_(workspace_buffer) {}

absl::Status CublasLtMatmulThunk::ExecuteOnStream(const ExecuteParams& params) {
  TF_ASSIGN_OR_RETURN(auto plan, GetMatmulPlan(params.stream));

  TF_ASSIGN_OR_RETURN(
      auto algorithm,
      GetMatmulAlgorithm(plan, workspace_buffer_.has_value()
                                   ? workspace_buffer_.value().size()
                                   : 0));

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

  std::optional<se::DeviceMemoryBase> workspace;
  if (workspace_buffer_.has_value()) {
    workspace = allocs.GetDeviceAddress(workspace_buffer_.value());
  }

  return plan->ExecuteOnStream(
      params.stream, allocs.GetDeviceAddress(a_buffer_),
      allocs.GetDeviceAddress(b_buffer_), allocs.GetDeviceAddress(c_buffer_),
      allocs.GetDeviceAddress(d_buffer_), bias, aux, a_scale, b_scale, c_scale,
      d_scale, d_amax, algorithm, workspace);
}

absl::StatusOr<se::gpu::BlasLt::MatmulPlan*> CublasLtMatmulThunk::GetMatmulPlan(
    const stream_executor::Stream* stream) {
  {
    absl::MutexLock lock(&matmul_plans_cache_mutex_);
    auto it = matmul_plans_cache_.find(stream);
    if (it != matmul_plans_cache_.end()) return it->second.get();
  }
  TF_ASSIGN_OR_RETURN(auto plan, se::gpu::BlasLt::GetMatmulPlan(
                                     stream, gemm_config_, epilogue_));

  absl::MutexLock lock(&matmul_plans_cache_mutex_);
  auto [it, _] = matmul_plans_cache_.emplace(stream, std::move(plan));
  return it->second.get();
}

absl::StatusOr<se::gpu::BlasLt::MatmulAlgorithm>
CublasLtMatmulThunk::GetMatmulAlgorithm(const se::gpu::BlasLt::MatmulPlan* plan,
                                        int64_t max_workspace) {
  {
    absl::MutexLock lock(&matmul_algorithm_cache_mutex_);
    auto it = matmul_algorithm_cache_.find(plan);
    if (it != matmul_algorithm_cache_.end()) return it->second;
  }
  TF_ASSIGN_OR_RETURN(
      auto algorithms,
      plan->GetAlgorithms(/*max_algorithm_count*/ 128,
                          /*max_workspace_size*/ max_workspace));
  TF_RET_CHECK(algorithm_idx_ >= 0 && algorithm_idx_ < algorithms.size());

  absl::MutexLock lock(&matmul_algorithm_cache_mutex_);
  auto [it, _] =
      matmul_algorithm_cache_.emplace(plan, algorithms[algorithm_idx_]);
  return it->second;
}

absl::Status CublasLtMatmulThunk::Initialize(const InitializeParams& params) {
  if (!params.executor->AsBlas()) {
    return absl::InternalError("Failed to initialize BLASLT support");
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla

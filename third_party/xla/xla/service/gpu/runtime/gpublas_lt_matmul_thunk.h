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

#ifndef XLA_SERVICE_GPU_RUNTIME_GPUBLAS_LT_MATMUL_THUNK_H_
#define XLA_SERVICE_GPU_RUNTIME_GPUBLAS_LT_MATMUL_THUNK_H_

#include <cstdint>
#include <optional>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/thunk.h"
#include "xla/status.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/stream.h"

namespace xla {
namespace gpu {

class CublasLtMatmulThunk : public Thunk {
 public:
  CublasLtMatmulThunk(ThunkInfo thunk_info, GemmConfig gemm_config,
                      se::gpu::BlasLt::Epilogue epilogue, int64_t algorithm_idx,
                      BufferAllocation::Slice a_buffer,
                      BufferAllocation::Slice b_buffer,
                      BufferAllocation::Slice c_buffer,
                      BufferAllocation::Slice d_buffer,
                      BufferAllocation::Slice bias_buffer /* may be null */,
                      BufferAllocation::Slice aux_buffer /* may be null */,
                      BufferAllocation::Slice a_scale_buffer /* may be null */,
                      BufferAllocation::Slice b_scale_buffer /* may be null */,
                      BufferAllocation::Slice c_scale_buffer /* may be null */,
                      BufferAllocation::Slice d_scale_buffer /* may be null */,
                      BufferAllocation::Slice d_amax_buffer /* may be null */);

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  absl::StatusOr<se::gpu::BlasLt::MatmulPlan*> GetMatmulPlan(
      const stream_executor::Stream* stream);
  absl::StatusOr<std::optional<se::gpu::BlasLt::MatmulAlgorithm> >
  GetMatmulAlgorithm(const se::gpu::BlasLt::MatmulPlan* plan);

  absl::Mutex matmul_plans_cache_mutex_;
  absl::flat_hash_map<const stream_executor::Stream*,
                      se::gpu::BlasLt::MatmulPlanPtr>
      matmul_plans_cache_ ABSL_GUARDED_BY(matmul_plans_cache_mutex_);

  absl::Mutex matmul_algorithm_cache_mutex_;
  absl::flat_hash_map<const se::gpu::BlasLt::MatmulPlan*,
                      se::gpu::BlasLt::MatmulAlgorithm>
      matmul_algorithm_cache_ ABSL_GUARDED_BY(matmul_algorithm_cache_mutex_);

  GemmConfig gemm_config_;
  se::gpu::BlasLt::Epilogue epilogue_;
  int64_t algorithm_idx_;
  BufferAllocation::Slice a_buffer_;
  BufferAllocation::Slice b_buffer_;
  BufferAllocation::Slice c_buffer_;
  BufferAllocation::Slice d_buffer_;
  BufferAllocation::Slice bias_buffer_;
  BufferAllocation::Slice aux_buffer_;
  BufferAllocation::Slice a_scale_buffer_;
  BufferAllocation::Slice b_scale_buffer_;
  BufferAllocation::Slice c_scale_buffer_;
  BufferAllocation::Slice d_scale_buffer_;
  BufferAllocation::Slice d_amax_buffer_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME_GPUBLAS_LT_MATMUL_THUNK_H_

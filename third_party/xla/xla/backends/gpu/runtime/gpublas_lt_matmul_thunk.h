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

#ifndef XLA_BACKENDS_GPU_RUNTIME_GPUBLAS_LT_MATMUL_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_GPUBLAS_LT_MATMUL_THUNK_H_

#include <cstdint>
#include <optional>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/stream.h"

namespace xla {
namespace gpu {

class CublasLtMatmulThunk : public Thunk {
 public:
  CublasLtMatmulThunk(const HloInstruction* instr, GemmConfig gemm_config,
                      se::gpu::BlasLt::Epilogue epilogue, int64_t algorithm_idx,
                      BufferAllocation::Slice a, BufferAllocation::Slice b,
                      BufferAllocation::Slice c, BufferAllocation::Slice d,
                      BufferAllocation::Slice bias /* may be null */,
                      BufferAllocation::Slice aux /* may be null */,
                      BufferAllocation::Slice a_scale /* may be null */,
                      BufferAllocation::Slice b_scale /* may be null */,
                      BufferAllocation::Slice c_scale /* may be null */,
                      BufferAllocation::Slice d_scale /* may be null */,
                      BufferAllocation::Slice d_amax /* may be null */,
                      std::optional<const BufferAllocation::Slice> workspace);

  absl::Status ExecuteOnStream(const ExecuteParams& params) override {
    return ExecuteOnStreamInternal(params.stream, params);
  }
  absl::Status Initialize(const InitializeParams& params) override;
  std::optional<const BufferAllocation::Slice> workspace() const {
    return workspace_;
  }

 protected:
  CublasLtMatmulThunk(const CublasLtMatmulThunk& rhs);

  absl::Status ExecuteOnStreamInternal(se::Stream* stream,
                                       const ExecuteParams& params);
  absl::StatusOr<se::gpu::BlasLt::MatmulPlan*> GetCachedMatmulPlan(
      const ExecuteParams& params);

 protected:
  GemmConfig gemm_config_;
  se::gpu::BlasLt::Epilogue epilogue_;
  int64_t algorithm_idx_;
  std::string canonical_hlo_;
  BufferAllocation::Slice a_;
  BufferAllocation::Slice b_;
  BufferAllocation::Slice c_;
  BufferAllocation::Slice d_;
  BufferAllocation::Slice bias_;
  BufferAllocation::Slice aux_;
  BufferAllocation::Slice a_scale_;
  BufferAllocation::Slice b_scale_;
  BufferAllocation::Slice c_scale_;
  BufferAllocation::Slice d_scale_;
  BufferAllocation::Slice d_amax_;
  std::optional<const BufferAllocation::Slice> workspace_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_GPUBLAS_LT_MATMUL_THUNK_H_

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
#include <memory>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/shaped_slice.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/stream.h"

namespace xla {
namespace gpu {

class CublasLtMatmulThunk : public Thunk {
 public:
  CublasLtMatmulThunk(
      Thunk::ThunkInfo thunk_info, std::string canonical_hlo,
      GemmConfig gemm_config, se::gpu::BlasLt::Epilogue epilogue,
      int64_t algorithm_idx, int64_t autotune_workspace_size, ShapedSlice a,
      ShapedSlice b, ShapedSlice c, ShapedSlice d,
      std::optional<ShapedSlice> bias, std::optional<ShapedSlice> aux,
      std::optional<ShapedSlice> a_scale, std::optional<ShapedSlice> b_scale,
      std::optional<ShapedSlice> c_scale, std::optional<ShapedSlice> d_scale,
      std::optional<ShapedSlice> d_amax,
      std::optional<const ShapedSlice> workspace);

  absl::Status ExecuteOnStream(const ExecuteParams& params) override {
    return ExecuteOnStreamInternal(params.stream, params);
  }
  absl::Status Initialize(const InitializeParams& params) override;
  std::optional<const BufferAllocation::Slice> workspace() const {
    if (workspace_.has_value()) {
      return workspace_->slice;
    }
    return std::nullopt;
  }

  BufferUses buffer_uses() const override;

  absl::StatusOr<ThunkProto> ToProto() const override;
  static absl::StatusOr<std::unique_ptr<Thunk>> FromProto(
      Thunk::ThunkInfo thunk_info, const CublasLtMatmulThunkProto& proto,
      absl::Span<const BufferAllocation> allocations);

  absl::Status AllocatePersistentBuffers(
      ThunkPassBufferAllocator& allocator) override;

 protected:
  CublasLtMatmulThunk(const CublasLtMatmulThunk& rhs);

  absl::Status ExecuteOnStreamInternal(se::Stream* stream,
                                       const ExecuteParams& params);
  absl::StatusOr<se::gpu::BlasLt::MatmulPlan*> GetCachedMatmulPlan(
      const ExecuteParams& params);

  GemmConfig gemm_config_;
  se::gpu::BlasLt::Epilogue epilogue_;
  int64_t algorithm_idx_;
  int64_t autotune_workspace_size_;
  std::string canonical_hlo_;
  ShapedSlice a_;
  ShapedSlice b_;
  ShapedSlice c_;
  ShapedSlice d_;
  std::optional<ShapedSlice> bias_;
  std::optional<ShapedSlice> aux_;
  std::optional<ShapedSlice> a_scale_;
  std::optional<ShapedSlice> b_scale_;
  std::optional<ShapedSlice> c_scale_;
  std::optional<ShapedSlice> d_scale_;
  std::optional<ShapedSlice> d_amax_;
  std::optional<ShapedSlice> workspace_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_GPUBLAS_LT_MATMUL_THUNK_H_

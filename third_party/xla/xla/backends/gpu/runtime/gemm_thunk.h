/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_GEMM_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_GEMM_THUNK_H_

#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/matmul_utils.h"

namespace xla {
namespace gpu {

// This is thread-compatible.
class GemmThunk : public Thunk {
 public:
  // Constructs a thunk that computes "output = (lhs <dot> rhs) * alpha" using
  // BLAS gemm (alpha is stored in the instruction GemmBackendConfig).
  GemmThunk(ThunkInfo thunk_info, GemmConfig config,
            const BufferAllocation::Slice& lhs_buffer,
            const BufferAllocation::Slice& rhs_buffer,
            const BufferAllocation::Slice& output_buffer,
            std::optional<const BufferAllocation::Slice> workspace,
            bool deterministic);

  GemmThunk(const GemmThunk&) = delete;
  GemmThunk& operator=(const GemmThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;
  absl::Status Initialize(const InitializeParams& params) override;

  GemmConfig config() const { return config_; }
  BufferAllocation::Slice lhs_buffer() const { return lhs_buffer_; }
  BufferAllocation::Slice rhs_buffer() const { return rhs_buffer_; }
  BufferAllocation::Slice output_buffer() const { return output_buffer_; }
  std::optional<const BufferAllocation::Slice> workspace() const {
    return workspace_;
  }
  bool deterministic() const { return deterministic_; }

  static absl::StatusOr<std::unique_ptr<GemmThunk>> FromProto(
      ThunkInfo thunk_info, const GemmThunkProto& proto,
      absl::Span<const BufferAllocation> buffer_allocations);

  absl::StatusOr<ThunkProto> ToProto() const override;

 private:
  const GemmConfig config_;
  const BufferAllocation::Slice lhs_buffer_;
  const BufferAllocation::Slice rhs_buffer_;
  const BufferAllocation::Slice output_buffer_;
  std::optional<const BufferAllocation::Slice> workspace_;
  // Whether to run deterministically.
  const bool deterministic_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_GEMM_THUNK_H_

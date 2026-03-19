/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_NORM_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_NORM_THUNK_H_

#include <memory>
#include <optional>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_norm_runner.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

class NormThunk : public Thunk {
 public:
  static absl::StatusOr<std::unique_ptr<NormThunk>> Create(
      ThunkInfo thunk_info, GpuNormDescriptor descriptor,
      BufferAllocation::Slice x, BufferAllocation::Slice scale,
      BufferAllocation::Slice y_or_dx,
      std::optional<BufferAllocation::Slice> bias,
      std::optional<BufferAllocation::Slice> expectation,
      std::optional<BufferAllocation::Slice> norm_factor,
      std::optional<BufferAllocation::Slice> dy,
      std::optional<BufferAllocation::Slice> dscale,
      std::optional<BufferAllocation::Slice> dbias,
      BufferAllocation::Slice scratch);

  NormThunk(const NormThunk&) = delete;
  NormThunk& operator=(const NormThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;
  absl::Status Initialize(const InitializeParams& params) override;

  BufferUses buffer_uses() const override;

  static absl::StatusOr<std::unique_ptr<NormThunk>> FromProto(
      ThunkInfo thunk_info, const NormThunkProto& proto,
      absl::Span<const BufferAllocation> buffer_allocations);

  absl::StatusOr<ThunkProto> ToProto() const override;

 private:
  NormThunk(ThunkInfo thunk_info, GpuNormConfig config,
            GpuNormDescriptor descriptor, BufferAllocation::Slice x,
            BufferAllocation::Slice scale, BufferAllocation::Slice y_or_dx,
            std::optional<BufferAllocation::Slice> bias,
            std::optional<BufferAllocation::Slice> expectation,
            std::optional<BufferAllocation::Slice> norm_factor,
            std::optional<BufferAllocation::Slice> dy,
            std::optional<BufferAllocation::Slice> dscale,
            std::optional<BufferAllocation::Slice> dbias,
            BufferAllocation::Slice scratch);

  BufferAllocation::Slice x_buffer_;
  BufferAllocation::Slice scale_buffer_;
  BufferAllocation::Slice y_or_dx_buffer_;
  std::optional<BufferAllocation::Slice> bias_buffer_;
  std::optional<BufferAllocation::Slice> expectation_buffer_;
  std::optional<BufferAllocation::Slice> norm_factor_buffer_;
  std::optional<BufferAllocation::Slice> dy_buffer_;
  std::optional<BufferAllocation::Slice> dscale_buffer_;
  std::optional<BufferAllocation::Slice> dbias_buffer_;
  BufferAllocation::Slice scratch_buffer_;
  NormRunner& GetOrCreateRunner(const stream_executor::Stream*);

  // Technically this is only needed during initialization to create the
  // GpuNormConfig, but the actual GpuNormConfig is hard to serialize. So we
  // keep the descriptor around for serialization purposes.
  GpuNormDescriptor descriptor_;
  GpuNormConfig config_;
  absl::Mutex mu_;
  absl::flat_hash_map<const stream_executor::Stream*,
                      std::unique_ptr<NormRunner>>
      runner_cache_ ABSL_GUARDED_BY(mu_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_NORM_THUNK_H_

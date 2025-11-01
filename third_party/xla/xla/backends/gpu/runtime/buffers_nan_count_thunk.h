/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_BUFFERS_NAN_COUNT_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_BUFFERS_NAN_COUNT_THUNK_H_

#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_buffer_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/gpu/buffer_debug_nan_count_kernel.h"

namespace xla::gpu {

class BuffersDebugNanCountThunk : public Thunk {
 public:
  explicit BuffersDebugNanCountThunk(
      ThunkInfo info, BufferAllocation::Slice log_slice,
      absl::flat_hash_map<ThunkBufferId, BufferAllocation::Slice> buffers)
      : Thunk(Thunk::Kind::kBuffersDebugNanCount, std::move(info)),
        log_slice_(log_slice),
        buffers_(std::move(buffers)) {}

  absl::Status Initialize(const InitializeParams& params) override;
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  std::string ToString(int indent) const override;

  BufferUses buffer_uses() const override {
    // Intentionally left empty to not nan-count the nan-counting thunk.
    return {};
  }

  const absl::flat_hash_map<ThunkBufferId, BufferAllocation::Slice>&
  buffer_slices() const {
    return buffers_;
  }

 private:
  // Loaded in Initialize.
  std::optional<stream_executor::gpu::BufferDebugNanCountF32Kernel::KernelType>
      kernel_f32_;
  std::optional<stream_executor::gpu::BufferDebugNanCountBf16Kernel::KernelType>
      kernel_bf16_;
  BufferAllocation::Slice log_slice_;
  absl::flat_hash_map<ThunkBufferId, BufferAllocation::Slice> buffers_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_BUFFERS_NAN_COUNT_THUNK_H_

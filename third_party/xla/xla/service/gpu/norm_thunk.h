/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_NORM_THUNK_H_
#define XLA_SERVICE_GPU_NORM_THUNK_H_

#include <memory>
#include <optional>

#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/gpu_norm_runner.h"
#include "xla/service/gpu/thunk.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/status.h"

namespace xla {
namespace gpu {

class NormThunk : public Thunk {
 public:
  NormThunk(ThunkInfo thunk_info, GpuNormConfig config,
            BufferAllocation::Slice input, BufferAllocation::Slice scale,
            BufferAllocation::Slice bias, BufferAllocation::Slice output,
            std::optional<BufferAllocation::Slice> expectation,
            std::optional<BufferAllocation::Slice> norm_factor,
            BufferAllocation::Slice scratch);

  NormThunk(const NormThunk&) = delete;
  NormThunk& operator=(const NormThunk&) = delete;

  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  BufferAllocation::Slice input_buffer_;
  BufferAllocation::Slice scale_buffer_;
  BufferAllocation::Slice bias_buffer_;
  BufferAllocation::Slice output_buffer_;
  std::optional<BufferAllocation::Slice> expectation_buffer_;
  std::optional<BufferAllocation::Slice> norm_factor_buffer_;
  BufferAllocation::Slice scratch_buffer_;
  NormRunner& GetOrCreateRunner(const stream_executor::Stream*);

  GpuNormConfig config_;
  absl::Mutex mu_;
  absl::flat_hash_map<const stream_executor::Stream*,
                      std::unique_ptr<NormRunner>>
      runner_cache_ ABSL_GUARDED_BY(mu_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_NORM_THUNK_H_

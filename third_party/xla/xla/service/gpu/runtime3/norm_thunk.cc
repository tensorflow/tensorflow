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

#include "xla/service/gpu/runtime3/norm_thunk.h"

#include <memory>
#include <optional>

#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

NormThunk::NormThunk(ThunkInfo thunk_info, GpuNormConfig config,
                     BufferAllocation::Slice input_slice,
                     BufferAllocation::Slice scale_slice,
                     BufferAllocation::Slice bias_slice,
                     BufferAllocation::Slice output_slice,
                     std::optional<BufferAllocation::Slice> expectation_slice,
                     std::optional<BufferAllocation::Slice> norm_factor_slice,
                     BufferAllocation::Slice scratch_slice)
    : Thunk(Kind::kNorm, thunk_info),
      input_buffer_(input_slice),
      scale_buffer_(scale_slice),
      bias_buffer_(bias_slice),
      output_buffer_(output_slice),
      expectation_buffer_(expectation_slice),
      norm_factor_buffer_(norm_factor_slice),
      scratch_buffer_(scratch_slice),
      config_(config) {}

NormRunner& NormThunk::GetOrCreateRunner(
    const stream_executor::Stream* stream) {
  absl::MutexLock lock(&mu_);
  auto it = runner_cache_.find(stream);
  if (it == runner_cache_.end()) {
    it = runner_cache_.insert({stream, std::make_unique<NormRunner>(config_)})
             .first;
  }
  return *it->second;
}

Status NormThunk::ExecuteOnStream(const ExecuteParams& params) {
  const auto& buffer_allocations = *params.buffer_allocations;

  se::DeviceMemoryBase input_se_buffer =
      buffer_allocations.GetDeviceAddress(input_buffer_);
  se::DeviceMemoryBase scale_se_buffer =
      buffer_allocations.GetDeviceAddress(scale_buffer_);
  se::DeviceMemoryBase bias_se_buffer =
      buffer_allocations.GetDeviceAddress(bias_buffer_);
  se::DeviceMemoryBase output_se_buffer =
      buffer_allocations.GetDeviceAddress(output_buffer_);

  std::optional<se::DeviceMemoryBase> expectation_se_buffer,
      norm_factor_se_buffer;
  if (expectation_buffer_) {
    expectation_se_buffer =
        buffer_allocations.GetDeviceAddress(expectation_buffer_.value());
  }
  if (norm_factor_buffer_) {
    norm_factor_se_buffer =
        buffer_allocations.GetDeviceAddress(norm_factor_buffer_.value());
  }

  se::DeviceMemoryBase scratch =
      buffer_allocations.GetDeviceAddress(scratch_buffer_);

  RunNormOptions opts;
  opts.norm_runner = &GetOrCreateRunner(params.stream);

  TF_RETURN_IF_ERROR(RunGpuNorm(config_, input_se_buffer, scale_se_buffer,
                                bias_se_buffer, output_se_buffer,
                                expectation_se_buffer, norm_factor_se_buffer,
                                scratch, params.stream, opts));

  if (!params.stream->ok()) {
    return InternalError("NormThunk::ExecuteOnStream failed.");
  }
  return OkStatus();
}

}  // namespace gpu
}  // namespace xla

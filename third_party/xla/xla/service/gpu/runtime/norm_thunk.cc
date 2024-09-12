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

#include "xla/service/gpu/runtime/norm_thunk.h"

#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

NormThunk::NormThunk(ThunkInfo thunk_info, GpuNormConfig config,
                     BufferAllocation::Slice x_slice,
                     BufferAllocation::Slice scale_slice,
                     BufferAllocation::Slice y_or_dx_slice,
                     std::optional<BufferAllocation::Slice> bias_slice,
                     std::optional<BufferAllocation::Slice> expectation_slice,
                     std::optional<BufferAllocation::Slice> norm_factor_slice,
                     std::optional<BufferAllocation::Slice> dy_slice,
                     std::optional<BufferAllocation::Slice> dscale_slice,
                     std::optional<BufferAllocation::Slice> dbias_slice,
                     BufferAllocation::Slice scratch_slice)
    : Thunk(Kind::kNorm, thunk_info),
      x_buffer_(x_slice),
      scale_buffer_(scale_slice),
      y_or_dx_buffer_(y_or_dx_slice),
      bias_buffer_(bias_slice),
      expectation_buffer_(expectation_slice),
      norm_factor_buffer_(norm_factor_slice),
      dy_buffer_(dy_slice),
      dscale_buffer_(dscale_slice),
      dbias_buffer_(dbias_slice),
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

absl::Status NormThunk::ExecuteOnStream(const ExecuteParams& params) {
  const auto& buffer_allocations = *params.buffer_allocations;

  se::DeviceMemoryBase x_se_buffer =
      buffer_allocations.GetDeviceAddress(x_buffer_);
  se::DeviceMemoryBase scale_se_buffer =
      buffer_allocations.GetDeviceAddress(scale_buffer_);
  se::DeviceMemoryBase y_or_dx_se_buffer =
      buffer_allocations.GetDeviceAddress(y_or_dx_buffer_);

  std::optional<se::DeviceMemoryBase> bias_se_buffer, expectation_se_buffer,
      norm_factor_se_buffer, dy_se_buffer, dscale_se_buffer, dbias_se_buffer;
  if (bias_buffer_) {
    bias_se_buffer = buffer_allocations.GetDeviceAddress(bias_buffer_.value());
  }
  if (expectation_buffer_) {
    expectation_se_buffer =
        buffer_allocations.GetDeviceAddress(expectation_buffer_.value());
    norm_factor_se_buffer =
        buffer_allocations.GetDeviceAddress(norm_factor_buffer_.value());
  }
  if (dscale_buffer_) {
    dy_se_buffer = buffer_allocations.GetDeviceAddress(dy_buffer_.value());
    dscale_se_buffer =
        buffer_allocations.GetDeviceAddress(dscale_buffer_.value());
    dbias_se_buffer =
        buffer_allocations.GetDeviceAddress(dbias_buffer_.value());
  }

  se::DeviceMemoryBase scratch =
      buffer_allocations.GetDeviceAddress(scratch_buffer_);

  RunNormOptions opts;
  opts.norm_runner = &GetOrCreateRunner(params.stream);

  TF_RETURN_IF_ERROR(RunGpuNorm(
      config_, x_se_buffer, scale_se_buffer, y_or_dx_se_buffer, bias_se_buffer,
      dy_se_buffer, expectation_se_buffer, norm_factor_se_buffer,
      dscale_se_buffer, dbias_se_buffer, scratch, params.stream, opts));

  if (!params.stream->ok()) {
    return Internal("NormThunk::ExecuteOnStream failed.");
  }
  return absl::OkStatus();
}

absl::Status NormThunk::Initialize(const InitializeParams& params) {
  // Create the runner at initialization time to avoid hangs if we try to build
  // the execution plan while a NCCL collective is running.
  se::dnn::LazyOpRunner<se::dnn::NormOp>* lazy_runner =
      GetOrCreateRunner(params.stream).AsNormRunner();
  TF_ASSIGN_OR_RETURN(auto ln_config, config_.AsDnnNormOpConfig());
  return lazy_runner->GetOrCreateRunner(ln_config, params.stream).status();
}

}  // namespace gpu
}  // namespace xla

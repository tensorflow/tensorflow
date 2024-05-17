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

#include "xla/service/gpu/gpu_norm_runner.h"

#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/lazy_op_runner.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

absl::Status RunGpuNorm(const gpu::GpuNormConfig& config,
                        const se::DeviceMemoryBase& x_buffer,
                        const se::DeviceMemoryBase& scale_buffer,
                        const se::DeviceMemoryBase& y_or_dx_buffer,
                        std::optional<se::DeviceMemoryBase> bias_buffer,
                        std::optional<se::DeviceMemoryBase> dy_buffer,
                        std::optional<se::DeviceMemoryBase> expectation_buffer,
                        std::optional<se::DeviceMemoryBase> norm_factor_buffer,
                        std::optional<se::DeviceMemoryBase> dscale_buffer,
                        std::optional<se::DeviceMemoryBase> dbias_buffer,
                        const se::DeviceMemoryBase& scratch_memory,
                        se::Stream* stream, RunNormOptions options) {
  se::dnn::LazyOpRunner<se::dnn::NormOp>* lazy_runner =
      options.norm_runner->AsNormRunner();
  TF_ASSIGN_OR_RETURN(se::dnn::NormOp::Config ln_config,
                      config.AsDnnNormOpConfig());
  TF_ASSIGN_OR_RETURN(auto* runner,
                      lazy_runner->GetOrCreateRunner(ln_config, stream));

  std::vector<se::DeviceMemoryBase> operands;
  operands.emplace_back(x_buffer);
  operands.emplace_back(scale_buffer);
  operands.emplace_back(y_or_dx_buffer);

  // The remaining operands are composed of inputs followed by outputs of the
  // library call. The expectation and norm factor are outputs of the forward
  // training layer norm, and inputs of the backward layer norm.
  if (config.kind == CudnnNormKind::kLayerForwardInfer ||
      config.kind == CudnnNormKind::kLayerForwardTrain) {
    operands.emplace_back(bias_buffer.value());
  }
  if (config.kind == CudnnNormKind::kLayerForwardTrain) {
    operands.emplace_back(expectation_buffer.value());
    operands.emplace_back(norm_factor_buffer.value());
  }
  if (config.kind == CudnnNormKind::kLayerBackward) {
    operands.emplace_back(dy_buffer.value());
    operands.emplace_back(expectation_buffer.value());
    operands.emplace_back(norm_factor_buffer.value());
    operands.emplace_back(dscale_buffer.value());
    operands.emplace_back(dbias_buffer.value());
  }

  return (*runner)(stream, options.profile_result, scratch_memory, operands);
}

}  // namespace gpu
}  // namespace xla

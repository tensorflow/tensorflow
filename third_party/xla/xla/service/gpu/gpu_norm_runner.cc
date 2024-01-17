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

#include "xla/service/gpu/gpu_norm_runner.h"

#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"

namespace xla {
namespace gpu {

absl::Status RunGpuNorm(const gpu::GpuNormConfig& config,
                        const se::DeviceMemoryBase& input_buffer,
                        const se::DeviceMemoryBase& scale_buffer,
                        const se::DeviceMemoryBase& bias_buffer,
                        const se::DeviceMemoryBase& output_buffer,
                        std::optional<se::DeviceMemoryBase> expectation_buffer,
                        std::optional<se::DeviceMemoryBase> norm_factor_buffer,
                        const se::DeviceMemoryBase& scratch_memory,
                        se::Stream* stream, RunNormOptions options) {
  se::dnn::LazyOpRunner<se::dnn::NormOp>* lazy_runner =
      options.norm_runner->AsNormRunner();
  std::optional<se::dnn::LazyOpRunner<se::dnn::NormOp>> local_runner;

  se::dnn::NormOp::Config ln_config{config.epsilon,
                                    config.input_descriptor,
                                    config.scale_descriptor,
                                    config.bias_descriptor,
                                    config.output_descriptor,
                                    config.expectation_descriptor,
                                    config.norm_factor_descriptor};
  TF_ASSIGN_OR_RETURN(auto* runner,
                      lazy_runner->GetOrCreateRunner(ln_config, stream));

  std::vector<se::DeviceMemoryBase> operands;
  operands.emplace_back(input_buffer);
  operands.emplace_back(scale_buffer);
  operands.emplace_back(bias_buffer);
  operands.emplace_back(output_buffer);
  if (expectation_buffer) {
    operands.emplace_back(expectation_buffer.value());
  }
  if (norm_factor_buffer) {
    operands.emplace_back(norm_factor_buffer.value());
  }

  return (*runner)(stream, options.profile_result, scratch_memory, operands);
}

}  // namespace gpu
}  // namespace xla

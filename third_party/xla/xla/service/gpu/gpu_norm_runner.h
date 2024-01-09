/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_GPU_NORM_RUNNER_H_
#define XLA_SERVICE_GPU_GPU_NORM_RUNNER_H_

#include <optional>
#include <string>
#include <vector>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/lazy_op_runner.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// Intermediate structure used as input to construct GpuNormConfig.
struct GpuNormDescriptor {
  CudnnNormBackendConfig backend_config;
  Shape input_shape;
  Shape scale_shape;
  Shape bias_shape;
  Shape output_shape;
  std::optional<Shape> expectation_shape;
  std::optional<Shape> norm_factor_shape;
  size_t scratch_size;
};

// Structure to describe static properties of a fused norm op.
struct GpuNormConfig {
  static StatusOr<GpuNormConfig> For(const GpuNormDescriptor& desc) {
    std::vector<PrimitiveType> output_types;

    GpuNormConfig config;
    config.epsilon = desc.backend_config.epsilon();
    config.algorithm = se::dnn::AlgorithmDesc(desc.backend_config.algorithm());

    auto tensor_descriptor_from_shape =
        [](Shape shape) -> StatusOr<se::dnn::TensorDescriptor> {
      TF_ASSIGN_OR_RETURN(
          se::dnn::DataType data_type,
          GetDNNDataTypeFromPrimitiveType(shape.element_type()));
      return se::dnn::TensorDescriptor::For(data_type, shape.dimensions(),
                                            shape.layout().minor_to_major());
    };

    TF_ASSIGN_OR_RETURN(config.input_descriptor,
                        tensor_descriptor_from_shape(desc.input_shape));
    TF_ASSIGN_OR_RETURN(config.scale_descriptor,
                        tensor_descriptor_from_shape(desc.scale_shape));
    TF_ASSIGN_OR_RETURN(config.bias_descriptor,
                        tensor_descriptor_from_shape(desc.bias_shape));
    TF_ASSIGN_OR_RETURN(config.output_descriptor,
                        tensor_descriptor_from_shape(desc.output_shape));
    if (desc.expectation_shape) {
      TF_ASSIGN_OR_RETURN(
          config.expectation_descriptor,
          tensor_descriptor_from_shape(desc.expectation_shape.value()));
    }
    if (desc.norm_factor_shape) {
      TF_ASSIGN_OR_RETURN(
          config.norm_factor_descriptor,
          tensor_descriptor_from_shape(desc.norm_factor_shape.value()));
    }
    return config;
  }

  double epsilon;
  se::dnn::AlgorithmDesc algorithm;
  se::dnn::TensorDescriptor input_descriptor;
  se::dnn::TensorDescriptor scale_descriptor;
  se::dnn::TensorDescriptor bias_descriptor;
  se::dnn::TensorDescriptor output_descriptor;
  std::optional<se::dnn::TensorDescriptor> expectation_descriptor;
  std::optional<se::dnn::TensorDescriptor> norm_factor_descriptor;
};

class NormRunner {
 public:
  NormRunner() = default;

  explicit NormRunner(
      std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::NormOp>> runner)
      : repr_(std::move(runner)) {}

  explicit NormRunner(const GpuNormConfig& config)
      : NormRunner(std::make_unique<se::dnn::LazyOpRunner<se::dnn::NormOp>>(
            config.algorithm)) {}

  se::dnn::AlgorithmDesc ToAlgorithmDesc() const {
    return repr_->ToAlgorithmDesc();
  }

  se::dnn::LazyOpRunner<se::dnn::NormOp>* AsNormRunner() { return repr_.get(); }

 private:
  std::unique_ptr<se::dnn::LazyOpRunner<se::dnn::NormOp>> repr_;
};

struct RunNormOptions {
  // Nullable output-parameter pointer for profiling results.
  se::dnn::ProfileResult* profile_result = nullptr;

  // Cannot be nullptr.
  NormRunner* norm_runner;
};

Status RunGpuNorm(const GpuNormConfig& conv_config,
                  const se::DeviceMemoryBase& input_buffer,
                  const se::DeviceMemoryBase& scale_buffer,
                  const se::DeviceMemoryBase& bias_buffer,
                  const se::DeviceMemoryBase& output_buffer,
                  std::optional<se::DeviceMemoryBase> exepctation_buffer,
                  std::optional<se::DeviceMemoryBase> norm_factor_buffer,
                  const se::DeviceMemoryBase& scratch_memory,
                  se::Stream* stream, RunNormOptions options = {});

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_NORM_RUNNER_H_

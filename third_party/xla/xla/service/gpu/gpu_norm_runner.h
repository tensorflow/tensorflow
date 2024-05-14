/* Copyright 2018 The OpenXLA Authors.

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

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/shape.h"
#include "xla/status.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/lazy_op_runner.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

inline absl::StatusOr<xla::gpu::CudnnNormKind> AsCudnnNormKind(
    xla::gpu::CudnnNormBackendConfig_Kind kind) {
  switch (kind) {
    case xla::gpu::CudnnNormBackendConfig::LAYER_FWD_INFER:
      return xla::gpu::CudnnNormKind::kLayerForwardInfer;
    case xla::gpu::CudnnNormBackendConfig::LAYER_FWD_TRAIN:
      return xla::gpu::CudnnNormKind::kLayerForwardTrain;
    case xla::gpu::CudnnNormBackendConfig::LAYER_BWD:
      return xla::gpu::CudnnNormKind::kLayerBackward;
    default:
      return xla::Internal("Unknown norm kind.");
  }
}

// Intermediate structure used as input to construct GpuNormConfig.
struct GpuNormDescriptor {
  CudnnNormBackendConfig backend_config;
  Shape x_shape;
  Shape scale_shape;
  std::optional<Shape> bias_shape;
  Shape y_or_dx_shape;
  std::optional<Shape> expectation_shape;
  std::optional<Shape> norm_factor_shape;
  std::optional<Shape> dy_shape;
  std::optional<Shape> dscale_shape;
  std::optional<Shape> dbias_shape;
  size_t scratch_size;
};

// Structure to describe static properties of a fused norm op.
struct GpuNormConfig {
  static absl::StatusOr<GpuNormConfig> For(const GpuNormDescriptor& desc) {
    std::vector<PrimitiveType> y_or_dx_types;

    GpuNormConfig config;
    config.epsilon = desc.backend_config.epsilon();
    config.algorithm = se::dnn::AlgorithmDesc(desc.backend_config.algorithm());
    TF_ASSIGN_OR_RETURN(config.kind,
                        AsCudnnNormKind(desc.backend_config.kind()));

    auto tensor_descriptor_from_shape =
        [](Shape shape) -> absl::StatusOr<se::dnn::TensorDescriptor> {
      TF_ASSIGN_OR_RETURN(
          se::dnn::DataType data_type,
          GetDNNDataTypeFromPrimitiveType(shape.element_type()));
      return se::dnn::TensorDescriptor::For(data_type, shape.dimensions(),
                                            shape.layout().minor_to_major());
    };

    TF_ASSIGN_OR_RETURN(config.x_descriptor,
                        tensor_descriptor_from_shape(desc.x_shape));
    TF_ASSIGN_OR_RETURN(config.scale_descriptor,
                        tensor_descriptor_from_shape(desc.scale_shape));
    TF_ASSIGN_OR_RETURN(config.y_or_dx_descriptor,
                        tensor_descriptor_from_shape(desc.y_or_dx_shape));
    if (desc.bias_shape) {
      TF_ASSIGN_OR_RETURN(config.bias_descriptor, tensor_descriptor_from_shape(
                                                      desc.bias_shape.value()));
    }
    if (desc.expectation_shape) {
      TF_ASSIGN_OR_RETURN(
          config.expectation_descriptor,
          tensor_descriptor_from_shape(desc.expectation_shape.value()));
      TF_ASSIGN_OR_RETURN(
          config.norm_factor_descriptor,
          tensor_descriptor_from_shape(desc.norm_factor_shape.value()));
    }
    if (desc.dscale_shape) {
      TF_ASSIGN_OR_RETURN(config.dy_descriptor,
                          tensor_descriptor_from_shape(desc.dy_shape.value()));
      TF_ASSIGN_OR_RETURN(
          config.dscale_descriptor,
          tensor_descriptor_from_shape(desc.dscale_shape.value()));
      TF_ASSIGN_OR_RETURN(
          config.dbias_descriptor,
          tensor_descriptor_from_shape(desc.dbias_shape.value()));
    }
    return config;
  }

  absl::StatusOr<se::dnn::NormOp::Config> AsDnnNormOpConfig() const {
    TF_ASSIGN_OR_RETURN(se::dnn::NormKind norm_kind,
                        GetDNNNormKindFromCudnnNormKind(kind));
    return se::dnn::NormOp::Config{norm_kind,
                                   epsilon,
                                   x_descriptor,
                                   scale_descriptor,
                                   y_or_dx_descriptor,
                                   bias_descriptor,
                                   dy_descriptor,
                                   expectation_descriptor,
                                   norm_factor_descriptor,
                                   dscale_descriptor,
                                   dbias_descriptor};
  }

  double epsilon;
  CudnnNormKind kind;
  se::dnn::AlgorithmDesc algorithm;
  se::dnn::TensorDescriptor x_descriptor;
  se::dnn::TensorDescriptor scale_descriptor;
  std::optional<se::dnn::TensorDescriptor> bias_descriptor;
  se::dnn::TensorDescriptor y_or_dx_descriptor;
  std::optional<se::dnn::TensorDescriptor> expectation_descriptor;
  std::optional<se::dnn::TensorDescriptor> norm_factor_descriptor;
  std::optional<se::dnn::TensorDescriptor> dy_descriptor;
  std::optional<se::dnn::TensorDescriptor> dscale_descriptor;
  std::optional<se::dnn::TensorDescriptor> dbias_descriptor;
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

absl::Status RunGpuNorm(const GpuNormConfig& conv_config,
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
                        se::Stream* stream, RunNormOptions options = {});

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_NORM_RUNNER_H_

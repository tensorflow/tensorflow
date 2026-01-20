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
#include "absl/status/statusor.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/gpu_norm_runner.pb.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/lazy_op_runner.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

absl::Status RunGpuNorm(const gpu::GpuNormConfig& config,
                        const se::DeviceAddressBase& x_buffer,
                        const se::DeviceAddressBase& scale_buffer,
                        const se::DeviceAddressBase& y_or_dx_buffer,
                        std::optional<se::DeviceAddressBase> bias_buffer,
                        std::optional<se::DeviceAddressBase> dy_buffer,
                        std::optional<se::DeviceAddressBase> expectation_buffer,
                        std::optional<se::DeviceAddressBase> norm_factor_buffer,
                        std::optional<se::DeviceAddressBase> dscale_buffer,
                        std::optional<se::DeviceAddressBase> dbias_buffer,
                        const se::DeviceAddressBase& scratch_memory,
                        se::Stream* stream, RunNormOptions options) {
  se::dnn::LazyOpRunner<se::dnn::NormOp>* lazy_runner =
      options.norm_runner->AsNormRunner();
  TF_ASSIGN_OR_RETURN(se::dnn::NormOp::Config ln_config,
                      config.AsDnnNormOpConfig());
  TF_ASSIGN_OR_RETURN(auto* runner,
                      lazy_runner->GetOrCreateRunner(ln_config, stream));

  std::vector<se::DeviceAddressBase> operands;
  operands.push_back(x_buffer);
  operands.push_back(scale_buffer);
  operands.push_back(y_or_dx_buffer);

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

GpuNormDescriptorProto GpuNormDescriptor::ToProto() const {
  GpuNormDescriptorProto proto;
  *proto.mutable_backend_config() = backend_config;
  *proto.mutable_x_shape() = x_shape.ToProto();
  *proto.mutable_scale_shape() = scale_shape.ToProto();
  if (bias_shape.has_value()) {
    *proto.mutable_bias_shape() = bias_shape->ToProto();
  }
  *proto.mutable_y_or_dx_shape() = y_or_dx_shape.ToProto();
  if (expectation_shape.has_value()) {
    *proto.mutable_expectation_shape() = expectation_shape->ToProto();
  }
  if (norm_factor_shape.has_value()) {
    *proto.mutable_norm_factor_shape() = norm_factor_shape->ToProto();
  }
  if (dy_shape.has_value()) {
    *proto.mutable_dy_shape() = dy_shape->ToProto();
  }
  if (dscale_shape.has_value()) {
    *proto.mutable_dscale_shape() = dscale_shape->ToProto();
  }
  if (dbias_shape.has_value()) {
    *proto.mutable_dbias_shape() = dbias_shape->ToProto();
  }
  *proto.mutable_scratch_shape() = scratch_shape.ToProto();
  return proto;
}

absl::StatusOr<GpuNormDescriptor> GpuNormDescriptor::FromProto(
    const GpuNormDescriptorProto& proto) {
  GpuNormDescriptor descriptor;
  descriptor.backend_config = proto.backend_config();

  TF_ASSIGN_OR_RETURN(descriptor.x_shape, Shape::FromProto(proto.x_shape()));
  TF_ASSIGN_OR_RETURN(descriptor.scale_shape,
                      Shape::FromProto(proto.scale_shape()));
  if (proto.has_bias_shape()) {
    TF_ASSIGN_OR_RETURN(descriptor.bias_shape,
                        Shape::FromProto(proto.bias_shape()));
  }
  TF_ASSIGN_OR_RETURN(descriptor.y_or_dx_shape,
                      Shape::FromProto(proto.y_or_dx_shape()));
  if (proto.has_expectation_shape()) {
    TF_ASSIGN_OR_RETURN(descriptor.expectation_shape,
                        Shape::FromProto(proto.expectation_shape()));
  }
  if (proto.has_norm_factor_shape()) {
    TF_ASSIGN_OR_RETURN(descriptor.norm_factor_shape,
                        Shape::FromProto(proto.norm_factor_shape()));
  }
  if (proto.has_dy_shape()) {
    TF_ASSIGN_OR_RETURN(descriptor.dy_shape,
                        Shape::FromProto(proto.dy_shape()));
  }
  if (proto.has_dscale_shape()) {
    TF_ASSIGN_OR_RETURN(descriptor.dscale_shape,
                        Shape::FromProto(proto.dscale_shape()));
  }
  if (proto.has_dbias_shape()) {
    TF_ASSIGN_OR_RETURN(descriptor.dbias_shape,
                        Shape::FromProto(proto.dbias_shape()));
  }
  TF_ASSIGN_OR_RETURN(descriptor.scratch_shape,
                      Shape::FromProto(proto.scratch_shape()));
  return descriptor;
}

}  // namespace gpu
}  // namespace xla

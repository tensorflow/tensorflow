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

#include "xla/backends/gpu/runtime/norm_thunk.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/per_device_state.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/util.h"

namespace xla::gpu {
namespace {

absl::StatusOr<se::dnn::NormKind> AsDnnNormKind(
    CudnnNormBackendConfig_Kind kind) {
  switch (kind) {
    case CudnnNormBackendConfig::LAYER_FWD_INFER:
      return se::dnn::LAYER_FWD_INFER;
    case CudnnNormBackendConfig::LAYER_FWD_TRAIN:
      return se::dnn::LAYER_FWD_TRAIN;
    case CudnnNormBackendConfig::LAYER_BWD:
      return se::dnn::LAYER_BWD;
    default:
      return Internal("Unknown norm kind.");
  }
}

absl::StatusOr<se::dnn::TensorDescriptor> TensorDescriptorFromShape(
    const Shape& shape) {
  ABSL_ASSIGN_OR_RETURN(se::dnn::DataType data_type,
                   GetDNNDataTypeFromPrimitiveType(shape.element_type()));
  return se::dnn::TensorDescriptor::For(data_type, shape.dimensions(),
                                        shape.layout().minor_to_major());
}

}  // namespace

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

  ABSL_ASSIGN_OR_RETURN(descriptor.x_shape, Shape::FromProto(proto.x_shape()));
  ABSL_ASSIGN_OR_RETURN(descriptor.scale_shape,
                   Shape::FromProto(proto.scale_shape()));
  if (proto.has_bias_shape()) {
    ABSL_ASSIGN_OR_RETURN(descriptor.bias_shape,
                     Shape::FromProto(proto.bias_shape()));
  }
  ABSL_ASSIGN_OR_RETURN(descriptor.y_or_dx_shape,
                   Shape::FromProto(proto.y_or_dx_shape()));
  if (proto.has_expectation_shape()) {
    ABSL_ASSIGN_OR_RETURN(descriptor.expectation_shape,
                     Shape::FromProto(proto.expectation_shape()));
  }
  if (proto.has_norm_factor_shape()) {
    ABSL_ASSIGN_OR_RETURN(descriptor.norm_factor_shape,
                     Shape::FromProto(proto.norm_factor_shape()));
  }
  if (proto.has_dy_shape()) {
    ABSL_ASSIGN_OR_RETURN(descriptor.dy_shape, Shape::FromProto(proto.dy_shape()));
  }
  if (proto.has_dscale_shape()) {
    ABSL_ASSIGN_OR_RETURN(descriptor.dscale_shape,
                     Shape::FromProto(proto.dscale_shape()));
  }
  if (proto.has_dbias_shape()) {
    ABSL_ASSIGN_OR_RETURN(descriptor.dbias_shape,
                     Shape::FromProto(proto.dbias_shape()));
  }
  ABSL_ASSIGN_OR_RETURN(descriptor.scratch_shape,
                   Shape::FromProto(proto.scratch_shape()));
  return descriptor;
}

absl::StatusOr<NormThunk::Config> NormThunk::Config::For(
    const GpuNormDescriptor& desc) {
  Config config;
  config.epsilon = desc.backend_config.epsilon();
  config.algorithm = se::dnn::AlgorithmDesc(desc.backend_config.algorithm());
  ABSL_ASSIGN_OR_RETURN(config.kind, AsDnnNormKind(desc.backend_config.kind()));

  ABSL_ASSIGN_OR_RETURN(config.x_descriptor,
                   TensorDescriptorFromShape(desc.x_shape));
  ABSL_ASSIGN_OR_RETURN(config.scale_descriptor,
                   TensorDescriptorFromShape(desc.scale_shape));
  ABSL_ASSIGN_OR_RETURN(config.y_or_dx_descriptor,
                   TensorDescriptorFromShape(desc.y_or_dx_shape));
  if (config.kind == se::dnn::LAYER_FWD_INFER ||
      config.kind == se::dnn::LAYER_FWD_TRAIN) {
    if (!desc.bias_shape.has_value()) {
      return InvalidArgument("bias_shape is required for forward norm.");
    }
    ABSL_ASSIGN_OR_RETURN(config.bias_descriptor,
                     TensorDescriptorFromShape(*desc.bias_shape));
  }
  if (config.kind == se::dnn::LAYER_FWD_TRAIN) {
    if (!desc.expectation_shape.has_value() ||
        !desc.norm_factor_shape.has_value()) {
      return InvalidArgument(
          "expectation_shape and norm_factor_shape are required for forward "
          "training norm.");
    }
    ABSL_ASSIGN_OR_RETURN(config.expectation_descriptor,
                     TensorDescriptorFromShape(*desc.expectation_shape));
    ABSL_ASSIGN_OR_RETURN(config.norm_factor_descriptor,
                     TensorDescriptorFromShape(*desc.norm_factor_shape));
  }
  if (config.kind == se::dnn::LAYER_BWD) {
    if (!desc.dy_shape.has_value() || !desc.expectation_shape.has_value() ||
        !desc.norm_factor_shape.has_value() || !desc.dscale_shape.has_value() ||
        !desc.dbias_shape.has_value()) {
      return InvalidArgument(
          "dy_shape, expectation_shape, norm_factor_shape, dscale_shape, and "
          "dbias_shape are required for backward norm.");
    }
    ABSL_ASSIGN_OR_RETURN(config.dy_descriptor,
                     TensorDescriptorFromShape(*desc.dy_shape));
    ABSL_ASSIGN_OR_RETURN(config.expectation_descriptor,
                     TensorDescriptorFromShape(*desc.expectation_shape));
    ABSL_ASSIGN_OR_RETURN(config.norm_factor_descriptor,
                     TensorDescriptorFromShape(*desc.norm_factor_shape));
    ABSL_ASSIGN_OR_RETURN(config.dscale_descriptor,
                     TensorDescriptorFromShape(*desc.dscale_shape));
    ABSL_ASSIGN_OR_RETURN(config.dbias_descriptor,
                     TensorDescriptorFromShape(*desc.dbias_shape));
  }
  return config;
}

absl::StatusOr<std::unique_ptr<NormThunk>> NormThunk::Create(
    ThunkInfo thunk_info, GpuNormDescriptor descriptor,
    BufferAllocation::Slice x_slice, BufferAllocation::Slice scale_slice,
    BufferAllocation::Slice y_or_dx_slice,
    std::optional<BufferAllocation::Slice> bias_slice,
    std::optional<BufferAllocation::Slice> expectation_slice,
    std::optional<BufferAllocation::Slice> norm_factor_slice,
    std::optional<BufferAllocation::Slice> dy_slice,
    std::optional<BufferAllocation::Slice> dscale_slice,
    std::optional<BufferAllocation::Slice> dbias_slice,
    BufferAllocation::Slice scratch_slice, int devices_per_host) {
  ABSL_ASSIGN_OR_RETURN(Config config, Config::For(descriptor));

  // Can't use make_unique because the constructor is private. go/totw/134
  return absl::WrapUnique(new NormThunk(
      thunk_info, std::move(config), std::move(descriptor), x_slice,
      scale_slice, y_or_dx_slice, bias_slice, expectation_slice,
      norm_factor_slice, dy_slice, dscale_slice, dbias_slice, scratch_slice,
      devices_per_host));
}

NormThunk::NormThunk(ThunkInfo thunk_info, Config config,
                     GpuNormDescriptor descriptor,
                     BufferAllocation::Slice x_slice,
                     BufferAllocation::Slice scale_slice,
                     BufferAllocation::Slice y_or_dx_slice,
                     std::optional<BufferAllocation::Slice> bias_slice,
                     std::optional<BufferAllocation::Slice> expectation_slice,
                     std::optional<BufferAllocation::Slice> norm_factor_slice,
                     std::optional<BufferAllocation::Slice> dy_slice,
                     std::optional<BufferAllocation::Slice> dscale_slice,
                     std::optional<BufferAllocation::Slice> dbias_slice,
                     BufferAllocation::Slice scratch_slice,
                     int devices_per_host)
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
      descriptor_(std::move(descriptor)),
      config_(std::move(config)),
      runner_states_(devices_per_host) {}

absl::StatusOr<const se::dnn::NormRunner*> NormThunk::GetOrCreateRunner(
    se::Stream* absl_nonnull stream) {
  int device_ordinal = stream->parent()->device_ordinal();
  ABSL_RETURN_IF_ERROR(runner_states_.GetOrCreateAndInitialize(
      device_ordinal,
      [&](std::unique_ptr<const se::dnn::NormRunner>* state) -> absl::Status {
        se::dnn::DnnSupport* dnn = stream->parent()->AsDnn();
        TF_RET_CHECK(dnn != nullptr) << "No DNN support for stream";
        ABSL_ASSIGN_OR_RETURN(
            *state,
            dnn->NormRunnerFromDesc(
                stream, config_.algorithm, config_.kind, config_.epsilon,
                config_.x_descriptor, config_.scale_descriptor,
                config_.y_or_dx_descriptor, config_.bias_descriptor,
                config_.dy_descriptor, config_.expectation_descriptor,
                config_.norm_factor_descriptor, config_.dscale_descriptor,
                config_.dbias_descriptor));
        TF_RET_CHECK(*state != nullptr)
            << "NormRunnerFromDesc returned null runner";
        return absl::OkStatus();
      }));
  return runner_states_.Find(device_ordinal)->get();
}

absl::Status NormThunk::ExecuteOnStream(const ExecuteParams& params) {
  const auto& buffer_allocations = *params.buffer_allocations;

  ABSL_ASSIGN_OR_RETURN(const se::dnn::NormRunner* runner,
                   GetOrCreateRunner(params.stream));

  std::vector<se::DeviceAddressBase> operands;
  operands.push_back(buffer_allocations.GetDeviceAddress(x_buffer_));
  operands.push_back(buffer_allocations.GetDeviceAddress(scale_buffer_));
  operands.push_back(buffer_allocations.GetDeviceAddress(y_or_dx_buffer_));

  // The remaining operands are composed of inputs followed by outputs of the
  // library call. The expectation and norm factor are outputs of the forward
  // training layer norm, and inputs of the backward layer norm.
  if (config_.kind == se::dnn::LAYER_FWD_INFER ||
      config_.kind == se::dnn::LAYER_FWD_TRAIN) {
    TF_RET_CHECK(bias_buffer_.has_value());
    operands.push_back(buffer_allocations.GetDeviceAddress(*bias_buffer_));
  }
  if (config_.kind == se::dnn::LAYER_FWD_TRAIN) {
    TF_RET_CHECK(expectation_buffer_.has_value());
    TF_RET_CHECK(norm_factor_buffer_.has_value());
    operands.push_back(
        buffer_allocations.GetDeviceAddress(*expectation_buffer_));
    operands.push_back(
        buffer_allocations.GetDeviceAddress(*norm_factor_buffer_));
  }
  if (config_.kind == se::dnn::LAYER_BWD) {
    TF_RET_CHECK(dy_buffer_.has_value());
    TF_RET_CHECK(expectation_buffer_.has_value());
    TF_RET_CHECK(norm_factor_buffer_.has_value());
    TF_RET_CHECK(dscale_buffer_.has_value());
    TF_RET_CHECK(dbias_buffer_.has_value());
    operands.push_back(buffer_allocations.GetDeviceAddress(*dy_buffer_));
    operands.push_back(
        buffer_allocations.GetDeviceAddress(*expectation_buffer_));
    operands.push_back(
        buffer_allocations.GetDeviceAddress(*norm_factor_buffer_));
    operands.push_back(buffer_allocations.GetDeviceAddress(*dscale_buffer_));
    operands.push_back(buffer_allocations.GetDeviceAddress(*dbias_buffer_));
  }

  se::DeviceAddressBase scratch =
      buffer_allocations.GetDeviceAddress(scratch_buffer_);
  if (scratch.size() < runner->GetWorkspaceSize()) {
    return InvalidArgument(
        "Scratch buffer is too small for cuDNN norm: got %d bytes, expected at "
        "least %d bytes.",
        scratch.size(), runner->GetWorkspaceSize());
  }
  ABSL_RETURN_IF_ERROR(
      (*runner)(params.stream, /*profile_result=*/nullptr, scratch, operands));

  if (!params.stream->ok()) {
    return Internal("NormThunk::ExecuteOnStream failed.");
  }
  return absl::OkStatus();
}

absl::Status NormThunk::Initialize(const InitializeParams& params) {
  // Create the runner at initialization time to avoid hangs if we try to build
  // the execution plan while a NCCL collective is running.
  return GetOrCreateRunner(params.stream).status();
}

Thunk::BufferUses NormThunk::buffer_uses() const {
  Thunk::BufferUses res{
      BufferUse::Read(x_buffer_, descriptor_.x_shape),
      BufferUse::Read(scale_buffer_, descriptor_.scale_shape),
      BufferUse::Write(y_or_dx_buffer_, descriptor_.y_or_dx_shape),
      BufferUse::Scratch(scratch_buffer_, descriptor_.scratch_shape),
  };
  if (bias_buffer_.has_value()) {
    res.push_back(BufferUse::Read(*bias_buffer_, *descriptor_.bias_shape));
  }
  if (expectation_buffer_.has_value()) {
    res.push_back(
        BufferUse::Write(*expectation_buffer_, *descriptor_.expectation_shape));
  }
  if (norm_factor_buffer_.has_value()) {
    res.push_back(
        BufferUse::Write(*norm_factor_buffer_, *descriptor_.norm_factor_shape));
  }
  if (dy_buffer_.has_value()) {
    res.push_back(BufferUse::Read(*dy_buffer_, *descriptor_.dy_shape));
  }
  if (dscale_buffer_.has_value()) {
    res.push_back(BufferUse::Write(*dscale_buffer_, *descriptor_.dscale_shape));
  }
  if (dbias_buffer_.has_value()) {
    res.push_back(BufferUse::Write(*dbias_buffer_, *descriptor_.dbias_shape));
  }
  return res;
}

absl::StatusOr<std::unique_ptr<NormThunk>> NormThunk::FromProto(
    ThunkInfo thunk_info, const NormThunkProto& proto,
    absl::Span<const BufferAllocation> buffer_allocations,
    int devices_per_host) {
  ABSL_ASSIGN_OR_RETURN(GpuNormDescriptor descriptor,
                   GpuNormDescriptor::FromProto(proto.norm_descriptor()));

  ABSL_ASSIGN_OR_RETURN(
      BufferAllocation::Slice x,
      BufferAllocation::Slice::FromProto(proto.x(), buffer_allocations));
  ABSL_ASSIGN_OR_RETURN(
      BufferAllocation::Slice scale,
      BufferAllocation::Slice::FromProto(proto.scale(), buffer_allocations));
  ABSL_ASSIGN_OR_RETURN(
      BufferAllocation::Slice y_or_dx,
      BufferAllocation::Slice::FromProto(proto.y_or_dx(), buffer_allocations));
  std::optional<BufferAllocation::Slice> bias;
  if (proto.has_bias()) {
    ABSL_ASSIGN_OR_RETURN(bias, BufferAllocation::Slice::FromProto(
                               proto.bias(), buffer_allocations));
  }
  std::optional<BufferAllocation::Slice> expectation;
  if (proto.has_expectation()) {
    ABSL_ASSIGN_OR_RETURN(expectation, BufferAllocation::Slice::FromProto(
                                      proto.expectation(), buffer_allocations));
  }
  std::optional<BufferAllocation::Slice> norm_factor;
  if (proto.has_norm_factor()) {
    ABSL_ASSIGN_OR_RETURN(norm_factor, BufferAllocation::Slice::FromProto(
                                      proto.norm_factor(), buffer_allocations));
  }
  std::optional<BufferAllocation::Slice> dy;
  if (proto.has_dy()) {
    ABSL_ASSIGN_OR_RETURN(
        dy, BufferAllocation::Slice::FromProto(proto.dy(), buffer_allocations));
  }
  std::optional<BufferAllocation::Slice> dscale;
  if (proto.has_dscale()) {
    ABSL_ASSIGN_OR_RETURN(dscale, BufferAllocation::Slice::FromProto(
                                 proto.dscale(), buffer_allocations));
  }
  std::optional<BufferAllocation::Slice> dbias;
  if (proto.has_dbias()) {
    ABSL_ASSIGN_OR_RETURN(dbias, BufferAllocation::Slice::FromProto(
                                proto.dbias(), buffer_allocations));
  }
  ABSL_ASSIGN_OR_RETURN(
      BufferAllocation::Slice scratch,
      BufferAllocation::Slice::FromProto(proto.scratch(), buffer_allocations));

  return Create(std::move(thunk_info), std::move(descriptor), x, scale, y_or_dx,
                bias, expectation, norm_factor, dy, dscale, dbias, scratch,
                devices_per_host);
}

absl::StatusOr<ThunkProto> NormThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  NormThunkProto* norm_proto = proto.mutable_norm_thunk();
  *norm_proto->mutable_norm_descriptor() = descriptor_.ToProto();

  ABSL_ASSIGN_OR_RETURN(*norm_proto->mutable_x(), x_buffer_.ToProto());
  ABSL_ASSIGN_OR_RETURN(*norm_proto->mutable_scale(), scale_buffer_.ToProto());
  ABSL_ASSIGN_OR_RETURN(*norm_proto->mutable_y_or_dx(), y_or_dx_buffer_.ToProto());
  if (bias_buffer_.has_value()) {
    ABSL_ASSIGN_OR_RETURN(*norm_proto->mutable_bias(), bias_buffer_->ToProto());
  }
  if (expectation_buffer_.has_value()) {
    ABSL_ASSIGN_OR_RETURN(*norm_proto->mutable_expectation(),
                     expectation_buffer_->ToProto());
  }
  if (norm_factor_buffer_.has_value()) {
    ABSL_ASSIGN_OR_RETURN(*norm_proto->mutable_norm_factor(),
                     norm_factor_buffer_->ToProto());
  }
  if (dy_buffer_.has_value()) {
    ABSL_ASSIGN_OR_RETURN(*norm_proto->mutable_dy(), dy_buffer_->ToProto());
  }
  if (dscale_buffer_.has_value()) {
    ABSL_ASSIGN_OR_RETURN(*norm_proto->mutable_dscale(), dscale_buffer_->ToProto());
  }
  if (dbias_buffer_.has_value()) {
    ABSL_ASSIGN_OR_RETURN(*norm_proto->mutable_dbias(), dbias_buffer_->ToProto());
  }
  ABSL_ASSIGN_OR_RETURN(*norm_proto->mutable_scratch(), scratch_buffer_.ToProto());

  return proto;
}

}  // namespace xla::gpu

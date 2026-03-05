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

#include "xla/backends/gpu/autotuner/miopen.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/gpu_conv_runner.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/engine_options.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {
namespace gpu {

namespace se = ::stream_executor;

using MIOpenBackendConfig = stream_executor::dnn::AlgorithmProto;

namespace {

bool IsCustomCallToDnnFusedConvolution(const HloInstruction& hlo) {
  if (hlo.opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  return hlo.custom_call_target() == kCudnnConvBiasActivationForwardCallTarget;
}

// Replaces the instruction with a new instruction with the same name in the
// parent computation. The given instruction will be replaced by a tuple of the
// convolution result and the workspace size. A few following instructions will
// be added to the parent computation to extract the convolution result from the
// new tuple.
absl::Status ApplyConfigAndUpdateWorkspaceInOutputTuple(
    HloInstruction& instr, const MIOpenBackendConfig& config) {
  HloComputation* computation = instr.parent();
  std::vector<Shape> new_call_element_shapes;
  // Add the shapes of the outputs of the convolution.
  new_call_element_shapes.reserve(instr.shape().tuple_shapes().size() - 1);
  for (int i = 0; i < instr.shape().tuple_shapes().size() - 1; ++i) {
    new_call_element_shapes.emplace_back(instr.shape().tuple_shapes(i));
  }
  // The final element is the size of the workspace.
  int64_t workspace_size = config.workspace_size().value();
  new_call_element_shapes.emplace_back(
      ShapeUtil::MakeShape(U8, {workspace_size}));
  Shape new_call_shape = ShapeUtil::MakeTupleShape(new_call_element_shapes);
  HloInstruction* new_call = computation->AddInstruction(
      instr.CloneWithNewOperands(new_call_shape, instr.operands()));
  new_call->SetAndSanitizeName(instr.name());

  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_backend_config,
                      instr.backend_config<GpuBackendConfig>());
  CudnnConvBackendConfig* cudnn_conv_config =
      gpu_backend_config.mutable_cudnn_conv_backend_config();
  *cudnn_conv_config->mutable_algorithm() = config;
  TF_RETURN_IF_ERROR(new_call->set_backend_config(gpu_backend_config));

  std::vector<HloInstruction*> new_tuple_elements;
  new_tuple_elements.reserve(new_call->shape().tuple_shapes().size() - 1);
  for (int i = 0; i < new_call->shape().tuple_shapes().size() - 1; ++i) {
    new_tuple_elements.emplace_back(
        computation->AddInstruction(HloInstruction::CreateGetTupleElement(
            new_call->shape().tuple_shapes(i), new_call, i)));
  }
  new_tuple_elements.emplace_back(computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<uint8_t>({}))));

  // Repackage new_call so it has the same shape as the original call, namely
  // (conv_result, u8[0]).
  HloInstruction* new_tuple = computation->AddInstruction(
      HloInstruction::CreateTuple(new_tuple_elements));

  TF_RETURN_IF_ERROR(instr.parent()->ReplaceInstruction(&instr, new_tuple));
  return absl::OkStatus();
}

absl::Status ApplyConfigToMIOpenCustomCall(HloInstruction& instr,
                                           const MIOpenBackendConfig& config) {
  if (config.has_workspace_size() && config.workspace_size().value() > 0) {
    return ApplyConfigAndUpdateWorkspaceInOutputTuple(instr, config);
  }
  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                      instr.backend_config<GpuBackendConfig>());
  CudnnConvBackendConfig* cudnn_conv_config =
      gpu_config.mutable_cudnn_conv_backend_config();
  *cudnn_conv_config->mutable_algorithm() = config;
  TF_RETURN_IF_ERROR(instr.set_backend_config(std::move(gpu_config)));
  return absl::OkStatus();
}

absl::Status ApplyConfigToFusedMIOpenCustomCall(
    HloInstruction& instr, const MIOpenBackendConfig& config) {
  if (config.algo_id() < 0) {
    return ApplyConfigToMIOpenCustomCall(instr, config);
  }

  // Decompose fused convs back to act(conv + broadcast(bias))
  ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   instr.backend_config<GpuBackendConfig>());

  CudnnConvBackendConfig& backend_config =
      *gpu_config.mutable_cudnn_conv_backend_config();

  CHECK(backend_config.conv_result_scale() == 1.0 &&
        backend_config.side_input_scale() == 0.0 &&
        instr.operands().size() == 3);

  HloInstruction* bias = instr.operands().back();
  const absl::InlinedVector<HloInstruction*, 1> users(instr.users().begin(),
                                                      instr.users().end());
  HloInstruction* conv_result =
      instr.AddInstruction(HloInstruction::CreateGetTupleElement(&instr, 0));
  HloInstruction* bcast_bias =
      instr.AddInstruction(HloInstruction::CreateBroadcast(
          conv_result->shape(), bias,
          {instr.convolution_dimension_numbers().output_feature_dimension()}));
  HloInstruction* conv_bias = instr.AddInstruction(HloInstruction::CreateBinary(
      conv_result->shape(), HloOpcode::kAdd, conv_result, bcast_bias));

  HloInstruction* conv_bias_act = nullptr;

  switch (backend_config.activation_mode()) {
    default:
      break;
    case se::dnn::ActivationMode::kNone:
      conv_bias_act = conv_bias;
      break;
    case se::dnn::ActivationMode::kRelu:
      conv_bias_act = instr.AddInstruction(HloInstruction::CreateBinary(
          conv_bias->shape(), HloOpcode::kMaximum,
          BroadcastZeros(instr.parent(), conv_bias->shape()), conv_bias));
      break;
    case se::dnn::ActivationMode::kElu:
      conv_bias_act = instr.AddInstruction(HloInstruction::CreateTernary(
          conv_bias->shape(), HloOpcode::kSelect,
          instr.AddInstruction(HloInstruction::CreateCompare(
              ShapeUtil::ChangeElementType(conv_bias->shape(), PRED), conv_bias,
              BroadcastZeros(instr.parent(), conv_bias->shape()),
              ComparisonDirection::kGt)),
          conv_bias,
          instr.AddInstruction(HloInstruction::CreateUnary(
              conv_bias->shape(), HloOpcode::kExpm1, conv_bias))));
      break;
    case se::dnn::ActivationMode::kRelu6:
      conv_bias_act = instr.AddInstruction(HloInstruction::CreateTernary(
          conv_bias->shape(), HloOpcode::kClamp,
          BroadcastZeros(instr.parent(), conv_bias->shape()), conv_bias,
          instr.AddInstruction(HloInstruction::CreateBroadcast(
              conv_bias->shape(),
              instr.AddInstruction(HloInstruction::CreateConstant(
                  LiteralUtil::CreateR0(conv_bias->shape().element_type(), 6))),
              {}))));
      break;
    case se::dnn::ActivationMode::kLeakyRelu:
      conv_bias_act = instr.AddInstruction(HloInstruction::CreateTernary(
          conv_bias->shape(), HloOpcode::kSelect,
          instr.AddInstruction(HloInstruction::CreateCompare(
              ShapeUtil::ChangeElementType(conv_bias->shape(), PRED), conv_bias,
              BroadcastZeros(instr.parent(), conv_bias->shape()),
              ComparisonDirection::kGt)),
          conv_bias,
          instr.AddInstruction(HloInstruction::CreateBinary(
              conv_bias->shape(), HloOpcode::kMultiply,
              instr.AddInstruction(HloInstruction::CreateBroadcast(
                  conv_bias->shape(),
                  instr.AddInstruction(HloInstruction::CreateConstant(
                      LiteralUtil::CreateR0(conv_bias->shape().element_type(),
                                            backend_config.leakyrelu_alpha()))),
                  {})),
              conv_bias))));
      break;
  }

  CHECK_NE(conv_bias_act, nullptr);

  HloInstruction* new_result = instr.AddInstruction(HloInstruction::CreateTuple(
      {conv_bias_act, instr.AddInstruction(HloInstruction::CreateConstant(
                          LiteralUtil::CreateR1<uint8_t>({})))}));

  for (auto user : users) {
    RETURN_IF_ERROR(instr.ReplaceUseWith(user, new_result));
  }

  absl::InlinedVector<HloInstruction*, 3> new_operands(instr.operands().begin(),
                                                       instr.operands().end());
  new_operands.pop_back();

  HloInstruction* new_conv = instr.AddInstruction(
      instr.CloneWithNewOperands(instr.shape(), new_operands));
  new_conv->set_custom_call_target(kCudnnConvForwardCallTarget);
  backend_config.set_activation_mode(se::dnn::ActivationMode::kNone);
  RETURN_IF_ERROR(new_conv->set_backend_config(gpu_config));
  // Preserve old name to make it obvious that we decomposed fused conv
  new_conv->SetAndSanitizeName(instr.name());
  RETURN_IF_ERROR(instr.parent()->ReplaceInstruction(&instr, new_conv));

  return ApplyConfigToMIOpenCustomCall(*new_conv, config);
}

}  // namespace

bool MIOpenBackend::IsSupported(const HloInstruction& instr) {
  return IsCustomCallToDnnConvolution(instr);
}

absl::StatusOr<std::unique_ptr<BackendConfig>> MIOpenBackend::GetDefaultConfig(
    const HloInstruction& instr) {
  if (IsSupported(instr)) {
    MIOpenBackendConfig config;
    config.set_algo_id(0);
    auto any = std::make_unique<google::protobuf::Any>();
    any->PackFrom(config);
    return any;
  }
  return absl::InvalidArgumentError(
      "MIOpen backend doesn't support getting a default config for this "
      "instruction.");
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
GetConvolutionCustomCallConfigs(const HloCustomCallInstruction* instr,
                                const HloModule* module,
                                se::StreamExecutor* stream_executor,
                                se::Stream* stream) {
  CHECK(instr->custom_call_target() != kCudnnConvForwardGraphCallTarget);
  ASSIGN_OR_RETURN(GpuConvConfig gpu_conv_config, GetGpuConvConfig(instr));
  se::dnn::ConvolutionKind conv_kind =
      CudnnConvKindToProto(gpu_conv_config.kind);
  ASSIGN_OR_RETURN(se::dnn::DataType input_type,
                   GetDNNDataTypeFromPrimitiveType(gpu_conv_config.input_type));
  ASSIGN_OR_RETURN(
      se::dnn::DataType output_type,
      GetDNNDataTypeFromPrimitiveType(gpu_conv_config.output_type));
  se::dnn::DnnSupport* dnn = stream_executor->AsDnn();
  se::StreamExecutorMemoryAllocator allocator(stream_executor);
  if (stream == nullptr) {
    TF_ASSIGN_OR_RETURN(stream,
                        allocator.GetStream(stream_executor->device_ordinal()));
  }
  bool allow_tf32 = absl::c_all_of(
      instr->precision_config().operand_precision(),
      [](int precision) { return precision <= PrecisionConfig::HIGH; });
  const se::EngineOptions engine_options{RequireDeterminism(module->config()),
                                         allow_tf32,
                                         /*require_command_buffer=*/false};

  se::OwningScratchAllocator<> scratch_allocator(
      stream_executor->device_ordinal(), &allocator);

  const auto initialize_buffer = [stream](se::DeviceAddressBase buffer) {
    // Although we don't have evidence this matters, zero out the buffers
    // before autotuning.  It's conceivable that using uninitialized memory as
    // the inputs might affect performance if e.g. the inputs contain
    // denormals, and this is easy enough.
    return stream->MemZero(&buffer, buffer.size());
  };

  std::vector<se::DeviceAddressBase> operand_buffers;
  operand_buffers.reserve(instr->operand_count());
  for (const auto* operand : instr->operands()) {
    ASSIGN_OR_RETURN(auto buffer, scratch_allocator.AllocateBytes(
                                      ShapeUtil::ByteSizeOf(operand->shape())));
    RETURN_IF_ERROR(initialize_buffer(buffer));
    operand_buffers.push_back(buffer);
  }

  std::vector<se::DeviceAddressBase> result_buffers;
  size_t result_buffers_count = instr->shape().tuple_shapes().size();
  result_buffers.reserve(result_buffers_count);
  for (int i = 0; i < result_buffers_count; ++i) {
    ASSIGN_OR_RETURN(auto buffer,
                     scratch_allocator.AllocateBytes(ShapeUtil::ByteSizeOf(
                         instr->shape().tuple_shapes(i))));
    result_buffers.push_back(buffer);
  }

  ASSIGN_OR_RETURN(
      GpuConvParams gpu_conv_params,
      GetGpuConvParams(gpu_conv_config, absl::MakeSpan(operand_buffers),
                       absl::MakeSpan(result_buffers)));

  std::vector<std::unique_ptr<const se::dnn::ConvRunner>> conv_runners;
  RETURN_IF_ERROR(dnn->GetConvolveRunners(
      conv_kind, input_type, output_type, stream,
      gpu_conv_config.input_descriptor, gpu_conv_params.input_buf,
      gpu_conv_config.filter_descriptor, gpu_conv_params.filter_buf,
      gpu_conv_config.output_descriptor, gpu_conv_params.output_buf,
      gpu_conv_config.conv_desc,
      /* use_fallback = */ false, &scratch_allocator, engine_options,
      &conv_runners));

  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.reserve(conv_runners.size());
  for (const auto& runner : conv_runners) {
    auto any = std::make_unique<google::protobuf::Any>();
    auto desc = runner->ToAlgorithmDesc();
    CHECK_GT(desc->algo_id(), 0);
    any->PackFrom(desc->ToProto());
    configs.push_back(std::move(any));
  }
  return configs;
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
GetFusedConvolutionCustomCallConfigs(const HloCustomCallInstruction* instr,
                                     const HloModule* module,
                                     se::StreamExecutor* stream_executor) {
  ASSIGN_OR_RETURN(GpuConvConfig gpu_conv_config, GetGpuConvConfig(instr));
  ASSIGN_OR_RETURN(se::dnn::DataType input_type,
                   GetDNNDataTypeFromPrimitiveType(gpu_conv_config.input_type));
  ASSIGN_OR_RETURN(
      se::dnn::DataType output_type,
      GetDNNDataTypeFromPrimitiveType(gpu_conv_config.output_type));
  se::dnn::DnnSupport* dnn = stream_executor->AsDnn();

  TF_ASSIGN_OR_RETURN(auto owned_stream, stream_executor->CreateStream());

  std::vector<std::unique_ptr<const se::dnn::FusedConvRunner>> runners;
  RETURN_IF_ERROR(dnn->GetFusedConvolveRunners(
      se::dnn::ConvolutionKind::FORWARD, input_type, output_type, output_type,
      gpu_conv_config.conv_result_scale,
      gpu_conv_config.fusion->side_input_scale,
      gpu_conv_config.fusion->leakyrelu_alpha, owned_stream.get(),
      gpu_conv_config.input_descriptor, gpu_conv_config.filter_descriptor,
      gpu_conv_config.bias_descriptor, gpu_conv_config.output_descriptor,
      gpu_conv_config.conv_desc,
      /* use_fallback */ false, gpu_conv_config.fusion->mode,
      /* numeric_options */ {}, &runners));

  if (!runners.empty()) {
    std::vector<std::unique_ptr<BackendConfig>> configs;
    configs.reserve(runners.size());
    for (const auto& runner : runners) {
      auto any = std::make_unique<google::protobuf::Any>();
      auto desc = runner->ToAlgorithmDesc();
      CHECK_LT(desc->algo_id(), 0);
      any->PackFrom(desc->ToProto());
      configs.push_back(std::move(any));
    }
    return configs;
  }

  // No algorithm found. Try finding algorithm for unfused conv.

  CHECK(gpu_conv_config.conv_result_scale == 1.0 &&
        gpu_conv_config.fusion->side_input_scale == 0.0 &&
        instr->operands().size() == 3);

  absl::InlinedVector<HloInstruction*, 3> new_operands(
      instr->operands().begin(), instr->operands().end());
  new_operands.pop_back();

  auto new_conv = instr->CloneWithNewOperands(instr->shape(), new_operands);
  new_conv->set_custom_call_target(kCudnnConvForwardCallTarget);

  ASSIGN_OR_RETURN(auto gpu_config,
                   new_conv->backend_config<GpuBackendConfig>());
  CudnnConvBackendConfig& backend_config =
      *gpu_config.mutable_cudnn_conv_backend_config();
  backend_config.set_activation_mode(se::dnn::ActivationMode::kNone);
  RETURN_IF_ERROR(new_conv->set_backend_config(gpu_config));

  return GetConvolutionCustomCallConfigs(
      static_cast<HloCustomCallInstruction*>(new_conv.get()), module,
      stream_executor, owned_stream.get());
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
MIOpenBackend::GetSupportedConfigs(const HloInstruction& instr) {
  if (IsSupported(instr)) {
    auto custom_call_instr = Cast<HloCustomCallInstruction>(&instr);
    if (IsCustomCallToDnnFusedConvolution(*custom_call_instr)) {
      return GetFusedConvolutionCustomCallConfigs(
          custom_call_instr, custom_call_instr->GetModule(), stream_executor());
    }

    if (do_not_autotune_) {
      ASSIGN_OR_RETURN(auto default_config, GetDefaultConfig(instr));
      std::vector<std::unique_ptr<BackendConfig>> configs;
      configs.push_back(std::move(default_config));
      return std::move(configs);
    }

    return GetConvolutionCustomCallConfigs(
        custom_call_instr, custom_call_instr->GetModule(), stream_executor(),
        /* stream */ nullptr);
  }
  return std::vector<std::unique_ptr<BackendConfig>>();
}

absl::Status MIOpenBackend::ApplyConfig(HloInstruction& instr,
                                        const BackendConfig& config) {
  MIOpenBackendConfig algorithm_config;
  if (!config.UnpackTo(&algorithm_config)) {
    return absl::InvalidArgumentError(
        "Failed to unpack MIOpenBackendConfig from Any.");
  }
  if (IsSupported(instr)) {
    if (IsCustomCallToDnnFusedConvolution(instr)) {
      return ApplyConfigToFusedMIOpenCustomCall(instr, algorithm_config);
    }
    return ApplyConfigToMIOpenCustomCall(instr, algorithm_config);
  }
  return absl::InvalidArgumentError(
      "MIOpen backend doesn't support this instruction.");
}

}  // namespace gpu
}  // namespace xla

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

#include "xla/backends/gpu/autotuner/cudnn.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/transforms/cudnn_fusion_compiler.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/literal_util.h"
#include "xla/service/algorithm_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/gpu_conv_runner.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/shape.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/engine_options.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace se = ::stream_executor;

using CudnnBackendConfig = stream_executor::dnn::AlgorithmProto;

namespace {

// Replaces the instruction with a new instruction with the same name in the
// parent computation. The given instruction will be replaced by a tuple of the
// convolution result and the workspace size. A few following instructions will
// be added to the parent computation to extract the convolution result from the
// new tuple.
absl::Status ApplyConfigAndUpdateWorkspaceInOutputTuple(
    HloInstruction& instr, const CudnnBackendConfig& config) {
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

bool IsSupportedCudnnFusion(const HloInstruction& instr,
                            se::StreamExecutor* stream_executor,
                            const DebugOptions& debug_options) {
  if (!instr.has_backend_config() ||
      !instr.backend_config<GpuBackendConfig>()->has_fusion_backend_config() ||
      instr.backend_config<GpuBackendConfig>()
              ->fusion_backend_config()
              .kind() != kCuDnnFusionKind) {
    VLOG(1) << "Instr is not a cudnn fusion.";
    return false;
  }

  const HloComputation* computation = instr.fused_instructions_computation();
  const HloInstruction* hero =
      hlo_query::GetFirstInstructionWithOpcode(*computation, HloOpcode::kDot);
  if (hero == nullptr) {
    hero = hlo_query::GetFirstInstructionWithOpcode(*computation,
                                                    HloOpcode::kConvolution);
  }
  if (hero == nullptr) {
    hero = hlo_query::GetFirstInstructionWithOpcode(*computation,
                                                    HloOpcode::kScaledDot);
  }

  if (hero == nullptr) {
    VLOG(1) << "Fusion does not contain a dot or convolution.";
    return false;
  }

  PrecisionConfig::Algorithm algorithm = PrecisionConfig::ALG_UNSET;
  if (auto* dot = DynCast<HloDotInstruction>(hero)) {
    algorithm = dot->precision_config().algorithm();
  } else if (auto* conv = DynCast<HloConvolutionInstruction>(hero)) {
    algorithm = conv->precision_config().algorithm();
  } else if (auto* scaled_dot = DynCast<HloScaledDotInstruction>(hero)) {
    algorithm = scaled_dot->precision_config().algorithm();
  }

  if (!algorithm_util::IsSupportedByCudnn(algorithm)) {
    VLOG(1) << "Fusion contains a precision config not supported by cudnn.";
    return false;
  }

  if (GetDnnVersionInfoOrDefault(stream_executor).major_version() < 9) {
    VLOG(1) << "Cudnn version is too old.";
    return false;
  }

  if (hero->opcode() == HloOpcode::kConvolution) {
    return true;
  }

  stream_executor::CudaComputeCapability compute_capability =
      stream_executor->GetDeviceDescription().cuda_compute_capability();
  if ((compute_capability.IsAtLeastAmpere() &&
       debug_options.xla_gpu_cudnn_gemm_fusion_level() > 1) ||
      (compute_capability.IsAtLeastBlackwell() &&
       debug_options.xla_gpu_cudnn_gemm_fusion_level() > 0)) {
    return true;
  }

  VLOG(1) << "Fusion is not supported by cudnn.";
  return false;
}

absl::StatusOr<std::vector<CudnnBackendConfig>> GetAlgorithms(
    se::dnn::DnnSupport* dnn, se::dnn::ConvolutionKind conv_kind,
    se::dnn::DataType input_type, se::dnn::DataType output_type,
    se::Stream* stream, const GpuConvConfig& gpu_conv_config,
    const se::EngineOptions& engine_options, bool use_fallback) {
  std::vector<std::unique_ptr<const se::dnn::ConvRunner>> conv_runners;
  std::vector<std::unique_ptr<const se::dnn::FusedConvRunner>>
      fused_conv_runners;
  std::vector<std::unique_ptr<const se::dnn::GraphConvRunner>>
      graph_conv_runners;
  switch (conv_kind) {
    case se::dnn::ConvolutionKind::FORWARD_BIAS_ACTIVATION: {
      if (!gpu_conv_config.fusion) {
        return absl::InvalidArgumentError(
            "GpuConvConfig had fusion ConvolutionKind but no FusionConfig.");
      }
      TF_RETURN_IF_ERROR(dnn->GetFusedConvolveRunners(
          se::dnn::ConvolutionKind::FORWARD, input_type,
          BiasTypeForInputType(input_type), output_type,
          gpu_conv_config.conv_result_scale,
          gpu_conv_config.fusion->side_input_scale,
          gpu_conv_config.fusion->leakyrelu_alpha, stream,
          gpu_conv_config.input_descriptor, gpu_conv_config.filter_descriptor,
          gpu_conv_config.bias_descriptor, gpu_conv_config.output_descriptor,
          gpu_conv_config.conv_desc, use_fallback, gpu_conv_config.fusion->mode,
          engine_options, &fused_conv_runners));
      break;
    }
    case se::dnn::ConvolutionKind::FORWARD_GRAPH: {
      TF_RETURN_IF_ERROR(dnn->GetGraphConvolveRunners(
          conv_kind, input_type, output_type, stream,
          gpu_conv_config.input_descriptor, gpu_conv_config.filter_descriptor,
          gpu_conv_config.output_descriptor, gpu_conv_config.conv_desc,
          use_fallback, engine_options, &graph_conv_runners,
          gpu_conv_config.serialized_graph));
      break;
    }
    case se::dnn::ConvolutionKind::FORWARD:
    case se::dnn::ConvolutionKind::BACKWARD_DATA:
    case se::dnn::ConvolutionKind::BACKWARD_FILTER: {
      TF_RETURN_IF_ERROR(dnn->GetConvolveRunners(
          conv_kind, input_type, output_type, stream,
          gpu_conv_config.input_descriptor,
          /*input_data=*/se::DeviceAddressBase(nullptr),
          gpu_conv_config.filter_descriptor,
          /*filter_data=*/se::DeviceAddressBase(nullptr),
          gpu_conv_config.output_descriptor,
          /*output_data=*/se::DeviceAddressBase(nullptr),
          gpu_conv_config.conv_desc, use_fallback,
          /*scratch_allocator=*/nullptr, engine_options, &conv_runners));
      break;
    }
    default:
      return absl::InvalidArgumentError(
          "Cudnn backend doesn't support this convolution kind.");
  }

  std::vector<CudnnBackendConfig> configs;
  if (!conv_runners.empty()) {
    configs.reserve(conv_runners.size());
    for (const auto& runner : conv_runners) {
      configs.push_back(runner->ToAlgorithmDesc()->ToProto());
    }
  } else if (!fused_conv_runners.empty()) {
    configs.reserve(fused_conv_runners.size());
    for (const auto& runner : fused_conv_runners) {
      configs.push_back(runner->ToAlgorithmDesc()->ToProto());
    }
  } else if (!graph_conv_runners.empty()) {
    configs.reserve(graph_conv_runners.size());
    for (const auto& runner : graph_conv_runners) {
      configs.push_back(runner->ToAlgorithmDesc()->ToProto());
    }
  }
  return configs;
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
GetCudnnFusionConfigs(const HloInstruction& instr,
                      se::StreamExecutor* stream_executor) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  int plan_count = CuDnnFusionCompiler::GetAvailablePlanCount(
      *stream_executor, *DynCast<HloFusionInstruction>(&instr));
  VLOG(2) << "Found " << plan_count << " plans for cudnn fusion.";
  configs.reserve(plan_count);
  for (int plan_id = 0; plan_id < plan_count; ++plan_id) {
    CudnnBackendConfig config;
    config.set_algo_id(plan_id);
    auto any = std::make_unique<google::protobuf::Any>();
    any->PackFrom(config);
    configs.push_back(std::move(any));
  }
  return configs;
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
GetConvolutionCustomCallConfigs(const HloCustomCallInstruction* instr,
                                se::StreamExecutor* stream_executor) {
  TF_ASSIGN_OR_RETURN(GpuConvConfig gpu_conv_config, GetGpuConvConfig(instr));
  se::dnn::ConvolutionKind conv_kind =
      CudnnConvKindToProto(gpu_conv_config.kind);
  TF_ASSIGN_OR_RETURN(
      se::dnn::DataType input_type,
      GetDNNDataTypeFromPrimitiveType(gpu_conv_config.input_type));
  TF_ASSIGN_OR_RETURN(
      se::dnn::DataType output_type,
      GetDNNDataTypeFromPrimitiveType(gpu_conv_config.output_type));
  se::dnn::DnnSupport* dnn = stream_executor->AsDnn();
  auto allocator =
      std::make_unique<stream_executor::StreamExecutorAddressAllocator>(
          stream_executor);
  TF_ASSIGN_OR_RETURN(se::Stream * stream,
                      allocator->GetStream(stream_executor->device_ordinal()));
  bool allow_tf32 = absl::c_all_of(
      instr->precision_config().operand_precision(),
      [](int precision) { return precision <= PrecisionConfig::HIGH; });
  const se::EngineOptions engine_options{
      RequireDeterminism(instr->GetModule()->config()), allow_tf32,
      /*require_command_buffer=*/false};

  // Try to get algorithms without fallback first, as fallback algorithms can be
  // very slow.
  std::vector<CudnnBackendConfig> algorithm_configs;
  TF_ASSIGN_OR_RETURN(
      algorithm_configs,
      GetAlgorithms(dnn, conv_kind, input_type, output_type, stream,
                    gpu_conv_config, engine_options, /*use_fallback=*/false));

  if (algorithm_configs.empty()) {
    TF_ASSIGN_OR_RETURN(
        algorithm_configs,
        GetAlgorithms(dnn, conv_kind, input_type, output_type, stream,
                      gpu_conv_config, engine_options, /*use_fallback=*/true));
  }

  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.reserve(algorithm_configs.size());
  for (const auto& algorithm_config : algorithm_configs) {
    auto any = std::make_unique<google::protobuf::Any>();
    any->PackFrom(algorithm_config);
    configs.push_back(std::move(any));
  }
  return configs;
}

absl::Status ApplyConfigToCudnnFusion(HloInstruction& instr,
                                      const CudnnBackendConfig& config) {
  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                      instr.backend_config<GpuBackendConfig>());
  FusionBackendConfig* backend_config =
      gpu_config.mutable_fusion_backend_config();
  backend_config->mutable_cudnn_fusion_config()->set_plan_id(config.algo_id());
  TF_RETURN_IF_ERROR(instr.set_backend_config(std::move(gpu_config)));
  return absl::OkStatus();
}

absl::Status ApplyConfigToCudnnCustomCall(HloInstruction& instr,
                                          const CudnnBackendConfig& config) {
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

}  // namespace

bool CudnnBackend::IsSupported(const HloInstruction& instr) {
  if (instr.opcode() == HloOpcode::kFusion) {
    return IsSupportedCudnnFusion(instr, stream_executor(), debug_options());
  }

  if (instr.opcode() == HloOpcode::kCustomCall) {
    return IsCustomCallToDnnConvolution(instr);
  }

  return false;
}

absl::StatusOr<std::unique_ptr<BackendConfig>> CudnnBackend::GetDefaultConfig(
    const HloInstruction& instr) {
  if (IsCustomCallToDnnConvolution(instr)) {
    // If the instruction is a custom call to a DnnConvolution, we can return
    // the default config.
    CudnnBackendConfig config;
    config.set_algo_id(-1);
    auto any = std::make_unique<google::protobuf::Any>();
    any->PackFrom(config);
    return any;
  }

  // Default config would require stream_executor to check if the fusion is
  // supported by Cudnn.
  return absl::InvalidArgumentError(
      "Cudnn backend doesn't support getting a default config.");
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
CudnnBackend::GetSupportedConfigs(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return std::vector<std::unique_ptr<BackendConfig>>();
  }
  if (instr.opcode() == HloOpcode::kFusion) {
    return GetCudnnFusionConfigs(instr, stream_executor());
  }
  if (IsCustomCallToDnnConvolution(instr)) {
    auto custom_call_instr = Cast<HloCustomCallInstruction>(&instr);
    return GetConvolutionCustomCallConfigs(custom_call_instr,
                                           stream_executor());
  }

  return std::vector<std::unique_ptr<BackendConfig>>();
}

absl::Status CudnnBackend::ApplyConfig(HloInstruction& instr,
                                       const BackendConfig& config) {
  CudnnBackendConfig algorithm_config;
  if (!config.UnpackTo(&algorithm_config)) {
    return absl::InvalidArgumentError(
        "Failed to unpack CudnnBackendConfig from Any.");
  }
  if (instr.opcode() == HloOpcode::kFusion) {
    return ApplyConfigToCudnnFusion(instr, algorithm_config);
  }

  if (instr.opcode() == HloOpcode::kCustomCall) {
    return ApplyConfigToCudnnCustomCall(instr, algorithm_config);
  }

  return absl::UnimplementedError(
      "Cudnn backend doesn't support this instruction.");
}

}  // namespace gpu
}  // namespace xla

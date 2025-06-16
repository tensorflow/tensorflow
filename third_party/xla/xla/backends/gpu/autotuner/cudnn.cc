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

#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/algorithm_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/gpu_conv_runner.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/gpu/transforms/cudnn_fusion_compiler.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/numeric_options.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/util.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

namespace se = ::stream_executor;

using CudnnBackendConfig = stream_executor::dnn::AlgorithmProto;

bool IsSupportedCudnnFusion(const HloInstruction& instr,
                            se::StreamExecutor* stream_executor,
                            const DebugOptions& debug_options) {
  if (!instr.has_backend_config() ||
      !instr.backend_config<GpuBackendConfig>()->has_fusion_backend_config() ||
      instr.backend_config<GpuBackendConfig>()
              ->fusion_backend_config()
              .kind() != kCuDnnFusionKind) {
    LOG(ERROR) << "Instr is not a cudnn fusion.";
    return false;
  }

  HloDotInstruction* dot =
      Cast<HloDotInstruction>(hlo_query::GetFirstInstructionWithOpcode(
          *instr.fused_instructions_computation(), HloOpcode::kDot));
  if (dot == nullptr) {
    LOG(ERROR) << "Fusion does not contain a dot.";
    return false;
  }
  if (dot->sparse_operands()) {
    LOG(ERROR) << "Fusion contains a sparse dot.";
    return false;
  }
  if (!algorithm_util::IsSupportedByCudnn(
          dot->precision_config().algorithm())) {
    LOG(ERROR) << "Fusion contains a precision config not supported by cudnn.";
    return false;
  }

  if (GetDnnVersionInfoOrDefault(stream_executor).major_version() < 9) {
    LOG(ERROR) << "Cudnn version is too old.";
    return false;
  }

  stream_executor::CudaComputeCapability compute_capability =
      stream_executor->GetDeviceDescription().cuda_compute_capability();
  if ((compute_capability.IsAtLeastAmpere() &&
       debug_options.xla_gpu_cudnn_gemm_fusion_level() > 1) ||
      (compute_capability.IsAtLeastBlackwell() &&
       debug_options.xla_gpu_cudnn_gemm_fusion_level() > 0)) {
    return true;
  }

  LOG(ERROR) << "Fusion is not supported by cudnn.";
  return false;
}

bool IsSupportedByCudnn(const HloInstruction& instr,
                        se::StreamExecutor* stream_executor,
                        const DebugOptions& debug_options) {
  if (instr.opcode() == HloOpcode::kFusion) {
    return IsSupportedCudnnFusion(instr, stream_executor, debug_options);
  }

  if (instr.opcode() == HloOpcode::kCustomCall) {
    return IsCustomCallToDnnConvolution(instr);
  }

  return false;
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
GetCudnnFusionConfigs(const HloInstruction& instr,
                      se::StreamExecutor* stream_executor) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  int plan_count = CuDnnFusionCompiler::GetAvailablePlanCount(
      *stream_executor, *DynCast<HloFusionInstruction>(&instr));
  configs.reserve(plan_count);
  for (int plan_id = 0; plan_id < plan_count; ++plan_id) {
    auto config = std::make_unique<CudnnBackendConfig>();
    config->set_algo_id(plan_id);
    configs.push_back(std::move(config));
  }
  return configs;
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
GetConvolutionCustomCallConfigs(const HloCustomCallInstruction* instr,
                                se::StreamExecutor* stream_executor) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  TF_ASSIGN_OR_RETURN(GpuConvConfig gpu_conv_config, GetGpuConvConfig(instr));
  TF_ASSIGN_OR_RETURN(se::dnn::ConvolutionKind conv_kind,
                      GetDNNConvKindFromCudnnConvKind(gpu_conv_config.kind));
  TF_ASSIGN_OR_RETURN(
      se::dnn::DataType input_type,
      GetDNNDataTypeFromPrimitiveType(gpu_conv_config.input_type));
  TF_ASSIGN_OR_RETURN(
      se::dnn::DataType output_type,
      GetDNNDataTypeFromPrimitiveType(gpu_conv_config.output_type));
  se::dnn::DnnSupport* dnn = stream_executor->AsDnn();
  auto allocator =
      std::make_unique<se::StreamExecutorMemoryAllocator>(stream_executor);
  TF_ASSIGN_OR_RETURN(se::Stream * stream,
                      allocator->GetStream(stream_executor->device_ordinal()));
  bool allow_tf32 = absl::c_all_of(
      instr->precision_config().operand_precision(),
      [](int precision) { return precision <= PrecisionConfig::HIGH; });
  const se::NumericOptions numeric_options{
      RequireDeterminism(instr->GetModule()->config()), allow_tf32};
  switch (conv_kind) {
    case se::dnn::ConvolutionKind::FORWARD_BIAS_ACTIVATION: {
      if (!gpu_conv_config.fusion) {
        return absl::InvalidArgumentError(
            "GpuConvConfig had fusion ConvolutionKind but no FusionConfig.");
      }
      std::vector<std::unique_ptr<const se::dnn::FusedConvRunner>> runners;
      TF_RETURN_IF_ERROR(dnn->GetFusedConvolveRunners(
          // This refers to the kind of convolution op inside the fusion, not
          // the whole fused graph.
          se::dnn::ConvolutionKind::FORWARD, input_type,
          BiasTypeForInputType(input_type), output_type,
          /*conv_input_scale=*/gpu_conv_config.conv_result_scale,
          /*side_input_scale=*/gpu_conv_config.fusion->side_input_scale,
          /*leakyrelu_alpha=*/gpu_conv_config.fusion->leakyrelu_alpha, stream,
          gpu_conv_config.input_descriptor, gpu_conv_config.filter_descriptor,
          gpu_conv_config.bias_descriptor, gpu_conv_config.output_descriptor,
          gpu_conv_config.conv_desc,
          /*use_fallback=*/false, gpu_conv_config.fusion->mode, numeric_options,
          &runners));
      for (const auto& runner : runners) {
        configs.push_back(std::make_unique<CudnnBackendConfig>(
            runner->ToAlgorithmDesc()->ToProto()));
      }
      return configs;
    }
    case se::dnn::ConvolutionKind::FORWARD_GRAPH: {
      std::vector<std::unique_ptr<const se::dnn::GraphConvRunner>> runners;
      // This path is cuDNN-only, where the DeviceMemoryBase arguments and the
      // allocator are unused; so, they're all provided as nullptr.
      TF_RETURN_IF_ERROR(dnn->GetGraphConvolveRunners(
          conv_kind, input_type, output_type, stream,
          gpu_conv_config.input_descriptor, gpu_conv_config.filter_descriptor,
          gpu_conv_config.output_descriptor, gpu_conv_config.conv_desc,
          /*use_fallback=*/false, numeric_options, &runners,
          gpu_conv_config.serialized_graph));
      for (const auto& runner : runners) {
        configs.push_back(std::make_unique<CudnnBackendConfig>(
            runner->ToAlgorithmDesc()->ToProto()));
      }
      return configs;
    }
    case se::dnn::ConvolutionKind::FORWARD:
    case se::dnn::ConvolutionKind::BACKWARD_DATA:
    case se::dnn::ConvolutionKind::BACKWARD_FILTER: {
      std::vector<std::unique_ptr<const se::dnn::ConvRunner>> runners;
      // This path is cuDNN-only, where the DeviceMemoryBase arguments and the
      // allocator are unused; so, they're all provided as nullptr.
      TF_RETURN_IF_ERROR(dnn->GetConvolveRunners(
          conv_kind, input_type, output_type, stream,
          gpu_conv_config.input_descriptor,
          /*input_data=*/se::DeviceMemoryBase(nullptr),
          gpu_conv_config.filter_descriptor,
          /*filter_data=*/se::DeviceMemoryBase(nullptr),
          gpu_conv_config.output_descriptor,
          /*output_data=*/se::DeviceMemoryBase(nullptr),
          gpu_conv_config.conv_desc,
          /*use_fallback=*/false, nullptr, numeric_options, &runners));
      for (const auto& runner : runners) {
        configs.push_back(std::make_unique<CudnnBackendConfig>(
            runner->ToAlgorithmDesc()->ToProto()));
      }
      return configs;
    }
    default:
      return absl::InvalidArgumentError(
          "Cudnn backend doesn't support this convolution kind.");
  }
  return configs;
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
CudnnBackend::GetSupportedConfigs(
    const HloInstruction& instr,
    stream_executor::StreamExecutor* stream_executor) {
  if (!IsSupportedByCudnn(instr, stream_executor, debug_options())) {
    return std::vector<std::unique_ptr<BackendConfig>>();
  }
  if (instr.opcode() == HloOpcode::kFusion) {
    return GetCudnnFusionConfigs(instr, stream_executor);
  }
  if (instr.opcode() == HloOpcode::kCustomCall) {
    auto custom_call_instr = Cast<HloCustomCallInstruction>(&instr);
    return GetConvolutionCustomCallConfigs(custom_call_instr, stream_executor);
  }

  return std::vector<std::unique_ptr<BackendConfig>>();
}

absl::StatusOr<std::unique_ptr<BackendConfig>> CudnnBackend::GetDefaultConfig(
    const HloInstruction& instr) {
  // Default config would require stream_executor to check if the fusion is
  // supported by Cudnn.
  return absl::InvalidArgumentError(
      "Cudnn backend doesn't support getting a default config.");
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
  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                      instr.backend_config<GpuBackendConfig>());
  CudnnConvBackendConfig* cudnn_conv_config =
      gpu_config.mutable_cudnn_conv_backend_config();
  *cudnn_conv_config->mutable_algorithm() = config;
  TF_RETURN_IF_ERROR(instr.set_backend_config(std::move(gpu_config)));
  return absl::OkStatus();
}

absl::Status CudnnBackend::ApplyConfig(HloInstruction& instr,
                                       const BackendConfig& config) {
  const CudnnBackendConfig& algorithm_config =
      static_cast<const CudnnBackendConfig&>(config);
  if (instr.opcode() == HloOpcode::kFusion) {
    return ApplyConfigToCudnnFusion(instr, algorithm_config);
  }

  if (instr.opcode() == HloOpcode::kCustomCall) {
    return ApplyConfigToCudnnCustomCall(instr, algorithm_config);
  }

  return absl::UnimplementedError(
      "Cudnn backend doesn't support this instruction.");
};

}  // namespace gpu
}  // namespace xla

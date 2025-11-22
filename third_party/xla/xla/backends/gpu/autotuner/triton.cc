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

#include "xla/backends/gpu/autotuner/triton.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/float_normalization.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/autotuning/dot_search_space.h"
#include "xla/service/gpu/autotuning/triton_configs.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_float_support.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/split_k_gemm_rewriter.h"
#include "xla/service/gpu/transforms/fusion_wrapper.h"
#include "xla/service/gpu/transforms/nest_gemm_fusion.h"
#include "xla/service/gpu/transforms/priority_fusion.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {
std::vector<TritonGemmConfig> GetDefaultTritonConfigs(
    se::GpuComputeCapability compute_capability, bool autotune_tma) {
  if (compute_capability.IsRocm()) {
    return *kDefaultRocmConfigs;
  }

  CHECK(compute_capability.IsCuda());
  auto* cuda_compute_capability = compute_capability.cuda_compute_capability();
  std::vector<TritonGemmConfig> configs;

  if (cuda_compute_capability->IsAtLeastBlackwell()) {
    configs = *kBlackwellConfigs;
  } else if (cuda_compute_capability->IsHopper() ||
             cuda_compute_capability->IsAmpere()) {
    configs = *kHopperAmpereConfigs;
  } else {
    configs = *kDefaultCudaConfigs;
  }

  if (!autotune_tma) {
    return configs;
  }

  // Hopper+ devices support TMA. Add TMA parameterized configs.
  std::vector<TritonGemmConfig> tma_parameterized_configs;
  for (auto& config : configs) {
    config.is_tma_allowed = false;
    tma_parameterized_configs.push_back(config);

    config.is_tma_allowed = true;
    tma_parameterized_configs.push_back(config);
  }
  return tma_parameterized_configs;
}

}  // namespace

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
TritonBackend::GetSupportedConfigs(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return std::vector<std::unique_ptr<BackendConfig>>();
  }
  const HloDotInstruction* dot =
      Cast<HloDotInstruction>(hlo_query::GetFirstInstructionWithOpcode(
          *instr.fused_instructions_computation(), HloOpcode::kDot));
  TritonDotFusionSearchSpace search_space(target_config().device_description,
                                          dot);

  bool supports_contracting_split =
      HloBfsFindAll({dot}, [&](const HloInstruction* node) {
        return node->opcode() == HloOpcode::kSlice;
      }).empty();
  bool autotune_contracting_split =
      supports_contracting_split &&
      debug_options().xla_gpu_enable_split_k_autotuning();

  // Allow TMA tuning for Hopper+ devices when TMA flag is passed.
  bool autotune_tma =
      debug_options().xla_gpu_experimental_enable_triton_tma() &&
      stream_executor::gpu::IsTmaAvailableForDevice(
          target_config().device_description);
  std::vector<std::unique_ptr<BackendConfig>> configs;
  VLOG(1) << "Generating configs from search space: "
          << search_space.ToString();
  // We don't need to consider small_dot here. The new search space will
  // already generate a unique config for small problems.
  std::vector<TritonGemmConfig> gemm_configs = search_space.GenerateConfigs(
      /*force_contracting_split=*/autotune_contracting_split
          ? std::nullopt
          : std::make_optional(1),
      /*autotune_tma=*/autotune_tma);

  if (!debug_options().xla_gpu_exhaustive_tiling_search()) {
    VLOG(1) << "Restricting configs to the default set.";
    gemm_configs = search_space.OptimizeConfigSet(
        gemm_configs, /*hints=*/GetDefaultTritonConfigs(
            target_config().device_description.gpu_compute_capability(),
            autotune_tma));
  }
  configs.reserve(gemm_configs.size());
  for (const auto& config : gemm_configs) {
    auto any = std::make_unique<google::protobuf::Any>();
    any->PackFrom(config.ToProto());
    configs.push_back(std::move(any));
  }
  return configs;
}

absl::StatusOr<std::unique_ptr<BackendConfig>> TritonBackend::GetDefaultConfig(
    const HloInstruction& instr) {
  TF_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<BackendConfig>> configs,
                      GetSupportedConfigs(instr));
  if (configs.empty()) {
    return absl::InvalidArgumentError(
        "TritonBackend does not support this instruction.");
  }
  return std::move(configs[0]);
}

absl::Status TritonBackend::ApplyConfig(HloInstruction& instr,
                                        const BackendConfig& config) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError(
        "TritonBackend does not support this instruction.");
  }
  AutotuneResult::TritonGemmKey triton_config_proto;
  if (!config.UnpackTo(&triton_config_proto)) {
    return absl::InvalidArgumentError(
        "Failed to unpack TritonBackendConfig from Any.");
  }

  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                      instr.backend_config<GpuBackendConfig>());
  FusionBackendConfig& backend_config =
      *gpu_config.mutable_fusion_backend_config();

  backend_config.set_kind(kTritonGemmFusionKind);
  *backend_config.mutable_triton_gemm_config() = triton_config_proto;
  TF_RETURN_IF_ERROR(instr.set_backend_config(gpu_config));

  TF_ASSIGN_OR_RETURN(TritonGemmConfig triton_config,
                      TritonGemmConfig::FromProto(triton_config_proto));
  if (triton_config.split_k > 1) {
    TF_RETURN_IF_ERROR(MakeDotSplitKBatch(&instr, triton_config));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<HloModule>> TritonBackend::RunHloPasses(
    std::unique_ptr<HloModule> hlo_module,
    const Compiler::CompileOptions& options) {
  auto gpu_device_info = target_config().device_description;
  for (PrimitiveType type :
       {BF16, F8E5M2, F8E4M3FN, F8E4M3B11FNUZ, F8E5M2FNUZ, F8E4M3FNUZ}) {
    GpuFloatSupport float_support(gpu_device_info.cuda_compute_capability(),
                                  type);
    FloatNormalization float_normalization(&float_support);
    TF_RETURN_IF_ERROR(float_normalization.Run(hlo_module.get()).status());
  }

  HloCostAnalysis::Options priority_fusion_options;
  priority_fusion_options.count_multiple_input_accesses = true;
  PriorityFusion priority_fusion(
      /*thread_pool=*/nullptr, gpu_device_info, priority_fusion_options,
      mlir_context_);
  TF_RETURN_IF_ERROR(priority_fusion.Run(hlo_module.get()).status());

  // If the priority fusion pass above skipped some instructions, turn them
  // into fusions.
  FusionWrapper fusion_wrapper(gpu_device_info);
  TF_RETURN_IF_ERROR(fusion_wrapper.Run(hlo_module.get()).status());

  NestGemmFusion nest_gemm_fusion(gpu_device_info, mlir_context_);
  TF_RETURN_IF_ERROR(nest_gemm_fusion.Run(hlo_module.get()).status());

  bool is_legacy_gemm_disabled = absl::c_contains(
      debug_options().xla_gpu_unsupported_generic_triton_emitter_features(),
      DebugOptions::GENERIC_TRITON_EMITTER_DISABLE_LEGACY_GEMM);
  bool is_triton_gemm_fusion =
      IsGpuFusionKind(*hlo_module->entry_computation()->root_instruction(),
                      kTritonGemmFusionKind);
  if (is_legacy_gemm_disabled && is_triton_gemm_fusion) {
    return absl::InternalError(
        absl::StrCat("Unexpected ", kTritonGemmFusionKind,
                     " fusion: ", hlo_module->ToString()));
  }
  return hlo_module;
}

bool TritonBackend::IsSupported(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return false;
  }
  auto gpu_config = instr.backend_config<GpuBackendConfig>();
  if (!gpu_config.ok()) {
    return false;
  }
  const FusionBackendConfig& backend_config =
      gpu_config->fusion_backend_config();
  return backend_config.kind() == kTritonGemmFusionKind ||
         backend_config.kind() == kCuDnnFusionKind ||
         backend_config.kind() == kCustomFusionKind;
}

}  // namespace gpu
}  // namespace xla

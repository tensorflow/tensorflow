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

#include <algorithm>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
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
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/split_k_gemm_rewriter.h"
#include "xla/service/gpu/transforms/convert_triton_gemm_config.h"
#include "xla/service/gpu/transforms/fusion_wrapper.h"
#include "xla/service/gpu/transforms/hoist_fused_bitcasts.h"
#include "xla/service/gpu/transforms/nest_gemm_fusion.h"
#include "xla/service/gpu/transforms/priority_fusion.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {
namespace gpu {

namespace {
std::vector<TritonGemmConfig> GetDefaultTritonConfigs(
    se::GpuComputeCapability compute_capability) {
  if (compute_capability.IsRocm()) {
    return GetTritonConfigsForPlatform(TritonConfigsPlatform::kDefaultRocm);
  }

  CHECK(compute_capability.IsCuda());
  auto* cuda_compute_capability = compute_capability.cuda_compute_capability();
  std::vector<TritonGemmConfig> configs;

  if (cuda_compute_capability->IsAtLeastBlackwell()) {
    configs = GetTritonConfigsForPlatform(TritonConfigsPlatform::kBlackwell);
  } else if (cuda_compute_capability->IsHopper()) {
    configs = GetTritonConfigsForPlatform(TritonConfigsPlatform::kHopper);
  } else if (cuda_compute_capability->IsAmpere()) {
    configs = GetTritonConfigsForPlatform(TritonConfigsPlatform::kAmpere);
  } else {
    configs = GetTritonConfigsForPlatform(TritonConfigsPlatform::kDefaultCuda);
  }

  return configs;
}

}  // namespace

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
TritonBackend::GetSupportedConfigs(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return std::vector<std::unique_ptr<BackendConfig>>();
  }
  const HloInstruction* dot_instr = hlo_query::GetFirstInstructionWithOpcode(
      *instr.fused_instructions_computation(), HloOpcode::kDot);
  if (dot_instr != nullptr) {
    return GetSupportedConfigsForDot(dot_instr);
  }
  const HloInstruction* scaled_dot_instr =
      hlo_query::GetFirstInstructionWithOpcode(
          *instr.fused_instructions_computation(), HloOpcode::kScaledDot);
  if (scaled_dot_instr != nullptr) {
    return GetSupportedConfigsForScaledDot(scaled_dot_instr);
  }
  return std::vector<std::unique_ptr<BackendConfig>>();
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
TritonBackend::GetSupportedConfigsForDot(const HloInstruction* instr) {
  const HloDotInstruction* dot = Cast<HloDotInstruction>(instr);
  TritonDotFusionSearchSpace search_space(target_config().device_description,
                                          dot);
  bool supports_contracting_split =
      HloBfsFindAll({dot}, [&](const HloInstruction* node) {
        return node->opcode() == HloOpcode::kSlice;
      }).empty();
  bool autotune_contracting_split =
      supports_contracting_split &&
      debug_options().xla_gpu_enable_split_k_autotuning();

  std::vector<std::unique_ptr<BackendConfig>> configs;
  VLOG(1) << "Generating configs from search space: "
          << search_space.ToString();
  // We don't need to consider small_dot here. The new search space will
  // already generate a unique config for small problems.
  std::vector<TritonGemmConfig> gemm_configs = search_space.GenerateConfigs(
      /*force_contracting_split=*/autotune_contracting_split
          ? std::nullopt
          : std::make_optional(1));

  if (!debug_options().xla_gpu_exhaustive_tiling_search()) {
    VLOG(1) << "Restricting configs to the default set.";
    gemm_configs = search_space.OptimizeConfigSet(
        gemm_configs, /*hints=*/GetDefaultTritonConfigs(
            target_config().device_description.gpu_compute_capability()));
  }
  configs.reserve(gemm_configs.size());
  for (const auto& config : gemm_configs) {
    auto any = std::make_unique<google::protobuf::Any>();
    any->PackFrom(config.ToProto());
    configs.push_back(std::move(any));
  }
  return configs;
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
TritonBackend::GetSupportedConfigsForScaledDot(const HloInstruction* instr) {
  std::vector<std::unique_ptr<BackendConfig>> configs;

  // TODO(b/436988479): fine tune the search space.
  for (int block_m = 128; block_m <= 256; block_m *= 2) {
    for (int block_n = 32; block_n <= 256; block_n *= 2) {
      for (int block_k = 128; block_k <= 256; block_k *= 2) {
        auto any = std::make_unique<google::protobuf::Any>();
        any->PackFrom(TritonGemmConfig(block_m, block_n,
                                       /*block_k=*/block_k, /*split_k=*/1,
                                       /*num_stages=*/1,
                                       /*num_warps=*/4,
                                       /*num_ctas=*/1,
                                       /*is_tma_allowed=*/false)
                          .ToProto());
        configs.push_back(std::move(any));
      }
    }
  }
  return configs;
}

absl::StatusOr<std::unique_ptr<BackendConfig>> TritonBackend::GetDefaultConfig(
    const HloInstruction& instr) {
  TF_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<BackendConfig>> configs,
                      GetSupportedConfigs(instr));
  // Filter split_k>1 configs. Split_k>1 is not guaranteed to be supported.
  configs.erase(
      std::remove_if(configs.begin(), configs.end(),
                     [](const std::unique_ptr<BackendConfig>& config) {
                       AutotuneResult::TritonGemmKey triton_config_proto;
                       config->UnpackTo(&triton_config_proto);
                       return triton_config_proto.split_k() > 1;
                     }),
      configs.end());
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
  TF_RETURN_IF_ERROR(HoistFusedBitcasts().Run(hlo_module.get()).status());
  if (debug_options().xla_gpu_unsupported_disable_nested_gemm_fusions()) {
    ConvertTritonGemmConfig convert_triton_gemm_config(gpu_device_info,
                                                       mlir_context_);
    RETURN_IF_ERROR(convert_triton_gemm_config.Run(hlo_module.get()).status());
  } else {
    NestGemmFusion nest_gemm_fusion(gpu_device_info, mlir_context_);
    RETURN_IF_ERROR(nest_gemm_fusion.Run(hlo_module.get()).status());
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

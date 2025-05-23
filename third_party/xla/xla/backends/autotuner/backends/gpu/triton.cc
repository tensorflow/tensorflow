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

#include "xla/backends/autotuner/backends/gpu/triton.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/float_normalization.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_float_support.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/split_k_gemm_rewriter.h"
#include "xla/service/gpu/transforms/fusion_wrapper.h"
#include "xla/service/gpu/transforms/nest_gemm_fusion.h"
#include "xla/service/gpu/transforms/priority_fusion.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

// Search space for exhaustive matmul autotuning.
constexpr std::array<int, 6> kBlockSizes = {16, 32, 64, 128, 256, 512};
constexpr std::array<int, 4> kNumStages = {1, 2, 3, 4};
constexpr std::array<int, 4> kNumWarps = {2, 4, 8, 16};
constexpr std::array<int, 5> kSplitK = {1, 2, 4, 8, 16};
constexpr std::array<int, 5> kNumCtas = {1, 2, 4, 8, 16};

}  // namespace

using TritonBackendConfig = AutotuneResult::TritonGemmKey;

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
TritonBackend::GetSupportedConfigs(
    const HloInstruction& instr,
    stream_executor::StreamExecutor* stream_executor) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError(
        "TritonBackend does not support this instruction.");
  }
  se::GpuComputeCapability gcc =
      target_config().device_description.gpu_compute_capability();
  bool is_rocm = std::holds_alternative<se::RocmComputeCapability>(gcc);
  auto cuda_compute_capability = std::get_if<se::CudaComputeCapability>(&gcc);

  bool tune_ctas = !is_rocm && cuda_compute_capability &&
                   cuda_compute_capability->IsAtLeastHopper();

  const int64_t threads_per_warp =
      target_config().device_description.threads_per_warp();
  std::vector<std::unique_ptr<BackendConfig>> configs;
  for (int num_stages : kNumStages) {
    for (int tile_m : kBlockSizes) {
      for (int tile_n : kBlockSizes) {
        for (int tile_k : kBlockSizes) {
          const int tile_lhs = tile_m * tile_k;
          const int tile_rhs = tile_k * tile_n;
          for (int num_warps : kNumWarps) {
            // Each thread should read at least one input element.
            if (num_warps * threads_per_warp > std::min(tile_lhs, tile_rhs)) {
              break;
            }
            for (int split_k : kSplitK) {
              // Split-K autotuning may be disabled by a flag.
              if (debug_options().xla_gpu_enable_split_k_autotuning() &&
                  split_k > 1) {
                break;
              }
              for (int num_ctas : kNumCtas) {
                // Clusters are only supported on Hopper.
                // Autotuning this parameter is enabled by a flag.
                if (!tune_ctas && num_ctas > 1) {
                  break;
                }
                if (num_ctas > num_warps) {
                  break;
                }
                configs.push_back(std::make_unique<TritonBackendConfig>(
                    TritonGemmConfig(tile_m, tile_n, tile_k, split_k,
                                     num_stages, num_warps, num_ctas)
                        .ToProto()));
              }
            }
          }
        }
      }
    }
  }
  return configs;
}

absl::StatusOr<std::unique_ptr<BackendConfig>> TritonBackend::GetDefaultConfig(
    const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError(
        "TritonBackend does not support this instruction.");
  }
  return std::make_unique<TritonBackendConfig>(
      TritonGemmConfig(64, 64, 64, 1, 1, 2, 1).ToProto());
}

absl::Status TritonBackend::ApplyConfig(HloInstruction& instr,
                                        const BackendConfig& config) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError(
        "TritonBackend does not support this instruction.");
  }
  if (config.GetDescriptor() != TritonBackendConfig::GetDescriptor()) {
    return absl::InvalidArgumentError(
        "Invalid backend config type for TritonBackend.");
  }
  const TritonBackendConfig& triton_config_proto =
      static_cast<const TritonBackendConfig&>(config);

  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                      instr.backend_config<GpuBackendConfig>());
  FusionBackendConfig& backend_config =
      *gpu_config.mutable_fusion_backend_config();

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
      /*thread_pool=*/nullptr, gpu_device_info, priority_fusion_options);
  TF_RETURN_IF_ERROR(priority_fusion.Run(hlo_module.get()).status());

  // If the priority fusion pass above skipped some instructions, turn them
  // into fusions.
  FusionWrapper fusion_wrapper(gpu_device_info);
  TF_RETURN_IF_ERROR(fusion_wrapper.Run(hlo_module.get()).status());

  if (debug_options()
          .xla_gpu_unsupported_enable_generic_triton_emitter_for_gemms()) {
    NestGemmFusion nest_gemm_fusion(gpu_device_info.gpu_compute_capability());
    TF_RETURN_IF_ERROR(nest_gemm_fusion.Run(hlo_module.get()).status());
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

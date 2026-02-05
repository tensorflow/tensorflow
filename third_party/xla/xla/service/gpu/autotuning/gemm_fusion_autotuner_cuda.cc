/* Copyright 2024 The OpenXLA Authors.

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

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "third_party/gpus/cuda/include/cublas_v2.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/autotuner/cudnn.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/algorithm_util.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/gpu/autotuning/gemm_fusion_autotuner.h"
#include "xla/service/gpu/autotuning/triton_configs.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/gpu/transforms/block_scaling_rewriter.h"
#include "xla/service/gpu/transforms/cudnn_fusion_compiler.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

const int64_t GemmFusionAutotunerImpl::BLAS_GEMM_DEFAULT = CUBLAS_GEMM_DEFAULT;

int GetCuDnnPlanCount(const HloInstruction& hlo,
                      const AutotuneConfig& autotune_config) {
  if (auto gpu_config = hlo.backend_config<GpuBackendConfig>();
      !gpu_config.ok() ||
      gpu_config->fusion_backend_config().has_cudnn_fusion_config()) {
    return {};
  }
  return CuDnnFusionCompiler::GetAvailablePlanCount(
      *autotune_config.GetExecutor(), *DynCast<HloFusionInstruction>(&hlo));
}

bool GemmFusionAutotunerImpl::AddLibConfigs(
    const HloFusionInstruction& fusion, const HloInstruction* dot,
    std::vector<BackendConfig>& configs) {
  // Add cuDNN plans, if available.
  stream_executor::CudaComputeCapability cc =
      *GetComputeCapability().cuda_compute_capability();
  auto dnn_version = GetDnnVersionInfoOrDefault(
      !config_.IsDeviceless() ? config_.GetExecutor() : nullptr);

  bool is_cudnn_fusion = IsGpuFusionKind(fusion, kCuDnnFusionKind);
  bool is_supported_triton_dot_fusion =
      IsGpuFusionKind(fusion, kTritonGemmFusionKind) &&
      dnn_version.major_version() >= 9 &&
      algorithm_util::IsSupportedByCudnn(dot->precision_config().algorithm()) &&
      ((cc.IsAtLeastAmpere() &&
        debug_options_.xla_gpu_cudnn_gemm_fusion_level() > 1) ||
       (cc.IsAtLeastBlackwell() &&
        debug_options_.xla_gpu_cudnn_gemm_fusion_level() > 0));
  bool is_cudnn_supported_scaled_dot_fusion =
      IsGpuFusionKind(fusion, kTritonNestedGemmFusionKind) &&
      dot->opcode() == HloOpcode::kScaledDot &&
      dnn_version >= kCudnnSupportsBlockScaledDot &&
      CudnnScaledDotHelper::IsSupported(Cast<HloScaledDotInstruction>(dot)) &&
      cc.IsAtLeastBlackwell();

  if (IsAutotuningEnabled() &&
      (is_cudnn_fusion || is_supported_triton_dot_fusion ||
       is_cudnn_supported_scaled_dot_fusion)) {
    const int plan_count = GetCuDnnPlanCount(fusion, config_);
    for (int plan_id = 0; plan_id < plan_count; ++plan_id) {
      configs.push_back(CuDnnConfig{plan_id});
    }
  }
  if (IsGpuFusionKind(fusion, kCuDnnFusionKind)) {
    if (!IsAutotuningEnabled()) {
      configs.push_back(CuDnnConfig{-1});
    }
    return true;
  }
  return false;
}

absl::StatusOr<std::vector<std::unique_ptr<CodegenBackend>>>
GemmFusionAutotuner::GetPlatformCodegenBackends(
    se::StreamExecutor* stream_exec, Compiler* compiler,
    const Compiler::GpuTargetConfig* target_config,
    const DebugOptions* debug_options) {
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::make_unique<CudnnBackend>(stream_exec, debug_options,
                                                    compiler, target_config));
  return backends;
}

std::vector<TritonGemmConfig> GemmFusionAutotunerImpl::GetDefaultTritonConfigs()
    const {
  stream_executor::CudaComputeCapability compute_capability =
      *GetComputeCapability().cuda_compute_capability();
  std::vector<TritonGemmConfig> configs;

  if (compute_capability.IsAtLeastBlackwell()) {
    configs = GetTritonConfigsForPlatform(TritonConfigsPlatform::kBlackwell);
  } else if (compute_capability.IsHopper()) {
    configs = GetTritonConfigsForPlatform(TritonConfigsPlatform::kHopper);
  } else if (compute_capability.IsAmpere()) {
    configs = GetTritonConfigsForPlatform(TritonConfigsPlatform::kAmpere);
  } else {
    configs = GetTritonConfigsForPlatform(TritonConfigsPlatform::kDefaultCuda);
  }

  // TODO(b/449668102): Currently only supporting warp specialization on
  // Blackwell+. Potentially extend support to Hopper.
  if (!debug_options_
           .xla_gpu_experimental_enable_triton_warp_specialization() ||
      !compute_capability.IsAtLeastBlackwell()) {
    return configs;
  }
  std::vector<TritonGemmConfig> warp_specialized_configs;
  for (auto& config : configs) {
    config.is_warp_specialization_allowed = false;
    warp_specialized_configs.push_back(config);

    if (config.is_tma_allowed && config.num_warps <= 16 &&
        config.num_warps % 4 == 0) {
      config.is_warp_specialization_allowed = true;
      warp_specialized_configs.push_back(config);
    }
  }

  return warp_specialized_configs;
}

}  // namespace gpu
}  // namespace xla

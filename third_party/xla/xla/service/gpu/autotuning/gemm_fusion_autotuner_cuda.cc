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
#include <vector>

#include "third_party/gpus/cuda/include/cublas_v2.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/algorithm_util.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/gpu/autotuning/gemm_fusion_autotuner.h"
#include "xla/service/gpu/autotuning/triton_configs.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/gpu/transforms/cudnn_fusion_compiler.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"

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
    const HloFusionInstruction& fusion, const HloDotInstruction* dot,
    std::vector<BackendConfig>& configs) {
  // Add cuDNN plans, if available.
  auto cc = std::get<se::CudaComputeCapability>(GetComputeCapability());
  bool is_cudnn_enabled =
      !config_.IsDeviceless() &&
      GetDnnVersionInfoOrDefault(config_.GetExecutor()).major_version() >= 9 &&
      ((cc.IsAtLeastAmpere() &&
        debug_options_.xla_gpu_cudnn_gemm_fusion_level() > 1) ||
       (cc.IsAtLeastBlackwell() &&
        debug_options_.xla_gpu_cudnn_gemm_fusion_level() > 0));
  if ((IsFusionKind(fusion, kCuDnnFusionKind) && IsAutotuningEnabled()) ||
      (IsFusionKind(fusion, kTritonGemmFusionKind) && is_cudnn_enabled &&
       algorithm_util::IsSupportedByCudnn(
           dot->precision_config().algorithm()) &&
       IsAutotuningEnabled())) {
    const int plan_count = GetCuDnnPlanCount(fusion, config_);
    for (int plan_id = 0; plan_id < plan_count; ++plan_id) {
      configs.push_back(CuDnnConfig{plan_id});
    }
  }
  if (IsFusionKind(fusion, kCuDnnFusionKind)) {
    if (!IsAutotuningEnabled()) {
      configs.push_back(CuDnnConfig{-1});
    }
    return true;
  }
  return false;
}

std::vector<TritonGemmConfig> GemmFusionAutotunerImpl::GetDefaultTritonConfigs()
    const {
  auto compute_capability =
      std::get<se::CudaComputeCapability>(GetComputeCapability());
  std::vector<TritonGemmConfig> configs;

  if (compute_capability.IsAtLeastBlackwell()) {
    configs = *kBlackwellConfigs;
  } else if (compute_capability.IsHopper() || compute_capability.IsAmpere()) {
    configs = *kHopperAmpereConfigs;
  } else {
    configs = *kDefaultCudaConfigs;
  }

  if (!debug_options_.xla_gpu_experimental_enable_triton_tma() ||
      !compute_capability.IsAtLeastHopper()) {
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

}  // namespace gpu
}  // namespace xla

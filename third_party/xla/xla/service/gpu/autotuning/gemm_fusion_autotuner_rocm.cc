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

#include "rocm/include/hipblas/hipblas.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/autotuning/gemm_fusion_autotuner.h"
#include "xla/service/gpu/matmul_utils.h"

namespace xla {
namespace gpu {

const int64_t GemmFusionAutotunerImpl::BLAS_GEMM_DEFAULT = HIPBLAS_GEMM_DEFAULT;

bool GemmFusionAutotunerImpl::AddLibConfigs(
    const HloFusionInstruction& fusion, const HloDotInstruction* dot,
    std::vector<BackendConfig>& configs) {
  return false;
}

std::vector<TritonGemmConfig> GemmFusionAutotunerImpl::GetDefaultTritonConfigs()
    const {
  using Config = TritonGemmConfig;
  std::vector<Config> configs = {
      Config(32, 32, 256, 1, 1, 4), Config(64, 32, 32, 16, 1, 4),
      Config(32, 64, 64, 4, 1, 4),  Config(128, 128, 64, 4, 1, 4),
      Config(16, 16, 256, 1, 1, 4), Config(16, 128, 32, 16, 1, 4),
  };
  return configs;
}

}  // namespace gpu
}  // namespace xla

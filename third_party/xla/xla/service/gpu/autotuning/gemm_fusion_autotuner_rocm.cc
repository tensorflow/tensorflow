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
#include "rocm/include/hipblas/hipblas.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/autotuning/gemm_fusion_autotuner.h"
#include "xla/service/gpu/autotuning/triton_configs.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

const int64_t GemmFusionAutotunerImpl::BLAS_GEMM_DEFAULT = HIPBLAS_GEMM_DEFAULT;

bool GemmFusionAutotunerImpl::AddLibConfigs(
    const HloFusionInstruction& fusion, const HloInstruction* dot,
    std::vector<BackendConfig>& configs) {
  return false;
}

absl::StatusOr<std::vector<std::unique_ptr<CodegenBackend>>>
GemmFusionAutotuner::GetPlatformCodegenBackends(
    se::StreamExecutor* stream_exec, Compiler* compiler,
    const Compiler::GpuTargetConfig* target_config,
    const DebugOptions* debug_options) {
  return std::vector<std::unique_ptr<CodegenBackend>>();
}

std::vector<TritonGemmConfig> GemmFusionAutotunerImpl::GetDefaultTritonConfigs()
    const {
  return GetTritonConfigsForPlatform(TritonConfigsPlatform::kDefaultRocm);
}

}  // namespace gpu
}  // namespace xla

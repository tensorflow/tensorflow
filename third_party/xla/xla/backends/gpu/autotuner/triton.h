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

#ifndef XLA_BACKENDS_GPU_AUTOTUNER_TRITON_H_
#define XLA_BACKENDS_GPU_AUTOTUNER_TRITON_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/autotuner/gpu_codegen_backend.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/mlir_context_pool.h"
#include "xla/xla.pb.h"

namespace xla {

namespace gpu {

class TritonBackend : public GpuCodegenBackend {
 public:
  explicit TritonBackend(const DebugOptions* debug_options, Compiler* compiler,
                         const Compiler::GpuTargetConfig* target_config,
                         const AliasInfo* alias_info,
                         MlirContextPool* absl_nonnull mlir_context_pool)
      : GpuCodegenBackend(autotuner::Backend::TRITON, debug_options, compiler,
                          target_config),
        alias_info_(alias_info),
        mlir_context_pool_(mlir_context_pool) {}

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
  GetSupportedConfigs(const HloInstruction& instr) override;

  absl::StatusOr<std::vector<EstimatedConfig>> GetSupportedConfigsWithEstimates(
      const HloInstruction& instr) override;

  absl::StatusOr<std::unique_ptr<BackendConfig>> GetDefaultConfig(
      const HloInstruction& instr) override;

  absl::Status ApplyConfig(HloInstruction& instr,
                           const BackendConfig& config) override;

  bool CanProduceWrongResults() const override { return true; }
  std::string version() const override;

 private:
  bool IsSupported(const HloInstruction& instr) override;

  absl::StatusOr<std::vector<TritonGemmConfig>> GetSupportedGemmConfigs(
      const HloInstruction& instr);

  absl::StatusOr<std::vector<TritonGemmConfig>> GetSupportedConfigsForDot(
      const HloInstruction* instr);

  absl::StatusOr<std::vector<TritonGemmConfig>> GetSupportedConfigsForScaledDot(
      const HloInstruction* instr);

  absl::StatusOr<std::vector<TritonGemmConfig>> GetOverriddenConfigs(
      const HloInstruction* instr);

  absl::StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> hlo_module,
      const Compiler::CompileOptions& options) override;

  const AliasInfo* alias_info_;
  MlirContextPool* absl_nonnull mlir_context_pool_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_AUTOTUNER_TRITON_H_

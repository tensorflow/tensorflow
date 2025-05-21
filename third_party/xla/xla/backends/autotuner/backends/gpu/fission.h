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

#ifndef XLA_BACKENDS_AUTOTUNER_BACKENDS_GPU_FISSION_H_
#define XLA_BACKENDS_AUTOTUNER_BACKENDS_GPU_FISSION_H_

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/autotuner/backends/gpu/cublas.h"
#include "xla/backends/autotuner/backends/gpu/cublaslt.h"
#include "xla/backends/autotuner/backends/gpu/custom_kernel.h"
#include "xla/backends/autotuner/backends/gpu/gpu_codegen_backend.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/compiler.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

// The FissionBackend tries to unfuse a fusion instruction.
// The resulting 'configurations" (HloModules) are equivalent to the original
// hlo graph but try to use a different backend for the dot operation: cublas,
// cublasLt, custom calls. If the CustomKernel registry matches a hlo
// subgraph, it will generate a config using the CustomKernel.
class FissionBackend : public GpuCodegenBackend {
 public:
  explicit FissionBackend(const Compiler::TargetConfig* target_config,
                          const DebugOptions* debug_options, Compiler* compiler)
      : GpuCodegenBackend("Fission", target_config, debug_options, compiler),
        cublas_backend_(target_config, debug_options, compiler),
        cublaslt_backend_(target_config, debug_options, compiler),
        custom_kernel_backend_(target_config, debug_options, compiler) {}

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
  GetSupportedConfigs(
      const HloInstruction& instr,
      stream_executor::StreamExecutor* stream_executor) override;

  absl::StatusOr<std::unique_ptr<BackendConfig>> GetDefaultConfig(
      const HloInstruction& instr) override;

  absl::Status ApplyConfig(HloInstruction& instr,
                           const BackendConfig& config) override {
    return absl::UnimplementedError("Not implemented.");
  }

 private:
  absl::StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> hlo_module,
      const Compiler::CompileOptions& options) override {
    return absl::UnimplementedError("Not implemented.");
  }

  CublasBackend cublas_backend_;
  CublasLtBackend cublaslt_backend_;
  CustomKernelBackend custom_kernel_backend_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_BACKENDS_GPU_FISSION_H_

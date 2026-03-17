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

#ifndef XLA_BACKENDS_GPU_AUTOTUNER_CUBLAS_H_
#define XLA_BACKENDS_GPU_AUTOTUNER_CUBLAS_H_

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/autotuner/gpu_codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/compiler.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

// A codegen backend for cuBLAS, with configurable fallback to cuBLAS LT for F8
// matmuls.
// This backend is used to autotune cuBLAS algorithms.
//
// Cublas calls are represented as custom-call instructions, with and
// configurable algorithm:
// ```
//   %custom-call.1 = .. custom-call(...), custom_call_target="__cublas$gemm",
//   backend_config={"
//     gemm_backend_config":{"selected_algorithm":"18"}
//   }
// ```

class CublasBackend : public GpuCodegenBackend {
 public:
  explicit CublasBackend(stream_executor::StreamExecutor* stream_executor,
                         const DebugOptions* debug_options, Compiler* compiler,
                         const Compiler::GpuTargetConfig* target_config,
                         bool fp8_lt_fallback = false)
      : GpuCodegenBackend(autotuner::Backend::CUBLAS, debug_options, compiler,
                          target_config, stream_executor),
        fp8_lt_fallback_(fp8_lt_fallback) {}

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
  GetSupportedConfigs(const HloInstruction& instr) override;

  absl::StatusOr<std::unique_ptr<BackendConfig>> GetDefaultConfig(
      const HloInstruction& instr) override;

  absl::Status ApplyConfig(HloInstruction& instr,
                           const BackendConfig& config) override;

 private:
  bool ShouldUseCublasLt(const HloInstruction& instr);

  bool IsSupported(const HloInstruction& instr) override;
  bool fp8_lt_fallback_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_AUTOTUNER_CUBLAS_H_

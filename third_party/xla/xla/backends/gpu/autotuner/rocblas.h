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

#ifndef XLA_BACKENDS_GPU_AUTOTUNER_ROCBLAS_H_
#define XLA_BACKENDS_GPU_AUTOTUNER_ROCBLAS_H_

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/autotuner/gpu_codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/compiler.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

// A codegen backend for rocBLAS GEMM autotuning on ROCm/AMD GPUs.
//
// This backend is used to autotune rocBLAS algorithms for matrix
// multiplications. rocBLAS calls are represented as custom-call instructions:
// ```
//   %custom-call.1 = .. custom-call(...), custom_call_target="__cublas$gemm",
//   backend_config={"
//     gemm_backend_config":{"selected_algorithm":"18"}
//   }
// ```

class RocblasBackend : public GpuCodegenBackend {
 public:
  explicit RocblasBackend(stream_executor::StreamExecutor* stream_executor,
                          const DebugOptions* debug_options, Compiler* compiler,
                          const Compiler::GpuTargetConfig* target_config,
                          bool fp8_lt_fallback = false)
      : GpuCodegenBackend("Rocblas", debug_options, compiler, target_config,
                          stream_executor),
        fp8_lt_fallback_(fp8_lt_fallback) {}

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
  GetSupportedConfigs(const HloInstruction& instr) override;

  absl::StatusOr<std::unique_ptr<BackendConfig>> GetDefaultConfig(
      const HloInstruction& instr) override;

  absl::Status ApplyConfig(HloInstruction& instr,
                           const BackendConfig& config) override;

 private:
  bool ShouldUseHipblasLt(const HloInstruction& instr);

  bool IsSupported(const HloInstruction& instr) override;
  bool fp8_lt_fallback_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_AUTOTUNER_ROCBLAS_H_

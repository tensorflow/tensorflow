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

#ifndef XLA_BACKENDS_AUTOTUNER_BACKENDS_GPU_CUDNN_H_
#define XLA_BACKENDS_AUTOTUNER_BACKENDS_GPU_CUDNN_H_

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/autotuner/backends/gpu/gpu_codegen_backend.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/compiler.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

// A codegen backend for cuDNN.
// Determines execution plan id. Requires a device with cuDNN >= 9.0.
// Note: We only support cudnn fusions containing a dot.
//
// A Cudnn fusion is a fusion with a custom call target of "__cudnn$fusion":
// ```
// fusion {
//   p0 = f32[3,28,32] parameter(0)
//   p1 = f32[3,28,32] parameter(1)
//   ROOT d = f32[3,32,32] dot(p0, p1),
//     lhs_batch_dims={0}, rhs_batch_dims={0},
//     lhs_contracting_dims={1}, rhs_contracting_dims={1}
// }

// ENTRY e {
//   p0 = f32[3,28,32] parameter(0)
//   p1 = f32[3,28,32] parameter(1)
//   ROOT _ = f32[3,32,32] fusion(p0, p1), kind=kCustom, calls=fusion,
//     backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
// })";
// ```

class CudnnBackend : public GpuCodegenBackend {
 public:
  explicit CudnnBackend(const Compiler::TargetConfig* target_config,
                        const DebugOptions* debug_options, Compiler* compiler)
      : GpuCodegenBackend("Cublas", target_config, debug_options, compiler) {}

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
  GetSupportedConfigs(
      const HloInstruction& instr,
      stream_executor::StreamExecutor* stream_executor) override;

  absl::StatusOr<std::unique_ptr<BackendConfig>> GetDefaultConfig(
      const HloInstruction& instr) override;

 private:
  absl::StatusOr<std::unique_ptr<HloModule>> WrapInModule(
      const HloInstruction& hlo_instruction,
      const BackendConfig& config) override;

  absl::StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> hlo_module,
      const Compiler::CompileOptions& options) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_BACKENDS_GPU_CUDNN_H_

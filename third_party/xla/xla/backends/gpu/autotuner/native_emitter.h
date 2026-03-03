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

#ifndef XLA_BACKENDS_GPU_AUTOTUNER_NATIVE_EMITTER_H_
#define XLA_BACKENDS_GPU_AUTOTUNER_NATIVE_EMITTER_H_

#include <memory>
#include <vector>

#include "absl/base/nullability.h"
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

// Codegen backend for XLA's native fusion emitters.
//
// This backend enables us to autotune XLA's native emitters against other
// backends.
class NativeEmitterBackend : public GpuCodegenBackend {
 public:
  explicit NativeEmitterBackend(const DebugOptions* absl_nonnull debug_options,
                                Compiler* absl_nonnull compiler,
                                const Compiler::GpuTargetConfig* target_config)
      : GpuCodegenBackend(autotuner::Backend::NATIVE_EMITTER, debug_options,
                          compiler, target_config) {}

  // Returns all supported configurations for the given instruction.
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
  GetSupportedConfigs(const HloInstruction& instr) override;

  // Returns a default configuration for the instruction.
  absl::StatusOr<std::unique_ptr<BackendConfig>> GetDefaultConfig(
      const HloInstruction& instr) override;

  // Applies a given fusion config to the instruction.
  absl::Status ApplyConfig(HloInstruction& instr,
                           const BackendConfig& config) override;

 private:
  bool IsSupported(const HloInstruction& instr) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_AUTOTUNER_NATIVE_EMITTER_H_

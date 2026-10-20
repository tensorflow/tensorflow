/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_AUTOTUNER_BLOCK_LEVEL_EMITTER_BACKEND_H_
#define XLA_BACKENDS_CPU_AUTOTUNER_BLOCK_LEVEL_EMITTER_BACKEND_H_

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/cpu/autotuner/cpu_codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/compiler.h"

namespace xla::cpu {

inline constexpr absl::string_view kCpuBlockLevelEmitterBackendName =
    "cpu_block_level_emitter";

// Codegen backend for the Xtile block-level fusion emitter on CPU.
//
// This backend enables autotuning of Xtile-based fusion computations at the
// block level. It generates tiling configurations, applies them to
// instructions, and prepares them for compilation using the Xtile emitter.
class BlockLevelEmitterBackend : public CpuCodegenBackend {
 public:
  static absl::StatusOr<std::unique_ptr<CodegenBackend>> Create(
      Compiler* compiler);

  explicit BlockLevelEmitterBackend(Compiler* compiler)
      : CpuCodegenBackend(compiler, kCpuBlockLevelEmitterBackendName) {}

  autotuner::Backend backend() const final {
    return autotuner::Backend::BLOCK_LEVEL_EMITTER_CPU;
  }

  // Returns all supported block-level tiling configurations for the given
  // instruction.
  absl::StatusOr<std::vector<std::unique_ptr<xla::BackendConfig>>>
  GetSupportedConfigs(const HloInstruction& instr) override;

  // Returns a default block-level configuration for the instruction.
  absl::StatusOr<std::unique_ptr<xla::BackendConfig>> GetDefaultConfig(
      const HloInstruction& instr) override;

  // Applies a given block-level fusion configuration to the instruction.
  absl::Status ApplyConfig(HloInstruction& instr,
                           const xla::BackendConfig& config) override;

  // Determines whether the given HLO instruction is supported by this backend.
  bool IsSupported(const HloInstruction& instr);
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_AUTOTUNER_BLOCK_LEVEL_EMITTER_BACKEND_H_

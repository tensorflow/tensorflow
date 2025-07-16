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

#ifndef XLA_BACKENDS_CPU_AUTOTUNER_LLVM_KERNEL_BACKEND_H_
#define XLA_BACKENDS_CPU_AUTOTUNER_LLVM_KERNEL_BACKEND_H_

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/cpu/autotuner/cpu_codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/backend_config.pb.h"

namespace xla::cpu {

inline constexpr absl::string_view kLlvmKernelBackendName =
    "llvm_kernel_backend";

class LlvmKernelBackend : public CpuCodegenBackend {
 public:
  static absl::StatusOr<std::unique_ptr<CodegenBackend>> Create(
      Compiler* compiler);

  using Config = LlvmKernelOptions;

  absl::StatusOr<std::vector<std::unique_ptr<xla::BackendConfig>>>
  GetSupportedConfigs(const HloInstruction& instr) final;

  absl::StatusOr<std::unique_ptr<xla::BackendConfig>> GetDefaultConfig(
      const HloInstruction& instr) final;

  absl::Status ApplyConfig(HloInstruction& instr,
                           const xla::BackendConfig& config) final;

 protected:
  bool IsSupported(const HloInstruction& instr);

  explicit LlvmKernelBackend(Compiler* compiler)
      : CpuCodegenBackend(compiler, kLlvmKernelBackendName) {}
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_AUTOTUNER_LLVM_KERNEL_BACKEND_H_

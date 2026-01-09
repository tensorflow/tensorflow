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

#ifndef XLA_BACKENDS_CPU_AUTOTUNER_CPU_CODEGEN_BACKEND_H_
#define XLA_BACKENDS_CPU_AUTOTUNER_CPU_CODEGEN_BACKEND_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/stream_executor/host/host_platform_id.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla.pb.h"

namespace xla {
namespace cpu {

// Abstract base class for CPU backends, implementing the CodegenBackend
// interface.
class CpuCodegenBackend : public CodegenBackend {
 public:
  // Helper method creating a compiler for the host platform.
  static absl::StatusOr<std::unique_ptr<Compiler>> CreateBackendCompiler() {
    return Compiler::GetForPlatform(stream_executor::host::kHostPlatformId);
  }

  CpuCodegenBackend(Compiler* compiler, absl::string_view name)
      : name_(name), compiler_(compiler) {}

  absl::string_view name() const override { return name_; }

  absl::StatusOr<std::unique_ptr<Executable>> Compile(
      const HloInstruction& hlo_instruction,
      const xla::BackendConfig& config) override {
    std::unique_ptr<HloModule> hlo_module =
        ExtractInstructionIntoNewModule(hlo_instruction);

    TF_RETURN_IF_ERROR(ApplyConfig(
        *hlo_module->entry_computation()->root_instruction(), config));

    return compiler_->RunBackend(std::move(hlo_module),
                                 /*executor=*/nullptr,
                                 /*device_allocator=*/nullptr);
  }

  bool CanProduceWrongResults() const override { return false; }

 private:
  std::string name_;
  Compiler* compiler_;
};

}  // namespace cpu
}  // namespace xla

#endif  // XLA_BACKENDS_CPU_AUTOTUNER_CPU_CODEGEN_BACKEND_H_

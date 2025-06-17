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

#include "xla/backends/cpu/autotuner/llvm_kernel_autotuner.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/autotuner/autotuner.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/backends/cpu/autotuner/cpu_codegen_backend.h"
#include "xla/backends/cpu/autotuner/cpu_profiler.h"
#include "xla/backends/cpu/autotuner/llvm_kernel_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

absl::StatusOr<bool> LlvmKernelAutotuner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  TF_ASSIGN_OR_RETURN(auto compiler,
                      CpuCodegenBackend::CreateBackendCompiler());
  TF_ASSIGN_OR_RETURN(auto backend, LlvmKernelBackend::Create(compiler.get()));

  std::unique_ptr<Profiler> profiler = CpuProfiler::Create(ProfileOptions());

  std::vector<std::unique_ptr<CodegenBackend>> codegen_backends;
  codegen_backends.push_back(std::move(backend));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<Autotuner> autotuner,
                      Autotuner::Create(std::move(codegen_backends),
                                        std::move(profiler), AutotuneConfig()));

  bool hlo_changed = false;
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      auto status = autotuner->Autotune(instruction);
      hlo_changed |= status.ok();
    }
  }

  return hlo_changed;
}

}  // namespace xla::cpu

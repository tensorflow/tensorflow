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

#include "xla/tools/xla_cpu_compile_lib.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/status/status_macros.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiled_module.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/service/executable.h"

namespace xla {

absl::StatusOr<std::string> AotCompileCpuExecutable(
    std::unique_ptr<HloModule> hlo_module,
    std::optional<cpu::TargetMachineOptions> target_config) {
  cpu::CpuCompiler cpu_compiler;
  Compiler::CompileOptions compile_options;
  if (target_config.has_value()) {
    compile_options.cpu_target_config =
        Compiler::CpuTargetConfig(*target_config);
  }
  ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<Executable>> executables,
      cpu_compiler.Compile(std::move(hlo_module), {nullptr}, compile_options));
  ASSIGN_OR_RETURN(std::unique_ptr<CompiledModule> aot_result,
                   cpu_compiler.Export(executables[0].get()));
  return aot_result->SerializeAsString();
}

}  // namespace xla

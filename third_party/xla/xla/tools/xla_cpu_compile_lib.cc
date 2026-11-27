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

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/TargetParser/Host.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiled_module.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/cpu_aot_compilation_result.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/service/executable.h"
#include "xla/service/xla_compile_result.pb.h"

namespace xla {

// Compiles `hlo_module` with msan instrumentation. Unlike Compile, which loads
// the generated code into this process and hence instruments it iff this
// binary is built with msan, CompileAheadOfTime only emits object files.
static absl::StatusOr<std::unique_ptr<CompiledModule>> CompileSanitized(
    cpu::CpuCompiler& cpu_compiler, std::unique_ptr<HloModule> hlo_module,
    std::optional<cpu::TargetMachineOptions> target_config,
    const CpuSanitizerOptions& sanitizer_options) {
  if (hlo_module->has_schedule()) {
    return absl::UnimplementedError(
        "Sanitizer instrumentation of scheduled HLO modules is not supported");
  }
  cpu::TargetMachineOptions target = target_config.value_or(
      cpu::TargetMachineOptions(hlo_module->config().debug_options()));
  // Like Compile, default to the host triple.
  std::string triple =
      target.triple().empty() ? llvm::sys::getProcessTriple() : target.triple();
  cpu::CpuAotCompilationOptions aot_options(
      triple, target.cpu(), target.GetTargetMachineFeatures(),
      hlo_module->name(),
      cpu::CpuAotCompilationOptions::RelocationModel::BigPic);
  aot_options.set_sanitize_memory(true);
  aot_options.set_sanitize_memory_track_origins(
      sanitizer_options.sanitize_memory_track_origins);

  ABSL_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<CompiledModule>> aot_results,
      cpu_compiler.CompileAheadOfTime(std::move(hlo_module), aot_options));
  if (aot_results.size() != 1) {
    return absl::InternalError(absl::StrCat(
        "Expected one AOT compilation result, got ", aot_results.size()));
  }
  return std::move(aot_results[0]);
}

absl::StatusOr<std::string> AotCompileCpuExecutable(
    std::unique_ptr<HloModule> hlo_module,
    std::optional<cpu::TargetMachineOptions> target_config,
    CompilationResult* result, const CpuSanitizerOptions& sanitizer_options) {
  if (sanitizer_options.sanitize_memory_track_origins < 0 ||
      sanitizer_options.sanitize_memory_track_origins > 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("sanitize_memory_track_origins must be 0, 1 or 2; got ",
                     sanitizer_options.sanitize_memory_track_origins));
  }
  cpu::CpuCompiler cpu_compiler;

  if (sanitizer_options.sanitize_memory ||
      sanitizer_options.sanitize_memory_track_origins > 0) {
    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<CompiledModule> aot_result,
        CompileSanitized(cpu_compiler, std::move(hlo_module),
                         std::move(target_config), sanitizer_options));
    if (result != nullptr) {
      *result->mutable_hlo_module() = aot_result->optimized_module()->ToProto();
    }
    return aot_result->SerializeAsString();
  }

  Compiler::CompileOptions compile_options;
  if (target_config.has_value()) {
    compile_options.cpu_target_config =
        Compiler::CpuTargetConfig(*target_config);
  }
  ABSL_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<Executable>> executables,
      cpu_compiler.Compile(std::move(hlo_module), {nullptr}, compile_options));
  if (result != nullptr && !executables.empty()) {
    *result->mutable_hlo_module() = executables[0]->module().ToProto();
  }
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<CompiledModule> aot_result,
                   cpu_compiler.Export(executables[0].get()));
  return aot_result->SerializeAsString();
}

}  // namespace xla

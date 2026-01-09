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

#ifndef XLA_SERVICE_CPU_CPU_AOT_LOADER_H_
#define XLA_SERVICE_CPU_CPU_AOT_LOADER_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/Target/TargetOptions.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_module_config.h"

namespace xla::cpu {

llvm::TargetOptions CompilerTargetOptions(const HloModuleConfig& module_config);

absl::StatusOr<std::unique_ptr<FunctionLibrary>> LoadFunctionLibrary(
    const std::vector<FunctionLibrary::Symbol>& compiled_symbols,
    absl::Span<const ObjFileProto> obj_files, const HloModule* hlo_module,
    const TargetMachineOptions& target_machine_options);

absl::StatusOr<std::vector<FunctionLibrary::Symbol>>
GetCompiledSymbolsFromProto(
    absl::Span<const SymbolProto> compiled_symbols_proto);

class CpuAotLoader {
 public:
  static absl::StatusOr<std::unique_ptr<Executable>> LoadExecutable(
      const std::string& serialized_aot_result);

  static absl::StatusOr<std::unique_ptr<Executable>> LoadExecutable(
      const xla::cpu::CompilationResultProto& aot_result_proto);

  static absl::StatusOr<std::unique_ptr<Executable>> LoadExecutable(
      CompiledModule&& compilation_result);

  static absl::StatusOr<std::unique_ptr<CompiledModule>>
  LoadAotCompilationResult(const std::string& serialized_aot_result);

  static absl::StatusOr<std::unique_ptr<CompiledModule>>
  LoadAotCompilationResult(
      const xla::cpu::CompilationResultProto& aot_result_proto);
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_CPU_AOT_LOADER_H_

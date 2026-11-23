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

#include "absl/status/statusor.h"
#include "llvm/Target/TargetOptions.h"
#include "xla/service/compiled_module.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/service/cpu/function_library_loader.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_module_config.h"

namespace xla::cpu {

llvm::TargetOptions CompilerTargetOptions(const HloModuleConfig& module_config);

class CpuAotLoader {
 public:
  static absl::StatusOr<std::unique_ptr<Executable>> LoadExecutable(
      const std::string& serialized_aot_result,
      FunctionLibraryLoader function_library_loader);

  static absl::StatusOr<std::unique_ptr<Executable>> LoadExecutable(
      const xla::cpu::CompilationResultProto& aot_result_proto,
      FunctionLibraryLoader function_library_loader);

  static absl::StatusOr<std::unique_ptr<Executable>> LoadExecutable(
      CompiledModule&& compilation_result);

  static absl::StatusOr<std::unique_ptr<CompiledModule>>
  LoadAotCompilationResult(const std::string& serialized_aot_result,
                           FunctionLibraryLoader function_library_loader);

  static absl::StatusOr<std::unique_ptr<CompiledModule>>
  LoadAotCompilationResult(
      const xla::cpu::CompilationResultProto& aot_result_proto,
      FunctionLibraryLoader function_library_loader);
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_CPU_AOT_LOADER_H_

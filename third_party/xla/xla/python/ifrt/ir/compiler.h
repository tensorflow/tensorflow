/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_IR_COMPILER_H_
#define XLA_PYTHON_IFRT_IR_COMPILER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/executable.h"
#include "xla/statusor.h"

namespace xla {
namespace ifrt {

struct IfrtIRProgram : llvm::RTTIExtends<IfrtIRProgram, Program> {
  IfrtIRProgram() = default;
  explicit IfrtIRProgram(mlir::ModuleOp mlir_module)
      : mlir_module(std::move(mlir_module)) {}

  mlir::ModuleOp mlir_module;

  static char ID;  // NOLINT
};

// CompileOptions for an IFRT IR program.
struct IfrtIRCompileOptions
    : llvm::RTTIExtends<IfrtIRCompileOptions, CompileOptions> {
  IfrtIRCompileOptions() = default;
  explicit IfrtIRCompileOptions(
      std::vector<int> device_assignments,
      absl::flat_hash_map<std::string, LoadedExecutable*> loaded_exec_binding =
          {},
      std::shared_ptr<absl::flat_hash_map<
          std::string, std::unique_ptr<xla::ifrt::CompileOptions>>>
          compile_options_overrides = {})
      : device_assignments(std::move(device_assignments)),
        loaded_exec_binding(std::move(loaded_exec_binding)),
        compile_options_overrides(std::move(compile_options_overrides)) {}

  // Map from logical device ids in MLIR module to runtime device ids obtained
  // from IFRT client.
  std::vector<int> device_assignments;

  // Map from `getSymName()` of declared LoadedExecutableOp in the `mlir_module`
  // to pre-compiled LoadedExecutable instance. The LoadedExecutables must
  // outlive the LoadedExecutable to be compiled.
  absl::flat_hash_map<std::string, LoadedExecutable*> loaded_exec_binding;

  // Mapping from values of `ifrt.compile_option_key` attribute of a `CallOp` to
  // compile options. If a `CallOp` does not have have the attribute set or does
  // not have an entry in this map then default compile options are used.
  std::shared_ptr<absl::flat_hash_map<
      std::string, std::unique_ptr<xla::ifrt::CompileOptions>>>
      compile_options_overrides;

  static char ID;  // NOLINT
};

// Gets `xla::ifrt::IfrtIRCompileOptions` from `xla::ifrt::CompileOptions`.
absl::StatusOr<std::unique_ptr<IfrtIRCompileOptions>> GetIfrtIRCompileOptions(
    std::unique_ptr<CompileOptions> options);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_COMPILER_H_

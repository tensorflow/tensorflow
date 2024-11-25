/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_CODEGEN_JIT_COMPILER_H_
#define XLA_BACKENDS_CPU_CODEGEN_JIT_COMPILER_H_

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetOptions.h"
#include "xla/backends/cpu/codegen/function_library.h"

namespace xla::cpu {

// Jit compiler that compiles LLVM modules added to it into a FunctionLibrary.
// Jit-compiled function library will be backed by multiple dynamic libraries
// compiled from LLVM modules using LLVM ORC APIs.
class JitCompiler {
 public:
  virtual ~JitCompiler() = default;

  struct Options {
    // Maximum CPU instruction set for wich the compiler should generate code.
    // If instruction set is empty, compiler will generate code for all ISA
    // extensions detected on the current machine.
    std::string max_cpu_isa;
  };

  // Creates a new instance of the JitCompiler.
  static absl::StatusOr<std::unique_ptr<JitCompiler>> Create(
      llvm::TargetOptions target_options, llvm::CodeGenOptLevel opt_level,
      const Options& options);

  // Adds a LLVM module to the dynamic library at `dylib_index`.
  virtual absl::Status AddModule(llvm::orc::ThreadSafeModule module,
                                 size_t dylib_index) = 0;

  absl::Status AddModule(llvm::orc::ThreadSafeModule module) {
    return AddModule(std::move(module), 0);
  }

  // Compiles all added LLVM modules into the FunctionLibrary.
  virtual absl::StatusOr<std::unique_ptr<FunctionLibrary>> Compile() && = 0;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_JIT_COMPILER_H_

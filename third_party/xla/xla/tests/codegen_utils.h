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

#ifndef XLA_TESTS_CODEGEN_UTILS_H_
#define XLA_TESTS_CODEGEN_UTILS_H_

#include <memory>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/llvm_compiler.h"

namespace xla {

// Compiles `hlo_module` with the provided `compiler` and `compile_options`. If
// `run_optimization_passes` is true, also the HLO optimization pass pipeline is
// run.
absl::StatusOr<std::unique_ptr<Executable>> CompileToExecutable(
    Compiler* compiler, const Compiler::CompileOptions& compile_options,
    std::unique_ptr<HloModule> hlo_module, bool run_optimization_passes);

// Compiles the given HLO module to LLVM IR and verifies the IR matches the
// given pattern. `pattern` is in the FileCheck pattern matching syntax
// (http://llvm.org/docs/CommandGuide/FileCheck.html).
//
// This function invokes the JIT compiler.
//
// If `match_optimized_ir` is true, match the version of the IR after internal
// optimizations are applied; otherwise, the IR before optimizations is
// matched.
absl::Status CompileAndVerifyIr(LLVMCompiler* compiler,
                                const Compiler::CompileOptions& compile_options,
                                std::unique_ptr<HloModule> hlo_module,
                                absl::string_view pattern,
                                bool match_optimized_ir,
                                bool run_optimization_passes = true);

}  // namespace xla

#endif  // XLA_TESTS_CODEGEN_UTILS_H_

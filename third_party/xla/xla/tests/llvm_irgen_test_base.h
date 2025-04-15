/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_TESTS_LLVM_IRGEN_TEST_BASE_H_
#define XLA_TESTS_LLVM_IRGEN_TEST_BASE_H_

#include <string>

#include "xla/service/llvm_compiler.h"
#include "xla/tests/codegen_test_base.h"

namespace xla {

// Tests that verify IR emitted by the CPU/GPU backend is as expected.
class LlvmIrGenTestBase : public CodegenTestBase {
 protected:
  // Compiles the given HLO module to LLVM IR and verifies the IR matches the
  // given pattern. `pattern` is in the FileCheck pattern matching syntax
  // (http://llvm.org/docs/CommandGuide/FileCheck.html).
  //
  // This function invokes the JIT compiler.
  //
  // If `match_optimized_ir` is true, match the version of the IR after internal
  // optimizations are applied; otherwise, the IR before optimizations is
  // matched.
  void CompileAndVerifyIr(std::unique_ptr<HloModule> hlo_module,
                          const std::string& pattern, bool match_optimized_ir,
                          bool run_optimization_passes = true);

  // A thin wrapper around CompileAndVerifyIr that parses `hlo_text` to create
  // an HLO module.
  void CompileAndVerifyIr(const std::string& hlo_text,
                          const std::string& expected_llvm_ir,
                          bool match_optimized_ir = false,
                          bool run_optimization_passes = true);

  // Compiles the given HLO module to LLVM IR and verifies the IR matches the
  // given pattern. `pattern` is in the FileCheck pattern matching syntax
  // (http://llvm.org/docs/CommandGuide/FileCheck.html).
  //
  // This function invokes the AOT compiler, with options in `options`.
  //
  // If `match_optimized_ir` is true, match the version of the IR after internal
  // optimizations are applied; otherwise, the IR before optimizations is
  // matched.
  void CompileAheadOfTimeAndVerifyIr(std::unique_ptr<HloModule> hlo_module,
                                     const AotCompilationOptions& options,
                                     const std::string& pattern,
                                     bool match_optimized_ir);

 private:
  LLVMCompiler* GetLLVMCompiler();

  void SetIrHook(bool match_optimized_ir);
  void ResetIrHook();

  std::string ir_;
  absl::Status IrHook(const llvm::Module& module);
};

}  // namespace xla

#endif  // XLA_TESTS_LLVM_IRGEN_TEST_BASE_H_

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_TESTS_LLVM_IRGEN_TEST_BASE_H_
#define TENSORFLOW_COMPILER_XLA_TESTS_LLVM_IRGEN_TEST_BASE_H_

#include <string>

#include "tensorflow/compiler/xla/service/llvm_compiler.h"
#include "tensorflow/compiler/xla/tests/codegen_test_base.h"

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
                          const string& pattern, bool match_optimized_ir);

  // A thin wrapper around CompileAndVerifyIr that parses `hlo_text` to create
  // an HLO module.
  void CompileAndVerifyIr(const string& hlo_text,
                          const string& expected_llvm_ir,
                          bool match_optimized_ir = false);

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
                                     const string& pattern,
                                     bool match_optimized_ir);

  // Compiles the given `hlo` with optimizations, and verifies that optimized
  // HLO matches the given FileCheck pattern.
  void MatchOptimizedHlo(absl::string_view hlo, absl::string_view pattern,
                         bool print_operand_shape = false);

  // LikeMatchOptimizedHlo, but checks operand shapes as well.
  void MatchOptimizedHloWithShapes(absl::string_view hlo,
                                   absl::string_view pattern) {
    MatchOptimizedHlo(hlo, pattern, /*print_operand_shape=*/true);
  }

  // Compiles and returns module with optimizations from a given HLO.
  StatusOr<std::unique_ptr<HloModule>> GetOptimizedModule(
      absl::string_view hlo);

 private:
  LLVMCompiler* GetLLVMCompiler();

  void SetIrHook(bool match_optimized_ir);
  void ResetIrHook();

  string ir_;
  Status IrHook(const llvm::Module& module);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TESTS_LLVM_IRGEN_TEST_BASE_H_

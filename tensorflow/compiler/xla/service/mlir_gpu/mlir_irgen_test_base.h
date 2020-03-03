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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_MLIR_GPU_MLIR_IRGEN_TEST_BASE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_MLIR_GPU_MLIR_IRGEN_TEST_BASE_H_

#include <string>

#include "tensorflow/compiler/xla/service/mlir_gpu/mlir_compiler.h"
#include "tensorflow/compiler/xla/tests/codegen_test_base.h"

namespace xla {
namespace mlir_gpu {

// Tests that verify IR emitted by the CPU/GPU backend is as expected.
class MlirIrGenTestBase : public CodegenTestBase {
 protected:
  using LoweringStage = MlirCompiler::IRHook::LoweringStage;

  // Compiles the given HLO module to MLIR IR and verifies the IR matches the
  // given pattern. `pattern` is in the FileCheck pattern matching syntax
  // (http://llvm.org/docs/CommandGuide/FileCheck.html).
  //
  // This function invokes the JIT compiler.
  //
  // If `match_lowered_ir` is true, match the version of the IR after lowering
  // steps to LLVM IR are applied; otherwise, the IR before lowering is
  // matched.
  void CompileAndVerifyIr(std::unique_ptr<HloModule> hlo_module,
                          const std::string& pattern_file,
                          LoweringStage printing_stage);

  // A thin wrapper around CompileAndVerifyIr that parses the hlo text in
  // `hlo_text_filename` to create an HLO module.
  void CompileAndVerifyIr(const std::string& hlo_text_filename,
                          LoweringStage printing_stage = LoweringStage::LHLO);

  // Adds the InjectErrorsForTestingPass to MLIRCompiler on the provided
  // lowering stage, compiles the given HLO module, and returns a std::string
  // representation of all the errors occurred during compiling.
  StatusOr<string> CompileAndInjectErrors(std::unique_ptr<HloModule> hlo_module,
                                          LoweringStage breaking_stage);

  // Adds the InjectErrorsForTestingPass to MLIRCompiler on the provided
  // lowering stage, parses and compiles `hlo_text`, and verifies that the
  // std::string representation of all the errors occurred during compiling
  // matches the given pattern.
  void CompileAndVerifyErrors(const std::string& hlo_text_filename,
                              LoweringStage breaking_stage);

 private:
  StatusOr<std::unique_ptr<VerifiedHloModule>> GetVerifiedHloModule(
      const std::string& hlo_text_filename);

  void CompileIr(std::unique_ptr<HloModule> hlo_module,
                 const MlirCompiler::IRHook& ir_hook);
  void PatternMatch(const std::string& str, const std::string& pattern_file);
  std::string CompileIr(std::unique_ptr<HloModule> hlo_module,
                        LoweringStage printing_stage);
  MlirCompiler::IRHook getIRHookBreakingLoweringStage(
      LoweringStage breaking_stage);
  MlirCompiler* GetMLIRCompiler();
};

}  // namespace mlir_gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_MLIR_GPU_MLIR_IRGEN_TEST_BASE_H_

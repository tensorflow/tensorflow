/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_MLIR_GPU_XLA_GPU_OPT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_MLIR_GPU_XLA_GPU_OPT_H_

#include <memory>
#include <string>

#include "llvm/Support/raw_ostream.h"
#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/mlir_gpu/mlir_compiler.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/verified_hlo_module.h"

namespace xla {
namespace mlir_gpu {

// Prints the IR created by the MLIR GPU backend at a certain lowering stage.
class XlaGpuOpt {
 public:
  using LoweringStage = MlirCompiler::IRHook::LoweringStage;
  XlaGpuOpt() {
    backend_ = std::move(Backend::CreateDefaultBackend().ValueOrDie());
  }

  // Compiles the HLO module given in 'hlo_text' to a GpuExecutable and prints
  // the IR at the lowering stage 'printing_stage' to the 'os' stream.
  //
  // This function invokes the JIT compiler.
  Status CompileAndOutputIr(const std::string& hlo_text, llvm::raw_ostream& os,
                            LoweringStage printing_stage = LoweringStage::LHLO);

  // Adds the InjectErrorsForTestingPass to MLIRCompiler on the provided
  // lowering stage 'breaking_stage', parses and compiles `hlo_text`, and prints
  // the resulting errors to the 'os' stream.
  Status CompileAndExpectErrors(const std::string& hlo_text,
                                llvm::raw_ostream& os,
                                LoweringStage breaking_stage);

 private:
  std::unique_ptr<Backend> backend_;
  StatusOr<std::unique_ptr<VerifiedHloModule>> GetVerifiedHloModule(
      const std::string& hlo_text_filename);

  Status CompileAndOutputIr(std::unique_ptr<HloModule> hlo_module,
                            llvm::raw_ostream& os,
                            LoweringStage printing_stage);
  Status CompileIr(std::unique_ptr<HloModule> hlo_module,
                   const MlirCompiler::IRHook& ir_hook);
  StatusOr<std::string> CompileIr(std::unique_ptr<HloModule> hlo_module,
                                  LoweringStage printing_stage);
  MlirCompiler::IRHook GetIRHookBreakingLoweringStage(
      LoweringStage breaking_stage);
  StatusOr<std::string> CompileAndInjectErrors(
      std::unique_ptr<HloModule> hlo_module, LoweringStage breaking_stage);
  MlirCompiler* GetMLIRCompiler();
};

}  // namespace mlir_gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_MLIR_GPU_XLA_GPU_OPT_H_

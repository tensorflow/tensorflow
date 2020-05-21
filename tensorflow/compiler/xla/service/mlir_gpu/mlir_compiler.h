/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_MLIR_GPU_MLIR_COMPILER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_MLIR_GPU_MLIR_COMPILER_H_

#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/mlir_gpu/emission_context.h"

namespace xla {
namespace mlir_gpu {

// A Compiler implementation that converts XLAs IR to a matching MLIR dialect,
// performs all lowering on the MLIR IR and finally converts MLIR to LLVMIR for
// generation of a thunk suitable for XLAs runtime. MlirCompilerImpl contains
// the implementation.
class MlirCompiler : public Compiler {
  using ErrorHandler =
      std::function<void(const EmissionContext::ErrorMap&, HloModule*)>;

 public:
  MlirCompiler();

  se::Platform::Id PlatformId() const override;

  struct IRHook {
    enum class LoweringStage { LHLO, GPU, LLVM, KERNEL };

    Status invoke(LoweringStage stage_, mlir::ModuleOp module) {
      if (callback && stage == stage_) {
        return callback(module);
      }
      return Status::OK();
    }

    std::function<Status(mlir::ModuleOp)> callback;
    LoweringStage stage;
  };

  void SetModuleHook(IRHook module_hook);
  void RemoveModuleHook();
  void SetErrorHandler(ErrorHandler error_handler);
  void RemoveErrorHandler();

 protected:
  ::mlir::MLIRContext context_;
  int64 pointer_size_;
  IRHook module_hook_;
  ErrorHandler error_handler_;
};

}  // namespace mlir_gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_MLIR_GPU_MLIR_COMPILER_H_

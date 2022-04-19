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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_XLA_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_XLA_PASSES_H_

#include <memory>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project

namespace mlir {

namespace func {
class FuncOp;
}  // namespace func
template <typename T>
class OperationPass;

namespace mhlo {

// Prepare module for export to XLA HLO protos/instruction.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareForExport();

// Wrap function with XLA:CPU's C interface.
std::unique_ptr<OperationPass<ModuleOp>> CreateOutlineWithXLAFrameworkPass();

// Convert XLAFramework operations to LLVM operations.
std::unique_ptr<OperationPass<ModuleOp>> CreateLegalizeXLAFrameworkToLLVMPass();

// Patterns to lower all XLAFramework operations and types to LLVM versions.
void PopulateLegalizeXLAFrameworkToLLVMPatterns(llvm::StringRef device_type,
                                                RewritePatternSet& patterns,
                                                MLIRContext* ctx,
                                                bool prefer_tf2xla = false);

#define GEN_PASS_REGISTRATION
#include "tensorflow/compiler/mlir/xla/transforms/xla_passes.h.inc"

}  // namespace mhlo
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_XLA_PASSES_H_

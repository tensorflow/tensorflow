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

#ifndef TENSORFLOW_COMPILER_MLIR_TFR_IR_TFR_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_TFR_IR_TFR_PASSES_H_

#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace TFR {

// Scans the func op and adds all the canonicalization patterns of the ops
// except the tf ops, inside the function.
void populateCanonicalizationPatterns(FuncOp func, RewritePatternSet &patterns);

// Decompose ops.
std::unique_ptr<OperationPass<FuncOp>> CreateDecomposeTFOpsPass(
    llvm::Optional<ModuleOp> tfr_module = llvm::None);

// Rewrites quantized operands and results with their storage types.
// This pass should be run at module level after decomposition, if there are
// quantized operands or results.
std::unique_ptr<OperationPass<ModuleOp>> CreateRewriteQuantizedIOPass();

// Raise to TF ops.
std::unique_ptr<OperationPass<FuncOp>> CreateRaiseToTFOpsPass(
    llvm::Optional<ModuleOp> tfr_module = llvm::None,
    bool materialize_derived_attrs = false);

}  // namespace TFR
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TFR_IR_TFR_PASSES_H_

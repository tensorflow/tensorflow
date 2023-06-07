/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_VISITOR_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_VISITOR_H_

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace TF {

// Walks the function by following function call chains and calling the callback
// for each reachable function (including `func`). Each function is visited only
// once even if it's called from multiple places and/or recursively.
//
// The current implementation follows direct calls to `mlir::func::FuncOp` only
// and returns a `mlir::WalkResult::interrupt()` when it encounters a call whose
// callee cannot be resolved to `mlir::func::FuncOp`.
mlir::WalkResult WalkReachableFunctions(
    mlir::func::FuncOp func,
    llvm::function_ref<mlir::WalkResult(mlir::func::FuncOp)> callback,
    mlir::SymbolTableCollection* symbol_table = nullptr);

// Creates a new MLIR module that contains only the given functions and all
// reachable functions from them.
mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> CreatePrunedModule(
    mlir::ModuleOp module, llvm::ArrayRef<llvm::StringRef> function_names);

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_VISITOR_H_

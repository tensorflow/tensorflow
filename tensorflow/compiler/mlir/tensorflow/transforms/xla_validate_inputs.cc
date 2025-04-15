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

#include <functional>
#include <memory>

#include "absl/log/log.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/call_graph_util.h"

namespace mlir {

namespace {

#define GEN_PASS_DEF_XLAVALIDATEINPUTSPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_device_passes.h.inc"

// Validate input graph.
struct XlaValidateInputsPass
    : public impl::XlaValidateInputsPassBase<XlaValidateInputsPass> {
  void runOnOperation() override;
};

LogicalResult HasNoNestedEntryFunctions(
    const llvm::SmallVector<func::FuncOp> &entry_funcs, SymbolTable &symtab) {
  auto calls_entry_functions = [&](SymbolUserOpInterface op) {
    llvm::SmallVector<func::FuncOp> callees;
    if (GetCallees(op, symtab, callees).failed()) {
      return false;
    }
    for (auto &callee : callees) {
      if (IsEntryFunction(callee)) {
        return true;
      }
    }
    return false;
  };

  for (auto &entry_func : entry_funcs) {
    llvm::SmallVector<SymbolUserOpInterface> calls;
    if (GetFirstOpsOfType<SymbolUserOpInterface>(
            entry_func, symtab, /*predicate*/ calls_entry_functions, calls)
            .failed()) {
      return failure();
    }
    if (!calls.empty()) {
      // This is not expected to happen in practice. We can add a pass after
      // GuaranteeAllFuncsOneUsePass to remove "tf.entry_function" or
      // "tf_saved_model.initializer_type" attribute from the callee of the
      // inner calls if the problem ever arises.
      entry_func->emitError()
          << "TF2XLA MLIR Non-replicated Phase 1 Bridge expects no nested calls"
             " of entry functions as they prevent graph traversal in some "
             "passes from "
             "working correctly";
      return failure();
    }
  }
  return success();
}

LogicalResult HasTopLevelCompilationMarker(
    llvm::SmallVector<func::FuncOp> &entry_funcs) {
  for (auto &entry_func : entry_funcs) {
    if (entry_func->hasAttr(mlir::TF::kCompileDeviceTypeAttr)) {
      entry_func->emitError() << "TF2XLA MLIR Non-replicated Phase 1 Bridge "
                                 "does not support top-level compilation "
                                 "marker.";
      return failure();
    }
  }
  return success();
}

void XlaValidateInputsPass::runOnOperation() {
  ModuleOp module = getOperation();
  SymbolTable symtab(module);
  llvm::SmallVector<func::FuncOp> entry_funcs = GetEntryFunctions(module);
  if (entry_funcs.empty()) {
    LOG(WARNING) << "missing entry functions";
  }

  if (HasNoNestedEntryFunctions(entry_funcs, symtab).failed()) {
    return signalPassFailure();
  }

  if (HasTopLevelCompilationMarker(entry_funcs).failed()) {
    return signalPassFailure();
  }
}

}  // namespace

namespace TFDevice {
std::unique_ptr<OperationPass<ModuleOp>> CreateXlaValidateInputsPass() {
  return std::make_unique<XlaValidateInputsPass>();
}
}  // namespace TFDevice

}  // namespace mlir

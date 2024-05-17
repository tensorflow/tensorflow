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

#include <vector>

#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

namespace mlir {

std::vector<llvm::StringRef> GetEntryFunctionAttributeNames() {
  return {"tf.entry_function",
          tf_saved_model::kTfSavedModelInitializerTypeAttr};
}

bool IsEntryFunction(func::FuncOp func) {
  for (const auto &attr : GetEntryFunctionAttributeNames()) {
    if (func->hasAttr(attr)) {
      return true;
    }
  }
  return false;
}

llvm::SmallVector<func::FuncOp> GetEntryFunctions(ModuleOp module) {
  llvm::SmallVector<func::FuncOp> entry_funcs;
  module.walk([&](func::FuncOp func) {
    // A model may have multiple graphs, with each graph having its own entry.
    // When a graph is imported to MLIR, `tf.entry_function` will be added to
    // each entry function. The one exception are initializer functions, which
    // have `tf_saved_model.initializer_type` instead.
    if (IsEntryFunction(func)) {
      entry_funcs.push_back(func);
    }
  });
  return entry_funcs;
}

LogicalResult GetCallees(SymbolUserOpInterface op, SymbolTable &symtab,
                         llvm::SmallVector<func::FuncOp> &callees) {
  for (auto attr : op->getAttrs()) {
    auto sym = mlir::dyn_cast<SymbolRefAttr>(attr.getValue());
    if (!sym) continue;
    auto callee = symtab.lookup<func::FuncOp>(sym.getRootReference());
    if (!callee) {
      // This is not expected to happen in practice.
      return op->emitError()
             << "Cannot find function " << sym.getRootReference();
    }
    callees.push_back(callee);
  }
  return success();
}

bool HasSingleBlock(func::FuncOp func) {
  return func->getNumRegions() == 1 && func.getBody().hasOneBlock();
}

}  // namespace mlir

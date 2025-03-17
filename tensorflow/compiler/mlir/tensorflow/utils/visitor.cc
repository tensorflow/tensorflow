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

#include "tensorflow/compiler/mlir/tensorflow/utils/visitor.h"

#include <utility>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace TF {

WalkResult WalkReachableFunctions(
    func::FuncOp func,
    llvm::function_ref<WalkResult(func::FuncOp)> callback,
    SymbolTableCollection* symbol_table) {
  llvm::SmallDenseSet<Operation*> visited;

  llvm::SmallVector<func::FuncOp> stack;
  stack.push_back(func);

  while (!stack.empty()) {
    func::FuncOp f = stack.back();
    stack.pop_back();

    if (!visited.insert(f).second) {
      continue;
    }

    WalkResult result = callback(f);
    if (result.wasInterrupted()) {
      return result;
    } else if (result.wasSkipped()) {
      continue;
    }

    result = f.walk([&](Operation* op) {
      const auto uses = SymbolTable::getSymbolUses(op);
      if (!uses.has_value()) {
        op->emitOpError() << "contains a potentially unknown symbol table";
        return WalkResult::interrupt();
      }

      for (const SymbolTable::SymbolUse& use : *uses) {
        func::FuncOp called_func =
            symbol_table != nullptr
                ? symbol_table->lookupNearestSymbolFrom<func::FuncOp>(
                      use.getUser(), use.getSymbolRef())
                : SymbolTable::lookupNearestSymbolFrom<
                      func::FuncOp>(use.getUser(), use.getSymbolRef());
        if (called_func == nullptr) {
          op->emitOpError()
              << "refers to an unknown symbol (expects a function)";
          return WalkResult::interrupt();
        }
        stack.push_back(called_func);
      }

      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      return result;
    }
  }

  return WalkResult::advance();
}

FailureOr<OwningOpRef<ModuleOp>> CreatePrunedModule(
    ModuleOp module, llvm::ArrayRef<llvm::StringRef> function_names) {
  SymbolTableCollection symbol_table;
  OpBuilder builder(module.getContext());

  OwningOpRef<ModuleOp> pruned =
      builder.create<ModuleOp>(module->getLoc());
  (*pruned)->setAttrs(module->getAttrs());
  builder.setInsertionPointToEnd(pruned->getBody());

  llvm::SmallDenseSet<func::FuncOp> added;
  for (const llvm::StringRef function_name : function_names) {
    auto func =
        llvm::dyn_cast_or_null<func::FuncOp>(symbol_table.lookupSymbolIn(
            module, builder.getStringAttr(function_name)));
    if (func == nullptr) {
      return module.emitError()
             << "Cannot find function '" << function_name << "'";
    }

    const WalkResult result = WalkReachableFunctions(
        func,
        [&](func::FuncOp f) {
          if (!added.insert(f).second) {
            return WalkResult::skip();
          }
          builder.clone(*f);
          return WalkResult::advance();
        },
        &symbol_table);
    if (result.wasInterrupted()) {
      return failure();
    }
  }

  return pruned;
}

}  // namespace TF
}  // namespace mlir

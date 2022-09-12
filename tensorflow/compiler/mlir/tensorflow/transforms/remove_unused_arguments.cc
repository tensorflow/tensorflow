/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <utility>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"

namespace mlir {
namespace TF {

namespace {

// Removes unused arguments from functions and their callers.
struct RemoveUnusedArgumentsPass
    : public RemoveUnusedArgumentsPassBase<RemoveUnusedArgumentsPass> {
  void runOnOperation() override;
};

// Return whether a given value is used.
bool isUsed(Value v) {
  // TODO(b/246310765): This doesn't handle recursion.
  return !v.use_empty();
}

void RemoveUnusedArgumentsPass::runOnOperation() {
  Operation* module = getOperation();

  llvm::DenseMap<Operation*, llvm::BitVector> args_to_erase;
  llvm::DenseSet<Operation*> do_not_touch;  // Funcs referenced by non-call ops

  // Find all users of functions that are not through a CallOp. Those
  // are functions we need to leave alone.
  module->walk([&](SymbolUserOpInterface op) {
    if (llvm::isa<CallOpInterface>(op.getOperation())) return;
    // SymbolUserOpInterface doesn't tell us which attributes contain
    // the symbols, so we have to scan through all of them.
    for (auto attr : op->getAttrs()) {
      if (auto sym = attr.getValue().dyn_cast<FlatSymbolRefAttr>()) {
        Operation* func = mlir::SymbolTable::lookupNearestSymbolFrom(op, sym);
        if (func) {
          do_not_touch.insert(func);
        }
      }
    }
  });

  // Find all functions
  module->walk([&](SymbolOpInterface op) {
    if (!op.isPrivate()) return;

    auto call = llvm::dyn_cast<CallableOpInterface>(op.getOperation());
    if (!call) return;
    Region* region = call.getCallableRegion();
    if (!region) return;  // happens e.g. for external functions

    auto func = llvm::dyn_cast<FunctionOpInterface>(op.getOperation());
    if (!func || do_not_touch.count(func)) return;
    llvm::BitVector unused(func.getNumArguments());
    for (BlockArgument arg : func.getArguments()) {
      if (!isUsed(arg)) {
        unused.set(arg.getArgNumber());
      }
    }
    func.eraseArguments(unused);
    args_to_erase.insert(std::make_pair(op, unused));
  });

  // Find all callers
  module->walk([&](CallOpInterface op) {
    auto callable = op.getCallableForCallee();
    mlir::SymbolRefAttr sym = callable.dyn_cast<mlir::SymbolRefAttr>();
    if (!sym) return;
    Operation* func = mlir::SymbolTable::lookupNearestSymbolFrom(op, sym);
    if (!args_to_erase.count(func)) return;
    op->eraseOperands(args_to_erase.lookup(func));
  });
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateRemoveUnusedArgumentsPass() {
  return std::make_unique<RemoveUnusedArgumentsPass>();
}

}  // namespace TF
}  // namespace mlir

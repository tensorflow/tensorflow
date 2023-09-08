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

#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/Interfaces/ControlFlowInterfaces.h"  // from @llvm-project
#include "mlir/Interfaces/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace TF {

namespace {

#define GEN_PASS_DEF_REMOVEUNUSEDARGUMENTSPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

// Removes unused arguments from functions and their callers.
struct RemoveUnusedArgumentsPass
    : public impl::RemoveUnusedArgumentsPassBase<RemoveUnusedArgumentsPass> {
  void runOnOperation() override;
};

// Return a bitvector that marks all return values that always come from
// the same value.
llvm::BitVector GetInvariantReturns(Region* region, int number_of_results) {
  llvm::BitVector invariant(number_of_results, true);
  std::vector<Value> used_argument(number_of_results, nullptr);
  for (Block& block : region->getBlocks()) {
    for (Operation& op : block.getOperations()) {
      if (!op.hasTrait<OpTrait::ReturnLike>()) continue;
      for (OpOperand& operand : op.getOpOperands()) {
        int i = operand.getOperandNumber();
        if (used_argument[i] && operand.get() != used_argument[i]) {
          assert(i < number_of_results);
          invariant.reset(i);
        }
        used_argument[i] = operand.get();
      }
    }
  }
  return invariant;
}

Operation* GetAnyReturn(Region* region) {
  Operation* result;
  // We only go one level deep, since "returns" in nested functions
  // and return-as-yield CF don't belong to us.
  for (Block& block : region->getBlocks()) {
    for (Operation& op : block.getOperations()) {
      if (op.hasTrait<OpTrait::ReturnLike>()) result = &op;
    }
  }
  return result;
}

void EraseReturnOperands(Region* region, llvm::BitVector& erase) {
  for (Block& block : region->getBlocks()) {
    for (Operation& op : block.getOperations()) {
      if (!op.hasTrait<OpTrait::ReturnLike>()) continue;
      op.eraseOperands(erase);
    }
  }
}

// Erases the given results from an operation, similar to what
// Operation::eraseArguments does (but for results).
// This is a lengthy bit of code, since it has to recreate the operation.
// TODO(kramm): Move this under utils/ somewhere.
void EraseResults(Operation* op, llvm::BitVector erase) {
  assert(!op->getNumRegions());
  std::vector<Type> result_types;
  for (auto result : op->getResults()) {
    if (!erase[result.getResultNumber()]) {
      result_types.push_back(result.getType());
    }
  }
  OpBuilder builder(op);
  builder.setInsertionPointAfter(op);
  OperationState state(op->getLoc(), op->getName().getStringRef(),
                       op->getOperands(), result_types, op->getAttrs());
  for (int i = 0; i < op->getNumRegions(); ++i) {
    state.addRegion();
  }
  Operation* new_op = builder.create(state);
  for (const auto& indexed_regions : llvm::enumerate(op->getRegions())) {
    Region& region = op->getRegion(indexed_regions.index());
    IRMapping mapping;
    indexed_regions.value().cloneInto(&region, mapping);
  }
  int new_position = 0;
  for (auto result : op->getResults()) {
    if (!erase[result.getResultNumber()]) {
      result.replaceAllUsesWith(new_op->getResult(new_position++));
    }
  }
  op->erase();
}

void RemoveUnusedArgumentsPass::runOnOperation() {
  Operation* module = getOperation();

  llvm::DenseMap<Operation*, llvm::BitVector> args_to_erase;
  llvm::DenseMap<Operation*, llvm::BitVector> results_to_erase;
  llvm::DenseMap<Operation*, llvm::DenseMap<int, int>> args_to_remap;
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
    llvm::BitVector unused_args(func.getNumArguments());
    llvm::BitVector unused_results(func.getNumResults());
    llvm::DenseMap<int, int> return_to_operand;
    llvm::BitVector invariant_returns =
        GetInvariantReturns(region, func.getNumResults());
    llvm::DenseMap<Value, int> argument_to_index;
    std::vector<int> use_count(func.getNumArguments(), 0);

    // Set up a use count for all function arguments. We'll use this to
    // determine whether we have taken care of all uses and can remove
    // the arg.
    for (BlockArgument arg : func.getArguments()) {
      auto uses = arg.getUses();
      use_count[arg.getArgNumber()] = std::distance(uses.begin(), uses.end());
      argument_to_index.insert({arg, arg.getArgNumber()});
    }

    // We're only considering return values that are the same (at this
    // position) across all returns, so we only need any single return
    // as reference.
    Operation* ret = GetAnyReturn(region);

    // Go through all return values and find the ones we can remove.
    for (mlir::OpOperand& r : ret->getOpOperands()) {
      int i = r.getOperandNumber();
      if (!invariant_returns[i]) continue;
      if (!argument_to_index.count(r.get())) continue;

      int arg = argument_to_index.lookup(r.get());

      // If we see an arg as a result, and we're going
      // to remove that result, we can assume one fewer use.
      use_count[arg]--;

      return_to_operand.insert({i, arg});
      unused_results.set(i);
    }

    // Now mark all args for deletion that don't have uses anymore.
    for (BlockArgument arg : func.getArguments()) {
      // TODO(b/246310765): This doesn't fully handle recursion.
      if (!use_count[arg.getArgNumber()]) unused_args.set(arg.getArgNumber());
    }

    EraseReturnOperands(region, unused_results);
    func.eraseResults(unused_results);
    func.eraseArguments(unused_args);

    args_to_erase.insert(std::make_pair(op, unused_args));
    results_to_erase.insert(std::make_pair(op, unused_results));
    args_to_remap.insert(std::make_pair(op, return_to_operand));
  });

  // Find all callers
  module->walk([&](CallOpInterface op) {
    auto callable = op.getCallableForCallee();
    mlir::SymbolRefAttr sym = callable.dyn_cast<mlir::SymbolRefAttr>();
    if (!sym) return;
    Operation* func = mlir::SymbolTable::lookupNearestSymbolFrom(op, sym);
    if (!args_to_erase.count(func)) return;

    auto map = args_to_remap.lookup(func);
    for (auto [from, to] : map) {
      op.getOperation()->getResult(from).replaceAllUsesWith(
          op.getOperation()->getOperand(to));
    }
    BitVector operands_to_erase(op->getNumOperands());
    int args_start = op->getNumOperands()
                         ? op.getArgOperands().getBase()->getOperandNumber()
                         : 0;
    operands_to_erase |= args_to_erase.lookup(func);
    operands_to_erase <<= args_start;
    op->eraseOperands(operands_to_erase);

    EraseResults(op, results_to_erase.lookup(func));
  });
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateRemoveUnusedArgumentsPass() {
  return std::make_unique<RemoveUnusedArgumentsPass>();
}

}  // namespace TF
}  // namespace mlir

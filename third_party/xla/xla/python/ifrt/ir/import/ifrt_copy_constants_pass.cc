/* Copyright 2026 The OpenXLA Authors.

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

#include <cstdint>
#include <iterator>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_interfaces.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/import/passes.h"  // IWYU pragma: keep

namespace xla {
namespace ifrt {

#define GEN_PASS_DEF_IFRTCOPYCONSTANTSPASS
#include "xla/python/ifrt/ir/import/passes.h.inc"

namespace {

// Traces `value` backward through zero or more `ifrt.Reshard` or
// `ifrt.CopyArrays` ops (matching the corresponding operand/result index) to
// a producer `ifrt.Call` result, and returns the defining op of the
// corresponding `func.return` operand inside the producer callee.
mlir::Operation* FindHloProducer(mlir::Value value,
                                 mlir::SymbolTableCollection& symbol_table) {
  while (mlir::OpResult op_result =
             llvm::dyn_cast_or_null<mlir::OpResult>(value)) {
    mlir::Operation* producer = op_result.getOwner();

    if (llvm::isa<ReshardOp, CopyArraysOp>(producer)) {
      value =
          llvm::cast<IfrtPairwiseArrayOpInterface>(producer).getInputForOutput(
              op_result);
      continue;
    }

    if (CallOp call = llvm::dyn_cast<CallOp>(producer)) {
      // Skip `ifrt.local_view` calls, which compile the callee with
      // `use_spmd_partitioning = false` and per-shard local shapes.
      if (call->hasAttr(kIfrtLocalViewAttrName)) {
        return nullptr;
      }
      mlir::func::FuncOp callee = call.getCalleeOp(symbol_table);
      // Skip external declarations and multi-block CFGs (which can have
      // multiple `func.return` terminators).
      if (!llvm::hasSingleElement(callee.getBody())) {
        return nullptr;
      }
      mlir::func::ReturnOp return_op = llvm::dyn_cast<mlir::func::ReturnOp>(
          callee.getBody().front().getTerminator());
      if (!return_op) {
        return nullptr;
      }
      return return_op.getOperand(op_result.getResultNumber()).getDefiningOp();
    }

    return nullptr;
  }
  return nullptr;
}

// Clones `callee` (or its enclosing private atom `ModuleOp`, if outlined) so
// `call_op` gets a dedicated callee before mutating its body.
mlir::func::FuncOp CloneCalleeForCall(
    CallOp call_op, mlir::func::FuncOp callee, mlir::ModuleOp root_module,
    mlir::SymbolTableCollection& symbol_table) {
  if (callee->getParentOp() == root_module) {
    mlir::func::FuncOp cloned_callee = callee.clone();
    symbol_table.getSymbolTable(root_module)
        .insert(cloned_callee, std::next(mlir::Block::iterator(callee)));
    call_op.setCalleeAttr(mlir::FlatSymbolRefAttr::get(cloned_callee));
    return cloned_callee;
  }
  if (mlir::ModuleOp callee_module =
          llvm::dyn_cast<mlir::ModuleOp>(callee->getParentOp());
      callee_module && callee_module->getParentOp() == root_module) {
    mlir::ModuleOp cloned_module = callee_module.clone();
    symbol_table.getSymbolTable(root_module)
        .insert(cloned_module, std::next(mlir::Block::iterator(callee_module)));
    mlir::func::FuncOp cloned_callee =
        llvm::dyn_cast_or_null<mlir::func::FuncOp>(
            symbol_table.getSymbolTable(cloned_module)
                .lookup(callee.getSymName()));
    if (!cloned_callee) {
      return nullptr;
    }
    call_op.setCalleeAttr(mlir::SymbolRefAttr::get(
        cloned_module.getSymNameAttr(),
        {mlir::SymbolRefAttr::get(cloned_callee.getSymNameAttr())}));
    return cloned_callee;
  }
  return nullptr;
}

class IfrtCopyConstantsPass
    : public impl::IfrtCopyConstantsPassBase<IfrtCopyConstantsPass> {
 public:
  using impl::IfrtCopyConstantsPassBase<
      IfrtCopyConstantsPass>::IfrtCopyConstantsPassBase;

  void runOnOperation() override {
    mlir::ModuleOp module_op = getOperation();
    mlir::SymbolTableCollection symbol_table;
    mlir::IRRewriter rewriter(&getContext());

    // Pre-count `ifrt.Call` references per callee so single-caller atom
    // functions can be mutated in place while shared callees are cloned on
    // demand only when constants are copied into them.
    llvm::DenseMap<mlir::func::FuncOp, int64_t> callee_call_counts;
    module_op.walk([&](CallOp call_op) {
      ++callee_call_counts[call_op.getCalleeOp(symbol_table)];
    });

    // Visit single-block IFRT functions and process consumer `ifrt.Call` ops in
    // reverse program order so consumers inspect producer callees before any
    // producer is mutated or cloned as a consumer itself.
    for (mlir::func::FuncOp func_op : module_op.getOps<mlir::func::FuncOp>()) {
      if (!func_op->hasAttr(kIfrtFunctionAttrName) ||
          !llvm::hasSingleElement(func_op.getBody())) {
        continue;
      }
      mlir::Block& block = func_op.getBody().front();
      for (CallOp call_op : llvm::reverse(block.getOps<CallOp>())) {
        // Skip `ifrt.local_view` calls: the callee runs with
        // `use_spmd_partitioning = false`, so a global view constant cannot be
        // cloned into a shared replica body.
        if (call_op->hasAttr(kIfrtLocalViewAttrName)) {
          continue;
        }
        mlir::func::FuncOp callee = call_op.getCalleeOp(symbol_table);
        // External declarations have no body blocks, whereas any defined
        // (single or multi block) SSACFG function has a unique dominating
        // entry block `callee.getBody().front()`.
        if (callee.isExternal()) {
          continue;
        }

        // Collect used consumer entry-block arguments whose corresponding
        // `ifrt.Call` input traces back (through zero or more `ifrt.Reshard` or
        // `ifrt.CopyArrays` ops) to a zero-operand, single-result
        // `ConstantLike` op of matching tensor type.
        mlir::Block& callee_entry = callee.getBody().front();
        llvm::SmallVector<std::pair<unsigned, mlir::Operation*>>
            constants_to_copy;
        for (const auto& [arg_idx, input] :
             llvm::enumerate(call_op.getInputs())) {
          mlir::BlockArgument consumer_arg = callee_entry.getArgument(arg_idx);
          if (consumer_arg.use_empty()) {
            continue;
          }
          mlir::Operation* hlo_producer = FindHloProducer(input, symbol_table);
          if (hlo_producer &&
              hlo_producer->hasTrait<mlir::OpTrait::ConstantLike>() &&
              hlo_producer->getNumOperands() == 0 &&
              hlo_producer->getNumResults() == 1 &&
              hlo_producer->getResult(0).getType() == consumer_arg.getType()) {
            constants_to_copy.emplace_back(arg_idx, hlo_producer);
          }
        }

        if (constants_to_copy.empty()) {
          continue;
        }

        // If `callee` is shared across multiple `ifrt.Call` ops, clone it (or
        // its enclosing atom `ModuleOp`) for `call_op` so other call sites are
        // unaffected.
        int64_t& call_count = callee_call_counts[callee];
        if (call_count > 1) {
          mlir::func::FuncOp cloned_callee =
              CloneCalleeForCall(call_op, callee, module_op, symbol_table);
          if (!cloned_callee) {
            continue;
          }
          --call_count;
          callee_call_counts[cloned_callee] = 1;
          callee = cloned_callee;
        }

        // Clone each `ConstantLike` op at the start of `callee`'s dominating
        // entry block and replace all uses of the corresponding block argument.
        // Dead arguments, `ifrt.Call` operands, and transfers are left for
        // `IfrtDcePass` to prune.
        mlir::Block& target_entry = callee.getBody().front();
        rewriter.setInsertionPointToStart(&target_entry);
        for (const auto& [arg_idx, hlo_producer] : constants_to_copy) {
          mlir::Operation* cloned_const = rewriter.clone(*hlo_producer);
          rewriter.replaceAllUsesWith(target_entry.getArgument(arg_idx),
                                      cloned_const->getResult(0));
        }
      }
    }
  }
};

}  // namespace
}  // namespace ifrt
}  // namespace xla

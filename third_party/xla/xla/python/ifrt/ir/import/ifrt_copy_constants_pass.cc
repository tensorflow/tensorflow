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

#include <optional>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_interfaces.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/import/callee_symbol_manager.h"
#include "xla/python/ifrt/ir/import/passes.h"  // IWYU pragma: keep
#include "xla/python/ifrt/ir/utils.h"

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
  while (mlir::OpResult op_result = llvm::dyn_cast<mlir::OpResult>(value)) {
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
      // Skip multi-block CFGs (which can have multiple `func.return`
      // terminators).
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

class IfrtCopyConstantsPass
    : public impl::IfrtCopyConstantsPassBase<IfrtCopyConstantsPass> {
 public:
  using impl::IfrtCopyConstantsPassBase<
      IfrtCopyConstantsPass>::IfrtCopyConstantsPassBase;

  void runOnOperation() override {
    mlir::ModuleOp module_op = getOperation();
    mlir::SymbolTableCollection symbol_table;
    mlir::FailureOr<CalleeSymbolManager> symbol_manager =
        CalleeSymbolManager::Create(module_op, &symbol_table);
    if (mlir::failed(symbol_manager)) {
      signalPassFailure();
      return;
    }
    mlir::IRRewriter rewriter(&getContext());

    // Visit IFRT functions and process consumer `ifrt.Call` ops in reverse
    // program order so consumers inspect producer callees before any producer
    // is mutated or cloned as a consumer itself.
    for (mlir::func::FuncOp func_op : GetIfrtFunctions(module_op)) {
      mlir::Block& block = func_op.getBody().front();
      for (CallOp call_op : llvm::reverse(block.getOps<CallOp>())) {
        // Skip `ifrt.local_view` calls: the callee runs with
        // `use_spmd_partitioning = false`, so a global view constant cannot be
        // cloned into a shared replica body.
        if (call_op->hasAttr(kIfrtLocalViewAttrName)) {
          continue;
        }
        // Any defined (single- or multi-block) SSACFG callee has a unique
        // dominating entry block `callee.getBody().front()`.
        mlir::func::FuncOp callee = call_op.getCalleeOp(symbol_table);

        // Collect used consumer entry-block arguments whose corresponding
        // `ifrt.Call` input traces back (through zero or more `ifrt.Reshard` or
        // `ifrt.CopyArrays` ops) to a zero-operand, single-result
        // `ConstantLike` op of matching tensor type.
        mlir::Block& callee_entry = callee.getBody().front();
        mlir::Operation* consumer_sym_table =
            mlir::SymbolTable::getNearestSymbolTable(callee);
        llvm::SmallVector<std::pair<unsigned, mlir::Operation*>>
            constants_to_copy;
        for (const auto& [arg_idx, input] :
             llvm::enumerate(call_op.getInputs())) {
          mlir::BlockArgument consumer_arg = callee_entry.getArgument(arg_idx);
          if (consumer_arg.use_empty()) {
            continue;
          }
          mlir::Operation* hlo_producer = FindHloProducer(input, symbol_table);
          if (hlo_producer == nullptr ||
              !hlo_producer->hasTrait<mlir::OpTrait::ConstantLike>() ||
              hlo_producer->getResult(0).getType() != consumer_arg.getType()) {
            continue;
          }
          // Skip constants carrying symbol references across `SymbolTable`
          // boundaries (e.g., `sdy.sharding` referencing a module-local
          // `sdy.mesh`), since those symbols would not resolve in the
          // consumer's `SymbolTable` or could collide with unrelated symbols.
          mlir::Operation* producer_sym_table =
              mlir::SymbolTable::getNearestSymbolTable(
                  hlo_producer->getParentOp());
          if (producer_sym_table != consumer_sym_table) {
            std::optional<mlir::SymbolTable::UseRange> uses =
                mlir::SymbolTable::getSymbolUses(hlo_producer);
            if (!uses.has_value() || !uses->empty()) {
              continue;
            }
          }
          constants_to_copy.push_back({arg_idx, hlo_producer});
        }

        if (constants_to_copy.empty()) {
          continue;
        }

        // If `callee` is shared or non-private, clone it (or its enclosing atom
        // `ModuleOp`) for `call_op` so other symbol users are unaffected.
        mlir::FailureOr<mlir::func::FuncOp> target_callee =
            symbol_manager->GetOrCloneCalleeForMutation({call_op});
        if (mlir::failed(target_callee)) {
          signalPassFailure();
          return;
        }
        callee = *target_callee;

        // Clone each `ConstantLike` op at the start of `callee`'s dominating
        // entry block and replace all uses of the corresponding block argument.
        // Dead arguments, `ifrt.Call` operands, and transfers are left for
        // `IfrtDcePass` to prune.
        mlir::Block& target_entry = callee.getBody().front();
        rewriter.setInsertionPointToStart(&target_entry);
        for (const auto& [arg_idx, hlo_producer] : constants_to_copy) {
          mlir::Operation* cloned_const = rewriter.clone(*hlo_producer);
          symbol_manager->RecordOpCreated(cloned_const);
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

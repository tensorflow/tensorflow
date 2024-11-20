/* Copyright 2024 The OpenXLA Authors.

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
#include <optional>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/transforms/passes.h"
#include "xla/python/ifrt/ir/transforms/utils.h"

namespace xla {
namespace ifrt {

namespace {

#define GEN_PASS_DEF_IFRTOUTLINEATOMPROGRAMTOMODULEPASS
#include "xla/python/ifrt/ir/transforms/passes.h.inc"

class IfrtOutlineAtomProgramToModulePass
    : public impl::IfrtOutlineAtomProgramToModulePassBase<
          IfrtOutlineAtomProgramToModulePass> {
 public:
  using impl::IfrtOutlineAtomProgramToModulePassBase<
      IfrtOutlineAtomProgramToModulePass>::
      IfrtOutlineAtomProgramToModulePassBase;

  void runOnOperation() override;
};

void IfrtOutlineAtomProgramToModulePass::runOnOperation() {
  mlir::SymbolTableCollection symbol_table;
  mlir::OpBuilder builder(&getContext());
  llvm::DenseSet<xla::ifrt::CallOp> visited;
  llvm::SmallVector<mlir::Operation*, 16> to_erase;
  mlir::ModuleOp module_op = getOperation();
  mlir::func::FuncOp main_func = GetMainFunction(module_op);
  auto result =
      main_func.walk([&](xla::ifrt::CallOp call_op) -> mlir::WalkResult {
        // Maybe visited by a previous CallOp with the same callee.
        if (visited.contains(call_op)) {
          return mlir::WalkResult::advance();
        }

        // Find the callee.
        mlir::func::FuncOp callee = call_op.getCalleeOp(symbol_table);
        if (callee.getSymName() == kCalleeMainFuncName &&
            llvm::isa<mlir::ModuleOp>(callee->getParentOp())) {
          // Atom program is already outlined in module. Do nothing.
          return mlir::WalkResult::advance();
        }

        // Create a ModuleOp and clone callee into it.
        builder.setInsertionPointAfter(callee);
        auto callee_module = builder.create<mlir::ModuleOp>(
            callee->getLoc(), callee.getSymName());
        callee_module.setVisibility(mlir::SymbolTable::Visibility::Private);

        mlir::func::FuncOp cloned_callee;
        // Find all symbols directly or indirectly referenced by callee and copy
        // them to the newly created module.
        {
          // Setup for DFS.
          llvm::DenseSet<mlir::func::FuncOp> visited_funcs;
          llvm::SmallVector<mlir::func::FuncOp, 8> func_stack = {callee};
          while (!func_stack.empty()) {
            mlir::func::FuncOp current_func = func_stack.back();
            func_stack.pop_back();
            if (!visited_funcs.insert(current_func).second) {
              continue;
            }

            // Copy function into the new module.
            mlir::func::FuncOp cloned_func =
                llvm::cast<mlir::func::FuncOp>(current_func->clone());
            if (current_func == callee) {
              cloned_callee = cloned_func;
              cloned_func.setSymName(kCalleeMainFuncName);
              cloned_func.setVisibility(mlir::SymbolTable::Visibility::Public);
            }
            builder.setInsertionPointToEnd(callee_module.getBody());
            builder.insert(cloned_func);

            // Check all symbols in function.
            std::optional<mlir::SymbolTable::UseRange> sym_uses =
                mlir::SymbolTable::getSymbolUses(current_func);
            if (!sym_uses.has_value()) {
              continue;
            }
            for (const mlir::SymbolTable::SymbolUse& sym_use : *sym_uses) {
              // Ensure the symbol represents a function.
              mlir::Operation* sym_op = module_op.lookupSymbol(
                  sym_use.getSymbolRef().getRootReference());
              if (sym_op == nullptr) {
                return sym_use.getUser()->emitOpError()
                       << "uses a symbol in attributes `"
                       << sym_use.getSymbolRef().getRootReference().str()
                       << "` that does not exist in the ModuleOp.";
              }
              auto func = llvm::dyn_cast<mlir::func::FuncOp>(sym_op);
              if (func == nullptr) {
                return sym_use.getUser()->emitOpError()
                       << "uses a symbol in attributes `"
                       << sym_use.getSymbolRef().getRootReference().str()
                       << "` that is not a FuncOp. Cannot handle such cases "
                          "for now.";
              }
              func_stack.push_back(func);
            }
          }
        }

        // Replace all uses of old callee.
        mlir::SymbolRefAttr new_symbol = mlir::SymbolRefAttr::get(
            callee_module.getSymNameAttr(),
            mlir::SymbolRefAttr::get(cloned_callee.getSymNameAttr()));
        // It is sufficient to get the symbols in the main func because
        // ifrt.Call nested within callees are not supported.
        std::optional<mlir::SymbolTable::UseRange> symbol_uses =
            callee.getSymbolUses(main_func);
        if (symbol_uses.has_value()) {
          for (const mlir::SymbolTable::SymbolUse symbol_use : *symbol_uses) {
            auto user = llvm::dyn_cast<xla::ifrt::CallOp>(symbol_use.getUser());
            if (user == nullptr) {
              return symbol_use.getUser()->emitOpError()
                     << "requires symbol `" << callee.getSymName()
                     << "` only used by ifrt.Call. Found use by `"
                     << user.getOperationName() << "`";
            }
            user.setCalleeAttr(new_symbol);
            visited.insert(user);
          }
        }

        // Can't erase callee yet during iteration.
        to_erase.push_back(callee);
        return mlir::WalkResult::advance();
      });

  if (result.wasInterrupted()) {
    signalPassFailure();
    return;
  }
  for (mlir::Operation* op : to_erase) {
    op->erase();
  }
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtOutlineAtomProgramToModulePass() {
  return std::make_unique<IfrtOutlineAtomProgramToModulePass>();
}

}  // namespace ifrt
}  // namespace xla

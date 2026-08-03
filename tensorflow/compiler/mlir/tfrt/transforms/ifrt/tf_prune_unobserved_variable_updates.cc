/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/host_runtime/tfrt_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

#define GEN_PASS_DEF_TFPRUNEUNOBSERVEDVARIABLEUPDATESPASS
#define GEN_PASS_DECL_TFPRUNEUNOBSERVEDVARIABLEUPDATESPASS
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/passes.h.inc"  // IWYU pragma: keep

// A variable is identified by the (container, shared_name) pair of its
// VarHandleOp.
using VariableKey = std::pair<mlir::StringAttr, mlir::StringAttr>;

bool IsResourceType(mlir::Type type) {
  return mlir::isa<mlir::TF::ResourceType>(mlir::getElementTypeOrSelf(type));
}

// Prunes self-update cycles of variables whose value is provably never
// observed, e.g. Keras metric accumulators
// (ReadVariableOp -> AddV2 -> AssignVariableOp on the same variable) that
// serving never reads back. Such variables block
// SinkVariableAsNamedArrayPass, which refuses to lower reads of any variable
// that is assigned somewhere in the module.
//
// The analysis fails closed: a variable is pruned only if every access to
// it, module-wide, is proven unobservable; any resource value that cannot be
// attributed to a VarHandleOp disables pruning for the whole module.
//
// Resource function arguments are attributed through their call sites (a
// SavedModel saved with a dedicated traced-restore/inference function passes
// variable handles as arguments): an argument gets a variable key only if
// every call site of the function passes a handle of the same variable.
class TfPruneUnobservedVariableUpdatesPass
    : public impl::TfPruneUnobservedVariableUpdatesPassBase<
          TfPruneUnobservedVariableUpdatesPass> {
 public:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // ------------------------------------------------------------------
    // Step 1: attribute every resource value in the module to a variable
    // key. Local attribution first; resource block arguments of functions
    // and Identity forwards are resolved by fixpoint below.
    // ------------------------------------------------------------------
    llvm::DenseMap<mlir::Value, VariableKey> value_to_key;
    // Resource values whose attribution is deferred to the fixpoint.
    llvm::SmallVector<mlir::TF::IdentityOp> pending_identities;
    llvm::SmallVector<mlir::BlockArgument> pending_args;
    // Resource values that can never be attributed.
    llvm::SmallVector<mlir::Value> unattributed;
    // Map from function symbol name to its call sites.
    llvm::DenseMap<llvm::StringRef, llvm::SmallVector<mlir::Operation*>>
        call_sites;

    module.walk([&](mlir::Operation* op) {
      if (auto call = llvm::dyn_cast<mlir::CallOpInterface>(op)) {
        if (auto sym = llvm::dyn_cast_or_null<mlir::SymbolRefAttr>(
                call.getCallableForCallee())) {
          call_sites[sym.getLeafReference().getValue()].push_back(op);
        }
      } else if (auto batch_func =
                     llvm::dyn_cast<mlir::TF::BatchFunctionOp>(op)) {
        call_sites[batch_func.getF().getLeafReference().getValue()].push_back(
            op);
      }
      for (mlir::Region& region : op->getRegions()) {
        for (mlir::Block& block : region) {
          for (mlir::BlockArgument arg : block.getArguments()) {
            if (!IsResourceType(arg.getType())) continue;
            // Only entry-block arguments of named functions can be resolved
            // through call sites. Region control-flow block arguments
            // (e.g. WhileRegion) are not supported and fail closed.
            if (llvm::isa<mlir::func::FuncOp>(op) &&
                &block == &region.front()) {
              pending_args.push_back(arg);
            } else {
              unattributed.push_back(arg);
            }
          }
        }
      }
      if (auto var_handle = llvm::dyn_cast<mlir::TF::VarHandleOp>(op)) {
        value_to_key[var_handle.getResult()] = {var_handle.getContainerAttr(),
                                                var_handle.getSharedNameAttr()};
        return;
      }
      if (auto identity = llvm::dyn_cast<mlir::TF::IdentityOp>(op)) {
        if (IsResourceType(identity.getOutput().getType())) {
          pending_identities.push_back(identity);
        }
        return;
      }
      // Only control-flow ops or call ops could return an aliased variable
      // handle that we cannot statically trace. All other resource-returning
      // ops (e.g. HashTableV2, FixtableSet, QuantileSet, IteratorV2, etc.)
      // produce non-variable resources and cannot alias a VarHandleOp.
      if (llvm::isa<mlir::TF::IfOp, mlir::TF::WhileOp, mlir::TF::CaseOp,
                    mlir::TF::LegacyCallOp, mlir::TF::PartitionedCallOp,
                    mlir::TF::StatefulPartitionedCallOp, mlir::TF::IfrtCallOp,
                    mlir::TF::BatchFunctionOp>(op)) {
        for (mlir::Value result : op->getResults()) {
          if (IsResourceType(result.getType())) {
            unattributed.push_back(result);
          }
        }
      }
    });

    // Fixpoint: resolve Identity forwards and function arguments until
    // nothing changes. An argument resolves only if the function has at
    // least one call site and every call site passes a handle attributed to
    // the same variable.
    bool changed = true;
    while (changed) {
      changed = false;
      llvm::erase_if(pending_identities, [&](mlir::TF::IdentityOp identity) {
        auto it = value_to_key.find(identity.getInput());
        if (it == value_to_key.end()) return false;
        value_to_key[identity.getOutput()] = it->second;
        changed = true;
        return true;
      });
      llvm::erase_if(pending_args, [&](mlir::BlockArgument arg) {
        auto func =
            llvm::cast<mlir::func::FuncOp>(arg.getOwner()->getParentOp());
        auto sites = call_sites.find(func.getSymName());
        if (sites == call_sites.end()) return false;
        std::optional<VariableKey> key;
        for (mlir::Operation* call : sites->second) {
          mlir::Operation::operand_range operands =
              llvm::isa<mlir::CallOpInterface>(call)
                  ? llvm::cast<mlir::CallOpInterface>(call).getArgOperands()
                  : call->getOperands();
          if (arg.getArgNumber() >= operands.size()) return false;
          auto it = value_to_key.find(operands[arg.getArgNumber()]);
          if (it == value_to_key.end()) return false;
          if (key.has_value() && *key != it->second) return false;
          key = it->second;
        }
        if (!key.has_value()) return false;
        value_to_key[arg] = *key;
        changed = true;
        return true;
      });
    }
    for (mlir::TF::IdentityOp identity : pending_identities) {
      unattributed.push_back(identity.getOutput());
    }
    for (mlir::BlockArgument arg : pending_args) {
      unattributed.push_back(arg);
    }

    if (!unattributed.empty()) {
      // Fail closed for the whole module: an unattributable resource may
      // alias any variable.
      constexpr int kMaxReported = 5;
      for (const auto& [i, value] : llvm::enumerate(unattributed)) {
        if (i >= kMaxReported) break;
        if (auto arg = llvm::dyn_cast<mlir::BlockArgument>(value)) {
          arg.getOwner()->getParentOp()->emitWarning()
              << "tf-prune-unobserved-variable-updates: resource block "
                 "argument #"
              << arg.getArgNumber()
              << " is not traceable to a VarHandleOp through its call sites";
        } else {
          value.getDefiningOp()->emitWarning()
              << "tf-prune-unobserved-variable-updates: resource result is "
                 "not traceable to a VarHandleOp";
        }
      }
      module.emitWarning()
          << "tf-prune-unobserved-variable-updates: found "
          << unattributed.size()
          << " resource value(s) not traceable to a VarHandleOp; "
             "conservatively keeping all variable updates";
      return;
    }

    // ------------------------------------------------------------------
    // Step 2: classify every use of every attributed resource value.
    // ------------------------------------------------------------------
    struct VariableInfo {
      bool observed = false;
      bool has_assign = false;
      // First op that blocked pruning, for diagnostics.
      mlir::Operation* blocker = nullptr;
      llvm::SmallVector<mlir::TF::ReadVariableOp> reads;
      llvm::SmallVector<mlir::Operation*> handles;
      mlir::StringAttr shared_name;
    };
    llvm::MapVector<VariableKey, VariableInfo> variables;

    for (const auto& [value, key] : value_to_key) {
      VariableInfo& info = variables[key];
      info.shared_name = key.second;
      if (llvm::isa_and_nonnull<mlir::TF::VarHandleOp>(value.getDefiningOp())) {
        info.handles.push_back(value.getDefiningOp());
      }
      for (mlir::Operation* user : value.getUsers()) {
        if (auto read = llvm::dyn_cast<mlir::TF::ReadVariableOp>(user)) {
          info.reads.push_back(read);
          continue;
        }
        if (llvm::isa<mlir::TF::AssignVariableOp>(user)) {
          // Writes never observe the variable.
          info.has_assign = true;
          continue;
        }
        if (llvm::isa<mlir::TF::IfrtRestoreVariableOp,
                      mlir::TF::VarIsInitializedOp, mlir::TF::DestroyResourceOp,
                      mlir::TF::IdentityOp>(user)) {
          // IfrtRestoreVariableOp is a write-only restore.
          // VarIsInitializedOp / DestroyResourceOp never observe the variable
          // value. Identity uses are analyzed through their own forwarded
          // value.
          continue;
        }
        if (auto call = llvm::dyn_cast<mlir::CallOpInterface>(user)) {
          // Passing the handle to a function whose corresponding argument is
          // attributed to the same variable is not an observation by itself:
          // the callee's uses of that argument are classified in this loop
          // like any other attributed value. (If any argument had failed to
          // attribute, the pass would have bailed out above.)
          bool all_args_match = true;
          mlir::Operation::operand_range operands = call.getArgOperands();
          for (const auto& [i, operand] : llvm::enumerate(operands)) {
            if (operand != value) continue;
            auto callee = llvm::dyn_cast_or_null<mlir::SymbolRefAttr>(
                call.getCallableForCallee());
            mlir::func::FuncOp callee_func =
                callee
                    ? mlir::SymbolTable::lookupNearestSymbolFrom<
                          mlir::func::FuncOp>(call, callee.getLeafReference())
                    : nullptr;
            if (!callee_func || i >= callee_func.getNumArguments()) {
              all_args_match = false;
              break;
            }
            auto it = value_to_key.find(callee_func.getArgument(i));
            if (it == value_to_key.end() || it->second != key) {
              all_args_match = false;
              break;
            }
          }
          if (all_args_match) continue;
        }
        if (!info.observed) info.blocker = user;
        info.observed = true;
      }
    }

    // ------------------------------------------------------------------
    // Step 3: for each unobserved variable, check that the value of every
    // read flows only through side-effect-free, region-free ops and
    // terminates exclusively in assignments back to the same variable. If
    // so, the reads, the intermediate pure ops and the assignments form a
    // cone that is unobservable and can be erased.
    // ------------------------------------------------------------------
    for (auto& [key, info] : variables) {
      if (info.reads.empty()) continue;

      llvm::SetVector<mlir::Operation*> cone;
      bool prunable = !info.observed;

      for (mlir::TF::ReadVariableOp read : info.reads) {
        if (!prunable) break;
        cone.insert(read);
        llvm::SmallVector<mlir::Value> worklist(read->result_begin(),
                                                read->result_end());
        while (prunable && !worklist.empty()) {
          mlir::Value value = worklist.pop_back_val();
          for (mlir::OpOperand& use : value.getUses()) {
            mlir::Operation* user = use.getOwner();
            if (auto assign =
                    llvm::dyn_cast<mlir::TF::AssignVariableOp>(user)) {
              auto it = value_to_key.find(assign.getResource());
              if (it != value_to_key.end() && it->second == key &&
                  use.get() == assign.getValue()) {
                cone.insert(assign);
                continue;
              }
              // The value is assigned to a different variable, which makes
              // it observable through that variable.
              if (!info.blocker) info.blocker = user;
              prunable = false;
              break;
            }
            if (mlir::isMemoryEffectFree(user) && user->getNumRegions() == 0 &&
                !user->hasTrait<mlir::OpTrait::IsTerminator>()) {
              if (cone.insert(user)) {
                worklist.append(user->result_begin(), user->result_end());
              }
              continue;
            }
            // Returned, passed to a call or side-effecting op, or enters
            // control flow: the value escapes.
            if (!info.blocker) info.blocker = user;
            prunable = false;
            break;
          }
        }
      }
      if (!prunable) {
        // Only mutable variables are interesting: an unpruned variable with
        // assigns will block SinkVariableAsNamedArrayPass from lowering its
        // reads, so say why it was kept.
        if (info.has_assign && info.blocker) {
          info.blocker->emitWarning()
              << "tf-prune-unobserved-variable-updates: variable '"
              << info.shared_name.getValue()
              << "' has assignments but is kept because its value escapes "
                 "through this op; its reads will not be lowered to "
                 "IfrtLoadVariableOp";
        }
        continue;
      }

      // The cone is closed under uses: every use of a cone op's result is
      // itself in the cone. Repeatedly erasing use-free ops therefore drains
      // the whole set without any topological bookkeeping.
      llvm::SmallVector<mlir::Operation*> pending(cone.begin(), cone.end());
      while (!pending.empty()) {
        llvm::SmallVector<mlir::Operation*> remaining;
        for (mlir::Operation* op : pending) {
          if (op->use_empty()) {
            op->erase();
          } else {
            remaining.push_back(op);
          }
        }
        if (remaining.size() == pending.size()) {
          module.emitError()
              << "tf-prune-unobserved-variable-updates: failed to erase a "
                 "closed use cone; this is a bug in the pass";
          return signalPassFailure();
        }
        pending = std::move(remaining);
      }

      // Clean up handles that no longer have any user.
      for (mlir::Operation* handle : info.handles) {
        if (handle->use_empty()) handle->erase();
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateTfPruneUnobservedVariableUpdatesPass() {
  return std::make_unique<TfPruneUnobservedVariableUpdatesPass>();
}

}  // namespace ifrt_serving
}  // namespace tensorflow

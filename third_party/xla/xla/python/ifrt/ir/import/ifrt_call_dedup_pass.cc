/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/import/passes.h"  // IWYU pragma: keep

namespace xla {
namespace ifrt {

#define GEN_PASS_DEF_IFRTCALLDEDUPPASS
#include "xla/python/ifrt/ir/import/passes.h.inc"

namespace {

// Returns the normalized `mlir::DictionaryAttr` at `index` from `attrs` (an
// `arg_attrs` or `res_attrs` `mlir::ArrayAttr`), or a null `mlir::Attribute` if
// `attrs` is null, `index` is out of bounds, or the dictionary is empty.
mlir::Attribute GetNormalizedDictAttrAt(mlir::ArrayAttr attrs, size_t index) {
  if (attrs == nullptr || index >= attrs.size()) {
    return mlir::Attribute();
  }
  auto dict = llvm::dyn_cast<mlir::DictionaryAttr>(attrs[index]);
  if (dict == nullptr || dict.empty()) {
    return mlir::Attribute();
  }
  return dict;
}

// Collects input indices of `call_op` that are donated or aliased to an output.
// These inputs cannot be deduplicated because each donated or aliased buffer
// has distinct ownership/in-place mutation semantics and cannot share an input
// slot.
llvm::SmallDenseSet<int32_t, 8> GetDonatedOrAliasedInputs(CallOp call_op) {
  llvm::SmallDenseSet<int32_t, 8> donated_or_aliased_idxs;
  donated_or_aliased_idxs.reserve(call_op.getDonatedInputIndices().size() +
                                  call_op.getIoAliases().size());
  donated_or_aliased_idxs.insert(call_op.getDonatedInputIndices().begin(),
                                 call_op.getDonatedInputIndices().end());
  for (const auto& io_alias :
       call_op.getIoAliases().getAsRange<mlir::DenseI32ArrayAttr>()) {
    donated_or_aliased_idxs.insert(io_alias.asArrayRef()[0]);
  }
  return donated_or_aliased_idxs;
}

// Collects output indices of `call_op` that are aliased to an input in
// `io_aliases`. Aliased outputs are excluded from output deduplication to
// preserve caller-visible buffer aliasing semantics.
llvm::SmallDenseSet<int32_t, 8> GetAliasedOutputs(CallOp call_op) {
  llvm::SmallDenseSet<int32_t, 8> aliased_idxs;
  aliased_idxs.reserve(call_op.getIoAliases().size());
  for (const auto& io_alias :
       call_op.getIoAliases().getAsRange<mlir::DenseI32ArrayAttr>()) {
    aliased_idxs.insert(io_alias.asArrayRef()[1]);
  }
  return aliased_idxs;
}

// Identifies duplicate non-donated, non-aliased inputs of `call_op`. Populates
// `first_idxs` so `first_idxs[idx]` is the index of the first compatible
// input matching `call_op.getInputs()[idx]` (or `idx` itself if unique or
// donated/aliased). Returns true if at least one duplicate `callee` block
// argument has uses that need to be redirected.
bool FindDuplicateArguments(CallOp call_op, mlir::func::FuncOp callee,
                            llvm::SmallVector<int32_t>& first_idxs) {
  llvm::SmallDenseSet<int32_t, 8> donated_or_aliased_idxs =
      GetDonatedOrAliasedInputs(call_op);
  mlir::Block& block = callee.getBody().front();
  mlir::ArrayAttr callee_arg_attrs = callee.getArgAttrsAttr();
  mlir::ArrayAttr call_arg_attrs = call_op.getArgAttrsAttr();
  bool found_duplicate_with_uses = false;
  // Two inputs are compatible duplicates iff they pass the same caller SSA
  // `mlir::Value`, have the same `callee` argument `mlir::Type`, and carry
  // identical normalized argument attributes on both `callee` and `call_op`.
  using Key = std::pair<std::pair<mlir::Value, mlir::Type>,
                        std::pair<mlir::Attribute, mlir::Attribute>>;
  llvm::DenseMap<Key, int32_t> index_of_first_occurrence;
  index_of_first_occurrence.reserve(call_op.getInputs().size());
  for (auto [idx, value] : llvm::enumerate(call_op.getInputs())) {
    // Donated or aliased inputs must remain distinct; map each to itself.
    if (donated_or_aliased_idxs.contains(idx)) {
      first_idxs.push_back(idx);
      continue;
    }
    Key key = {{value, block.getArgument(idx).getType()},
               {GetNormalizedDictAttrAt(callee_arg_attrs, idx),
                GetNormalizedDictAttrAt(call_arg_attrs, idx)}};
    auto [it, inserted] = index_of_first_occurrence.try_emplace(key, idx);
    if (inserted) {
      first_idxs.push_back(idx);
    } else {
      // Only redirect duplicate block arguments that are actually used inside
      // `callee`. Keeping unused duplicate arguments mapped to themselves
      // avoids unnecessary callee cloning.
      if (!block.getArgument(idx).use_empty()) {
        first_idxs.push_back(it->second);
        found_duplicate_with_uses = true;
      } else {
        first_idxs.push_back(idx);
      }
    }
  }
  return found_duplicate_with_uses;
}

// Returns true if `operand_num` falls within `inputs`.
bool IsInOperandRange(int32_t operand_num, mlir::OperandRange inputs) {
  return !inputs.empty() && operand_num >= inputs.getBeginOperandIndex() &&
         operand_num < inputs.getBeginOperandIndex() + inputs.size();
}

// Returns true if `operand` is donated or aliased to its owner operation.
bool IsDonatedOrAliasedOperand(mlir::OpOperand& operand) {
  mlir::Operation* user = operand.getOwner();
  const int32_t operand_num = operand.getOperandNumber();
  return llvm::TypeSwitch<mlir::Operation*, bool>(user)
      .Case<CallOp, CallLoadedExecutableOp>([&](auto call_like_op) {
        mlir::OperandRange inputs = call_like_op.getInputs();
        if (!IsInOperandRange(operand_num, inputs)) {
          return false;
        }
        const int32_t input_idx = operand_num - inputs.getBeginOperandIndex();
        if (llvm::is_contained(call_like_op.getDonatedInputIndices(),
                               input_idx)) {
          return true;
        }
        return llvm::any_of(call_like_op.getIoAliases()
                                .template getAsRange<mlir::DenseI32ArrayAttr>(),
                            [input_idx](mlir::DenseI32ArrayAttr io_alias) {
                              return io_alias.asArrayRef()[0] == input_idx;
                            });
      })
      .Case<CopyArraysOp>([&](CopyArraysOp copy_op) {
        // `reuse = true` aliases the input buffer, while `donated = true`
        // donates it.
        return IsInOperandRange(operand_num, copy_op.getInputs()) &&
               (copy_op.getDonated() || copy_op.getReuse());
      })
      .Case<ReshardOp>([&](ReshardOp reshard_op) {
        return IsInOperandRange(operand_num, reshard_op.getInputs()) &&
               reshard_op.getDonated();
      })
      .Case<BitcastArraysOp, RemapArraysOp, AssembleOp>([&](auto alias_op) {
        // `BitcastArrays`, `RemapArrays`, and `Assemble` are metadata/view ops
        // that always either donate (`donated = true`) or alias
        // (`donated = false`) their input buffers without copying.
        return IsInOperandRange(operand_num, alias_op.getInputs());
      })
      .Case<DisassembleOp>([&](DisassembleOp) {
        // `Disassemble` creates per-device shard views aliasing `input`
        // (operand 0).
        return operand_num == 0;
      })
      .Default(false);
}

// Returns true if `value` is donated or aliased in any user `mlir::OpOperand`.
bool IsDonatedOrAliasedByUser(mlir::Value value) {
  return llvm::any_of(value.getUses(), IsDonatedOrAliasedOperand);
}

// Redirects caller uses of duplicate non-aliased, non-donated `call_op` outputs
// to the first compatible output returning the same SSA value in `callee`. The
// now-unused duplicate outputs on `call_op` and `callee` are left in place to
// be pruned by `IfrtDcePass`.
void RemoveDuplicatedResults(CallOp call_op, mlir::func::FuncOp callee) {
  auto return_op = llvm::dyn_cast<mlir::func::ReturnOp>(
      callee.getBody().front().getTerminator());
  if (return_op == nullptr ||
      return_op.getNumOperands() != call_op.getOutputs().size()) {
    return;
  }

  llvm::SmallDenseSet<int32_t, 8> aliased_out_idxs = GetAliasedOutputs(call_op);
  mlir::ArrayAttr callee_res_attrs = callee.getResAttrsAttr();
  mlir::ArrayAttr call_res_attrs = call_op.getResAttrsAttr();
  // Two outputs are compatible duplicates iff they return the same `callee`
  // SSA `mlir::Value`, have the same caller `!ifrt.array` `mlir::Type`
  // (including sharding, devices, and memory kind), and carry identical
  // normalized result attributes on both `callee` and `call_op`.
  using Key = std::pair<std::pair<mlir::Value, mlir::Type>,
                        std::pair<mlir::Attribute, mlir::Attribute>>;
  llvm::DenseMap<Key, int32_t> index_of_first_occurrence;
  index_of_first_occurrence.reserve(call_op.getOutputs().size());
  llvm::SmallVector<int32_t> first_idxs;
  first_idxs.reserve(call_op.getOutputs().size());
  bool found_duplicate = false;
  for (auto [idx, ret_and_out] : llvm::enumerate(
           llvm::zip(return_op.getOperands(), call_op.getOutputs()))) {
    auto [ret_val, call_out] = ret_and_out;
    // Outputs aliased to an input of `call_op` or donated/aliased to a
    // downstream consumer must retain their own distinct buffers; map each to
    // itself so they neither merge into an earlier output nor absorb later
    // duplicates.
    if (aliased_out_idxs.contains(idx) || IsDonatedOrAliasedByUser(call_out)) {
      first_idxs.push_back(idx);
      continue;
    }
    Key key = {{ret_val, call_out.getType()},
               {GetNormalizedDictAttrAt(callee_res_attrs, idx),
                GetNormalizedDictAttrAt(call_res_attrs, idx)}};
    auto [it, inserted] = index_of_first_occurrence.try_emplace(key, idx);
    if (inserted) {
      first_idxs.push_back(idx);
    } else {
      found_duplicate = true;
      first_idxs.push_back(it->second);
    }
  }

  if (!found_duplicate) {
    return;
  }

  // Rewrite caller uses of each duplicate output to its canonical first
  // occurrence without mutating `callee` (so shared callees do not need to be
  // cloned for result deduplication).
  for (auto [idx, output] : llvm::enumerate(call_op.getOutputs())) {
    int32_t first_idx = first_idxs[idx];
    if (idx != first_idx) {
      output.replaceAllUsesWith(call_op.getOutputs()[first_idx]);
    }
  }
}

// Returns the top-level symbol operation in `root_module` corresponding to
// `callee` (`callee` itself when top-level, or its enclosing child
// `mlir::ModuleOp` when outlined), or `nullptr` if `callee` has an unsupported
// parent hierarchy.
mlir::Operation* GetRootSymbolOp(mlir::func::FuncOp callee,
                                 mlir::ModuleOp root_module) {
  mlir::Operation* parent = callee->getParentOp();
  if (parent == root_module) {
    return callee.getOperation();
  }
  if (auto callee_module = llvm::dyn_cast_or_null<mlir::ModuleOp>(parent);
      callee_module != nullptr && callee_module->getParentOp() == root_module) {
    return callee_module.getOperation();
  }
  return nullptr;
}

// Returns true if `callee` must be cloned before mutating its arguments in
// place: i.e., if top-level symbol uses are unknown, the root symbol is not
// private, the root symbol has uses other than this `ifrt.Call` (such as
// another `ifrt.Call` or a `func.call`), or an outlined `callee` is also
// referenced inside its enclosing `mlir::ModuleOp`.
bool ShouldCloneCallee(
    mlir::func::FuncOp callee, mlir::Operation* root_sym_op,
    const llvm::DenseMap<mlir::StringAttr, int>& symbol_use_counts,
    bool symbol_uses_known) {
  if (!symbol_uses_known || root_sym_op == nullptr) {
    return true;
  }
  if (mlir::SymbolTable::getSymbolVisibility(root_sym_op) !=
      mlir::SymbolTable::Visibility::Private) {
    return true;
  }
  mlir::StringAttr root_sym_name =
      mlir::SymbolTable::getSymbolName(root_sym_op);
  if (symbol_use_counts.lookup(root_sym_name) != 1) {
    return true;
  }
  if (auto callee_module = llvm::dyn_cast<mlir::ModuleOp>(root_sym_op);
      callee_module != nullptr &&
      !mlir::SymbolTable::symbolKnownUseEmpty(callee, callee_module)) {
    return true;
  }
  return false;
}

// Clones `callee` (or its enclosing nested `mlir::ModuleOp` when outlined) as a
// private symbol into `root_module` immediately after the original symbol,
// updates `call_op` to reference the clone, updates `symbol_use_counts` and
// `symbol_uses_known`, and returns the cloned `mlir::func::FuncOp`. Emits an op
// error on `call_op` and returns `nullptr` if `callee` cannot be cloned.
mlir::func::FuncOp CloneCalleeForCallOp(
    CallOp call_op, mlir::func::FuncOp callee, mlir::ModuleOp root_module,
    mlir::SymbolTableCollection& symbol_table,
    llvm::DenseMap<mlir::StringAttr, int>& symbol_use_counts,
    bool& symbol_uses_known) {
  mlir::Operation* parent = callee->getParentOp();
  // Case 1: `callee` is a top-level `func.func` in `root_module` (`@callee`).
  if (parent == root_module) {
    mlir::func::FuncOp cloned = callee.clone();
    mlir::SymbolTable::setSymbolVisibility(
        cloned, mlir::SymbolTable::Visibility::Private);
    mlir::StringAttr new_name =
        symbol_table.getSymbolTable(root_module)
            .insert(cloned, ++mlir::Block::iterator(callee));
    call_op.setCalleeAttr(mlir::FlatSymbolRefAttr::get(new_name));
    if (symbol_uses_known) {
      --symbol_use_counts[callee.getSymNameAttr()];
      symbol_use_counts[new_name] = 1;
      if (std::optional<mlir::SymbolTable::UseRange> cloned_uses =
              mlir::SymbolTable::getSymbolUses(cloned);
          cloned_uses.has_value()) {
        for (const mlir::SymbolTable::SymbolUse& use : *cloned_uses) {
          ++symbol_use_counts[use.getSymbolRef().getRootReference()];
        }
      } else {
        symbol_uses_known = false;
      }
    }
    return cloned;
  }
  // Case 2: `callee` is outlined inside a child `builtin.module` of
  // `root_module` (`@callee_module::@main`). Clone the entire nested module so
  // module-level attributes and helper symbols stay intact.
  if (auto callee_module = llvm::dyn_cast_or_null<mlir::ModuleOp>(parent);
      callee_module != nullptr && callee_module->getParentOp() == root_module) {
    mlir::ModuleOp cloned_module = callee_module.clone();
    mlir::SymbolTable::setSymbolVisibility(
        cloned_module, mlir::SymbolTable::Visibility::Private);
    mlir::SymbolTable& root_sym_table =
        symbol_table.getSymbolTable(root_module);
    mlir::StringAttr new_mod_name = root_sym_table.insert(
        cloned_module, ++mlir::Block::iterator(callee_module));
    mlir::SymbolTable& cloned_mod_sym_table =
        symbol_table.getSymbolTable(cloned_module);
    auto cloned_callee = llvm::dyn_cast_or_null<mlir::func::FuncOp>(
        cloned_mod_sym_table.lookup(callee.getSymNameAttr()));
    if (cloned_callee == nullptr) {
      symbol_table.invalidateSymbolTable(cloned_module);
      root_sym_table.erase(cloned_module);
      call_op.emitOpError()
          << "cannot clone callee " << call_op.getCalleeAttr()
          << " because target function was not found in cloned module";
      return nullptr;
    }
    // If `cloned_callee` is also referenced inside `cloned_module` (e.g., via
    // an internal or recursive `func.call`), keep `cloned_callee` as the
    // `@main` public entrypoint required by downstream IFRT passes and move the
    // unmutated copy into a private helper symbol, rewriting internal uses to
    // target the private helper.
    if (!mlir::SymbolTable::symbolKnownUseEmpty(cloned_callee, cloned_module)) {
      mlir::func::FuncOp unmutated_callee = cloned_callee.clone();
      mlir::SymbolTable::setSymbolVisibility(
          unmutated_callee, mlir::SymbolTable::Visibility::Private);
      mlir::StringAttr helper_name = cloned_mod_sym_table.insert(
          unmutated_callee, ++mlir::Block::iterator(cloned_callee));
      if (mlir::failed(mlir::SymbolTable::replaceAllSymbolUses(
              cloned_callee, helper_name, cloned_module))) {
        symbol_table.invalidateSymbolTable(cloned_module);
        root_sym_table.erase(cloned_module);
        call_op.emitOpError()
            << "cannot clone callee " << call_op.getCalleeAttr()
            << " because internal symbol uses could not be updated";
        return nullptr;
      }
    }
    call_op.setCalleeAttr(mlir::SymbolRefAttr::get(
        new_mod_name,
        {mlir::FlatSymbolRefAttr::get(cloned_callee.getSymNameAttr())}));
    if (symbol_uses_known) {
      --symbol_use_counts[callee_module.getSymNameAttr()];
      symbol_use_counts[new_mod_name] = 1;
    }
    return cloned_callee;
  }
  call_op.emitOpError()
      << "cannot clone callee " << call_op.getCalleeAttr()
      << " because only top-level functions and functions nested one "
         "module deep are supported";
  return nullptr;
}

// Memoized specialized callee created for a `(callee, first_arg_idxs)` pair so
// multiple `ifrt.Call` sites with identical duplicate-argument patterns share a
// single cloned callee.
struct SpecializedCallee {
  mlir::SymbolRefAttr callee_ref;
  mlir::func::FuncOp callee;
};

// Pass that deduplicates compatible non-donated/non-aliased operands and
// non-aliased/non-donated results of `ifrt.Call` ops, leaving dead operands and
// results to be erased by `IfrtDcePass`.
class IfrtCallDedupPass
    : public impl::IfrtCallDedupPassBase<IfrtCallDedupPass> {
 public:
  using impl::IfrtCallDedupPassBase<IfrtCallDedupPass>::IfrtCallDedupPassBase;

  void runOnOperation() override {
    mlir::ModuleOp module_op = getOperation();
    mlir::SymbolTableCollection symbol_table;

    // Count all top-level symbol uses across `module_op` (including non-IFRT
    // `func.call` references inside atom helpers) so shared or non-private
    // callees are cloned before mutating their block arguments.
    llvm::DenseMap<mlir::StringAttr, int> symbol_use_counts;
    bool symbol_uses_known = false;
    if (std::optional<mlir::SymbolTable::UseRange> uses =
            mlir::SymbolTable::getSymbolUses(&module_op.getBodyRegion());
        uses.has_value()) {
      symbol_uses_known = true;
      for (const mlir::SymbolTable::SymbolUse& use : *uses) {
        ++symbol_use_counts[use.getSymbolRef().getRootReference()];
      }
    }

    // Collect all `(call_op, callee)` pairs in IFRT entry functions upfront
    // before mutating `module_op` so cloning shared callees cannot invalidate
    // the top-level `FuncOp` iterator and each callee is resolved only once.
    llvm::SmallVector<std::pair<CallOp, mlir::func::FuncOp>> calls;
    for (mlir::func::FuncOp func_op : module_op.getOps<mlir::func::FuncOp>()) {
      if (!IsIfrtFunction(func_op)) {
        continue;
      }
      func_op.walk([&](CallOp call_op) {
        if (mlir::func::FuncOp callee = call_op.getCalleeOp(symbol_table);
            callee != nullptr && callee.getBody().hasOneBlock()) {
          calls.push_back({call_op, callee});
        }
      });
    }

    llvm::BumpPtrAllocator arg_idxs_allocator;
    llvm::DenseMap<std::pair<mlir::func::FuncOp, llvm::ArrayRef<int32_t>>,
                   SpecializedCallee>
        specialized_callees;

    // Process calls in program order so deduplicated producer outputs (e.g.,
    // `%0#1` rewritten to `%0#0`) become identical SSA inputs at downstream
    // consumer `ifrt.Call` ops in a single pass.
    for (auto [call_op, callee] : calls) {
      // 1. Deduplicate compatible `call_op` inputs by redirecting duplicate
      // block arguments inside `callee` to the first matching argument.
      llvm::SmallVector<int32_t> first_arg_idxs;
      first_arg_idxs.reserve(call_op.getInputs().size());
      if (FindDuplicateArguments(call_op, callee, first_arg_idxs)) {
        mlir::Operation* root_sym_op = GetRootSymbolOp(callee, module_op);
        if (auto it = specialized_callees.find({callee, first_arg_idxs});
            it != specialized_callees.end()) {
          // Reuse the existing specialized clone when another `ifrt.Call`
          // invokes the same `callee` with the same `first_arg_idxs`.
          const SpecializedCallee& specialized = it->second;
          call_op.setCalleeAttr(specialized.callee_ref);
          if (symbol_uses_known && root_sym_op != nullptr) {
            --symbol_use_counts[mlir::SymbolTable::getSymbolName(root_sym_op)];
            ++symbol_use_counts[specialized.callee_ref.getRootReference()];
          }
          callee = specialized.callee;
        } else {
          // Clone `callee` on write if it is non-private or referenced by any
          // other symbol user (such as another `ifrt.Call` or `func.call`).
          const bool cloned_on_write = ShouldCloneCallee(
              callee, root_sym_op, symbol_use_counts, symbol_uses_known);
          mlir::func::FuncOp orig_callee = callee;
          if (cloned_on_write) {
            mlir::func::FuncOp cloned =
                CloneCalleeForCallOp(call_op, callee, module_op, symbol_table,
                                     symbol_use_counts, symbol_uses_known);
            if (cloned == nullptr) {
              signalPassFailure();
              return;
            }
            callee = cloned;
          }
          mlir::Block& block = callee.getBody().front();
          for (int32_t idx = 0; idx < first_arg_idxs.size(); ++idx) {
            int32_t first_idx = first_arg_idxs[idx];
            if (idx != first_idx) {
              block.getArgument(idx).replaceAllUsesWith(
                  block.getArgument(first_idx));
            }
          }
          if (cloned_on_write) {
            int32_t* saved_idxs =
                arg_idxs_allocator.Allocate<int32_t>(first_arg_idxs.size());
            llvm::copy(first_arg_idxs, saved_idxs);
            specialized_callees.insert(
                {{orig_callee,
                  llvm::ArrayRef<int32_t>(saved_idxs, first_arg_idxs.size())},
                 {call_op.getCalleeAttr(), callee}});
          }
        }
      }

      // 2. Deduplicate compatible `call_op` outputs in the caller. Running
      // this after input deduplication also catches outputs that became
      // identical once duplicate `callee` block arguments were merged.
      RemoveDuplicatedResults(call_op, callee);
    }
  }
};

}  // namespace
}  // namespace ifrt
}  // namespace xla

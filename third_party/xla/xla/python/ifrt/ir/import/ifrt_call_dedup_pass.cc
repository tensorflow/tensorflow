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
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
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
#include "mlir/Support/LogicalResult.h"
#include "xla/python/ifrt/ir/ifrt_interfaces.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/import/callee_symbol_manager.h"
#include "xla/python/ifrt/ir/import/passes.h"  // IWYU pragma: keep
#include "xla/python/ifrt/ir/utils.h"

namespace xla {
namespace ifrt {

#define GEN_PASS_DEF_IFRTCALLDEDUPPASS
#include "xla/python/ifrt/ir/import/passes.h.inc"

namespace {

// A deduplicated callee variant cached for a `(orig_callee, first_arg_idxs)`
// specialization pattern.
struct SpecializedCallee {
  mlir::SymbolRefAttr callee_ref;
  mlir::func::FuncOp callee;
};

// Returns the normalized `mlir::DictionaryAttr` at `index` from `attrs` (an
// `arg_attrs` or `res_attrs` `mlir::ArrayAttr`), or a null `mlir::Attribute` if
// `attrs` is null or the dictionary at `index` is empty.
mlir::Attribute GetNormalizedDictAttrAt(mlir::ArrayAttr attrs, size_t index) {
  if (attrs == nullptr) {
    return mlir::Attribute();
  }
  CHECK_LT(index, attrs.size());
  auto dict = llvm::cast<mlir::DictionaryAttr>(attrs[index]);
  return dict.empty() ? mlir::Attribute() : mlir::Attribute(dict);
}

// Identifies duplicate non-donated, non-aliased inputs of `call_op`. Populates
// `first_idxs` so `first_idxs[idx]` is the index of the first compatible
// input matching `call_op.getInputs()[idx]` (or `idx` itself if unique or
// donated/aliased). Returns true if at least one duplicate `callee` block
// argument has uses that need to be redirected.
bool FindDuplicateArguments(CallOp call_op, mlir::func::FuncOp callee,
                            absl::InlinedVector<int32_t, 8>& first_idxs) {
  llvm::SmallDenseSet<int32_t, 8> donated_or_aliased_idxs =
      GetDonatedOrAliasedInputIndices(call_op);
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

  llvm::SmallDenseSet<int32_t, 8> aliased_out_idxs =
      GetAliasedOutputIndices(call_op);
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
    mlir::FailureOr<CalleeSymbolManager> symbol_manager =
        CalleeSymbolManager::Create(module_op, &symbol_table);
    if (mlir::failed(symbol_manager)) {
      signalPassFailure();
      return;
    }
    absl::flat_hash_map<
        std::pair<mlir::Operation*, absl::InlinedVector<int32_t, 8>>,
        SpecializedCallee>
        specialized_callees;

    // Collect all `(call_op, callee)` pairs in IFRT entry functions upfront
    // before mutating `module_op` so cloning shared callees cannot invalidate
    // the top-level `FuncOp` iterator and each callee is resolved only once.
    llvm::SmallVector<std::pair<CallOp, mlir::func::FuncOp>> calls;
    for (mlir::func::FuncOp func_op : GetIfrtFunctions(module_op)) {
      for (CallOp call_op : func_op.getOps<CallOp>()) {
        if (mlir::func::FuncOp callee = call_op.getCalleeOp(symbol_table);
            callee != nullptr && callee.getBody().hasOneBlock()) {
          calls.push_back({call_op, callee});
        }
      }
    }

    // Process calls in program order so deduplicated producer outputs (e.g.,
    // `%0#1` rewritten to `%0#0`) become identical SSA inputs at downstream
    // consumer `ifrt.Call` ops in a single pass.
    for (auto [call_op, callee] : calls) {
      // 1. Deduplicate compatible `call_op` inputs by redirecting duplicate
      // block arguments inside `callee` to the first matching argument.
      absl::InlinedVector<int32_t, 8> first_arg_idxs;
      first_arg_idxs.reserve(call_op.getInputs().size());
      if (FindDuplicateArguments(call_op, callee, first_arg_idxs)) {
        std::pair<mlir::Operation*, absl::InlinedVector<int32_t, 8>> key{
            callee.getOperation(), std::move(first_arg_idxs)};
        if (auto it = specialized_callees.find(key);
            it != specialized_callees.end()) {
          const SpecializedCallee& specialized = it->second;
          symbol_manager->RedirectCallToCallee(call_op, specialized.callee_ref);
          callee = specialized.callee;
        } else {
          mlir::func::FuncOp orig_callee = callee;
          mlir::FailureOr<mlir::func::FuncOp> target_callee =
              symbol_manager->GetOrCloneCalleeForMutation({call_op});
          if (mlir::failed(target_callee)) {
            signalPassFailure();
            return;
          }
          mlir::Block& block = target_callee->getBody().front();
          for (auto [idx, first_idx] : llvm::enumerate(key.second)) {
            if (static_cast<int32_t>(idx) != first_idx) {
              block.getArgument(idx).replaceAllUsesWith(
                  block.getArgument(first_idx));
            }
          }
          if (*target_callee != orig_callee) {
            specialized_callees.insert(
                {std::move(key),
                 SpecializedCallee{call_op.getCalleeAttr(), *target_callee}});
          }
          callee = *target_callee;
        }
      }

      // 2. Deduplicate compatible `call_op` outputs in the caller. Running
      // this after input deduplication also catches outputs that became
      // identical once duplicate `callee` block arguments were merged.
      RemoveDuplicatedResults(call_op, callee);
    }

    symbol_manager->EraseUnusedTrackedRootSymbols();
  }
};

}  // namespace
}  // namespace ifrt
}  // namespace xla

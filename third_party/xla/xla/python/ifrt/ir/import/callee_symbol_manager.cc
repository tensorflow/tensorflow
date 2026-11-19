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

#include "xla/python/ifrt/ir/import/callee_symbol_manager.h"

#include <optional>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/utils.h"

namespace xla {
namespace ifrt {
namespace {

// Converts an `llvm::StringRef` to an `absl::string_view` without copying.
absl::string_view ToStringView(llvm::StringRef str) {
  return absl::string_view(str.data(), str.size());
}

// Scans root-scope symbol uses in `op`, CHECK-failing on unregistered
// region-bearing ops.
mlir::SymbolTable::UseRange GetRootSymbolUses(mlir::Operation* op) {
  std::optional<mlir::SymbolTable::UseRange> uses =
      mlir::SymbolTable::getSymbolUses(op);
  CHECK(uses.has_value())
      << "Unregistered region-bearing op encountered in root module symbol "
         "scope while scanning `"
      << ToStringView(op->getName().getStringRef()) << "` ("
      << mlir::debugString(op->getLoc()) << ").";
  return *std::move(uses);
}

}  // namespace

mlir::FailureOr<CalleeSymbolManager> CalleeSymbolManager::Create(
    mlir::ModuleOp root_module,
    mlir::SymbolTableCollection* absl_nonnull symbol_table) {
  CHECK_NE(root_module, nullptr);
  CHECK_NE(symbol_table, nullptr);
  // Seed initial root-scope use counts from `root_module`'s body region
  // (stopping at nested `SymbolTable` boundaries). Fails if an unregistered
  // region-bearing op prevents determining symbol uses.
  std::optional<mlir::SymbolTable::UseRange> uses =
      mlir::SymbolTable::getSymbolUses(&root_module.getBodyRegion());
  if (!uses.has_value()) {
    root_module.emitOpError() << "cannot determine root-scope symbol uses";
    return mlir::failure();
  }
  CalleeSymbolManager manager(root_module, symbol_table);
  manager.AdjustSymbolUses(*uses, /*delta=*/1);
  return manager;
}

CalleeSymbolManager::CalleeSymbolManager(
    mlir::ModuleOp root_module,
    mlir::SymbolTableCollection* absl_nonnull symbol_table)
    : root_module_(root_module), symbol_table_(symbol_table) {}

mlir::FailureOr<mlir::func::FuncOp> CalleeSymbolManager::GetCallee(
    CallOp call_op) {
  CHECK_NE(call_op, nullptr);
  mlir::SymbolRefAttr callee_ref = call_op.getCalleeAttr();
  if (!IsInRootSymbolScope(call_op)) {
    call_op.emitOpError()
        << "cannot resolve callee " << callee_ref
        << " because the call is not in the root module's symbol scope";
    return mlir::failure();
  }
  mlir::func::FuncOp callee = call_op.getCalleeOp(*symbol_table_);
  if (callee == nullptr || callee.isExternal()) {
    call_op.emitOpError() << "failed to resolve defined callee " << callee_ref;
    return mlir::failure();
  }
  if (GetCalleeRootSymbolOp(callee, root_module_) == nullptr) {
    call_op.emitOpError()
        << "cannot resolve callee " << callee_ref
        << " because only top-level functions and functions nested one module "
           "deep are supported";
    return mlir::failure();
  }
  return callee;
}

mlir::FailureOr<mlir::func::FuncOp>
CalleeSymbolManager::GetOrCloneCalleeForMutation(
    absl::Span<const CallOp> calls) {
  CHECK(!calls.empty());
  CallOp first_call = calls.front();
  CHECK_NE(first_call, nullptr);
  mlir::SymbolRefAttr callee_ref = first_call.getCalleeAttr();
  llvm::SmallDenseSet<CallOp, 4> seen_calls;
  seen_calls.reserve(calls.size());
  for (CallOp call_op : calls) {
    CHECK_NE(call_op, nullptr);
    CHECK_EQ(call_op.getCalleeAttr(), callee_ref);
    CHECK(seen_calls.insert(call_op).second);
  }
  for (CallOp call_op : calls) {
    if (!IsInRootSymbolScope(call_op)) {
      call_op.emitOpError()
          << "cannot prepare callee " << call_op.getCalleeAttr()
          << " for mutation because the call is not in the root module's "
             "symbol scope";
      return mlir::failure();
    }
  }

  mlir::FailureOr<mlir::func::FuncOp> resolved_callee = GetCallee(first_call);
  if (mlir::failed(resolved_callee)) {
    return mlir::failure();
  }
  mlir::func::FuncOp callee = *resolved_callee;
  mlir::Operation* root_sym_op = GetCalleeRootSymbolOp(callee, root_module_);
  CHECK_NE(root_sym_op, nullptr);
  if (IsIfrtFunction(root_sym_op)) {
    first_call.emitOpError()
        << "cannot prepare callee " << callee_ref
        << " for mutation because top-level IFRT functions cannot be mutated";
    return mlir::failure();
  }

  bool has_internal_uses = false;
  if (mlir::ModuleOp orig_module = llvm::dyn_cast<mlir::ModuleOp>(root_sym_op);
      orig_module != nullptr) {
    // Check that all internal symbol uses inside `orig_module` are known
    // before cloning or rewriting (`SymbolTable::replaceAllSymbolUses` does
    // not fail on unknown potential symbol-table ops).
    std::optional<mlir::SymbolTable::UseRange> uses =
        mlir::SymbolTable::getSymbolUses(callee, orig_module);
    if (!uses.has_value()) {
      first_call.emitOpError()
          << "cannot prepare callee " << callee_ref
          << " for mutation because internal symbol uses could not be "
             "determined";
      return mlir::failure();
    }
    has_internal_uses = !uses->empty();
  }

  mlir::StringAttr root_sym_name =
      mlir::SymbolTable::getSymbolName(root_sym_op);
  const bool must_be_cloned =
      mlir::SymbolTable::getSymbolVisibility(root_sym_op) !=
          mlir::SymbolTable::Visibility::Private ||
      symbol_use_counts_.lookup(root_sym_name) !=
          static_cast<int>(calls.size());

  mlir::Operation* target_root = root_sym_op;
  mlir::func::FuncOp target_callee = callee;
  mlir::SymbolRefAttr new_callee_ref;
  if (must_be_cloned) {
    // Clone `root_sym_op` as a private symbol inserted immediately after it in
    // `root_module_`, resolving the cloned `FuncOp` and its `SymbolRefAttr`.
    target_root = root_sym_op->clone();
    mlir::SymbolTable::setSymbolVisibility(
        target_root, mlir::SymbolTable::Visibility::Private);
    mlir::StringAttr new_root_name =
        symbol_table_->getSymbolTable(root_module_)
            .insert(target_root, ++mlir::Block::iterator(root_sym_op));
    if (mlir::ModuleOp cloned_module =
            llvm::dyn_cast<mlir::ModuleOp>(target_root);
        cloned_module != nullptr) {
      target_callee = symbol_table_->lookupSymbolIn<mlir::func::FuncOp>(
          cloned_module, callee.getSymNameAttr());
      CHECK_NE(target_callee, nullptr)
          << "Cloned module @" << ToStringView(new_root_name.getValue())
          << " is missing expected callee @"
          << ToStringView(callee.getSymNameAttr().getValue()) << ".";
      new_callee_ref =
          MakeOutlinedCalleeSymbolRef(cloned_module, target_callee);
    } else {
      target_callee = llvm::cast<mlir::func::FuncOp>(target_root);
      new_callee_ref = mlir::FlatSymbolRefAttr::get(new_root_name);
    }
  }

  if (has_internal_uses) {
    mlir::ModuleOp callee_module = llvm::cast<mlir::ModuleOp>(target_root);
    // Keep `target_callee` under its original name (e.g. `@main` entrypoint)
    // and move the unmutated copy into a private helper symbol, rewriting
    // internal uses inside `callee_module` to target it.
    mlir::SymbolTable& mod_sym_table =
        symbol_table_->getSymbolTable(callee_module);
    mlir::func::FuncOp unmutated_callee = target_callee.clone();
    mlir::SymbolTable::setSymbolVisibility(
        unmutated_callee, mlir::SymbolTable::Visibility::Private);
    mlir::StringAttr helper_name = mod_sym_table.insert(
        unmutated_callee, ++mlir::Block::iterator(target_callee));
    // Cannot fail: `GetOrCloneCalleeForMutation` already verified that all
    // symbol uses in `callee_module` are known.
    CHECK(mlir::succeeded(mlir::SymbolTable::replaceAllSymbolUses(
        target_callee, helper_name, callee_module)))
        << "Failed to rewrite internal uses of @"
        << ToStringView(target_callee.getSymNameAttr().getValue()) << " to @"
        << ToStringView(helper_name.getValue()) << " inside module @"
        << ToStringView(callee_module.getSymName().value_or("<unnamed>"))
        << ".";
  }

  if (must_be_cloned) {
    for (CallOp call_op : calls) {
      RedirectCallToCallee(call_op, new_callee_ref);
    }
    AdjustSymbolUses(GetRootSymbolUses(target_root), /*delta=*/1);
  }
  return target_callee;
}

void CalleeSymbolManager::RedirectCallToCallee(
    CallOp call_op, mlir::SymbolRefAttr new_callee_ref) {
  CHECK(IsInRootSymbolScope(call_op));
  CHECK(new_callee_ref != nullptr);
  mlir::SymbolRefAttr old_callee_ref = call_op.getCalleeAttr();
  if (old_callee_ref == new_callee_ref) {
    return;
  }
  call_op.setCalleeAttr(new_callee_ref);
  mlir::StringAttr old_root = old_callee_ref.getRootReference();
  mlir::StringAttr new_root = new_callee_ref.getRootReference();
  if (old_root != new_root) {
    int& old_count = symbol_use_counts_[old_root];
    --old_count;
    CHECK_GE(old_count, 0) << "Symbol use count underflow for root symbol @"
                           << ToStringView(old_root.getValue())
                           << " while redirecting to @"
                           << ToStringView(new_root.getValue()) << ".";
    ++symbol_use_counts_[new_root];
    detached_root_symbols_.insert(old_root);
  }
}

void CalleeSymbolManager::RecordOpCreated(mlir::Operation* op) {
  CHECK_NE(op, nullptr);
  CHECK_NE(op->getParentOp(), nullptr);
  if (!IsInRootSymbolScope(op)) {
    return;
  }
  AdjustSymbolUses(GetRootSymbolUses(op), /*delta=*/1);
}

void CalleeSymbolManager::RecordOpErased(mlir::Operation* op) {
  CHECK_NE(op, nullptr);
  CHECK_NE(op->getParentOp(), nullptr);
  if (!IsInRootSymbolScope(op)) {
    return;
  }
  mlir::SymbolTable::UseRange uses = GetRootSymbolUses(op);
  AdjustSymbolUses(uses, /*delta=*/-1);
  for (const mlir::SymbolTable::SymbolUse& use : uses) {
    detached_root_symbols_.insert(use.getSymbolRef().getRootReference());
  }
}

void CalleeSymbolManager::RecordOpMutation(
    mlir::Operation* op, llvm::function_ref<void()> mutate_fn) {
  CHECK_NE(op, nullptr);
  CHECK_NE(op->getParentOp(), nullptr);
  if (!IsInRootSymbolScope(op)) {
    mutate_fn();
    return;
  }
  llvm::SmallDenseMap<mlir::StringAttr, int, 4> net_deltas;
  llvm::SmallVector<mlir::StringAttr, 4> before_roots;
  {
    mlir::SymbolTable::UseRange before_uses = GetRootSymbolUses(op);
    AdjustSymbolUses(before_uses, /*delta=*/-1);
    for (const mlir::SymbolTable::SymbolUse& use : before_uses) {
      mlir::StringAttr root_ref = use.getSymbolRef().getRootReference();
      auto [it, inserted] = net_deltas.insert({root_ref, 0});
      if (inserted) {
        before_roots.push_back(root_ref);
      }
      --it->second;
    }
  }
  mutate_fn();
  mlir::SymbolTable::UseRange after_uses = GetRootSymbolUses(op);
  AdjustSymbolUses(after_uses, /*delta=*/1);
  if (net_deltas.empty()) {
    return;
  }
  for (const mlir::SymbolTable::SymbolUse& use : after_uses) {
    mlir::StringAttr root_ref = use.getSymbolRef().getRootReference();
    if (auto it = net_deltas.find(root_ref); it != net_deltas.end()) {
      ++it->second;
    }
  }
  // Iterate `before_roots` to insert into `detached_root_symbols_` in
  // deterministic IR order regardless of `SmallDenseMap` hash ordering.
  for (mlir::StringAttr root_ref : before_roots) {
    if (net_deltas.lookup(root_ref) < 0) {
      detached_root_symbols_.insert(root_ref);
    }
  }
}

bool CalleeSymbolManager::EraseCallAndMaybeEraseCallee(
    CallOp call_op, mlir::RewriterBase& rewriter) {
  CHECK(IsInRootSymbolScope(call_op));
  CHECK(call_op->use_empty());
  mlir::StringAttr root_sym_name = call_op.getCalleeAttr().getRootReference();
  RecordOpErased(call_op);
  rewriter.eraseOp(call_op);
  return MaybeEraseUnusedRootSymbol(root_sym_name, &rewriter);
}

void CalleeSymbolManager::EraseUnusedTrackedRootSymbols() {
  llvm::SmallVector<mlir::StringAttr, 8> candidates =
      detached_root_symbols_.takeVector();
  for (mlir::StringAttr root_sym_name : candidates) {
    MaybeEraseUnusedRootSymbol(root_sym_name, /*rewriter=*/nullptr);
  }
  // Drain symbols transitively detached during the loop above so entries
  // erased by a later candidate in the same sweep are not left tracked.
  llvm::SmallVector<mlir::StringAttr, 8> newly_detached =
      detached_root_symbols_.takeVector();
  candidates.append(newly_detached.begin(), newly_detached.end());
  for (mlir::StringAttr root_sym_name : candidates) {
    if (symbol_use_counts_.lookup(root_sym_name) > 0) {
      detached_root_symbols_.insert(root_sym_name);
    }
  }
}

bool CalleeSymbolManager::IsInRootSymbolScope(mlir::Operation* op) const {
  if (op == nullptr || op->getParentOp() == nullptr) {
    return false;
  }
  mlir::ModuleOp root_module = root_module_;
  return mlir::SymbolTable::getNearestSymbolTable(op->getParentOp()) ==
         root_module.getOperation();
}

void CalleeSymbolManager::AdjustSymbolUses(
    const mlir::SymbolTable::UseRange& uses, int delta) {
  for (const mlir::SymbolTable::SymbolUse& use : uses) {
    mlir::StringAttr root_ref = use.getSymbolRef().getRootReference();
    int& count = symbol_use_counts_[root_ref];
    count += delta;
    CHECK_GE(count, 0) << "Symbol use count underflow for root symbol @"
                       << ToStringView(root_ref.getValue())
                       << " (delta=" << delta << ", missing RecordOpCreated?).";
  }
}

bool CalleeSymbolManager::MaybeEraseUnusedRootSymbol(
    mlir::StringAttr root_sym_name,
    mlir::RewriterBase* absl_nullable rewriter) {
  if (root_sym_name == nullptr) {
    return false;
  }
  mlir::SymbolTable& root_sym_table =
      symbol_table_->getSymbolTable(root_module_);
  bool erased_root = false;
  llvm::SmallVector<mlir::StringAttr, 4> worklist;
  llvm::SmallVector<mlir::StringAttr, 4> decremented;
  worklist.push_back(root_sym_name);
  while (!worklist.empty()) {
    mlir::StringAttr curr_name = worklist.pop_back_val();
    mlir::Operation* curr_op = root_sym_table.lookup(curr_name);
    if (curr_op == nullptr || IsIfrtFunction(curr_op) ||
        mlir::SymbolTable::getSymbolVisibility(curr_op) !=
            mlir::SymbolTable::Visibility::Private ||
        symbol_use_counts_.lookup(curr_name) != 0) {
      continue;
    }
    std::optional<mlir::SymbolTable::UseRange> internal_uses =
        mlir::SymbolTable::getSymbolUses(curr_op);
    CHECK(internal_uses.has_value())
        << "Unregistered region-bearing op encountered in root module symbol "
           "scope while erasing @"
        << ToStringView(curr_name.getValue()) << ".";
    for (const mlir::SymbolTable::SymbolUse& use : *internal_uses) {
      mlir::StringAttr referenced = use.getSymbolRef().getRootReference();
      int& count = symbol_use_counts_[referenced];
      --count;
      CHECK_GE(count, 0) << "Symbol use count underflow for root symbol @"
                         << ToStringView(referenced.getValue())
                         << " while erasing @"
                         << ToStringView(curr_name.getValue());
      if (count == 0) {
        worklist.push_back(referenced);
      } else {
        decremented.push_back(referenced);
      }
    }
    symbol_use_counts_.erase(curr_name);
    detached_root_symbols_.remove(curr_name);

    // Invalidate any symbol tables at or nested within `curr_op` before erasing
    // `curr_op` from the root symbol table.
    curr_op->walk([&](mlir::Operation* op) {
      if (op->hasTrait<mlir::OpTrait::SymbolTable>()) {
        symbol_table_->invalidateSymbolTable(op);
      }
    });
    if (rewriter != nullptr) {
      root_sym_table.remove(curr_op);
      rewriter->eraseOp(curr_op);
    } else {
      root_sym_table.erase(curr_op);
    }

    if (curr_name == root_sym_name) {
      erased_root = true;
    }
  }
  for (mlir::StringAttr ref : decremented) {
    if (symbol_use_counts_.lookup(ref) > 0) {
      detached_root_symbols_.insert(ref);
    }
  }
  return erased_root;
}

}  // namespace ifrt
}  // namespace xla

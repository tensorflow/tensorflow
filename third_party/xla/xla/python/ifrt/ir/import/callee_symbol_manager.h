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

#ifndef XLA_PYTHON_IFRT_IR_IMPORT_CALLEE_SYMBOL_MANAGER_H_
#define XLA_PYTHON_IFRT_IR_IMPORT_CALLEE_SYMBOL_MANAGER_H_

#include "absl/base/nullability.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"

namespace xla {
namespace ifrt {

// Tracks top-level symbol use counts in a root `ModuleOp` and provides
// copy-on-write callee cloning, call redirection, and dead-callee cleanup for
// passes that mutate or erase `ifrt.Call` ops and their atom-program callees.
//
// Intended for direct-IR passes using a plain `mlir::IRRewriter` without
// attached rewrite listeners: methods such as `GetOrCloneCalleeForMutation`,
// `RedirectCallToCallee`, and `EraseUnusedTrackedRootSymbols` mutate or erase
// ops outside `mlir::RewriterBase`, so listener-attached `PatternRewriter`s
// are not supported.
class CalleeSymbolManager {
 public:
  // Creates a `CalleeSymbolManager` after indexing root-scope symbol uses in
  // `root_module`. Emits an op error on `root_module` and returns
  // `mlir::failure()` if root-scope symbol uses cannot be determined (for
  // example, due to an unregistered region-bearing op).
  //
  // REQUIRES: `root_module` and `symbol_table` are non-null and outlive the
  // returned `CalleeSymbolManager`. Any root-level symbol insertions or
  // erasures done outside this manager must go through `symbol_table` so its
  // cached root `SymbolTable` stays in sync (otherwise clone names can collide
  // and erasure can look up a dangling op).
  static mlir::FailureOr<CalleeSymbolManager> Create(
      mlir::ModuleOp root_module,
      mlir::SymbolTableCollection* absl_nonnull symbol_table);

  CalleeSymbolManager(const CalleeSymbolManager&) = delete;
  CalleeSymbolManager& operator=(const CalleeSymbolManager&) = delete;
  CalleeSymbolManager(CalleeSymbolManager&&) = default;
  CalleeSymbolManager& operator=(CalleeSymbolManager&&) = default;

  // Returns `call_op`'s `FuncOp` callee from `symbol_table_` if it is either a
  // top-level function in `root_module_` or nested one `ModuleOp` deep and
  // `call_op` resides in `root_module_`'s symbol scope. Emits an op error on
  // `call_op` and returns `mlir::failure()` otherwise.
  //
  // REQUIRES: `call_op` is non-null, and its callee resolves to a defined
  // (non-external) `mlir::func::FuncOp` in `symbol_table_` (as guaranteed on
  // verified IFRT IR).
  mlir::FailureOr<mlir::func::FuncOp> GetCallee(CallOp call_op);

  // Returns a `FuncOp` that can be safely mutated for `calls` together: returns
  // the resolved callee directly when `calls` account for all uses of its
  // private root symbol, or clones a private copy and redirects every `CallOp`
  // in `calls` to the clone otherwise. When the callee is outlined inside a
  // child `ModuleOp`, any internal symbol references to it (such as recursive
  // calls) are isolated by splitting the unmutated copy into a private helper
  // in place (or in the clone) so internal callers are unaffected. Self-
  // references of a top-level `FuncOp` callee count as root-scope uses and
  // therefore force a clone. Emits an op error and returns `mlir::failure()` if
  // any `CallOp` is outside `root_module_`'s symbol scope, if the callee's root
  // symbol is a top-level IFRT function (`ifrt.function` or
  // `ifrt.reshard_function`, matching the non-erasure rule in
  // `EraseCallAndMaybeEraseCallee`), or if the callee cannot be prepared for
  // mutation (for example, nested more than one `ModuleOp` deep or inside an
  // outlined module with unregistered region-bearing ops).
  //
  // REQUIRES: `calls` is non-empty, and all `CallOp`s in `calls` are non-null,
  // distinct, and share the same callee `SymbolRefAttr`, which resolves to a
  // defined (non-external) `mlir::func::FuncOp` in `symbol_table_` (as
  // guaranteed on verified IFRT IR).
  mlir::FailureOr<mlir::func::FuncOp> GetOrCloneCalleeForMutation(
      absl::Span<const CallOp> calls);

  // Redirects `call_op` to an existing `new_callee_ref` in `root_module_` and
  // updates tracked symbol use counts.
  //
  // REQUIRES: `call_op` is non-null and resides in `root_module_`'s symbol
  // scope, and `new_callee_ref` is non-null and resolves to a valid callee
  // `mlir::func::FuncOp` in `symbol_table_`.
  void RedirectCallToCallee(CallOp call_op, mlir::SymbolRefAttr new_callee_ref);

  // Increments tracked root symbol use counts for any `SymbolRefAttr`s in `op`
  // when `op` resides in `root_module_`'s symbol scope (no-op for ops inside
  // nested `SymbolTable`s such as outlined atom-program modules). Passes that
  // insert or clone symbol-bearing ops outside `CalleeSymbolManager` must call
  // this immediately after inserting `op` into the IR; omitting it under-counts
  // uses, which can mutate a shared callee in place or erase a still-referenced
  // private symbol.
  //
  // REQUIRES: `op` is non-null, attached to a parent op, and introduces no
  // unregistered region-bearing ops in `root_module_`'s symbol scope.
  void RecordOpCreated(mlir::Operation* op);

  // Decrements tracked root symbol use counts for any `SymbolRefAttr`s in `op`
  // when `op` resides in `root_module_`'s symbol scope (no-op for ops inside
  // nested `SymbolTable`s), and records each decremented root symbol for
  // deferred cleanup by `EraseUnusedTrackedRootSymbols()`. Must be called
  // immediately before erasing `op`; omitting it over-counts uses, causing
  // extra clones or leftover private symbols.
  //
  // REQUIRES: `op` is non-null, attached to a parent op, and contains no
  // unregistered region-bearing ops in `root_module_`'s symbol scope.
  void RecordOpErased(mlir::Operation* op);

  // Decrements tracked root symbol use counts for `op`, runs `mutate_fn`, and
  // re-increments counts for `op`'s updated `SymbolRefAttr` uses. Root symbols
  // with a net negative count delta across `mutate_fn` are recorded for
  // deferred cleanup by `EraseUnusedTrackedRootSymbols()`; unchanged and newly
  // added symbol references are not added, and pre-existing zero-use private
  // symbols remain untouched. Use this to keep cached counts in sync across
  // in-place mutations to `op` or its nested regions (for example, region
  // simplification or modifying/erasing attributes carrying symbol references)
  // when individual op edits cannot be tracked via `RecordOpCreated` /
  // `RecordOpErased`. No-op wrapper for ops inside nested `SymbolTable`s.
  //
  // REQUIRES: `op` is non-null and attached to a parent op. `mutate_fn` must
  // not erase `op`, invoke `CalleeSymbolManager` methods, or introduce
  // unregistered region-bearing ops in `root_module_`'s symbol scope.
  void RecordOpMutation(mlir::Operation* op,
                        llvm::function_ref<void()> mutate_fn);

  // Updates tracked symbol use counts for `call_op` (via `RecordOpErased`),
  // erases `call_op` via `rewriter`, and erases `call_op`'s top-level root
  // symbol (plus any transitively unused private non-IFRT root symbols) from
  // `root_module_` via `rewriter` if it is private, not an IFRT function, and
  // now unused. Any other root symbols decremented by `call_op` (for example,
  // discardable symbol-reference attributes on `call_op`) remain tracked for
  // deferred cleanup by `EraseUnusedTrackedRootSymbols()`. Erasing through
  // `rewriter` adjusts `rewriter`'s insertion point if it sits on an erased
  // root symbol. Returns true if `call_op`'s root symbol was erased.
  //
  // REQUIRES: `call_op` is non-null, resides in `root_module_`'s symbol scope,
  // and all of its results (including the control output) have no remaining
  // uses. `rewriter`'s insertion point is not inside a block nested within an
  // erased root symbol.
  bool EraseCallAndMaybeEraseCallee(CallOp call_op,
                                    mlir::RewriterBase& rewriter);

  // Sweeps all root symbols previously detached by `RedirectCallToCallee`,
  // `RecordOpErased`, `RecordOpMutation` (net negative delta),
  // `EraseCallAndMaybeEraseCallee`, or transitive symbol erasure and erases
  // any private non-IFRT root symbols (plus transitively unused private
  // non-IFRT root symbols) whose use counts have reached zero. Symbols that
  // still have remaining uses stay tracked for later sweeps. Pre-existing
  // zero-use private symbols that were never decremented remain untouched.
  void EraseUnusedTrackedRootSymbols();

 private:
  CalleeSymbolManager(mlir::ModuleOp root_module,
                      mlir::SymbolTableCollection* absl_nonnull symbol_table);

  // Returns true if `op` is non-null, attached, and its nearest enclosing
  // `SymbolTable` is `root_module_`.
  bool IsInRootSymbolScope(mlir::Operation* op) const;

  // Adds `delta` to `symbol_use_counts_` for each root symbol referenced in
  // `uses`.
  void AdjustSymbolUses(const mlir::SymbolTable::UseRange& uses, int delta);

  // Erases `root_sym_name` (and any transitively unused private non-IFRT root
  // symbols) from `root_module_` if it is private, not an IFRT function, and
  // has zero uses. When `rewriter` is non-null, unregisters each erased symbol
  // from the root `SymbolTable` and deletes it via `rewriter->eraseOp` so
  // `rewriter`'s insertion point is adjusted if it sits on an erased op.
  // Returns true if `root_sym_name` itself was erased.
  bool MaybeEraseUnusedRootSymbol(mlir::StringAttr root_sym_name,
                                  mlir::RewriterBase* absl_nullable rewriter);

  mlir::ModuleOp root_module_;
  mlir::SymbolTableCollection* absl_nonnull symbol_table_;
  // Cached count of root-scope `SymbolRefAttr` uses per top-level symbol.
  // Callers must keep this cache in sync via `RecordOpCreated`,
  // `RecordOpErased`, and `RecordOpMutation`.
  llvm::DenseMap<mlir::StringAttr, int> symbol_use_counts_;
  // Root symbols detached by redirection, erasure, mutation (net negative
  // delta), or transitive symbol erasure, swept by
  // `EraseUnusedTrackedRootSymbols()`.
  llvm::SmallSetVector<mlir::StringAttr, 8> detached_root_symbols_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_IMPORT_CALLEE_SYMBOL_MANAGER_H_

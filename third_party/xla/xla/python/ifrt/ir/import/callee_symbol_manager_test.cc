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

#include <cstddef>
#include <optional>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/python/ifrt/ir/ifrt_ir_test_base.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/utils.h"

namespace xla {
namespace ifrt {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::AnyOf;
using ::testing::HasSubstr;
using ::testing::Ne;

constexpr absl::string_view kPrivateF0 =
    "  func.func private @f0() { return }\n";
constexpr absl::string_view kPrivateF1 =
    "  func.func private @f1() { return }\n";
constexpr absl::string_view kPublicFn = "  func.func @pub() { return }\n";
constexpr absl::string_view kUnregRegionOp =
    R"mlir("unregistered.region_op"() ({ ^bb0: }) : () -> ())mlir";
constexpr absl::string_view kMultiFuncMod = R"mlir(
  module @multi_mod <sym_visibility = "private"> {
    func.func @main() { return }
    func.func @other() { return }
  }
)mlir";
constexpr absl::string_view kNestedCallMod = R"mlir(
  module @nested_call_mod <sym_visibility = "private"> {
    func.func @f0() { return }
    func.func @nested_ifrt() attributes {ifrt.function} {
      %0 = ifrt.Call @f0() on devices [0] : () -> ()
      return
    }
  }
)mlir";

class CalleeSymbolManagerTest : public test_util::IfrtIrTestBase {
 protected:
  // Parses a module containing `decls` and a `@main` function with `main_body`,
  // constructs `symbol_manager_`, and collects `@main`'s `ifrt.Call` ops.
  absl::Status InitModule(absl::string_view decls,
                          absl::string_view main_body) {
    ABSL_ASSIGN_OR_RETURN(module_,
                     LoadFromSource(absl::StrCat(
                         "module {\n", decls,
                         "\nfunc.func @main() attributes {ifrt.function} {\n",
                         main_body, "\nreturn\n}\n}\n")));
    mlir::BaseScopedDiagnosticHandler diag(&mlir_context_);
    mlir::FailureOr<CalleeSymbolManager> symbol_manager =
        CalleeSymbolManager::Create(*module_, &symbol_table_);
    if (mlir::failed(symbol_manager)) {
      return diag.Combine(
          absl::InvalidArgumentError("Failed to create CalleeSymbolManager."));
    }
    ABSL_RETURN_IF_ERROR(diag.ConsumeStatus());
    symbol_manager_ = *std::move(symbol_manager);
    calls_.assign(GetMainFunction(*module_).getOps<CallOp>().begin(),
                  GetMainFunction(*module_).getOps<CallOp>().end());
    return absl::OkStatus();
  }

  // Parses a module containing `decls` and one `ifrt.Call` in `@main` per entry
  // in `callees`, verifying `calls_.size() == callees.size()`.
  absl::Status InitModuleWithCalls(
      absl::string_view decls, absl::Span<const absl::string_view> callees) {
    std::string main_body;
    for (size_t i = 0; i < callees.size(); ++i) {
      absl::StrAppend(&main_body, "    %", i, " = ifrt.Call ", callees[i],
                      "() on devices [0] : () -> ()\n");
    }
    ABSL_RETURN_IF_ERROR(InitModule(decls, main_body));
    if (calls_.size() != callees.size()) {
      return absl::InternalError(absl::StrCat("Expected ", callees.size(),
                                              " calls, got ", calls_.size()));
    }
    return absl::OkStatus();
  }

  void TearDown() override {
    if (module_.get() != nullptr) {
      EXPECT_TRUE(mlir::succeeded(mlir::verify(*module_)));
    }
  }

  // Converts a `FailureOr<FuncOp>` and any diagnostics captured in `diag` into
  // `absl::StatusOr<FuncOp>`, failing if the result failed or an error
  // diagnostic was emitted.
  static absl::StatusOr<mlir::func::FuncOp> CaptureResult(
      mlir::FailureOr<mlir::func::FuncOp> result,
      mlir::BaseScopedDiagnosticHandler& diag) {
    absl::Status status = diag.ConsumeStatus();
    if (mlir::failed(result)) {
      if (status.ok()) {
        return absl::InternalError("Operation failed without diagnostic.");
      }
      return status;
    }
    ABSL_RETURN_IF_ERROR(status);
    return *result;
  }

  // Resolves `call_op`'s callee via `CalleeSymbolManager::GetCallee`, capturing
  // any emitted MLIR diagnostic into the returned `absl::Status`.
  absl::StatusOr<mlir::func::FuncOp> GetCallee(CallOp call_op) {
    mlir::BaseScopedDiagnosticHandler diag(&mlir_context_);
    return CaptureResult(symbol_manager_->GetCallee(call_op), diag);
  }

  // Prepares `calls`'s callee for mutation via
  // `CalleeSymbolManager::GetOrCloneCalleeForMutation`, capturing any emitted
  // MLIR diagnostic into the returned `absl::Status`.
  absl::StatusOr<mlir::func::FuncOp> GetOrClone(
      absl::Span<const CallOp> calls) {
    mlir::BaseScopedDiagnosticHandler diag(&mlir_context_);
    return CaptureResult(symbol_manager_->GetOrCloneCalleeForMutation(calls),
                         diag);
  }

  // Erases `call_op` via `CalleeSymbolManager::EraseCallAndMaybeEraseCallee`
  // using `rewriter_`.
  bool EraseCall(CallOp call_op) {
    return symbol_manager_->EraseCallAndMaybeEraseCallee(call_op, rewriter_);
  }

  // Returns true if `module_` contains a top-level symbol named `name`.
  bool HasSymbol(llvm::StringRef name) const {
    return module_.get().lookupSymbol(name) != nullptr;
  }

  // Returns the single `ifrt.Call` inside `@nested_call_mod::@nested_ifrt`.
  CallOp FindNestedCall() const {
    mlir::ModuleOp nested_mod =
        module_.get().lookupSymbol<mlir::ModuleOp>("nested_call_mod");
    if (nested_mod == nullptr) {
      return nullptr;
    }
    mlir::func::FuncOp nested_ifrt =
        nested_mod.lookupSymbol<mlir::func::FuncOp>("nested_ifrt");
    if (nested_ifrt == nullptr) {
      return nullptr;
    }
    llvm::SmallVector<CallOp> nested_calls(nested_ifrt.getOps<CallOp>());
    return nested_calls.size() == 1u ? nested_calls.front() : nullptr;
  }

  mlir::IRRewriter rewriter_{&mlir_context_};
  mlir::SymbolTableCollection symbol_table_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  std::optional<CalleeSymbolManager> symbol_manager_;
  llvm::SmallVector<CallOp> calls_;
};

TEST_F(CalleeSymbolManagerTest, ValidatesCalleeScope) {
  ASSERT_OK(InitModuleWithCalls(
      absl::StrCat(kPrivateF0, kMultiFuncMod, kNestedCallMod),
      {"@f0", "@multi_mod::@main"}));

  EXPECT_OK(GetCallee(calls_[0]));
  EXPECT_OK(GetCallee(calls_[1]));

  // Both `GetCallee` and `GetOrClone` reject calls outside root symbol scope.
  CallOp nested_call = FindNestedCall();
  ASSERT_NE(nested_call, nullptr);
  constexpr absl::string_view kScopeError =
      "not in the root module's symbol scope";
  EXPECT_THAT(GetCallee(nested_call),
              StatusIs(absl::StatusCode::kUnknown, HasSubstr(kScopeError)));
  EXPECT_THAT(GetOrClone({nested_call}),
              StatusIs(absl::StatusCode::kUnknown, HasSubstr(kScopeError)));
  EXPECT_THAT(GetOrClone({calls_[0], nested_call}),
              StatusIs(absl::StatusCode::kUnknown, HasSubstr(kScopeError)));
}

TEST_F(CalleeSymbolManagerTest,
       PrivateFlatCalleeMutatesInPlaceAndPublicClonesAsPrivate) {
  ASSERT_OK(InitModuleWithCalls(absl::StrCat(kPrivateF0, kPublicFn),
                                {"@f0", "@pub"}));

  ASSERT_OK_AND_ASSIGN(mlir::func::FuncOp priv_fn, GetCallee(calls_[0]));
  ASSERT_OK_AND_ASSIGN(mlir::func::FuncOp pub_fn, GetCallee(calls_[1]));

  EXPECT_THAT(GetOrClone({calls_[0]}), IsOkAndHolds(priv_fn));

  ASSERT_OK_AND_ASSIGN(mlir::func::FuncOp cloned_pub, GetOrClone({calls_[1]}));
  EXPECT_NE(cloned_pub, pub_fn);
  EXPECT_EQ(mlir::SymbolTable::getSymbolVisibility(cloned_pub),
            mlir::SymbolTable::Visibility::Private);
}

TEST_F(CalleeSymbolManagerTest,
       MultiCallGroupMutatesInPlaceWhenAllUsesCoveredAndClonesOnceWhenShared) {
  ASSERT_OK(InitModuleWithCalls(kPrivateF0, {"@f0", "@f0", "@f0"}));

  ASSERT_OK_AND_ASSIGN(mlir::func::FuncOp f0, GetCallee(calls_[0]));

  // Two of three calls clone once and redirect both calls to the same clone.
  ASSERT_OK_AND_ASSIGN(mlir::func::FuncOp group_clone,
                       GetOrClone({calls_[0], calls_[1]}));
  EXPECT_NE(group_clone, f0);
  EXPECT_EQ(calls_[0].getCalleeAttr(), calls_[1].getCalleeAttr());

  // Both calls now cover all uses of `group_clone`, so they mutate in place.
  EXPECT_THAT(GetOrClone({calls_[0], calls_[1]}), IsOkAndHolds(group_clone));

  // The sole remaining call claims `@f0` in place.
  EXPECT_THAT(GetOrClone({calls_[2]}), IsOkAndHolds(f0));
}

TEST_F(CalleeSymbolManagerTest, RedirectCallAndEraseUnusedTrackedRootSymbols) {
  ASSERT_OK(InitModuleWithCalls(absl::StrCat(kPrivateF0, kPrivateF1),
                                {"@f0", "@f1"}));

  // Redirecting a call to its current callee is a no-op.
  mlir::SymbolRefAttr f0_ref = calls_[0].getCalleeAttr();
  symbol_manager_->RedirectCallToCallee(calls_[0], f0_ref);
  EXPECT_EQ(calls_[0].getCalleeAttr(), f0_ref);
  symbol_manager_->EraseUnusedTrackedRootSymbols();
  EXPECT_TRUE(HasSymbol("f0"));

  symbol_manager_->RedirectCallToCallee(calls_[1], f0_ref);
  EXPECT_EQ(calls_[1].getCalleeAttr(), f0_ref);
  symbol_manager_->EraseUnusedTrackedRootSymbols();
  EXPECT_FALSE(HasSymbol("f1"));
  EXPECT_TRUE(HasSymbol("f0"));
  // Sweeping again is an idempotent no-op.
  symbol_manager_->EraseUnusedTrackedRootSymbols();

  EXPECT_FALSE(EraseCall(calls_[0]));
  EXPECT_TRUE(HasSymbol("f0"));
  EXPECT_TRUE(EraseCall(calls_[1]));
  EXPECT_FALSE(HasSymbol("f0"));
}

TEST_F(CalleeSymbolManagerTest,
       NonCallReferencesPreventInPlaceMutationUntilErased) {
  ASSERT_OK(InitModule(absl::StrCat(kPrivateF0, kPrivateF1, R"mlir(
  func.func private @caller_helper() {
    func.call @f0() : () -> ()
    func.call @f1() : () -> ()
    return
  }
  func.func private @attr_dup() { return }
  func.func private @attr_dup_erase() { return }
)mlir"),
                       R"mlir(
    %0 = ifrt.Call @f0() on devices [0] : () -> ()
    %1 = ifrt.Call @f1() on devices [0] : () -> ()
    %2 = ifrt.Call @caller_helper() on devices [0] : () -> ()
    %3 = ifrt.Call @attr_dup() on devices [0] {test.origin = @attr_dup}
        : () -> ()
    %4 = ifrt.Call @attr_dup_erase() on devices [0]
        {test.origin = @attr_dup_erase} : () -> ()
)mlir"));

  ASSERT_EQ(calls_.size(), 5u);
  ASSERT_OK_AND_ASSIGN(mlir::func::FuncOp f0, GetCallee(calls_[0]));
  ASSERT_OK_AND_ASSIGN(mlir::func::FuncOp f1, GetCallee(calls_[1]));
  ASSERT_OK_AND_ASSIGN(mlir::func::FuncOp attr_dup, GetCallee(calls_[3]));

  // While `@caller_helper` holds `func.call @f1`, `GetOrClone` must clone `@f1`
  // rather than mutating it in place, and `@f1` remains alive until
  // `@caller_helper` is erased.
  EXPECT_THAT(GetOrClone({calls_[1]}), IsOkAndHolds(Ne(f1)));
  symbol_manager_->EraseUnusedTrackedRootSymbols();
  EXPECT_TRUE(HasSymbol("f1"));

  // Erasing `@caller_helper` decrements internal `func.call` uses on `@f0`
  // (making it mutable in place) and transitively erases the now-unused `@f1`.
  EXPECT_TRUE(EraseCall(calls_[2]));
  EXPECT_FALSE(HasSymbol("caller_helper"));
  EXPECT_FALSE(HasSymbol("f1"));
  EXPECT_THAT(GetOrClone({calls_[0]}), IsOkAndHolds(f0));

  // While `calls_[3]` holds `{test.origin = @attr_dup}`, `GetOrClone` must
  // clone `@attr_dup` and keep the original alive until the discardable
  // attribute is removed.
  EXPECT_THAT(GetOrClone({calls_[3]}), IsOkAndHolds(Ne(attr_dup)));
  symbol_manager_->EraseUnusedTrackedRootSymbols();
  EXPECT_TRUE(HasSymbol("attr_dup"));

  symbol_manager_->RecordOpMutation(
      calls_[3], [&]() { calls_[3]->removeAttr("test.origin"); });
  symbol_manager_->EraseUnusedTrackedRootSymbols();
  EXPECT_FALSE(HasSymbol("attr_dup"));

  // Extra discardable `SymbolRefAttr` (`test.origin = @attr_dup_erase`) raises
  // the count to 2; erasing `calls_[4]` decrements both references and erases
  // `@attr_dup_erase` in one step.
  EXPECT_TRUE(EraseCall(calls_[4]));
  EXPECT_FALSE(HasSymbol("attr_dup_erase"));
}

TEST_F(CalleeSymbolManagerTest, EraseCallTransitivelyErasesPrivateChain) {
  ASSERT_OK(InitModuleWithCalls(absl::StrCat(kPublicFn, R"mlir(
  func.func private @priv_ifrt() attributes {ifrt.function} { return }
  func.func private @priv_reshard() attributes {ifrt.reshard_function} {
    return
  }
  func.func private @leaf()
      attributes {test.ref = @priv_ifrt, test.ref2 = @priv_reshard} {
    func.call @pub() : () -> ()
    return
  }
  func.func private @mid() {
    func.call @leaf() : () -> ()
    return
  }
  func.func private @top() {
    func.call @mid() : () -> ()
    return
  }
)mlir"),
                                {"@top"}));

  EXPECT_TRUE(EraseCall(calls_[0]));
  EXPECT_FALSE(HasSymbol("top"));
  EXPECT_FALSE(HasSymbol("mid"));
  EXPECT_FALSE(HasSymbol("leaf"));
  EXPECT_TRUE(HasSymbol("pub"));
  EXPECT_TRUE(HasSymbol("priv_ifrt"));
  EXPECT_TRUE(HasSymbol("priv_reshard"));
}

TEST_F(CalleeSymbolManagerTest,
       EraseUnusedTrackedRootSymbolsSweepsDeferredAndRedirectedCallees) {
  ASSERT_OK(InitModuleWithCalls(absl::StrCat(kPrivateF0, kPrivateF1, R"mlir(
  func.func private @redirect_target() { return }
  func.func private @attr_holder() attributes {test.ref = @f0} { return }
)mlir"),
                                {"@f0", "@f1", "@f1", "@redirect_target"}));

  mlir::func::FuncOp attr_holder =
      module_->lookupSymbol<mlir::func::FuncOp>("attr_holder");
  ASSERT_NE(attr_holder, nullptr);

  // Erasing `calls_[0]` cannot erase `@f0` yet because `@attr_holder` still
  // references it; redirecting `calls_[1]` detaches `@f1` while `calls_[2]`
  // still uses it.
  EXPECT_FALSE(EraseCall(calls_[0]));
  symbol_manager_->RedirectCallToCallee(calls_[1], calls_[3].getCalleeAttr());

  // First sweep leaves both `@f0` and `@f1` alive (retaining them as tracked).
  symbol_manager_->EraseUnusedTrackedRootSymbols();
  EXPECT_TRUE(HasSymbol("f0"));
  EXPECT_TRUE(HasSymbol("f1"));

  // Once the remaining references drop to zero, a second sweep erases both.
  symbol_manager_->RecordOpMutation(
      attr_holder, [&]() { attr_holder->removeAttr("test.ref"); });
  symbol_manager_->RedirectCallToCallee(calls_[2], calls_[3].getCalleeAttr());
  symbol_manager_->EraseUnusedTrackedRootSymbols();
  EXPECT_FALSE(HasSymbol("f0"));
  EXPECT_FALSE(HasSymbol("f1"));
}

TEST_F(CalleeSymbolManagerTest,
       EraseUnusedTrackedRootSymbolsTracksTransitivelyDetachedCallees) {
  ASSERT_OK(InitModuleWithCalls(
      R"mlir(
  func.func private @redirect_target() { return }
  func.func private @leaf() { return }
  func.func private @caller() {
    func.call @leaf() : () -> ()
    return
  }
  func.func private @attr_holder() attributes {test.ref = @leaf} { return }
  func.func private @dup_leaf() { return }
  func.func private @dup_caller() {
    func.call @dup_leaf() : () -> ()
    func.call @dup_leaf() : () -> ()
    return
  }
  func.func private @shared_leaf() { return }
  func.func private @c0() {
    func.call @shared_leaf() : () -> ()
    return
  }
  func.func private @c1() {
    func.call @shared_leaf() : () -> ()
    return
  }
)mlir",
      {"@caller", "@dup_caller", "@c0", "@c1", "@redirect_target"}));

  mlir::func::FuncOp attr_holder =
      module_->lookupSymbol<mlir::func::FuncOp>("attr_holder");
  ASSERT_NE(attr_holder, nullptr);
  mlir::func::FuncOp redirect_target =
      module_->lookupSymbol<mlir::func::FuncOp>("redirect_target");
  ASSERT_NE(redirect_target, nullptr);
  auto insert_fresh_private_fn = [&](llvm::StringRef name) {
    mlir::func::FuncOp fn = redirect_target.clone();
    fn.setSymName(name);
    symbol_table_.getSymbolTable(*module_).insert(fn);
  };

  // 2-hop transitive deferred erasure: erasing `@caller` decrements `@leaf`
  // from 2 to 1 (leaving `@attr_holder` as the sole user) and must register
  // `@leaf` for subsequent sweeps.
  EXPECT_TRUE(EraseCall(calls_[0]));
  EXPECT_FALSE(HasSymbol("caller"));
  symbol_manager_->EraseUnusedTrackedRootSymbols();
  EXPECT_TRUE(HasSymbol("leaf"));

  symbol_manager_->RecordOpMutation(
      attr_holder, [&]() { attr_holder->removeAttr("test.ref"); });
  symbol_manager_->EraseUnusedTrackedRootSymbols();
  EXPECT_FALSE(HasSymbol("leaf"));

  // A symbol decremented to >0 and then to 0 within the same `EraseCall`
  // worklist must not remain tracked after erasure.
  EXPECT_TRUE(EraseCall(calls_[1]));
  EXPECT_FALSE(HasSymbol("dup_caller"));
  EXPECT_FALSE(HasSymbol("dup_leaf"));
  insert_fresh_private_fn("dup_leaf");
  symbol_manager_->EraseUnusedTrackedRootSymbols();
  EXPECT_TRUE(HasSymbol("dup_leaf"));

  // A symbol transitively detached by one candidate and erased by a later
  // candidate in the same `EraseUnusedTrackedRootSymbols` sweep must be
  // drained and not remain tracked after the sweep.
  mlir::SymbolRefAttr target_ref = calls_[4].getCalleeAttr();
  symbol_manager_->RedirectCallToCallee(calls_[2], target_ref);
  symbol_manager_->RedirectCallToCallee(calls_[3], target_ref);
  symbol_manager_->EraseUnusedTrackedRootSymbols();
  EXPECT_FALSE(HasSymbol("c0"));
  EXPECT_FALSE(HasSymbol("c1"));
  EXPECT_FALSE(HasSymbol("shared_leaf"));
  insert_fresh_private_fn("shared_leaf");
  symbol_manager_->EraseUnusedTrackedRootSymbols();
  EXPECT_TRUE(HasSymbol("shared_leaf"));
}

TEST_F(CalleeSymbolManagerTest, OutlinedRecursiveModuleIsolatesInternalUses) {
  ASSERT_OK(InitModuleWithCalls(R"mlir(
  module @rec_mod <sym_visibility = "private"> {
    func.func @main() {
      func.call @main() : () -> ()
      return
    }
  }
  module @pub_rec_mod {
    func.func @main() {
      func.call @main() : () -> ()
      return
    }
  }
)mlir",
                                {"@rec_mod::@main", "@pub_rec_mod::@main"}));

  ASSERT_OK_AND_ASSIGN(mlir::func::FuncOp rec_main, GetCallee(calls_[0]));
  ASSERT_OK_AND_ASSIGN(mlir::func::FuncOp pub_main, GetCallee(calls_[1]));

  mlir::ModuleOp rec_mod = llvm::cast<mlir::ModuleOp>(rec_main->getParentOp());
  EXPECT_FALSE(mlir::SymbolTable::symbolKnownUseEmpty(rec_main, rec_mod));
  EXPECT_THAT(GetOrClone({calls_[0]}), IsOkAndHolds(rec_main));
  EXPECT_TRUE(mlir::SymbolTable::symbolKnownUseEmpty(rec_main, rec_mod));
  EXPECT_EQ(rec_mod.getBody()->getOperations().size(), 2u);
  // Splitting is idempotent on subsequent calls once internal uses are
  // isolated.
  EXPECT_THAT(GetOrClone({calls_[0]}), IsOkAndHolds(rec_main));
  EXPECT_EQ(rec_mod.getBody()->getOperations().size(), 2u);

  mlir::ModuleOp pub_mod = llvm::cast<mlir::ModuleOp>(pub_main->getParentOp());
  ASSERT_OK_AND_ASSIGN(mlir::func::FuncOp cloned_split,
                       GetOrClone({calls_[1]}));
  EXPECT_NE(cloned_split, pub_main);
  mlir::ModuleOp cloned_mod =
      llvm::cast<mlir::ModuleOp>(cloned_split->getParentOp());
  EXPECT_EQ(mlir::SymbolTable::getSymbolVisibility(cloned_mod),
            mlir::SymbolTable::Visibility::Private);
  EXPECT_TRUE(mlir::SymbolTable::symbolKnownUseEmpty(cloned_split, cloned_mod));
  EXPECT_EQ(cloned_mod.getBody()->getOperations().size(), 2u);
  EXPECT_FALSE(mlir::SymbolTable::symbolKnownUseEmpty(pub_main, pub_mod));
  EXPECT_EQ(pub_mod.getBody()->getOperations().size(), 1u);
}

TEST_F(CalleeSymbolManagerTest,
       FlatSelfRecursiveFuncOpClonesAndPreservesDeadCycle) {
  ASSERT_OK(InitModuleWithCalls(R"mlir(
  func.func private @rec() {
    func.call @rec() : () -> ()
    return
  }
  func.func private @rec_dead() {
    func.call @rec_dead() : () -> ()
    return
  }
)mlir",
                                {"@rec", "@rec_dead"}));

  ASSERT_OK_AND_ASSIGN(mlir::func::FuncOp rec, GetCallee(calls_[0]));

  // Root-scope self-recursion gives `@rec` count == 2 (`ifrt.Call` + internal
  // `func.call`), so `GetOrClone` must clone rather than mutate in place. The
  // clone still calls `@rec`, keeping `@rec`'s count at 2 (self-call + clone).
  ASSERT_OK_AND_ASSIGN(mlir::func::FuncOp cloned_rec, GetOrClone({calls_[0]}));
  EXPECT_NE(cloned_rec, rec);

  // The clone itself has only `calls_[0]` referencing it (count == 1), so a
  // second `GetOrClone` mutates it in place.
  EXPECT_THAT(GetOrClone({calls_[0]}), IsOkAndHolds(cloned_rec));
  symbol_manager_->EraseUnusedTrackedRootSymbols();
  EXPECT_TRUE(HasSymbol("rec"));

  // Erasing `calls_[0]` erases the clone and decrements `@rec`'s count from 2
  // to 1 (its own self-call), leaving the dead self-recursive cycle intact.
  mlir::StringAttr cloned_rec_name = cloned_rec.getSymNameAttr();
  EXPECT_TRUE(EraseCall(calls_[0]));
  EXPECT_FALSE(HasSymbol(cloned_rec_name.getValue()));
  EXPECT_TRUE(HasSymbol("rec"));
  symbol_manager_->EraseUnusedTrackedRootSymbols();
  EXPECT_TRUE(HasSymbol("rec"));

  // Directly erasing a call to a self-recursive flat callee returns false
  // because the self-call keeps its use count at 1.
  EXPECT_FALSE(EraseCall(calls_[1]));
  symbol_manager_->EraseUnusedTrackedRootSymbols();
  EXPECT_TRUE(HasSymbol("rec_dead"));
}

TEST_F(CalleeSymbolManagerTest,
       MultiFunctionOutlinedModuleClonesAndErasesIdempotently) {
  ASSERT_OK(InitModuleWithCalls(kMultiFuncMod,
                                {"@multi_mod::@main", "@multi_mod::@other"}));

  ASSERT_OK_AND_ASSIGN(mlir::func::FuncOp multi_main, GetCallee(calls_[0]));
  mlir::SymbolRefAttr multi_main_ref = calls_[0].getCalleeAttr();

  EXPECT_THAT(GetOrClone({calls_[0]}), IsOkAndHolds(Ne(multi_main)));
  symbol_manager_->EraseUnusedTrackedRootSymbols();
  EXPECT_TRUE(HasSymbol("multi_mod"));

  // Redirecting within the same root module updates the callee reference
  // without changing root-symbol use counts or detaching `@multi_mod`.
  symbol_manager_->RedirectCallToCallee(calls_[1], multi_main_ref);
  EXPECT_EQ(calls_[1].getCalleeAttr(), multi_main_ref);
  symbol_manager_->EraseUnusedTrackedRootSymbols();
  EXPECT_TRUE(HasSymbol("multi_mod"));

  symbol_manager_->RedirectCallToCallee(calls_[1], calls_[0].getCalleeAttr());
  symbol_manager_->EraseUnusedTrackedRootSymbols();
  EXPECT_FALSE(HasSymbol("multi_mod"));
  symbol_manager_->EraseUnusedTrackedRootSymbols();
}

TEST_F(CalleeSymbolManagerTest, CloningOutlinedModuleTracksRootSymbolAttrUses) {
  ASSERT_OK(
      InitModuleWithCalls(absl::StrCat(kPrivateF0, R"mlir(
  module @mod_with_ref <sym_visibility = "private">
      attributes {test.ref = @f0} {
    func.func @main() { return }
  }
)mlir"),
                          {"@mod_with_ref::@main", "@mod_with_ref::@main"}));

  ASSERT_OK_AND_ASSIGN(mlir::func::FuncOp mod_main, GetCallee(calls_[0]));

  EXPECT_THAT(GetOrClone({calls_[0]}), IsOkAndHolds(Ne(mod_main)));
  EXPECT_TRUE(EraseCall(calls_[0]));
  EXPECT_TRUE(HasSymbol("f0"));
  EXPECT_TRUE(EraseCall(calls_[1]));
  EXPECT_FALSE(HasSymbol("f0"));
}

TEST_F(CalleeSymbolManagerTest,
       MutationTrackingUpdatesCountsAndIgnoresNestedSymbolTables) {
  ASSERT_OK(
      InitModuleWithCalls(absl::StrCat(kMultiFuncMod, kPrivateF0, kPrivateF1),
                          {"@multi_mod::@main", "@f0", "@f1"}));

  ASSERT_OK_AND_ASSIGN(mlir::func::FuncOp nested_main, GetCallee(calls_[0]));
  ASSERT_OK_AND_ASSIGN(mlir::func::FuncOp f0, GetCallee(calls_[1]));
  ASSERT_OK_AND_ASSIGN(mlir::func::FuncOp f1, GetCallee(calls_[2]));

  mlir::func::CallOp added_call;
  symbol_manager_->RecordOpMutation(f0, [&]() {
    rewriter_.setInsertionPointToStart(&f0.getBody().front());
    added_call = mlir::func::CallOp::create(rewriter_, f0.getLoc(), f1,
                                            mlir::ValueRange{});
  });
  EXPECT_THAT(GetOrClone({calls_[2]}), IsOkAndHolds(Ne(f1)));
  symbol_manager_->EraseUnusedTrackedRootSymbols();
  EXPECT_TRUE(HasSymbol("f1"));

  symbol_manager_->RecordOpMutation(f0,
                                    [&]() { rewriter_.eraseOp(added_call); });
  symbol_manager_->EraseUnusedTrackedRootSymbols();
  EXPECT_FALSE(HasSymbol("f1"));

  // Ops inside a nested `SymbolTable` must not affect `root_module_`'s counts.
  mlir::ModuleOp multi_mod =
      llvm::cast<mlir::ModuleOp>(nested_main->getParentOp());
  mlir::func::FuncOp nested_other =
      multi_mod.lookupSymbol<mlir::func::FuncOp>("other");
  ASSERT_NE(nested_other, nullptr);
  mlir::FlatSymbolRefAttr f0_ref =
      mlir::FlatSymbolRefAttr::get(rewriter_.getStringAttr("f0"));

  symbol_manager_->RecordOpMutation(
      nested_main, [&]() { nested_main->setAttr("test.ref", f0_ref); });
  EXPECT_THAT(GetOrClone({calls_[1]}), IsOkAndHolds(f0));
  symbol_manager_->RecordOpMutation(
      nested_main, [&]() { nested_main->removeAttr("test.ref"); });

  rewriter_.setInsertionPointToStart(&nested_main.getBody().front());
  mlir::func::CallOp nested_call = mlir::func::CallOp::create(
      rewriter_, nested_main.getLoc(), nested_other, mlir::ValueRange{});
  nested_call->setAttr("test.ref", f0_ref);
  symbol_manager_->RecordOpCreated(nested_call);
  EXPECT_THAT(GetOrClone({calls_[1]}), IsOkAndHolds(f0));
  symbol_manager_->RecordOpErased(nested_call);
  rewriter_.eraseOp(nested_call);
  EXPECT_THAT(GetOrClone({calls_[1]}), IsOkAndHolds(f0));

  // Root-scope `RecordOpCreated` and `RecordOpErased` update counts directly.
  mlir::func::FuncOp main_func = GetMainFunction(*module_);
  rewriter_.setInsertionPointToStart(&main_func.getBody().front());
  mlir::func::CallOp root_call = mlir::func::CallOp::create(
      rewriter_, main_func.getLoc(), f0, mlir::ValueRange{});
  symbol_manager_->RecordOpCreated(root_call);
  EXPECT_THAT(GetOrClone({calls_[1]}), IsOkAndHolds(Ne(f0)));
  symbol_manager_->EraseUnusedTrackedRootSymbols();
  EXPECT_TRUE(HasSymbol("f0"));

  symbol_manager_->RecordOpErased(root_call);
  rewriter_.eraseOp(root_call);
  symbol_manager_->EraseUnusedTrackedRootSymbols();
  EXPECT_FALSE(HasSymbol("f0"));
}

TEST_F(CalleeSymbolManagerTest,
       RecordOpErasedAndMutationTrackNetDecrementedRootSymbols) {
  ASSERT_OK(InitModule(absl::StrCat(kPrivateF0, R"mlir(
  func.func private @orphan() { return }
  func.func private @mut_leaf() { return }
  func.func private @mut_helper() {
    func.call @mut_leaf() : () -> ()
    return
  }
  func.func private @caller_mut() {
    func.call @mut_helper() : () -> ()
    return
  }
  func.func private @erased_only() { return }
  func.func private @attr_only() { return }
  func.func private @keep() { return }
  func.func private @drop() { return }
  func.func private @dup() { return }
  func.func private @added() { return }
  func.func private @reused() { return }
  func.func private @delta_holder() attributes {
    test.keep = @keep, test.drop = @drop, test.dup0 = @dup,
    test.dup1 = @dup, test.reused = @reused, test.main = @main
  } { return }
)mlir"),
                       R"mlir(
    %0 = ifrt.Call @caller_mut() on devices [0] : () -> ()
    %1 = ifrt.Call @f0() on devices [0] {test.extra = @attr_only} : () -> ()
    func.call @erased_only() : () -> ()
)mlir"));

  ASSERT_EQ(calls_.size(), 2u);
  mlir::func::FuncOp main_func = GetMainFunction(*module_);

  // `RecordOpMutation` dropping the last reference (without prior `GetOrClone`
  // or `RedirectCallToCallee`) sweeps `@mut_helper` and transitively
  // `@mut_leaf`.
  mlir::func::FuncOp caller_mut =
      module_->lookupSymbol<mlir::func::FuncOp>("caller_mut");
  ASSERT_NE(caller_mut, nullptr);
  llvm::SmallVector<mlir::func::CallOp> caller_calls(
      caller_mut.getOps<mlir::func::CallOp>());
  ASSERT_EQ(caller_calls.size(), 1u);
  symbol_manager_->RecordOpMutation(
      caller_mut, [&]() { rewriter_.eraseOp(caller_calls.front()); });
  EXPECT_TRUE(HasSymbol("mut_helper"));
  EXPECT_TRUE(HasSymbol("mut_leaf"));
  symbol_manager_->EraseUnusedTrackedRootSymbols();
  EXPECT_FALSE(HasSymbol("mut_helper"));
  EXPECT_FALSE(HasSymbol("mut_leaf"));
  EXPECT_TRUE(HasSymbol("caller_mut"));

  // `RecordOpErased` dropping the last reference (without prior `GetOrClone`)
  // sweeps `@erased_only`.
  llvm::SmallVector<mlir::func::CallOp> main_calls(
      main_func.getOps<mlir::func::CallOp>());
  ASSERT_EQ(main_calls.size(), 1u);
  symbol_manager_->RecordOpErased(main_calls.front());
  rewriter_.eraseOp(main_calls.front());
  symbol_manager_->EraseUnusedTrackedRootSymbols();
  EXPECT_FALSE(HasSymbol("erased_only"));

  // `EraseCallAndMaybeEraseCallee` tracks extra non-callee `SymbolRefAttr`s on
  // `call_op` for deferred sweeping.
  EXPECT_TRUE(EraseCall(calls_[1]));
  EXPECT_FALSE(HasSymbol("f0"));
  EXPECT_TRUE(HasSymbol("attr_only"));
  symbol_manager_->EraseUnusedTrackedRootSymbols();
  EXPECT_FALSE(HasSymbol("attr_only"));

  // `RecordOpMutation` with > 4 distinct symbols (spilling `SmallDenseMap`
  // inline storage) tracks net-negative deltas (`@drop` `1 -> 0`, `@dup`
  // `2 -> 1`) while keeping unchanged (`@keep`, `@reused`, `@main`) and newly
  // added (`@added` `0 -> 1`) references alive.
  mlir::func::FuncOp delta_holder =
      module_->lookupSymbol<mlir::func::FuncOp>("delta_holder");
  ASSERT_NE(delta_holder, nullptr);
  symbol_manager_->RecordOpMutation(delta_holder, [&]() {
    delta_holder->removeAttr("test.drop");
    delta_holder->removeAttr("test.dup0");
    delta_holder->setAttr("test.added", mlir::FlatSymbolRefAttr::get(
                                            rewriter_.getStringAttr("added")));
  });
  symbol_manager_->EraseUnusedTrackedRootSymbols();
  EXPECT_FALSE(HasSymbol("drop"));
  EXPECT_TRUE(HasSymbol("dup"));
  EXPECT_TRUE(HasSymbol("keep"));
  EXPECT_TRUE(HasSymbol("added"));

  symbol_manager_->RecordOpMutation(
      delta_holder, [&]() { delta_holder->removeAttr("test.dup1"); });
  symbol_manager_->EraseUnusedTrackedRootSymbols();
  EXPECT_FALSE(HasSymbol("dup"));

  // Dropping the last reference to `@reused` in `RecordOpMutation` and then
  // re-referencing it via `RecordOpCreated` before sweeping keeps `@reused`
  // alive and tracked until the new reference is erased.
  mlir::func::FuncOp reused_fn =
      module_->lookupSymbol<mlir::func::FuncOp>("reused");
  ASSERT_NE(reused_fn, nullptr);
  symbol_manager_->RecordOpMutation(
      delta_holder, [&]() { delta_holder->removeAttr("test.reused"); });
  rewriter_.setInsertionPointToStart(&main_func.getBody().front());
  mlir::func::CallOp new_reused_call = mlir::func::CallOp::create(
      rewriter_, main_func.getLoc(), reused_fn, mlir::ValueRange{});
  symbol_manager_->RecordOpCreated(new_reused_call);
  symbol_manager_->EraseUnusedTrackedRootSymbols();
  EXPECT_TRUE(HasSymbol("reused"));
  symbol_manager_->RecordOpErased(new_reused_call);
  rewriter_.eraseOp(new_reused_call);
  symbol_manager_->EraseUnusedTrackedRootSymbols();
  EXPECT_FALSE(HasSymbol("reused"));

  // Pre-existing zero-use `@orphan` was never decremented and remains alive.
  EXPECT_TRUE(HasSymbol("orphan"));
}

TEST_F(CalleeSymbolManagerTest, GetOrCloneFailsOnUnregisteredRegionOp) {
  mlir_context_.allowUnregisteredDialects(/*allow=*/true);
  ASSERT_OK(InitModuleWithCalls(
      absl::StrCat("  module @priv_unreg_mod <sym_visibility = \"private\"> {\n"
                   "    func.func @main() {\n      ",
                   kUnregRegionOp,
                   "\n      return\n    }\n  }\n"
                   "  module @pub_unreg_mod {\n"
                   "    func.func @main() {\n      ",
                   kUnregRegionOp, "\n      return\n    }\n  }\n"),
      {"@priv_unreg_mod::@main", "@pub_unreg_mod::@main"}));

  // Pre-checking internal symbol uses fails before cloning or adding a helper
  // (on both the in-place and clone paths) without redirecting or mutating IR.
  constexpr absl::string_view kInternalUsesError =
      "internal symbol uses could not be determined";
  const size_t initial_root_ops = module_->getBody()->getOperations().size();
  mlir::ModuleOp priv_unreg_mod =
      module_->lookupSymbol<mlir::ModuleOp>("priv_unreg_mod");
  ASSERT_NE(priv_unreg_mod, nullptr);
  EXPECT_THAT(GetOrClone({calls_[0]}), StatusIs(absl::StatusCode::kUnknown,
                                                HasSubstr(kInternalUsesError)));
  EXPECT_EQ(priv_unreg_mod.getBody()->getOperations().size(), 1u);

  mlir::SymbolRefAttr orig_pub_ref = calls_[1].getCalleeAttr();
  EXPECT_THAT(GetOrClone({calls_[1]}), StatusIs(absl::StatusCode::kUnknown,
                                                HasSubstr(kInternalUsesError)));
  EXPECT_EQ(calls_[1].getCalleeAttr(), orig_pub_ref);
  EXPECT_EQ(module_->getBody()->getOperations().size(), initial_root_ops);
}

TEST_F(CalleeSymbolManagerTest, CreateFailsOnUnregisteredRegionOp) {
  mlir_context_.allowUnregisteredDialects(/*allow=*/true);
  ASSERT_OK_AND_ASSIGN(
      module_, LoadFromSource(absl::StrCat(
                   "module {\n", kPrivateF0,
                   "  func.func @main() attributes {ifrt.function} {\n    ",
                   kUnregRegionOp,
                   "\n    %0 = ifrt.Call @f0() on devices [0] : () -> ()\n"
                   "    return\n  }\n}\n")));

  mlir::BaseScopedDiagnosticHandler diag(&mlir_context_);
  EXPECT_TRUE(
      mlir::failed(CalleeSymbolManager::Create(*module_, &symbol_table_)));
  EXPECT_THAT(diag.ConsumeStatus(),
              StatusIs(absl::StatusCode::kUnknown,
                       HasSubstr("cannot determine root-scope symbol uses")));
}

TEST_F(CalleeSymbolManagerTest, PreconditionChecksFailFast) {
  ASSERT_OK(InitModuleWithCalls(
      absl::StrCat(kPrivateF0, kPrivateF1, kNestedCallMod), {"@f0", "@f1"}));

  ASSERT_OK_AND_ASSIGN(mlir::func::FuncOp func_op, GetCallee(calls_[0]));
  mlir::OwningOpRef<mlir::Operation*> detached_op(func_op->clone());
  CallOp nested_call = FindNestedCall();
  ASSERT_NE(nested_call, nullptr);

  mlir::SymbolTableCollection* null_symbol_table = nullptr;
  EXPECT_DEATH(static_cast<void>(CalleeSymbolManager::Create(mlir::ModuleOp{},
                                                             &symbol_table_)),
               "Check failed");
  // Under ASan/UBSan, `absl_nonnull` aborts with a nullability trap before
  // reaching `CHECK_NE`.
  EXPECT_DEATH(
      static_cast<void>(
          CalleeSymbolManager::Create(*module_, null_symbol_table)),
      AnyOf(HasSubstr("Check failed"), HasSubstr("null pointer passed")));
  EXPECT_DEATH(symbol_manager_->RecordOpCreated(nullptr), "Check failed");
  EXPECT_DEATH(symbol_manager_->RecordOpCreated(detached_op.get()),
               "Check failed");
  EXPECT_DEATH(symbol_manager_->RecordOpErased(nullptr), "Check failed");
  EXPECT_DEATH(symbol_manager_->RecordOpErased(detached_op.get()),
               "Check failed");
  EXPECT_DEATH(symbol_manager_->RecordOpMutation(nullptr, []() {}),
               "Check failed");
  EXPECT_DEATH(symbol_manager_->RecordOpMutation(detached_op.get(), []() {}),
               "Check failed");
  EXPECT_DEATH(static_cast<void>(GetCallee(CallOp{})), "Check failed");
  EXPECT_DEATH(static_cast<void>(GetOrClone({})), "Check failed");
  EXPECT_DEATH(static_cast<void>(GetOrClone({CallOp{}})), "Check failed");
  EXPECT_DEATH(static_cast<void>(GetOrClone({calls_[0], CallOp{}})),
               "Check failed");
  EXPECT_DEATH(static_cast<void>(GetOrClone({calls_[0], calls_[1]})),
               "Check failed");
  EXPECT_DEATH(static_cast<void>(GetOrClone({calls_[0], calls_[0]})),
               "Check failed");
  // Preconditions are enforced even when a call is outside root symbol scope.
  EXPECT_DEATH(static_cast<void>(GetOrClone({nested_call, CallOp{}})),
               "Check failed");
  EXPECT_DEATH(static_cast<void>(GetOrClone({nested_call, calls_[1]})),
               "Check failed");
  EXPECT_DEATH(static_cast<void>(GetOrClone({calls_[1], nested_call})),
               "Check failed");
  EXPECT_DEATH(static_cast<void>(GetOrClone({nested_call, nested_call})),
               "Check failed");
  EXPECT_DEATH(symbol_manager_->RedirectCallToCallee(nested_call,
                                                     calls_[0].getCalleeAttr()),
               "Check failed");
  EXPECT_DEATH(symbol_manager_->RedirectCallToCallee(CallOp{},
                                                     calls_[0].getCalleeAttr()),
               "Check failed");
  EXPECT_DEATH(symbol_manager_->RedirectCallToCallee(
                   calls_[0], /*new_callee_ref=*/mlir::SymbolRefAttr{}),
               "Check failed");
  EXPECT_DEATH(
      symbol_manager_->EraseCallAndMaybeEraseCallee(nested_call, rewriter_),
      "Check failed");

  mlir_context_.allowUnregisteredDialects(/*allow=*/true);
  ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> unreg_module,
      LoadFromSource(absl::StrCat("module { func.func @main() {\n",
                                  kUnregRegionOp, "\nreturn\n} }\n")));
  mlir::Operation* unreg_op =
      &unreg_module->lookupSymbol<mlir::func::FuncOp>("main")
           .getBody()
           .front()
           .front();
  mlir::Operation* cloned_unreg = unreg_op->clone();
  rewriter_.setInsertionPointToStart(
      &GetMainFunction(*module_).getBody().front());
  rewriter_.insert(cloned_unreg);
  EXPECT_DEATH(symbol_manager_->RecordOpCreated(cloned_unreg), "Check failed");
  EXPECT_DEATH(symbol_manager_->RecordOpErased(cloned_unreg), "Check failed");
  EXPECT_DEATH(symbol_manager_->RecordOpMutation(cloned_unreg, []() {}),
               "Check failed");
  rewriter_.eraseOp(cloned_unreg);
}

}  // namespace
}  // namespace ifrt
}  // namespace xla

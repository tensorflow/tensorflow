/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Rewrite/FrozenRewritePatternSet.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TF {

namespace {

#define GEN_PASS_DEF_DECOMPOSEOPTIONALSPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

struct DecomposeOptionalsPass
    : public impl::DecomposeOptionalsPassBase<DecomposeOptionalsPass> {
  void runOnOperation() override;
};

class HandleOptionalFrom : public OpRewritePattern<TF::OptionalFromValueOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TF::OptionalFromValueOp op,
                                PatternRewriter& rewriter) const override {
    Value value = nullptr;
    for (auto v : op.getComponents()) {
      value = v;
    }
    if (!value) return failure();
    rewriter.replaceOpWithNewOp<TF::IdentityOp>(op, value.getType(), value);
    return success();
  }
};

class HandleOptionalGet : public OpRewritePattern<TF::OptionalGetValueOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TF::OptionalGetValueOp op,
                                PatternRewriter& rewriter) const override {
    auto input = op.getOptional();
    auto elementType = getElementTypeOrSelf(input.getType());
    if (isa<TF::VariantType>(elementType)) {
      // We can only replace OptionalGetValue after the inputs have been
      // replaced.
      return failure();
    }
    rewriter.replaceOpWithNewOp<TF::CastOp>(op, op.getResult(0).getType(),
                                            input);
    return success();
  }
};

class HandleOptionalNone : public OpRewritePattern<TF::OptionalNoneOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TF::OptionalNoneOp op,
                                PatternRewriter& rewriter) const override {
    Type elementType = getElementTypeOrSelf(op->getResult(0));
    Type newType = nullptr;
    if (auto variant = dyn_cast<TF::VariantType>(elementType)) {
      ArrayRef<TensorType> sub = variant.getSubtypes();
      if (sub.size() == 1) {
        auto inner = sub[0];
        if (!isa<TF::VariantType>(inner)) {
          newType = inner;
        }
      }
    }
    if (!newType) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, newType,
                                                            op->getOperands());
    return success();
  }
};

class HandleCall : public OpInterfaceRewritePattern<CallOpInterface> {
  // Optional-agnostic pattern that propagates types across the program.
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(CallOpInterface call,
                                PatternRewriter& rewriter) const override {
    // Adjust the function arguments to match the types of the caller.
    // So e.g.
    //
    // func.func @f(%x : foo) {
    // }
    // ...
    // func.call @f(%x : bar)
    //
    // will be transformed to
    //
    // func.func @f(%x : bar) {
    //   ...
    // }
    // ...
    // func.call @f(%x : bar)

    CallInterfaceCallable callable = call.getCallableForCallee();
    mlir::SymbolRefAttr sym = callable.dyn_cast<mlir::SymbolRefAttr>();
    auto symbol =
        mlir::SymbolTable::lookupNearestSymbolFrom(call.getOperation(), sym);
    if (!symbol) return failure();
    auto f = llvm::dyn_cast<mlir::func::FuncOp>(symbol);

    if (call.getOperation()->getNumOperands() !=
        f.getBody().front().getNumArguments()) {
      return failure();  // RemoteCall et al
    }

    rewriter.startOpModification(f);
    bool changed = false;
    for (auto [call_arg, body_arg] :
         llvm::zip(call.getOperation()->getOperands(),
                   f.getBody().front().getArguments())) {
      if (call_arg.getType() != body_arg.getType()) {
        body_arg.setType(call_arg.getType());
        changed = true;
      }
    }
    if (changed) {
      rewriter.finalizeOpModification(f);
      return success();
    } else {
      rewriter.cancelOpModification(f);
      return failure();
    }
  }
};

class HandleIf : public OpRewritePattern<TF::IfOp> {
  // Optional-agnostic pattern that propagates types across the program.
  using OpRewritePattern::OpRewritePattern;

  LogicalResult adjustBranch(TF::IfOp ifop, func::FuncOp branch,
                             PatternRewriter& rewriter) const {
    bool changed = false;
    rewriter.startOpModification(branch);
    for (auto [call_arg, body_arg] :
         llvm::zip(llvm::drop_begin(ifop.getOperation()->getOperands()),
                   branch.getBody().front().getArguments())) {
      if (call_arg.getType() != body_arg.getType()) {
        body_arg.setType(call_arg.getType());
        changed = true;
      }
    }
    if (changed) {
      rewriter.finalizeOpModification(branch);
      return success();
    } else {
      rewriter.cancelOpModification(branch);
      return failure();
    }
  }

  LogicalResult matchAndRewrite(TF::IfOp ifop,
                                PatternRewriter& rewriter) const override {
    bool success =
        succeeded(adjustBranch(ifop, ifop.then_function(), rewriter)) ||
        succeeded(adjustBranch(ifop, ifop.else_function(), rewriter));
    return LogicalResult::success(success);
  }
};

class HandleFunc : public OpRewritePattern<func::FuncOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp f,
                                PatternRewriter& rewriter) const override {
    // Adjust the function type to match the block args. So this e.g. transforms
    //
    // func.func @f(%x : foo) -> foo {
    //   ^main(%x : bar):
    //     yield %x : bar
    // }
    //
    // to
    //
    // func.func @f(%x : bar) -> bar {
    //   ...
    // }
    auto ret = f.getBody().front().getTerminator();
    std::vector<Type> argument_types;
    for (auto arg : f.getBody().front().getArguments()) {
      argument_types.push_back(arg.getType());
    }
    std::vector<Type> return_types;
    for (auto ret_op : ret->getOperands()) {
      return_types.push_back(ret_op.getType());
    }
    auto newType =
        FunctionType::get(rewriter.getContext(), argument_types, return_types);
    if (f.getFunctionType() == newType) {
      return failure();
    }
    rewriter.modifyOpInPlace(f, [&] { f.setType(newType); });

    // Adjust the type of the return values callers of the function to
    // match the "func.return" within the function.
    //
    // So this would transform
    // func.func @f(...) -> ... {
    //     yield %x : foo
    // }
    // ...
    // func.call @f(...) -> bar
    //
    // to
    //
    // func.func @f(...) -> ... {
    //     yield %x : foo
    // }
    // ...
    // func.call f(...) -> foo
    auto symbol_uses = f.getSymbolUses(f->getParentOp());
    if (!symbol_uses.has_value()) {
      return failure();
    }
    for (auto use : *symbol_uses) {
      Operation* caller = use.getUser();
      bool changed = false;
      rewriter.startOpModification(caller);
      for (auto [result, type] :
           llvm::zip(caller->getResults(), return_types)) {
        if (result.getType() != type) {
          result.setType(type);
          changed = true;
        }
      }
      if (changed) {
        rewriter.finalizeOpModification(caller);
      } else {
        rewriter.cancelOpModification(caller);
      }
    }
    return success();
  }
};

void DecomposeOptionalsPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();

  RewritePatternSet pattern_list(&getContext());
  pattern_list.add<HandleOptionalFrom>(&getContext());
  pattern_list.add<HandleOptionalGet>(&getContext());
  pattern_list.add<HandleOptionalNone>(&getContext());
  pattern_list.add<HandleFunc>(&getContext());
  pattern_list.add<HandleCall>(&getContext());
  pattern_list.add<HandleIf>(&getContext());
  FrozenRewritePatternSet patterns(std::move(pattern_list));

  if (failed(applyPatternsGreedily(module, patterns))) {
    signalPassFailure();
  }
}
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateDecomposeOptionalsPass() {
  return std::make_unique<DecomposeOptionalsPass>();
}

}  // namespace TF
}  // namespace mlir

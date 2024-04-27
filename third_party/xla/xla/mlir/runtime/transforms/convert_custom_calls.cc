/* Copyright 2022 The OpenXLA Authors.

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
#include <string>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "xla/mlir/runtime/ir/rt_dialect.h"
#include "xla/mlir/runtime/ir/rt_interfaces.h"
#include "xla/mlir/runtime/ir/rt_ops.h"
#include "xla/mlir/runtime/transforms/passes.h"

namespace xla {
namespace runtime {

using namespace mlir;  // NOLINT

#define GEN_PASS_DEF_CONVERTCUSTOMCALLS
#include "xla/mlir/runtime/transforms/passes.h.inc"

class ConvertCustomCallsPass
    : public impl::ConvertCustomCallsBase<ConvertCustomCallsPass> {
  void runOnOperation() override;
};

//===----------------------------------------------------------------------====/

class CallOpLowering : public OpRewritePattern<func::CallOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  CallOpLowering(MLIRContext* ctx, SymbolTable sym_table)
      : OpRewritePattern(ctx), sym_table_(std::move(sym_table)) {}

  LogicalResult matchAndRewrite(func::CallOp op,
                                PatternRewriter& rewriter) const override {
    // Check if callee is a custom call declaration.
    auto callee = sym_table_.lookup<func::FuncOp>(op.getCallee());
    StringAttr target = callee->getAttrOfType<StringAttr>("rt.custom_call");
    if (!target) return failure();

    // Check if call operation is inside the exported runtime function.
    auto exported = op->getParentOfType<func::FuncOp>();
    if (!exported || !exported->hasAttr(kExportedAttrName))
      return rewriter.notifyMatchFailure(
          op, "func.call is not inside the exported runtime function");

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value exec_ctx = exported.getArgument(0);

    // Custom call operation always returns the status flag.
    llvm::SmallVector<Type> results = {StatusType::get(getContext())};
    results.append(op->getResultTypes().begin(), op->getResultTypes().end());

    // Build a runtime call operation, maybe inside the trace region.
    auto build_custom_call = [&](ImplicitLocOpBuilder b) -> CallOp {
      // Rewrite function call with a runtime call, and check the return status.
      bool dynamic = callee->hasAttr("rt.dynamic");
      auto call = b.create<CallOp>(results, exec_ctx, target, dynamic,
                                   op.getOperands());

      // Copy optional attributes from the custom call function declaration.
      llvm::ArrayRef<llvm::StringRef> callee_attrs = callee.getAttributeNames();
      for (auto& attr : callee->getAttrs()) {
        if (isa_and_nonnull<RuntimeDialect>(attr.getNameDialect())) continue;
        if (llvm::find(callee_attrs, attr.getName()) == callee_attrs.end())
          call->setAttr(attr.getName(), attr.getValue());
      }

      // Copy optional attributes from the call operation to the custom call.
      llvm::ArrayRef<llvm::StringRef> orig_attrs = op.getAttributeNames();
      for (auto& attr : op->getAttrs()) {
        if (llvm::find(orig_attrs, attr.getName()) == orig_attrs.end())
          call->setAttr(attr.getName(), attr.getValue());
      }

      return call;
    };

    // Builds the trace operation body region.
    auto build_trace = [&](OpBuilder& builder, Location loc) {
      ImplicitLocOpBuilder b(loc, builder);
      auto call = build_custom_call(b);
      call->removeAttr("rt.trace");
      b.create<YieldOp>(call->getOpResults());
    };

    Value status;                // custom call status
    SmallVector<OpResult> rets;  // custom call results

    // Check if we must trace the custom call execution.
    auto attrs = op->getAttrDictionary();
    if (auto traced = attrs.getAs<TraceAnnotationAttrInterface>("rt.trace")) {
      auto trace = b.create<TraceOp>(results, exec_ctx, traced, build_trace);
      status = trace.getResult(0);
      rets = llvm::to_vector(llvm::drop_begin(trace->getResults()));
    } else {
      auto call = build_custom_call(b);
      status = call.getStatus();
      rets = llvm::to_vector(call.getResults());
    }

    b.create<cf::AssertOp>(
        b.create<IsOkOp>(TypeRange(b.getI1Type()), status),
        b.getStringAttr("custom call '" + target.str() + "' failed"));

    // Forward users of the original results to custom call results.
    llvm::for_each(llvm::zip(op->getResults(), rets), [](auto ret) {
      std::get<0>(ret).replaceAllUsesWith(std::get<1>(ret));
    });

    // Erase the original function call operation.
    rewriter.eraseOp(op);

    return success();
  }

 private:
  SymbolTable sym_table_;
};

//===----------------------------------------------------------------------====/

void ConvertCustomCallsPass::runOnOperation() {
  ModuleOp op = getOperation();

  RewritePatternSet patterns(&getContext());
  patterns.insert<CallOpLowering>(&getContext(), SymbolTable(op));

  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
    return signalPassFailure();
  }

  Region& body = op.getBodyRegion();

  // Erase all unused custom call declarations.
  auto is_unused = [](func::FuncOp decl) {
    return decl->hasAttr("rt.custom_call") && decl.symbolKnownUseEmpty(decl);
  };
  for (auto op : llvm::make_early_inc_range(body.getOps<func::FuncOp>())) {
    if (is_unused(op)) op.erase();
  }
}

std::unique_ptr<OperationPass<ModuleOp>> CreateConvertCustomCallsPass() {
  return std::make_unique<ConvertCustomCallsPass>();
}

}  // namespace runtime
}  // namespace xla

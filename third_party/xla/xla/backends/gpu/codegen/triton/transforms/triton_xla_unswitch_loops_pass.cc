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
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/triton/transforms/passes.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLAUNSWITCHLOOPSPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

class TritonXLAUnswitchLoopsPass
    : public impl::TritonXLAUnswitchLoopsPassBase<TritonXLAUnswitchLoopsPass> {
 public:
  void runOnOperation() override;
};

struct UnswitchLoop : mlir::OpRewritePattern<mlir::scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::scf::ForOp for_op, mlir::PatternRewriter& rewriter) const override {
    mlir::scf::IfOp if_op = nullptr;
    for (Operation& body_op : for_op.getBody()->getOperations()) {
      if (auto current_if = mlir::dyn_cast<mlir::scf::IfOp>(body_op)) {
        // Unswitching loops with multiple ifs can lead to exponential code
        // blow.
        if (if_op != nullptr) {
          return rewriter.notifyMatchFailure(
              for_op, "multiple ifs are found inside the loop");
        }
        if_op = current_if;
      }
    }
    if (!if_op) {
      return rewriter.notifyMatchFailure(for_op, "no if found inside the loop");
    }
    if (mlir::matchPattern(if_op.getCondition(), mlir::m_Constant())) {
      return rewriter.notifyMatchFailure(for_op, "condition is a constant");
    }
    // Check if the condition is loop invariant. We rely on
    // loop-invariant-code-motion pass to run before and hoist the condition out
    // of the loop.
    if (!for_op.isDefinedOutsideOfLoop(if_op.getCondition())) {
      return rewriter.notifyMatchFailure(
          for_op, "condition depends on values defined inside the loop");
    }
    auto true_cst = rewriter.create<mlir::arith::ConstantOp>(
        for_op.getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
    auto false_cst = rewriter.create<mlir::arith::ConstantOp>(
        for_op.getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
    rewriter.setInsertionPoint(for_op);
    mlir::IRMapping mapping;
    mapping.map(if_op.getCondition(), false_cst);
    auto false_branch_loop = for_op->clone(mapping);
    auto new_if = rewriter.create<mlir::scf::IfOp>(
        for_op.getLoc(), for_op.getResultTypes(), if_op.getCondition(),
        /*addThenBlock=*/true, /*addElseBlock=*/true);
    rewriter.replaceAllUsesWith(for_op.getResults(), new_if.getResults());

    auto then_builder = new_if.getThenBodyBuilder(rewriter.getListener());
    auto then_yield = then_builder.create<mlir::scf::YieldOp>(
        for_op.getLoc(), for_op.getResults());
    rewriter.moveOpBefore(for_op, then_yield);
    rewriter.modifyOpInPlace(if_op, [&]() { if_op->setOperand(0, true_cst); });

    auto else_builder = new_if.getElseBodyBuilder(rewriter.getListener());
    else_builder.insert(false_branch_loop);
    else_builder.create<mlir::scf::YieldOp>(for_op.getLoc(),
                                            false_branch_loop->getResults());

    return mlir::success();
  }
};

void TritonXLAUnswitchLoopsPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<UnswitchLoop>(&getContext());
  mlir::scf::ForOp::getCanonicalizationPatterns(patterns, &getContext());
  mlir::scf::IfOp::getCanonicalizationPatterns(patterns, &getContext());
  if (mlir::failed(
          mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<mlir::Pass> CreateTritonXLAUnswitchLoopsPass() {
  return std::make_unique<TritonXLAUnswitchLoopsPass>();
}

}  // namespace mlir::triton::xla

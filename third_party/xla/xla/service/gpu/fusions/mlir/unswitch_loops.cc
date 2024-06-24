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

#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace xla {
namespace gpu {

#define GEN_PASS_DEF_UNSWITCHLOOPSPASS
#include "xla/service/gpu/fusions/mlir/passes.h.inc"

namespace {

class UnswitchLoopsPass
    : public impl::UnswitchLoopsPassBase<UnswitchLoopsPass> {
 public:
  void runOnOperation() override;
};

struct UnswitchLoop : mlir::OpRewritePattern<mlir::scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::scf::ForOp op, mlir::PatternRewriter& rewriter) const override {
    if (op.getBody()->getOperations().size() != 2) {
      return rewriter.notifyMatchFailure(
          op, "loop body is not a single instruction");
    }
    auto if_op = mlir::dyn_cast<mlir::scf::IfOp>(op.getBody()->front());
    if (!if_op) {
      return rewriter.notifyMatchFailure(op, "no if found inside the loop");
    }
    if (mlir::matchPattern(if_op.getCondition(), mlir::m_Constant())) {
      return rewriter.notifyMatchFailure(op, "condition is a constant");
    }

    auto true_cst = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
    auto false_cst = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
    rewriter.setInsertionPoint(op);
    mlir::IRMapping mapping;
    mapping.map(if_op.getCondition(), false_cst);
    auto false_branch_loop = op->clone(mapping);
    auto new_if = rewriter.create<mlir::scf::IfOp>(
        op.getLoc(), op.getResultTypes(), if_op.getCondition(), true, true);
    rewriter.replaceAllUsesWith(op.getResults(), new_if.getResults());

    auto then_builder = new_if.getThenBodyBuilder(rewriter.getListener());
    auto then_yield =
        then_builder.create<mlir::scf::YieldOp>(op.getLoc(), op.getResults());
    rewriter.moveOpBefore(op, then_yield);
    rewriter.modifyOpInPlace(if_op, [&]() { if_op->setOperand(0, true_cst); });

    auto else_builder = new_if.getElseBodyBuilder(rewriter.getListener());
    else_builder.insert(false_branch_loop);
    else_builder.create<mlir::scf::YieldOp>(op.getLoc(),
                                            false_branch_loop->getResults());

    return mlir::success();
  }
};

void UnswitchLoopsPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<UnswitchLoop>(&getContext());
  mlir::scf::ForOp::getCanonicalizationPatterns(patterns, &getContext());
  mlir::scf::IfOp::getCanonicalizationPatterns(patterns, &getContext());
  if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                      std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateUnswitchLoopsPass() {
  return std::make_unique<UnswitchLoopsPass>();
}

}  // namespace gpu
}  // namespace xla

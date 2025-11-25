/* Copyright 2025 The OpenXLA Authors.

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
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/triton/transforms/passes.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLAUNSWITCHLOOPSPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

LogicalResult UnswitchLoop(mlir::scf::ForOp for_op,
                           mlir::PatternRewriter& rewriter) {
  // Walk the body of the loop, including nested blocks, and count all scf::IfOp
  // instances.
  scf::IfOp if_op;
  int if_count = 0;
  int max_if_count = 2;
  for_op->walk([&](scf::IfOp op) -> WalkResult {
    if (matchPattern(op.getCondition(), m_Constant())) {
      // Do not match if with constant conditions - they are left from
      // our previous transformations and will be optimized away later.
      return WalkResult::advance();
    }
    if (!for_op.isDefinedOutsideOfLoop(op.getCondition())) {
      // Condition is not loop invariant.
      // We rely on loop-invariant-code-motion pass to run before and
      // hoist the condition out of the loop.
      return WalkResult::advance();
    }
    if_op = op;
    ++if_count;
    return failure(if_count > max_if_count);
  });
  if (if_count > max_if_count) {
    // We don't want to explode the code size too much by unswitching
    // multiple times. 2 is the current need for cases we have seen
    // for two concats used in dot. You might want to increase this
    // number. In this case it might make sense to make it a parameter
    // of the pass.
    return rewriter.notifyMatchFailure(
        for_op, "more than 2 ifs are found inside the loop");
  }
  if (!if_op) {
    return rewriter.notifyMatchFailure(for_op, "no if found inside the loop");
  }

  scf::IfOp new_if = scf::IfOp::create(
      rewriter, for_op.getLoc(), for_op.getResultTypes(), if_op.getCondition(),
      /*addThenBlock=*/true, /*addElseBlock=*/true);
  for (int body_index : {0, 1}) {
    auto builder = OpBuilder::atBlockEnd(new_if.getBody(body_index),
                                         rewriter.getListener());
    arith::ConstantOp condition = arith::ConstantOp::create(
        builder, for_op.getLoc(),
        rewriter.getIntegerAttr(rewriter.getI1Type(), body_index == 0));
    IRMapping mapping;
    mapping.map(if_op.getCondition(), condition);
    Operation* new_for = builder.clone(*for_op, mapping);
    scf::YieldOp::create(builder, for_op.getLoc(), new_for->getResults());
  }
  rewriter.replaceOp(for_op, new_if);
  return success();
}

class TritonXLAUnswitchLoopsPass
    : public impl::TritonXLAUnswitchLoopsPassBase<TritonXLAUnswitchLoopsPass> {
 public:
  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add(UnswitchLoop);
    mlir::scf::ForOp::getCanonicalizationPatterns(patterns, &getContext());
    mlir::scf::IfOp::getCanonicalizationPatterns(patterns, &getContext());
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateTritonXLAUnswitchLoopsPass() {
  return std::make_unique<TritonXLAUnswitchLoopsPass>();
}

}  // namespace mlir::triton::xla

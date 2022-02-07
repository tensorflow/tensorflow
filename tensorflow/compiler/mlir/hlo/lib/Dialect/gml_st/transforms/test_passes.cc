/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir-hlo/Dialect/gml_st/transforms/test_passes.h"

#include <utility>

#include "mlir-hlo/Dialect/gml_st/transforms/transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_CLASSES
#include "mlir-hlo/Dialect/gml_st/transforms/test_passes.h.inc"

static constexpr char kPeeledLoopsLabel[] = "__peeled_loops__";
static constexpr char kPartialIterationLabel[] = "__partial_iteration__";

/// Peel LoopOps, i.e., split them into two loops: One loop where the
/// `idx`-th loop contains only "full" iterations and a second loop for the
/// remaining partial iteration (if any).
struct TiledLoopPeelingPattern : public OpRewritePattern<LoopOp> {
  TiledLoopPeelingPattern(MLIRContext *ctx, int64_t idx, bool skip_partial)
      : OpRewritePattern<LoopOp>(ctx), idx(idx), skip_partial(skip_partial) {}

  LogicalResult matchAndRewrite(LoopOp loopOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<int64_t> peeledLoops;
    if (loopOp->hasAttr(kPeeledLoopsLabel)) {
      auto attr = loopOp->getAttr(kPeeledLoopsLabel).cast<ArrayAttr>();
      peeledLoops =
          llvm::to_vector<4>(llvm::map_range(attr, [](Attribute attr) {
            return attr.cast<IntegerAttr>().getInt();
          }));
      // Check if the loop was already peeled.
      if (llvm::find(peeledLoops, idx) != peeledLoops.end()) return failure();
    }
    if (skip_partial && loopOp->hasAttr(kPartialIterationLabel))
      // No peeling of loop nests with a partial iteration.
      return failure();

    if (static_cast<int64_t>(loopOp.iterator_types().size()) <= idx)
      return failure();

    // Peel loop and canonicalize.
    LoopOp result;
    if (failed(peelAndCanonicalizeGmlStLoop(rewriter, loopOp, idx, result)))
      return failure();

    // Apply label, so that the same loop is not rewritten a second time.
    peeledLoops.push_back(idx);
    rewriter.updateRootInPlace(loopOp, [&]() {
      loopOp->setAttr(kPeeledLoopsLabel, rewriter.getI64ArrayAttr(peeledLoops));
    });
    result->setAttr(kPeeledLoopsLabel, rewriter.getI64ArrayAttr(peeledLoops));
    result->setAttr(kPartialIterationLabel, rewriter.getUnitAttr());

    return success();
  }

  /// Index of loop to peel.
  int64_t idx;

  /// If set to true, do not peel LoopOps with a partial iteration.
  bool skip_partial;
};

class TestGmlStLoopPeelingPass
    : public TestGmlStLoopPeelingBase<TestGmlStLoopPeelingPass> {
  void runOnOperation() final {
    auto funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);
    for (unsigned idx : dims)
      patterns.add<TiledLoopPeelingPattern>(ctx, idx, skip_partial);

    (void)(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)));

    // Drop the markers.
    funcOp.walk([](LoopOp op) {
      op->removeAttr(kPeeledLoopsLabel);
      op->removeAttr(kPartialIterationLabel);
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createTestGmlStLoopPeelingPass() {
  return std::make_unique<TestGmlStLoopPeelingPass>();
}

}  // namespace gml_st
}  // namespace mlir

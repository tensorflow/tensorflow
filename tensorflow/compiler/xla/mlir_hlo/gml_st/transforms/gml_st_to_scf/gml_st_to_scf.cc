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

#include <iterator>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_DEF_GMLSTTOSCF
#include "gml_st/transforms/passes.h.inc"

/// Converts gml_st.parallel to SCF loop nest.
struct ParallelOpToSCFPattern : public OpRewritePattern<ParallelOp> {
  using OpRewritePattern<ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ParallelOp loop,
                                PatternRewriter &rewriter) const override {
    // Fail conversion if the loop has not been bufferized.
    if (!loop.hasBufferSemantics()) return failure();

    auto cloneBody = [&](OpBuilder &builder, Location /*loc*/, ValueRange ivs) {
      IRMapping bvm;
      bvm.map(loop.getInductionVars(), ivs);

      for (auto &op : loop.getBody()->without_terminator())
        builder.clone(op, bvm);
    };

    rewriter.create<scf::ParallelOp>(loop.getLoc(), loop.getLowerBound(),
                                     loop.getUpperBound(), loop.getStep(),
                                     cloneBody);

    rewriter.eraseOp(loop);
    return success();
  }
};

struct GmlStToScfPass : public impl::GmlStToScfBase<GmlStToScfPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ParallelOpToSCFPattern>(patterns.getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createGmlStToScfPass() {
  return std::make_unique<GmlStToScfPass>();
}

}  // namespace gml_st
}  // namespace mlir

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
#include <utility>
#include <vector>

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_DEF_GMLSTTOSCF
#include "gml_st/transforms/passes.h.inc"

/// Converts gml_st.loop to SCF loop nest. All parallel dimensions are collected
/// into an scf.parallel loop and all sequential dimensions will result in a
/// nested scf.for loop nest. The pattern assumes that a gml_st.loop with
/// iterator_types ["reduction", "parallel", "reduction"] can be reordered.
struct LoopToSCFPattern : public OpRewritePattern<LoopOp> {
  using OpRewritePattern<LoopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoopOp loop,
                                PatternRewriter &rewriter) const override {
    // Fail conversion if the `gml_st.loop` has not been bufferized.
    if (!loop.hasBufferSemantics()) return failure();

    // Collect loop control parameters for parallel and sequential dimensions.
    SmallVector<Value, 3> seqLBs, seqUBs, seqSteps, seqIVs;
    SmallVector<Value, 3> parLBs, parUBs, parSteps, parIVs;
    for (const auto &en :
         llvm::enumerate(llvm::zip(loop.getLowerBound(), loop.getUpperBound(),
                                   loop.getStep(), loop.getInductionVars()))) {
      Value lb, ub, step, iv;
      std::tie(lb, ub, step, iv) = en.value();
      if (loop.isParallelDimension(en.index())) {
        parLBs.push_back(lb);
        parUBs.push_back(ub);
        parSteps.push_back(step);
        parIVs.push_back(iv);
      } else {
        seqLBs.push_back(lb);
        seqUBs.push_back(ub);
        seqSteps.push_back(step);
        seqIVs.push_back(iv);
      }
    }

    Location loc = loop.getLoc();
    auto generateForLoopNestAndCloneBody = [&](OpBuilder &builder, Location loc,
                                               ValueRange ivs) {
      BlockAndValueMapping bvm;
      bvm.map(parIVs, ivs);
      bvm.map(loop.getRegionInputArgs(), loop.getInputs());
      bvm.map(loop.getRegionOutputArgs(), loop.getOutputs());

      // If not all dimensions of the gml_st.loop are parallel, an scf.for loop
      // nest is generated.
      if (!seqIVs.empty()) {
        scf::LoopNest nest =
            scf::buildLoopNest(builder, loc, seqLBs, seqUBs, seqSteps,
                               [&](OpBuilder & /*builder*/, Location /*loc*/,
                                   ValueRange ivs) { bvm.map(seqIVs, ivs); });
        builder.setInsertionPointToStart(nest.loops.back().getBody());
      }
      for (auto &op : loop.getBody()->without_terminator())
        builder.clone(op, bvm);
    };

    if (parIVs.empty()) {
      generateForLoopNestAndCloneBody(rewriter, loc, llvm::None);
    } else {
      rewriter.create<scf::ParallelOp>(loc, parLBs, parUBs, parSteps,
                                       generateForLoopNestAndCloneBody);
    }
    rewriter.eraseOp(loop);
    return success();
  }
};

/// Converts gml_st.parallel to SCF loop nest.
struct ParallelOpToSCFPattern : public OpRewritePattern<ParallelOp> {
  using OpRewritePattern<ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ParallelOp loop,
                                PatternRewriter &rewriter) const override {
    // Fail conversion if the loop has not been bufferized.
    if (!loop.hasBufferSemantics()) return failure();

    auto cloneBody = [&](OpBuilder &builder, Location /*loc*/, ValueRange ivs) {
      BlockAndValueMapping bvm;
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

/// Converts gml_st.for to SCF loop nest.
struct ForOpToSCFPattern : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp loop,
                                PatternRewriter &rewriter) const override {
    auto cloneBody = [&](OpBuilder &builder, Location /*loc*/, ValueRange ivs,
                         ValueRange iterArgs) {
      BlockAndValueMapping bvm;
      bvm.map(loop.getInductionVars(), ivs);
      bvm.map(loop.getRegionOutputArgs(), iterArgs);

      for (auto &op : loop.getBody()->without_terminator())
        builder.clone(op, bvm);

      scf::ValueVector result;
      llvm::transform(loop.getTerminator().getSrcs(),
                      std::back_inserter(result),
                      [&](Value src) { return bvm.lookupOrDefault(src); });
      return result;
    };

    scf::LoopNest nest = scf::buildLoopNest(
        rewriter, loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(),
        loop.getStep(), loop.getOutputs(), cloneBody);
    rewriter.replaceOp(loop, nest.results);
    return success();
  }
};

struct GmlStToScfPass : public impl::GmlStToScfBase<GmlStToScfPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ForOpToSCFPattern, LoopToSCFPattern, ParallelOpToSCFPattern>(
        patterns.getContext());
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

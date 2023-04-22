/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <iterator>
#include <memory>

#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"  // from @llvm-project
#include "mlir/Dialect/SCF/SCF.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/InliningUtils.h"  // from @llvm-project
#include "mlir/Transforms/LoopUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tfr/ir/tfr_ops.h"
#include "tensorflow/compiler/mlir/tfr/passes/passes.h"

//===----------------------------------------------------------------------===//
// Canonicalization patterns for the scf.for and scf.if ops. They are used to
// optimize the control flow in the tfr function. Technically, both patterns
// should be upstreamed to be part of the op definition.
// TODO(fengliuai): sync with the llvm upstream for both patterns.
//
namespace mlir {
namespace TFR {

namespace {

struct UnrollSCFForOp : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForOp for_op,
                                PatternRewriter &rewriter) const override {
    Location loc = for_op.getLoc();
    APInt lower_bound, upper_bound, step;
    if (!matchPattern(for_op.lowerBound(), m_ConstantInt(&lower_bound)) ||
        !matchPattern(for_op.upperBound(), m_ConstantInt(&upper_bound)) ||
        !matchPattern(for_op.step(), m_ConstantInt(&step))) {
      return failure();
    }
    uint64_t trip_count = (upper_bound - lower_bound).sdiv(step).getZExtValue();
    if (trip_count <= 0) return failure();

    // TODO(fengliuai): use loopUnrollByFactor once the iter_arg is supported

    Block *single_block = for_op.getBody();
    BlockAndValueMapping mapping;
    Value iv = for_op.getInductionVar();
    for (auto iter_op :
         llvm::zip(for_op.getRegionIterArgs(), for_op.initArgs())) {
      mapping.map(std::get<0>(iter_op), std::get<1>(iter_op));
    }
    mapping.map(iv, for_op.lowerBound());
    for (auto i = 0; i < trip_count; ++i) {
      if (!iv.use_empty()) {
        // iv' = iv + step * i;
        Value iter = rewriter.create<ConstantIndexOp>(loc, i);
        Value step_cst =
            rewriter.create<ConstantIndexOp>(loc, step.getSExtValue());
        Value stride = rewriter.create<MulIOp>(loc, step_cst, iter);
        Value iv_unroll =
            rewriter.create<AddIOp>(loc, mapping.lookup(iv), stride);
        mapping.map(iv, iv_unroll);
      }

      Operation *terminator_op;
      for (auto it = single_block->begin(); it != single_block->end(); ++it) {
        terminator_op = rewriter.clone(*it, mapping);
      }
      // Map the block arguments to the yield results.
      for (auto iter_op : llvm::zip(for_op.getRegionIterArgs(),
                                    terminator_op->getOperands())) {
        mapping.map(std::get<0>(iter_op), std::get<1>(iter_op));
      }
      rewriter.eraseOp(terminator_op);
    }
    SmallVector<Value, 4> returned;
    for (Value arg : for_op.getRegionIterArgs()) {
      returned.push_back(mapping.lookup(arg));
    }
    rewriter.replaceOp(for_op, returned);
    return success();
  }
};

// TODO(fengliuai): up stream this pattern.
struct SimplifySCFIfOp : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::IfOp if_op,
                                PatternRewriter &rewriter) const override {
    // Then branch
    if (matchPattern(if_op.condition(), m_NonZero())) {
      return InlineRegion(if_op.getLoc(), rewriter, if_op, &if_op.thenRegion());
    }

    // Else branch
    if (matchPattern(if_op.condition(), m_Zero())) {
      if (if_op.elseRegion().empty()) {
        // Remove the op
        rewriter.eraseOp(if_op);
        return success();
      } else {
        return InlineRegion(if_op.getLoc(), rewriter, if_op,
                            &if_op.elseRegion());
      }
    }

    // Not a constant condition
    return failure();
  }

 private:
  LogicalResult InlineRegion(Location loc, PatternRewriter &rewriter,
                             Operation *inline_point, Region *region) const;
};

LogicalResult SimplifySCFIfOp::InlineRegion(Location loc,
                                            PatternRewriter &rewriter,
                                            Operation *inline_point,
                                            Region *region) const {
  InlinerInterface interface(loc.getContext());
  if (failed(inlineRegion(interface, region, inline_point, {},
                          inline_point->getResults(), loc,
                          /*shouldCloneInlinedRegion=*/true))) {
    return failure();
  }

  // If the inlining was successful then erase the scf.if op.
  rewriter.eraseOp(inline_point);
  return success();
}

}  // namespace

void populateCanonicalizationPatterns(FuncOp func,
                                      OwningRewritePatternList &patterns) {
  MLIRContext *context = func.getContext();
  mlir::Dialect *tf = context->getLoadedDialect<mlir::TF::TensorFlowDialect>();
  // Load all official canonicalization patterns. Here we skip the
  // canonicalization of the ops in the tf dialect, because they couldn't
  // propagate the attributes correctly. These optimization will be played by
  // bridge.
  func->walk([&](Operation *op) {
    if (op->getDialect() != tf) {
      op->getAbstractOperation()->getCanonicalizationPatterns(patterns,
                                                              context);
    }
  });
  patterns.insert<UnrollSCFForOp, SimplifySCFIfOp>(context);
}

}  // namespace TFR
}  // namespace mlir

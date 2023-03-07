/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "gml_st/transforms/passes.h"

#include <optional>

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::gml_st {
namespace {

using scf::ForallOp;

struct CollapseForallOpDimensions : public OpRewritePattern<ForallOp> {
  using OpRewritePattern<ForallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForallOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Compute new loop bounds that omit all single-iteration loop dimensions.
    SmallVector<OpFoldResult> newMixedLowerBounds, newMixedUpperBounds,
        newMixedSteps;
    IRMapping mapping;
    for (auto [lowerBound, upperBound, step, iv] :
         llvm::zip(op.getMixedLowerBound(), op.getMixedUpperBound(),
                   op.getMixedStep(), op.getInductionVars())) {
      // Collect the statically known loop bounds.
      std::optional<int64_t> lowerBoundConstant =
          getConstantIntValue(lowerBound);
      std::optional<int64_t> upperBoundConstant =
          getConstantIntValue(upperBound);
      std::optional<int64_t> stepConstant = getConstantIntValue(step);
      // Remove the loop if it performs zero iterations.
      if (lowerBoundConstant && upperBoundConstant &&
          *lowerBoundConstant == *upperBoundConstant) {
        rewriter.replaceOp(op, op.getOutputs());
        return success();
      }
      // Replace the loop induction variable by the lower bound if the loop
      // performs a single iteration. Otherwise, copy the loop bounds.
      if (lowerBoundConstant && upperBoundConstant && stepConstant &&
          (*upperBoundConstant - *lowerBoundConstant) > 0 &&
          (*upperBoundConstant - *lowerBoundConstant) <= *stepConstant) {
        mapping.map(iv,
                    getValueOrCreateConstantIndexOp(rewriter, loc, lowerBound));
      } else {
        newMixedLowerBounds.push_back(lowerBound);
        newMixedUpperBounds.push_back(upperBound);
        newMixedSteps.push_back(step);
      }
    }
    // Exit if none of the loop dimensions perform a single iteration.
    if (newMixedLowerBounds.size() == static_cast<unsigned>(op.getRank())) {
      return failure();
    }

    // All of the loop dimensions perform a single iteration. Inline loop body.
    if (newMixedLowerBounds.empty()) {
      mapping.map(op.getOutputBlockArguments(), op.getOutputs());
      for (auto &bodyOp : op.getBody()->without_terminator())
        rewriter.clone(bodyOp, mapping);
      SmallVector<Value> results;
      results.reserve(op.getResults().size());
      scf::InParallelOp terminator = op.getTerminator();
      for (auto &yieldingOp : terminator.getYieldingOps()) {
        auto parallelInsertSliceOp =
            cast<tensor::ParallelInsertSliceOp>(yieldingOp);

        Value dst = parallelInsertSliceOp.getDest();
        Value src = parallelInsertSliceOp.getSource();

        auto getMappedValues = [&](ValueRange values) {
          return llvm::to_vector(llvm::map_range(values, [&](Value value) {
            return mapping.lookupOrDefault(value);
          }));
        };

        Value srcVal = mapping.lookupOrDefault(src);
        if (srcVal.getType().isa<TensorType>()) {
          results.push_back(rewriter.create<tensor::InsertSliceOp>(
              op.getLoc(), dst.getType(), srcVal, mapping.lookupOrDefault(dst),
              getMappedValues(parallelInsertSliceOp.getOffsets()),
              getMappedValues(parallelInsertSliceOp.getSizes()),
              getMappedValues(parallelInsertSliceOp.getStrides()),
              parallelInsertSliceOp.getStaticOffsets(),
              parallelInsertSliceOp.getStaticSizes(),
              parallelInsertSliceOp.getStaticStrides()));
        }
      }
      rewriter.replaceOp(op, results);
      return success();
    }

    // Replace the loop by a lower-dimensional loop.
    ForallOp newOp;
    newOp = rewriter.create<ForallOp>(loc, newMixedLowerBounds,
                                      newMixedUpperBounds, newMixedSteps,
                                      op.getOutputs(), std::nullopt, nullptr);
    newOp.getBodyRegion().getBlocks().clear();
    // The new loop needs to keep all attributes from the old one, except for
    // "operand_segment_sizes" which captures the outdated information of the
    // old iteration domain.
    SmallVector<StringAttr> elidedAttrs{newOp.getOperandSegmentSizesAttrName(),
                                        newOp.getStaticLowerBoundAttrName(),
                                        newOp.getStaticUpperBoundAttrName(),
                                        newOp.getStaticStepAttrName()};
    for (const auto &namedAttr : op->getAttrs()) {
      if (llvm::is_contained(elidedAttrs, namedAttr.getName())) continue;
      newOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    // Clone the loop body and remap the block arguments of the collapsed loops
    // (inlining does not support a cancellable block argument mapping).
    rewriter.cloneRegionBefore(op.getRegion(), newOp.getRegion(),
                               newOp.getRegion().begin(), mapping);
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

}  // namespace

void populateCollapseForallOpDimensionsPattern(RewritePatternSet &patterns) {
  patterns.add<CollapseForallOpDimensions>(patterns.getContext());
}

}  // namespace mlir::gml_st

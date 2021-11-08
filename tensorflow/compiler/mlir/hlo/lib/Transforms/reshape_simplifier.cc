/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>

#include "mlir-hlo/Analysis/shape_component_analysis.h"
#include "mlir-hlo/Transforms/PassDetail.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

namespace {

// Returns true if `reshape` only adds `1` dimensions.
bool isExpandShape(ShapeComponentAnalysis &shapeComponentAnalysis,
                   mhlo::DynamicReshapeOp reshape) {
  auto output_shape =
      shapeComponentAnalysis.dimensionsForShapeTensor(reshape.output_shape());
  auto operand_shape =
      shapeComponentAnalysis.dimensionsForShape(reshape.operand());
  if (!output_shape || !operand_shape ||
      output_shape->size() <= operand_shape->size())
    return false;
  int inputDim = 0;
  // Check if the reshape only inserts 1 dimensions and everything else is the
  // same.
  for (const auto &dim : *output_shape) {
    if (dim.isConstant(1)) continue;
    if (inputDim >= operand_shape->size() || dim != (*operand_shape)[inputDim])
      return false;
    ++inputDim;
  }
  return inputDim == operand_shape->size();
}

// Rewrite dynamic reshapes that only insert one dimensions into
// linalg.tensor_expand_shape.
struct ReshapeToExpandShape final
    : public OpRewritePattern<mhlo::DynamicReshapeOp> {
  ReshapeToExpandShape(MLIRContext *ctx,
                       ShapeComponentAnalysis &shapeComponentAnalysis)
      : OpRewritePattern(ctx), shapeComponentAnalysis(shapeComponentAnalysis) {}
  LogicalResult matchAndRewrite(mhlo::DynamicReshapeOp op,
                                PatternRewriter &rewriter) const override {
    if (!isExpandShape(shapeComponentAnalysis, op)) return failure();
    auto output_shape =
        shapeComponentAnalysis.dimensionsForShapeTensor(op.output_shape());
    SmallVector<ReassociationExprs> reassociations(output_shape->size());
    auto it = reassociations.begin();
    int64_t runningIndex = 0;
    for (const auto &dim : *output_shape) {
      it->push_back(rewriter.getAffineDimExpr(runningIndex++));
      if (!dim.isConstant(1)) ++it;
    }
    // If the last dimension was a 1 expand it from the penultimate dim.
    if (output_shape->back().isConstant(1)) std::prev(it)->append(*it);
    reassociations.erase(it, reassociations.end());

    rewriter.replaceOpWithNewOp<linalg::TensorExpandShapeOp>(
        op, op.getResult().getType(), op.operand(), reassociations);
    shapeComponentAnalysis.reset();
    return success();
  }
  ShapeComponentAnalysis &shapeComponentAnalysis;
};

// Remove compute_reshape_shape if we can prove that the dynamic shape does not
// contain a `-1` dimension.
struct RemoveComputeReshapeShape final
    : public OpRewritePattern<mhlo::ComputeReshapeShapeOp> {
  RemoveComputeReshapeShape(MLIRContext *ctx,
                            ShapeComponentAnalysis &shapeComponentAnalysis)
      : OpRewritePattern(ctx), shapeComponentAnalysis(shapeComponentAnalysis) {}
  LogicalResult matchAndRewrite(mhlo::ComputeReshapeShapeOp op,
                                PatternRewriter &rewriter) const override {
    auto dynamic_shape =
        shapeComponentAnalysis.dimensionsForShapeTensor(op.dynamic_shape());
    if (!dynamic_shape) return failure();

    if (llvm::any_of(*dynamic_shape, [](const auto &dim) {
          return !dim.isKnownNotNegativeOne();
        }))
      return failure();
    rewriter.replaceOp(op, op.dynamic_shape());
    shapeComponentAnalysis.reset();
    return success();
  }
  ShapeComponentAnalysis &shapeComponentAnalysis;
};

class ReshapeSimplifierPass final
    : public ReshapeSimplifierBase<ReshapeSimplifierPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnFunction() override;

 private:
};
}  // end namespace

void ReshapeSimplifierPass::runOnFunction() {
  MLIRContext *ctx = &getContext();
  mlir::RewritePatternSet patterns(ctx);

  ShapeComponentAnalysis shapeComponentAnalysis;
  patterns.insert<ReshapeToExpandShape>(ctx, shapeComponentAnalysis);
  patterns.insert<RemoveComputeReshapeShape>(ctx, shapeComponentAnalysis);

  (void)mlir::applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
}

std::unique_ptr<FunctionPass> createReshapeSimplifierPass() {
  return std::make_unique<ReshapeSimplifierPass>();
}

}  // end namespace mlir

/* Copyright 2021 The OpenXLA Authors.

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

// This file contains the patterns to simplify shape ops that were deemed not
// suitable for shape op canonicalization in MLIR Core.

#include <memory>
#include <utility>

#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_SHAPESIMPLIFICATION
#include "mhlo/transforms/mhlo_passes.h.inc"

namespace {

using shape::BroadcastOp;
using shape::ConstShapeOp;
using shape::ShapeOfOp;

// Try to remove operands from broadcasts that don't contribute to the final
// result.
struct BroadcastRemoveSubsumedOperandsPattern
    : public OpRewritePattern<BroadcastOp> {
  using OpRewritePattern<BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    // First collect the static components when joining all shapes. The
    // resulting vector contains a static dimension if any operand has a static
    // non-1 dimension in that position. The remaining dimensions are set to
    // dynamic size.
    SmallVector<int64_t> knownExtents;
    SmallVector<SmallVector<int64_t, 4>, 4> operandExtents;
    for (Value shape : op.getShapes()) {
      auto &extents = operandExtents.emplace_back();
      if (failed(shape::getShapeVec(shape, extents))) return failure();

      // Prepend dynamic dims if sizes don't match.
      if (extents.size() > knownExtents.size()) {
        knownExtents.insert(knownExtents.begin(),
                            extents.size() - knownExtents.size(),
                            ShapedType::kDynamic);
      }

      for (size_t i = 0, e = extents.size(); i != e; ++i) {
        int64_t extent = extents[e - i - 1];
        if (extent != ShapedType::kDynamic && extent != 1) {
          int64_t &knownExtent = knownExtents[knownExtents.size() - i - 1];
          // A dynamic dimension is subsumed by a static one, but bail out for
          // known conflicting shapes.
          if (knownExtent != extent && knownExtent != ShapedType::kDynamic)
            return failure();
          knownExtent = extent;
        }
      }
    }

    // If we've figured out all shapes to be constants we're done.
    if (!llvm::is_contained(knownExtents, ShapedType::kDynamic)) {
      rewriter.replaceOpWithNewOp<ConstShapeOp>(
          op, op->getResultTypes(), rewriter.getIndexTensorAttr(knownExtents));
      return success();
    }

    // If only some dimensions are known see if any of the operands can be
    // removed without affecting the result.
    SmallVector<Value, 4> filteredOperands;
    for (auto tuple : llvm::zip(op.getShapes(), operandExtents)) {
      Value shape = std::get<0>(tuple);
      auto &extents = std::get<1>(tuple);

      // An operand can't be dead if it's the only operand of the maximum rank.
      // Removing it would reduce the rank of the output.
      if (llvm::count_if(operandExtents, [&](ArrayRef<int64_t> op) {
            return op.size() >= extents.size();
          }) <= 1) {
        filteredOperands.push_back(shape);
        continue;
      }

      for (size_t i = 0, e = extents.size(); i != e; ++i) {
        int64_t extent = extents[e - i - 1];
        // A dimension of an operand can be subsumed if it's
        //   - a 1 dimension. All other operands will have 1 dims or better.
        if (extent == 1) continue;

        //   - a dynamic dim but the result is known to be constant.
        int64_t knownExtent = knownExtents[knownExtents.size() - i - 1];
        assert(knownExtent != 1);
        if (knownExtent != ShapedType::kDynamic &&
            extent == ShapedType::kDynamic)
          continue;

        //   - a constant non-1 dimension equal to the "known" dim.
        // In this case we also have to check whether this operand is the only
        // contributor of that constant.
        if (knownExtent != ShapedType::kDynamic && extent == knownExtent &&
            llvm::count_if(operandExtents, [&](ArrayRef<int64_t> operandShape) {
              return i < operandShape.size() &&
                     operandShape[operandShape.size() - i - 1] == knownExtent;
            }) > 1)
          continue;

        filteredOperands.push_back(shape);
        break;
      }
    }
    if (filteredOperands.size() != op.getShapes().size()) {
      rewriter.replaceOpWithNewOp<BroadcastOp>(op, op->getResultTypes(),
                                               filteredOperands);
      return success();
    }
    return failure();
  }
};

// Convert cases like:
// ```
//  %1 = shape.shape_of %arg0 : tensor<?x?x?xf64> -> tensor<3xindex>
//  %2 = shape.shape_of %arg1 : tensor<?x?x1xf64> -> tensor<3xindex>
//  %3 = shape.broadcast %1, %2 : tensor<3xindex>, tensor<3xindex>
//                                -> tensor<3xindex>
//  %result = tensor.extract %3[%c2] : tensor<3xindex>
// ```
// to
//
// ```
//  %result = tensor.dim %arg0[%c2] : tensor<?x?x2048xf64>
// ```
struct ExtractFromBroadcastedTensorCanonicalizationPattern
    : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter &rewriter) const override {
    auto broadcastOp = op.getTensor().getDefiningOp<BroadcastOp>();
    if (!broadcastOp) return failure();

    // Confirm that there is a constant index. This is required, so we can
    // confirm the DimOp's input will define the resulting broadcasted shape in
    // that dimension.
    auto index =
        op.getIndices().front().getDefiningOp<arith::ConstantIndexOp>();
    if (!index) return failure();
    auto idx = index.value();

    // Iterate through the operands with 3 considerations in this order:
    // 1. If a static, non-1 dimension is seen, we know this to be the
    // broadcasted result
    // 2. If a single dynamic dimension is seen, we know this to be the
    // broadcasted result (with a possibly 1 or non-1 result)
    // 3. If no dynamic dimensions and no non-1 static dimensions are seen, we
    // know the result to be 1
    //
    // Iterate through all operands, keeping track of dynamic dimensions and
    // returning immediately if a non-1 static dimension is seen.
    ShapeOfOp dynamicShape;
    int64_t numDynamic = 0;
    for (auto shape : broadcastOp.getShapes()) {
      auto shapeOfOp = shape.getDefiningOp<ShapeOfOp>();
      if (!shapeOfOp) return failure();
      auto shapedType =
          mlir::cast<ShapedType>(shapeOfOp->getOperandTypes().front());

      // Abort on the existence of unranked shapes as they require more logic.
      if (!shapedType.hasRank()) return failure();
      if (shapedType.getRank() <= idx) continue;

      // Only consider dynamic dimensions after the loop because any non-1
      // static dimension takes precedence.
      if (shapedType.isDynamicDim(idx)) {
        dynamicShape = shapeOfOp;
        numDynamic++;
        continue;
      }

      if (shapedType.getDimSize(idx) == 1) continue;

      // Return as soon as we see a non-1 static dim.
      rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(
          op, shapedType.getDimSize(idx));
      return success();
    }
    if (numDynamic > 1) return failure();

    // Replace with the single dynamic dimension or 1.
    if (dynamicShape) {
      rewriter.replaceOpWithNewOp<tensor::DimOp>(op, dynamicShape.getArg(),
                                                 index);
    } else {
      rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, 1);
    }
    return success();
  }
};

struct ShapeSimplification
    : public impl::ShapeSimplificationBase<ShapeSimplification> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mhlo::MhloDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<shape::ShapeDialect>();
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());

    for (auto op : context->getRegisteredOperations()) {
      if (isa<shape::ShapeDialect, mhlo::MhloDialect>(op.getDialect()))
        op.getCanonicalizationPatterns(patterns, context);
    }

    patterns.add<BroadcastRemoveSubsumedOperandsPattern,
                 ExtractFromBroadcastedTensorCanonicalizationPattern>(context);

    auto func = getOperation();
    if (failed(applyPatternsGreedily(func, std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createShapeSimplification() {
  return std::make_unique<ShapeSimplification>();
}

}  // namespace mhlo
}  // namespace mlir

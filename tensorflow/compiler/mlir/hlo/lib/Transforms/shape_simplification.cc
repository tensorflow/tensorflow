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

// This file contains the patterns to simplify shape ops that were deemed not
// suitable for shape op canonicalization in MLIR Core.

#include <memory>
#include <utility>

#include "llvm/ADT/Optional.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Transforms/PassDetail.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

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
    SmallVector<int64_t> known_extents;
    SmallVector<SmallVector<int64_t, 4>, 4> operand_extents;
    for (Value shape : op.getShapes()) {
      auto &extents = operand_extents.emplace_back();
      if (failed(shape::getShapeVec(shape, extents))) return failure();

      // Prepend dynamic dims if sizes don't match.
      if (extents.size() > known_extents.size()) {
        known_extents.insert(known_extents.begin(),
                             extents.size() - known_extents.size(),
                             ShapedType::kDynamicSize);
      }

      for (size_t i = 0, e = extents.size(); i != e; ++i) {
        int64_t extent = extents[e - i - 1];
        if (extent != ShapedType::kDynamicSize && extent != 1) {
          int64_t &known_extent = known_extents[known_extents.size() - i - 1];
          // A dynamic dimension is subsumed by a static one, but bail out for
          // known conflicting shapes.
          if (known_extent != extent &&
              known_extent != ShapedType::kDynamicSize)
            return failure();
          known_extent = extent;
        }
      }
    }

    // If we've figured out all shapes to be constants we're done.
    if (!llvm::is_contained(known_extents, ShapedType::kDynamicSize)) {
      rewriter.replaceOpWithNewOp<ConstShapeOp>(
          op, op->getResultTypes(), rewriter.getIndexTensorAttr(known_extents));
      return success();
    }

    // If only some dimensions are known see if any of the operands can be
    // removed without affecting the result.
    SmallVector<Value, 4> filtered_operands;
    for (auto tuple : llvm::zip(op.getShapes(), operand_extents)) {
      Value shape = std::get<0>(tuple);
      auto &extents = std::get<1>(tuple);

      // An operand can't be dead if it's the only operand of the maximum rank.
      // Removing it would reduce the rank of the output.
      if (llvm::count_if(operand_extents, [&](ArrayRef<int64_t> op) {
            return op.size() >= extents.size();
          }) <= 1) {
        filtered_operands.push_back(shape);
        continue;
      }

      for (size_t i = 0, e = extents.size(); i != e; ++i) {
        int64_t extent = extents[e - i - 1];
        // A dimension of an operand can be subsumed if it's
        //   - a 1 dimension. All other operands will have 1 dims or better.
        if (extent == 1) continue;

        //   - a dynamic dim but the result is known to be constant.
        int64_t known_extent = known_extents[known_extents.size() - i - 1];
        assert(known_extent != 1);
        if (known_extent != ShapedType::kDynamicSize &&
            extent == ShapedType::kDynamicSize)
          continue;

        //   - a constant non-1 dimension equal to the "known" dim.
        // In this case we also have to check whether this operand is the only
        // contributor of that constant.
        if (known_extent != ShapedType::kDynamicSize &&
            extent == known_extent &&
            llvm::count_if(
                operand_extents, [&](ArrayRef<int64_t> operand_shape) {
                  return i < operand_shape.size() &&
                         operand_shape[operand_shape.size() - i - 1] ==
                             known_extent;
                }) > 1)
          continue;

        filtered_operands.push_back(shape);
        break;
      }
    }
    if (filtered_operands.size() != op.getShapes().size()) {
      rewriter.replaceOpWithNewOp<BroadcastOp>(op, op->getResultTypes(),
                                               filtered_operands);
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
    auto broadcast_op = op.tensor().getDefiningOp<BroadcastOp>();
    if (!broadcast_op) return failure();

    // Confirm that there is a constant index. This is required, so we can
    // confirm the DimOp's input will define the resulting broadcasted shape in
    // that dimension.
    auto index = op.indices().front().getDefiningOp<arith::ConstantIndexOp>();
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
    ShapeOfOp dynamic_shape;
    int64_t num_dynamic = 0;
    for (auto shape : broadcast_op.getShapes()) {
      auto shape_of_op = shape.getDefiningOp<ShapeOfOp>();
      if (!shape_of_op) return failure();
      auto shaped_type =
          shape_of_op->getOperandTypes().front().cast<ShapedType>();

      // Abort on the existence of unranked shapes as they require more logic.
      if (!shaped_type.hasRank()) return failure();
      if (shaped_type.getRank() <= idx) continue;

      // Only consider dynamic dimensions after the loop because any non-1
      // static dimension takes precedence.
      if (shaped_type.isDynamicDim(idx)) {
        dynamic_shape = shape_of_op;
        num_dynamic++;
        continue;
      }

      if (shaped_type.getDimSize(idx) == 1) continue;

      // Return as soon as we see a non-1 static dim.
      rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(
          op, shaped_type.getDimSize(idx));
      return success();
    }
    if (num_dynamic > 1) return failure();

    // Replace with the single dynamic dimension or 1.
    if (dynamic_shape) {
      rewriter.replaceOpWithNewOp<tensor::DimOp>(op, dynamic_shape.getArg(),
                                                 index);
    } else {
      rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, 1);
    }
    return success();
  }
};

struct ShapeSimplification
    : public ShapeSimplificationBase<ShapeSimplification> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithmeticDialect>();
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
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateShapeSimplification() {
  return std::make_unique<ShapeSimplification>();
}

}  // namespace mlir

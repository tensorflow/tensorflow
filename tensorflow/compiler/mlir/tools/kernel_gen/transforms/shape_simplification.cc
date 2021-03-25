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

#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {

namespace {

using shape::BroadcastOp;
using shape::ConstShapeOp;
using shape::ShapeOfOp;

// Given an input shape Value, try to obtain the shape's values.
LogicalResult getShapeVec(Value input, SmallVectorImpl<int64_t> &shape_values) {
  if (auto input_op = input.getDefiningOp<ShapeOfOp>()) {
    auto type = input_op.arg().getType().dyn_cast<ShapedType>();
    if (!type.hasRank()) return failure();
    shape_values = llvm::to_vector<6>(type.getShape());
    return success();
  }
  if (auto input_op = input.getDefiningOp<ConstShapeOp>()) {
    shape_values = llvm::to_vector<6>(input_op.shape().getValues<int64_t>());
    return success();
  }
  return failure();
}

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
    for (Value shape : op.shapes()) {
      auto &extents = operand_extents.emplace_back();
      if (failed(getShapeVec(shape, extents))) return failure();

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
    for (auto tuple : llvm::zip(op.shapes(), operand_extents)) {
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
    if (filtered_operands.size() != op.shapes().size()) {
      rewriter.replaceOpWithNewOp<BroadcastOp>(op, op->getResultTypes(),
                                               filtered_operands);
      return success();
    }
    return failure();
  }
};

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

struct ShapeSimplification
    : public ShapeSimplificationBase<ShapeSimplification> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mhlo::MhloDialect>();
    registry.insert<shape::ShapeDialect>();
  }

  void runOnFunction() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());

    Dialect *shape_dialect = context->getLoadedDialect<shape::ShapeDialect>();
    Dialect *mhlo_dialect = context->getLoadedDialect<mhlo::MhloDialect>();
    for (auto *op : context->getRegisteredOperations()) {
      if (op->dialect.getTypeID() == shape_dialect->getTypeID() ||
          op->dialect.getTypeID() == mhlo_dialect->getTypeID())
        op->getCanonicalizationPatterns(patterns, context);
    }

    patterns.insert<BroadcastRemoveSubsumedOperandsPattern>(context);

    auto func = getFunction();
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<FunctionPass> CreateShapeSimplification() {
  return std::make_unique<ShapeSimplification>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir

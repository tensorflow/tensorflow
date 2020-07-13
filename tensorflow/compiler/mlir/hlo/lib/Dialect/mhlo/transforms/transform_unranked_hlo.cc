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

#include "absl/memory/memory.h"
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h"

namespace mlir {
namespace mhlo {
namespace {

// TODO(frgossen): Make it variadic.
template <typename OpTy>
inline void AddLegalOpOnRankedTensor(ConversionTarget *target) {
  target->addDynamicallyLegalOp<OpTy>([](OpTy op) {
    return llvm::all_of((op.getOperation())->getOperandTypes(),
                        [&](Type t) { return t.isa<RankedTensorType>(); });
  });
}

/// Unary element-wise operations on unranked tensors can be applied to the
/// flattened tensor with the same effect.
/// This pattern rewrites every such operation to
///   (i)   flatten the input tensor,
///   (ii)  apply the unary operation, and
///   (iii) restore the original shape.
template <typename OpTy>
struct UnaryElementwiseOpConversion : public OpRewritePattern<OpTy> {
  explicit UnaryElementwiseOpConversion(MLIRContext *context)
      : OpRewritePattern<OpTy>(context) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // Don't apply conversion to ops with statically shaped operands.
    Value operand = op.getOperand();
    auto operandTy = operand.getType().dyn_cast<TensorType>();
    if (operandTy.hasRank()) return failure();

    // Generate IR to flatten the operand.
    auto loc = op.getLoc();
    Value shape = rewriter.create<shape::ShapeOfOp>(loc, operand);
    Value numElements = rewriter.create<shape::NumElementsOp>(
        loc, rewriter.getType<shape::SizeType>(), shape);
    Value numElementsAsIndex = rewriter.create<shape::SizeToIndexOp>(
        loc, rewriter.getIndexType(), numElements);
    Value flatShapeAsDimTensor =
        rewriter.create<TensorFromElementsOp>(loc, numElementsAsIndex);
    auto flatTensorTy = RankedTensorType::get({ShapedType::kDynamicSize},
                                              operandTy.getElementType());
    Value flatOperand = rewriter.create<mhlo::DynamicReshapeOp>(
        loc, flatTensorTy, operand, flatShapeAsDimTensor);

    // Generate IR for the actual operation.
    Value flatResult = rewriter.create<OpTy>(loc, flatTensorTy, flatOperand);

    // Generate IR to restore the original shape.
    auto extentTensorTy = RankedTensorType::get({ShapedType::kDynamicSize},
                                                rewriter.getIndexType());
    Value shapeAsExtentTensor =
        rewriter.create<shape::ToExtentTensorOp>(loc, extentTensorTy, shape);
    Value result = rewriter.create<mhlo::DynamicReshapeOp>(
        loc, operandTy, flatResult, shapeAsExtentTensor);
    rewriter.replaceOp(op, result);

    return success();
  }
};

/// Binary element-wise operation on unranked tensors can be applied to the
/// flattened operand tensors with the same effect.
/// This pattern rewrites every such operation to
///   (i)   flatten the operand tensors,
///   (ii)  apply the binary operation, and
//    (iii) restore the original shape.
template <typename OpTy>
struct BinaryElementwiseOpConversion : public OpRewritePattern<OpTy> {
  explicit BinaryElementwiseOpConversion(MLIRContext *context)
      : OpRewritePattern<OpTy>(context) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // Don't apply conversion unless both operands are unranked.
    if (op.lhs().getType().template isa<RankedTensorType>() ||
        op.rhs().getType().template isa<RankedTensorType>()) {
      return failure();
    }

    // Flatten operands.
    Type shapeTy = shape::ShapeType::get(rewriter.getContext());
    auto loc = op.getLoc();
    Value shapeLhs = rewriter.create<shape::ShapeOfOp>(loc, op.lhs());
    Value shapeRhs = rewriter.create<shape::ShapeOfOp>(loc, op.rhs());
    Value shape = rewriter.create<shape::AnyOp>(loc, shapeTy,
                                                ValueRange{shapeLhs, shapeRhs});
    Value numElements = rewriter.create<shape::NumElementsOp>(loc, shape);
    Value numElementsAsIndex =
        rewriter.create<shape::SizeToIndexOp>(loc, numElements);
    Value flatShape =
        rewriter.create<TensorFromElementsOp>(loc, numElementsAsIndex);
    TensorType lhsTy = op.lhs().getType().template cast<TensorType>();
    Type flatLhsTy = RankedTensorType::get({ShapedType::kDynamicSize},
                                           lhsTy.getElementType());
    Value flatLhs =
        rewriter.create<DynamicReshapeOp>(loc, flatLhsTy, op.lhs(), flatShape);
    TensorType rhsTy = op.rhs().getType().template cast<TensorType>();
    Type flatRhsTy = RankedTensorType::get({ShapedType::kDynamicSize},
                                           rhsTy.getElementType());
    Value flatRhs =
        rewriter.create<DynamicReshapeOp>(loc, flatRhsTy, op.rhs(), flatShape);

    // Apply actual operation to flattened operands.
    Value flatResult = rewriter.create<OpTy>(loc, flatLhs, flatRhs);

    // Restore original shape.
    auto extentTensorTy = RankedTensorType::get({ShapedType::kDynamicSize},
                                                rewriter.getIndexType());
    Value shapeAsExtentTensor =
        rewriter.create<shape::ToExtentTensorOp>(loc, extentTensorTy, shape);
    Value result = rewriter.create<DynamicReshapeOp>(
        loc, op.getType(), flatResult, shapeAsExtentTensor);
    rewriter.replaceOp(op, result);

    return success();
  }
};

struct TransformUnrankedHloPass
    : public PassWrapper<TransformUnrankedHloPass, FunctionPass> {
  void runOnFunction() override {
    // Setup conversion target.
    MLIRContext &ctx = getContext();
    ConversionTarget target(ctx);
    target.addLegalDialect<MhloDialect, StandardOpsDialect,
                           shape::ShapeDialect>();
    target.addLegalOp<FuncOp>();
    AddLegalOpOnRankedTensor<SqrtOp>(&target);
    AddLegalOpOnRankedTensor<AddOp>(&target);

    // Populate rewrite patterns.
    OwningRewritePatternList patterns;
    PopulateTransformUnrankedHloPatterns(&ctx, &patterns);

    // Apply transformation.
    if (failed(applyFullConversion(getFunction(), target, patterns)))
      return signalPassFailure();
  }
};

}  // namespace

void PopulateTransformUnrankedHloPatterns(MLIRContext *context,
                                          OwningRewritePatternList *patterns) {
  // TODO(frgossen): Populate all unary and binary operations.
  // clang-format off
  patterns->insert<
      BinaryElementwiseOpConversion<AddOp>,
      UnaryElementwiseOpConversion<SqrtOp>>(context);
  // clang-format on
}

static PassRegistration<TransformUnrankedHloPass> transform_unranked_hlo_pass(
    "transform-unranked-hlo",
    "Realize element-wise operations on ranked tensors where possible");

}  // namespace mhlo
}  // namespace mlir

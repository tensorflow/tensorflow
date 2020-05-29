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
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"

namespace mlir {
namespace xla_hlo {
namespace {

template <typename OpTy>
inline void AddLegalOpOnRankedTensor(ConversionTarget *conversionTarget) {
  conversionTarget->addDynamicallyLegalOp<OpTy>([](OpTy op) {
    return op.getOperand().getType().template cast<TensorType>().hasRank();
  });
}

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
    Value flatOperand = rewriter.create<xla_hlo::DynamicReshapeOp>(
        loc, flatTensorTy, operand, flatShapeAsDimTensor);

    // Generate IR for the actual operation.
    Value flatResult = rewriter.create<OpTy>(loc, flatTensorTy, flatOperand);

    // Generate IR to restore the original shape.
    auto extentTensorTy = RankedTensorType::get({ShapedType::kDynamicSize},
                                                rewriter.getIndexType());
    Value shapeAsExtentTensor =
        rewriter.create<shape::ToExtentTensorOp>(loc, extentTensorTy, shape);
    Value result = rewriter.create<xla_hlo::DynamicReshapeOp>(
        loc, operandTy, flatResult, shapeAsExtentTensor);
    rewriter.replaceOp(op, result);

    return success();
  }
};

struct TransformUnrankedHloPass
    : public PassWrapper<TransformUnrankedHloPass, FunctionPass> {
  void runOnFunction() override {
    ConversionTarget conversionTarget(getContext());
    OwningRewritePatternList conversionPatterns;
    SetupTransformUnrankedHloLegality(&getContext(), &conversionTarget);
    PopulateTransformUnrankedHloPatterns(&getContext(), &conversionPatterns);

    if (failed(applyPartialConversion(getFunction(), conversionTarget,
                                      conversionPatterns)))
      return signalPassFailure();
  }
};

}  // namespace

void SetupTransformUnrankedHloLegality(MLIRContext *context,
                                       ConversionTarget *conversionTarget) {
  conversionTarget->addLegalDialect<XlaHloDialect, StandardOpsDialect,
                                    shape::ShapeDialect>();

  // Targeted operations are only legal when they operate on ranked tensors.
  AddLegalOpOnRankedTensor<SqrtOp>(conversionTarget);
}

void PopulateTransformUnrankedHloPatterns(MLIRContext *context,
                                          OwningRewritePatternList *patterns) {
  patterns->insert<UnaryElementwiseOpConversion<SqrtOp>>(context);
}

std::unique_ptr<OperationPass<FuncOp>> createTransformUnrankedHloPass() {
  return absl::make_unique<TransformUnrankedHloPass>();
}

static PassRegistration<TransformUnrankedHloPass> transform_unranked_hlo_pass(
    "transform-unranked-hlo",
    "Realize element-wise operations on ranked tensors where possible");

}  // namespace xla_hlo
}  // namespace mlir

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

#include <memory>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/utils/validators.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace odml {

class FuseMhloMulAndConvolutionPattern : public OpRewritePattern<mhlo::MulOp> {
 public:
  using OpRewritePattern<mhlo::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::MulOp mul_op,
                                PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops.
    mhlo::ConvolutionOp conv_op;
    mhlo::BroadcastInDimOp broadcast_op;
    mhlo::ConstantOp filter;
    mhlo::ConstantOp multiplier;
    mlir::ElementsAttr filter_value, mul_value;
    mlir::DenseIntElementsAttr broadcast_dims;

    // Match and capture values/attributes.
    Value lhs = mul_op.getLhs();
    Value rhs = mul_op.getRhs();
    conv_op = lhs.getDefiningOp<mhlo::ConvolutionOp>();
    if (conv_op == nullptr) {
      return failure();
    }
    filter = conv_op.getRhs().getDefiningOp<mhlo::ConstantOp>();
    if (filter == nullptr) {
      return failure();
    }
    broadcast_op = rhs.getDefiningOp<mhlo::BroadcastInDimOp>();
    multiplier =
        (broadcast_op == nullptr)
            ? rhs.getDefiningOp<mhlo::ConstantOp>()
            : broadcast_op.getOperand().getDefiningOp<mhlo::ConstantOp>();
    if (multiplier == nullptr) {
      return failure();
    }
    auto result_type = OpTrait::util::getBroadcastedType(filter.getType(),
                                                         multiplier.getType());
    if (!result_type) {
      return rewriter.notifyMatchFailure(mul_op, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'filter, multiplier' failed to satisfy constraint: "
                "non-broadcastable operands";
      });
    }
    filter_value = filter.getValue();
    mul_value = multiplier.getValue();
    // In MHLO, Conv filter is in HWIO format, Depthwise conv filter is in HW1O
    // format and backprop input conv filter is in HWOI format.
    // Only fuses multiplier if all dimensions other than the out channel
    // dimension are equal to 1.
    if (!TFL::IsDimensionsDegenerateExceptLastOne(
            mul_value.getType().getShape())) {
      return rewriter.notifyMatchFailure(mul_op, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'mul_value' failed to satisfy constraint: "
                "unsupported dimensions";
      });
    }
    if (!((*conv_op.getODSResults(0).begin()).hasOneUse())) {
      return rewriter.notifyMatchFailure(mul_op, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'conv' failed to satisfy constraint: has one use";
      });
    }

    // Rewrite
    broadcast_dims = broadcast_op.getBroadcastDimensions();
    if (broadcast_dims == nullptr) {
      const auto filter_rank = filter_value.getType().getRank();
      auto dimsType = RankedTensorType::get({1}, rewriter.getIntegerType(64));
      broadcast_dims = DenseIntElementsAttr::get(dimsType, {filter_rank - 1});
    }
    Value broadcast_multiplier = rewriter.create<mhlo::BroadcastInDimOp>(
        mul_op.getLoc(), filter.getType(), multiplier, broadcast_dims);
    Value new_filter = rewriter.create<mhlo::MulOp>(
        mul_op.getLoc(), filter.getType(), filter, broadcast_multiplier);
    Value new_conv = rewriter.create<mhlo::ConvolutionOp>(
        mul_op.getLoc(), conv_op.getType(), conv_op.getLhs(), new_filter,
        conv_op.getWindowStridesAttr(), conv_op.getPaddingAttr(),
        conv_op.getLhsDilationAttr(), conv_op.getRhsDilationAttr(),
        conv_op.getWindowReversalAttr(), conv_op.getDimensionNumbers(),
        conv_op.getFeatureGroupCount(), conv_op.getBatchGroupCount(),
        conv_op.getPrecisionConfigAttr());
    rewriter.replaceOp(mul_op, {new_conv});

    return success();
  }
};

class FuseMhloConvolutionPass
    : public PassWrapper<FuseMhloConvolutionPass, OperationPass<func::FuncOp>> {
 public:
  StringRef getArgument() const final { return "fuse-mhlo-convolution-pass"; }
  StringRef getDescription() const final {
    return "Fuses MHLO binary element-wise ops and convolution op";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseMhloMulAndConvolutionPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createFuseConvolutionPass() {
  return std::make_unique<FuseMhloConvolutionPass>();
}

static PassRegistration<FuseMhloConvolutionPass> pass;

}  // namespace odml
}  // namespace mlir

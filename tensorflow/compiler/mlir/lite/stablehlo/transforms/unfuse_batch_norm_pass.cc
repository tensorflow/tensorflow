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
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace odml {

// Broadcasts the 1D value tensor 'value_1d' to the shape of 'result_type'. If
// 'shape_value' is initialized, creates a dynamic broadcast, otherwise creates
// a static broadcast.
Value broadcastToFeatureDim(Location loc, RankedTensorType result_type,
                            Value value1d, Value shape_value,
                            int64_t feature_dim, PatternRewriter &rewriter) {
  auto dims_type =
      RankedTensorType::get(/*shape=*/{1}, rewriter.getIntegerType(64));
  auto dims = DenseIntElementsAttr::get(dims_type, {feature_dim});
  if (shape_value) {
    return rewriter.createOrFold<mhlo::DynamicBroadcastInDimOp>(
        loc, result_type, value1d, shape_value, dims);
  }
  assert(result_type.hasStaticShape());
  return rewriter.create<mhlo::BroadcastInDimOp>(loc, result_type, value1d,
                                                 dims);
}

// Gets the shape of operand, assuming it is a dynamic shape with static rank.
Value getShapeValue(Location loc, Value operand, PatternRewriter &rewriter) {
  RankedTensorType resultType = operand.getType().dyn_cast<RankedTensorType>();
  return rewriter.create<shape::ShapeOfOp>(
      loc,
      RankedTensorType::get(/*shape=*/{resultType.getRank()},
                            rewriter.getIndexType()),
      operand);
}

Value materializeEpsilon(Operation *op, FloatAttr epsilon_attr,
                         FloatType fp_type, Value broadcast_to,
                         RankedTensorType broadcast_to_type,
                         PatternRewriter &rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  if (epsilon_attr.getType() != fp_type) {
    // Need to convert.
    bool loses_info;
    APFloat epsilon_float = epsilon_attr.getValue();
    auto status = epsilon_float.convert(
        fp_type.getFloatSemantics(), APFloat::rmNearestTiesToEven, &loses_info);
    if ((status & (~APFloat::opInexact)) != APFloat::opOK) {
      op->emitWarning() << "Could not convert batch_norm epsilon to target fp "
                           "type: opStatus = "
                        << static_cast<int>(status);
      return nullptr;
    }
    if (loses_info) {
      op->emitWarning("Conversion of epsilon loses precision");
    }
    epsilon_attr = b.getFloatAttr(fp_type, epsilon_float);
  }

  auto scalar_type = RankedTensorType::get(/*shape=*/{}, fp_type);
  auto epsilon_tensor_attr =
      DenseElementsAttr::get(scalar_type, {epsilon_attr.cast<Attribute>()});
  Value epsilon = b.create<mhlo::ConstantOp>(epsilon_tensor_attr);
  auto dims_type = RankedTensorType::get(/*shape=*/{0}, b.getIntegerType(64));
  auto dims = DenseIntElementsAttr::get(dims_type, SmallVector<int64_t, 1>{});
  if (broadcast_to_type.hasStaticShape()) {
    return b.create<mhlo::BroadcastInDimOp>(broadcast_to_type, epsilon, dims);
  }
  Value shape_value = getShapeValue(op->getLoc(), broadcast_to, rewriter);
  return b.createOrFold<mhlo::DynamicBroadcastInDimOp>(
      broadcast_to_type, epsilon, shape_value, dims);
}

class UnfuseBatchNormInferencePattern
    : public OpRewritePattern<mhlo::BatchNormInferenceOp> {
 public:
  using OpRewritePattern<mhlo::BatchNormInferenceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::BatchNormInferenceOp bn_op,
                                PatternRewriter &rewriter) const override {
    // Enforce type invariants.
    // Note that we deduce the actual element type from the variance,
    // which should not be subject to quantization at a higher level.
    auto input_type = bn_op.getOperand().getType().dyn_cast<RankedTensorType>();
    auto variance_type =
        bn_op.getVariance().getType().dyn_cast<RankedTensorType>();
    if (!input_type || !variance_type) {
      return failure();
    }
    auto fp_type = variance_type.getElementType().dyn_cast<FloatType>();
    if (!fp_type) {
      return failure();
    }

    // result = (x - mean) * scale / sqrt(variance + epsilon) + offset
    // Let multiplier = scale / sqrt(variance + epsilon), to compute
    // (x - mean) * scale / sqrt(variance + epsilon) + offset,
    // is then to compute (x * multiplier) + (offset - mean * multiplier).

    auto epsilon = materializeEpsilon(
        bn_op.getOperation(), bn_op.getEpsilonAttr(), fp_type,
        bn_op.getVariance(), variance_type, rewriter);
    if (!epsilon) {
      return failure();
    }

    // Compute multiplier = scale / sqrt(variance + epsilon)
    Value multiplier = rewriter.create<mhlo::AddOp>(
        bn_op.getLoc(), bn_op.getVariance(), epsilon);
    multiplier = rewriter.create<mhlo::RsqrtOp>(bn_op.getLoc(), multiplier);
    multiplier = rewriter.create<mhlo::MulOp>(bn_op.getLoc(), multiplier,
                                              bn_op.getScale());

    // Compute rhs = offset - mean * multiplier
    Value rhs = rewriter.create<mhlo::MulOp>(bn_op.getLoc(), multiplier,
                                             bn_op.getMean());
    rhs = rewriter.create<mhlo::SubtractOp>(bn_op.getLoc(), bn_op.getOffset(),
                                            rhs);

    // Broadcast `multiplier` and `rhs`
    Value shape_value;
    if (!input_type.hasStaticShape()) {
      shape_value = getShapeValue(bn_op.getLoc(), bn_op.getOperand(), rewriter);
    }
    int64_t feature_dim = bn_op.getFeatureIndex();
    auto broadcast_multiplier =
        broadcastToFeatureDim(bn_op.getLoc(), input_type, multiplier,
                              shape_value, feature_dim, rewriter);
    auto broadcast_rhs = broadcastToFeatureDim(
        bn_op.getLoc(), input_type, rhs, shape_value, feature_dim, rewriter);

    // Computes x * multiplier + rhs
    Value lhs = rewriter.create<mhlo::MulOp>(bn_op.getLoc(), bn_op.getOperand(),
                                             broadcast_multiplier);
    rewriter.replaceOpWithNewOp<mhlo::AddOp>(bn_op, lhs, broadcast_rhs);

    return success();
  }
};

class UnfuseMhloBatchNormPass
    : public PassWrapper<UnfuseMhloBatchNormPass, OperationPass<func::FuncOp>> {
 public:
  StringRef getArgument() const final { return "unfuse-mhlo-batch-norm-pass"; }
  StringRef getDescription() const final {
    return "Unfuses MHLO batch norm inference op into arithmetic ops";
  }
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<shape::ShapeDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<UnfuseBatchNormInferencePattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createUnfuseBatchNormPass() {
  return std::make_unique<UnfuseMhloBatchNormPass>();
}

static PassRegistration<UnfuseMhloBatchNormPass> pass;

}  // namespace odml
}  // namespace mlir

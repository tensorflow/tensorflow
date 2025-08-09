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

#include "tensorflow/compiler/mlir/stablehlo/transforms/mhlo_passes/unfuse_batch_norm_pass.h"

#include <cassert>
#include <cstdint>
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
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

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
  RankedTensorType resultType =
      mlir::dyn_cast<RankedTensorType>(operand.getType());
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
  auto epsilon_tensor_attr = DenseElementsAttr::get(
      scalar_type, {mlir::cast<Attribute>(epsilon_attr)});
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

class UnfuseBatchNormTrainingPattern
    : public OpRewritePattern<mhlo::BatchNormTrainingOp> {
 public:
  using OpRewritePattern<mhlo::BatchNormTrainingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::BatchNormTrainingOp bn_op,
                                PatternRewriter &rewriter) const override {
    auto inputs = bn_op.getOperand();
    auto input_type = mlir::dyn_cast<RankedTensorType>(inputs.getType());
    if (!input_type) {
      return failure();
    }
    auto feature_index = bn_op.getFeatureIndex();

    // Compute mean
    int64_t input_last_dim = input_type.getRank() - 1;
    auto dims_type = RankedTensorType::get(/*shape=*/{input_last_dim},
                                           rewriter.getIntegerType(32));
    ::mlir::SmallVector<int32_t> reduce_dim_axes;
    for (int i = 0; i < input_type.getRank(); ++i) {
      if (i != feature_index) {
        reduce_dim_axes.push_back(i);
      }
    }
    auto mean_dims = DenseIntElementsAttr::get(dims_type, reduce_dim_axes);
    ::mlir::TF::ConstOp reduce_dim_op =
        rewriter.create<TF::ConstOp>(bn_op.getLoc(), mean_dims);
    int64_t feature_dim_size = input_type.getDimSize(feature_index);
    auto mean_var_type = RankedTensorType::get(/*shape=*/{feature_dim_size},
                                               rewriter.getF32Type());
    ::mlir::Value mean = rewriter.create<TF::MeanOp>(
        bn_op.getLoc(), mean_var_type, inputs, reduce_dim_op,
        /*keep_dims=*/rewriter.getBoolAttr(false));

    // Compute variance
    Value shape_value =
        getShapeValue(bn_op.getLoc(), bn_op.getOperand(), rewriter);
    auto broadcast_mean = broadcastToFeatureDim(
        bn_op.getLoc(), input_type, mean, shape_value, feature_index, rewriter);
    ::mlir::Value square_diff = rewriter.create<TF::SquaredDifferenceOp>(
        bn_op.getLoc(), inputs, broadcast_mean);
    ::mlir::Value variance = rewriter.create<TF::MeanOp>(
        bn_op.getLoc(), mean_var_type, square_diff, reduce_dim_op,
        /*keep_dims=*/rewriter.getBoolAttr(false));

    // Invoke BatchNormInferenceOp
    ::mlir::FloatAttr epsilon = bn_op.getEpsilonAttr();
    ::mlir::Value batch_norm = rewriter.create<mhlo::BatchNormInferenceOp>(
        bn_op.getLoc(), inputs, bn_op.getScale(), bn_op.getOffset(), mean,
        variance, epsilon, rewriter.getI64IntegerAttr(feature_index));

    // Return normalized values, mean, variable.
    rewriter.replaceOp(bn_op, ::mlir::ValueRange{batch_norm, mean, variance});
    return success();
  }
};

class UnfuseBatchNormInferencePattern
    : public OpRewritePattern<mhlo::BatchNormInferenceOp> {
 public:
  using OpRewritePattern<mhlo::BatchNormInferenceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::BatchNormInferenceOp bn_op,
                                PatternRewriter &rewriter) const override {
    // Enforce type invariants.
    // Note that we deduce the actual element type from the variance,
    // which should not be subject to quantization at a higher level.
    auto input_type =
        mlir::dyn_cast<RankedTensorType>(bn_op.getOperand().getType());
    auto variance_type =
        mlir::dyn_cast<RankedTensorType>(bn_op.getVariance().getType());
    if (!input_type || !variance_type) {
      return failure();
    }
    auto fp_type = mlir::dyn_cast<FloatType>(variance_type.getElementType());
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

    // Computes x * multiplier + rhs
    Value lhs = rewriter.create<mhlo::MulOp>(bn_op.getLoc(), bn_op.getOperand(),
                                             broadcast_multiplier);
    auto broadcast_rhs = broadcastToFeatureDim(
        bn_op.getLoc(), input_type, rhs, shape_value, feature_dim, rewriter);
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
    // TODO(b/299514833): Remove TensorFlowDialect usage.
    registry.insert<shape::ShapeDialect, mlir::TF::TensorFlowDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<UnfuseBatchNormTrainingPattern>(&getContext());
    patterns.add<UnfuseBatchNormInferencePattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "llvm/ADT/SmallVector.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace mhlo {

namespace {

// Broadcasts the 1D value tensor 'value_1d' to the shape of 'result_type'. If
// 'shape_value' is initialized, creates a dynamic broadcast, otherwise creates
// a static broadcast.
Value BroadcastToFeatureDim(Location loc, RankedTensorType result_type,
                            Value value_1d, Value shape_value,
                            int64_t feature_dim,
                            PatternRewriter& rewriter) {  // NOLINT
  Builder b(rewriter.getContext());
  auto dims_type = RankedTensorType::get({1}, b.getIntegerType(64));
  auto dims = DenseIntElementsAttr::get(dims_type, {feature_dim});
  if (shape_value) {
    return rewriter.createOrFold<mhlo::DynamicBroadcastInDimOp>(
        loc, result_type, value_1d, shape_value, dims);
  }
  assert(result_type.hasStaticShape());
  return rewriter.create<mhlo::BroadcastInDimOp>(loc, result_type, value_1d,
                                                 dims);
}

// Calculate the shape value of operand, assuming it is a dynamic shape with
// static rank.
Value CalculateShapeValue(Location loc, Value operand,
                          PatternRewriter& rewriter) {  // NOLINT
  RankedTensorType result_type = operand.getType().dyn_cast<RankedTensorType>();
  llvm::SmallVector<Value, 4> shape_values;
  int64_t rank = result_type.getRank();
  shape_values.reserve(rank);
  for (int64_t i = 0; i < rank; ++i) {
    shape_values.push_back(
        rewriter.create<mlir::tensor::DimOp>(loc, operand, i));
  }
  return rewriter.create<tensor::FromElementsOp>(loc, shape_values);
}

Value MaterializeEpsilon(Operation* op, FloatAttr epsilon_attr,
                         FloatType fp_type, Value variance,
                         RankedTensorType broadcast_to_type,
                         PatternRewriter& rewriter) {  // NOLINT
  Builder b(rewriter.getContext());
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

  auto scalar_type = RankedTensorType::get({}, fp_type);
  auto epsilon_tensor_attr =
      DenseElementsAttr::get(scalar_type, {epsilon_attr.cast<Attribute>()});
  Value epsilon =
      rewriter.create<mhlo::ConstOp>(op->getLoc(), epsilon_tensor_attr);
  auto dims_type = RankedTensorType::get({0}, b.getIntegerType(64));
  auto dims = DenseIntElementsAttr::get(dims_type, SmallVector<int64_t, 1>{});
  if (broadcast_to_type.hasStaticShape()) {
    return rewriter.create<mhlo::BroadcastInDimOp>(
        op->getLoc(), broadcast_to_type, epsilon, /*broadcast_dims=*/dims);
  }
  Value shape_value = CalculateShapeValue(op->getLoc(), variance, rewriter);
  return rewriter.createOrFold<mhlo::DynamicBroadcastInDimOp>(
      op->getLoc(), broadcast_to_type, epsilon, shape_value,
      /*broadcast_dims=*/dims);
}

class UnfuseBatchNormInferencePattern
    : public OpRewritePattern<mhlo::BatchNormInferenceOp> {
 public:
  using OpRewritePattern<mhlo::BatchNormInferenceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::BatchNormInferenceOp bn_op,
                                PatternRewriter& rewriter) const override {
    // Enforce type invariants.
    // Note that we deduce the actual element type from the variance,
    // which should not be subject to quantization at a higher level.
    auto input_type = bn_op.operand().getType().dyn_cast<RankedTensorType>();
    auto variance_type =
        bn_op.variance().getType().dyn_cast<RankedTensorType>();
    if (!input_type || !variance_type) {
      return failure();
    }
    auto fp_type = variance_type.getElementType().dyn_cast<FloatType>();
    if (!fp_type) {
      return failure();
    }
    int64_t feature_dim = bn_op.feature_index();

    // Add epsilon to the variance and sqrt to get stddev:
    // stddev = sqrt(variance + epsilon)
    auto epsilon =
        MaterializeEpsilon(bn_op.getOperation(), bn_op.epsilonAttr(), fp_type,
                           bn_op.variance(), variance_type, rewriter);
    if (!epsilon) {
      return failure();
    }
    Value stddev =
        rewriter.create<mhlo::AddOp>(bn_op.getLoc(), bn_op.variance(), epsilon);
    stddev = rewriter.create<mhlo::SqrtOp>(bn_op.getLoc(), stddev);

    // Broadcast all terms.
    Value shape_value;
    if (!input_type.hasStaticShape()) {
      shape_value =
          CalculateShapeValue(bn_op.getLoc(), bn_op.operand(), rewriter);
    }
    auto broadcast_scale =
        BroadcastToFeatureDim(bn_op.getLoc(), input_type, bn_op.scale(),
                              shape_value, feature_dim, rewriter);
    auto broadcast_offset =
        BroadcastToFeatureDim(bn_op.getLoc(), input_type, bn_op.offset(),
                              shape_value, feature_dim, rewriter);
    auto broadcast_mean =
        BroadcastToFeatureDim(bn_op.getLoc(), input_type, bn_op.mean(),
                              shape_value, feature_dim, rewriter);
    auto broadcast_stddev = BroadcastToFeatureDim(
        bn_op.getLoc(), input_type, stddev, shape_value, feature_dim, rewriter);

    // Compute:
    // scale * (input - mean) / stddev + offset
    Value result = rewriter.create<mhlo::SubOp>(bn_op.getLoc(), bn_op.operand(),
                                                broadcast_mean);
    result =
        rewriter.create<mhlo::MulOp>(bn_op.getLoc(), result, broadcast_scale);
    result =
        rewriter.create<mhlo::DivOp>(bn_op.getLoc(), result, broadcast_stddev);
    rewriter.replaceOpWithNewOp<mhlo::AddOp>(bn_op, result, broadcast_offset);

    return success();
  }
};

}  // namespace

// Populates conversion patterns to unfuse batch normalization operations.
// In combination with marking such ops as illegal, this allows backends that
// do not have special support for fused batchnorm to use simpler arithmetic
// primitives.
void PopulateUnfuseBatchNormPatterns(MLIRContext* context,
                                     OwningRewritePatternList* patterns) {
  patterns->insert<UnfuseBatchNormInferencePattern>(context);
}

}  // namespace mhlo
}  // namespace mlir

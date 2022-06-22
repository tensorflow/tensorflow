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
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
                         FloatType fp_type, Value broadcast_to,
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
  Value shape_value = CalculateShapeValue(op->getLoc(), broadcast_to, rewriter);
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

// Create "mhlo.reduce", "operand" is reduce input and "zero" is init value,
// reduce sum from operand to operand[feature_index].
Value CreateReduce(Location loc, Value operand, Value zero,
                   SmallVector<int64_t>& reduce_dims, int64_t feature_index,
                   PatternRewriter& rewriter) {
  auto operand_type = operand.getType().cast<RankedTensorType>();
  Type reduce_result_type = RankedTensorType::get(
      {operand_type.getDimSize(feature_index)}, operand_type.getElementType());
  mhlo::ReduceOp reduce =
      rewriter.create<mhlo::ReduceOp>(loc, reduce_result_type, operand, zero,
                                      rewriter.getI64TensorAttr(reduce_dims));

  // setup "mhlo.reduce"'s body
  Region& region = reduce.body();
  Block& block = region.emplaceBlock();
  RankedTensorType block_argument_type =
      RankedTensorType::get({}, operand_type.getElementType());
  block.addArgument(block_argument_type, loc);
  block.addArgument(block_argument_type, loc);
  auto* first_argument = block.args_begin();
  auto second_argument = block.args_rbegin();
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);
    Value add_result =
        rewriter.create<mhlo::AddOp>(loc, *first_argument, *second_argument);
    rewriter.create<mhlo::ReturnOp>(loc, add_result);
  }

  return reduce.getResult(0);
}

// Calculate total reduce size, assuming it is a dynamic shape with static rank.
// Reduce from operand to operand[feature_index]
Value CalculateReduceSize(Operation* op, Value operand,
                          RankedTensorType operand_type,
                          RankedTensorType scale_type, int64_t feature_index,
                          PatternRewriter& rewriter) {
  Location loc = op->getLoc();
  if (!operand_type.hasStaticShape()) {
    // the "operand" has dynamic shape with static rank
    llvm::SmallVector<Value, 4> reduce_values;
    for (int64_t i = 0, e = operand_type.getRank(); i < e; i++) {
      if (i != feature_index) {
        reduce_values.push_back(
            rewriter.create<tensor::DimOp>(loc, operand, i));
      }
    }
    assert(!reduce_values.empty());
    Value reduce_size = reduce_values[0];
    for (size_t i = 1, e = reduce_values.size(); i < e; i++) {
      reduce_size =
          rewriter.create<arith::MulIOp>(loc, reduce_size, reduce_values[i]);
    }
    reduce_size = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), reduce_size);
    reduce_size = rewriter.create<tensor::FromElementsOp>(loc, reduce_size);
    reduce_size = rewriter.create<mhlo::ConvertOp>(
        loc, RankedTensorType::get({1}, operand_type.getElementType()),
        reduce_size);
    reduce_size = rewriter.create<mhlo::ReshapeOp>(
        loc, RankedTensorType::get({}, operand_type.getElementType()),
        reduce_size);
    Value feature_size =
        rewriter.create<tensor::DimOp>(loc, operand, feature_index);
    feature_size = rewriter.create<tensor::FromElementsOp>(loc, feature_size);

    return rewriter.createOrFold<mhlo::DynamicBroadcastInDimOp>(
        loc, scale_type, reduce_size, feature_size,
        rewriter.getI64TensorAttr({}));
  }

  // the "operand" has static shape
  int64_t reduce_dims_size = 1;
  for (int64_t i = 0, e = operand_type.getRank(); i < e; i++) {
    if (i != feature_index) {
      reduce_dims_size *= operand_type.getDimSize(i);
    }
  }
  llvm::APFloat float_value(static_cast<double>(reduce_dims_size));
  bool loses_info;
  float_value.convert(
      scale_type.getElementType().cast<FloatType>().getFloatSemantics(),
      APFloat::rmNearestTiesToEven, &loses_info);
  if (loses_info) {
    op->emitWarning("Conversion of reduce_dims_size loses precision");
  }
  Value reduce_size = rewriter.create<mhlo::ConstOp>(
      loc, DenseFPElementsAttr::get(scale_type, float_value));
  return reduce_size;
}

// BatchNormTraining(X, scale, offset) =
//    ((X - E[X]) / Sqrt(Var[X] + epsilon)) * scale + offset.
class UnfuseBatchNormTrainingPattern
    : public OpRewritePattern<mhlo::BatchNormTrainingOp> {
 public:
  using OpRewritePattern<mhlo::BatchNormTrainingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::BatchNormTrainingOp bn_op,
                                PatternRewriter& rewriter) const override {
    auto operand_type = bn_op.operand().getType().dyn_cast<RankedTensorType>();
    auto scale_type = bn_op.scale().getType().dyn_cast<RankedTensorType>();
    if (!operand_type || !scale_type) {
      return failure();
    }
    auto fp_type = operand_type.getElementType().dyn_cast<FloatType>();
    if (!fp_type) {
      return failure();
    }
    int64_t feature_index = bn_op.feature_index();
    SmallVector<int64_t> dimensions_without_feature;
    for (int64_t i = 0, e = operand_type.getRank(); i < e; i++) {
      if (i != feature_index) {
        dimensions_without_feature.push_back(i);
      }
    }

    // zero constant
    Value const_zero = rewriter.create<mhlo::ConstOp>(
        bn_op.getLoc(), DenseFPElementsAttr::get(
                            RankedTensorType::get({}, fp_type),
                            APFloat::getZero(fp_type.getFloatSemantics())));
    // epsilon
    auto epsilon =
        MaterializeEpsilon(bn_op.getOperation(), bn_op.epsilonAttr(), fp_type,
                           bn_op.scale(), scale_type, rewriter);
    if (!epsilon) {
      return failure();
    }
    // reduce size constant
    Value reduce_size =
        CalculateReduceSize(bn_op.getOperation(), bn_op.operand(), operand_type,
                            scale_type, feature_index, rewriter);
    if (!reduce_size) {
      return failure();
    }
    // Sum[X]
    Value sum =
        CreateReduce(bn_op.getLoc(), bn_op.operand(), const_zero,
                     dimensions_without_feature, feature_index, rewriter);
    // X^2
    Value operand_square = rewriter.create<mhlo::MulOp>(
        bn_op.getLoc(), bn_op.operand(), bn_op.operand());
    // Sum[X^2]
    Value square_sum =
        CreateReduce(bn_op.getLoc(), operand_square, const_zero,
                     dimensions_without_feature, feature_index, rewriter);
    // E[X]
    Value mean = rewriter.create<mhlo::DivOp>(bn_op.getLoc(), sum, reduce_size);
    // E[X^2]
    Value square_mean =
        rewriter.create<mhlo::DivOp>(bn_op.getLoc(), square_sum, reduce_size);
    // E^2[X]
    Value mean_square =
        rewriter.create<mhlo::MulOp>(bn_op.getLoc(), mean, mean);
    // Var[X]
    Value var =
        rewriter.create<mhlo::SubOp>(bn_op.getLoc(), square_mean, mean_square);
    // Var[X] + epsilon
    Value var_add_epsilon =
        rewriter.create<mhlo::AddOp>(bn_op.getLoc(), var, epsilon);
    // Sqrt(Var[X] + epsilon)
    Value sqrt_var =
        rewriter.create<mhlo::SqrtOp>(bn_op.getLoc(), var_add_epsilon);

    Value shape_value;
    if (!operand_type.hasStaticShape()) {
      shape_value =
          CalculateShapeValue(bn_op.getLoc(), bn_op.operand(), rewriter);
    }
    // X - E[X]
    Value mean_broadcast =
        BroadcastToFeatureDim(bn_op.getLoc(), operand_type, mean, shape_value,
                              feature_index, rewriter);
    Value operand_minus_mean = rewriter.create<mhlo::SubOp>(
        bn_op.getLoc(), bn_op.operand(), mean_broadcast);
    // (X - E[X]) / Sqrt(Var[X] + epsilon)
    Value sqrt_var_broadcast =
        BroadcastToFeatureDim(bn_op.getLoc(), operand_type, sqrt_var,
                              shape_value, feature_index, rewriter);
    Value normalized = rewriter.create<mhlo::DivOp>(
        bn_op.getLoc(), operand_minus_mean, sqrt_var_broadcast);

    // ((X - E[X]) / Sqrt(Var[X] + epsilon)) * scale
    Value scale_broadcast =
        BroadcastToFeatureDim(bn_op.getLoc(), operand_type, bn_op.scale(),
                              shape_value, feature_index, rewriter);
    Value scaled_normalized = rewriter.create<mhlo::MulOp>(
        bn_op.getLoc(), normalized, scale_broadcast);
    // ((X - E[X]) / Sqrt(Var[X] + epsilon)) * scale + offset.
    Value offset_broadcast =
        BroadcastToFeatureDim(bn_op.getLoc(), operand_type, bn_op.offset(),
                              shape_value, feature_index, rewriter);
    Value shifted_normalized = rewriter.create<mhlo::AddOp>(
        bn_op.getLoc(), scaled_normalized, offset_broadcast);

    // results
    SmallVector<Value> results = {shifted_normalized, mean, var};
    rewriter.replaceOp(bn_op, results);

    return success();
  }
};

}  // namespace

// Populates conversion patterns to unfuse batch normalization operations.
// In combination with marking such ops as illegal, this allows backends that
// do not have special support for fused batchnorm to use simpler arithmetic
// primitives.
void PopulateUnfuseBatchNormPatterns(MLIRContext* context,
                                     RewritePatternSet* patterns) {
  patterns->add<UnfuseBatchNormInferencePattern>(context);
  patterns->add<UnfuseBatchNormTrainingPattern>(context);
}

}  // namespace mhlo
}  // namespace mlir

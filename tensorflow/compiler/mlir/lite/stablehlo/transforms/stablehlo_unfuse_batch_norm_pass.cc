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

#include <cassert>
#include <cstdint>
#include <utility>

#include "stablehlo/dialect/StablehloOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h"  // NOLINT: Used in stablehlo_passes.h.inc

namespace mlir {
namespace odml {

#define GEN_PASS_DEF_STABLEHLOUNFUSEBATCHNORMPASS
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h.inc"

namespace {

// Broadcasts the 1D value tensor 'value_1d' to the shape of 'result_type'. If
// 'shape_value' is initialized, creates a dynamic broadcast, otherwise creates
// a static broadcast.
Value broadcastToFeatureDim(Location loc, RankedTensorType result_type,
                            Value value1d, Value shape_value,
                            int64_t feature_dim, PatternRewriter &rewriter) {
  auto dims = mlir::DenseI64ArrayAttr::get(loc.getContext(), {feature_dim});
  if (shape_value) {
    return rewriter.createOrFold<stablehlo::DynamicBroadcastInDimOp>(
        loc, result_type, value1d, shape_value, dims);
  }
  assert(result_type.hasStaticShape());
  return stablehlo::BroadcastInDimOp::create(rewriter, loc, result_type,
                                             value1d, dims);
}

// Gets the shape of operand, assuming it is a dynamic shape with static rank.
Value getShapeValue(Location loc, Value operand, PatternRewriter &rewriter) {
  RankedTensorType resultType =
      mlir::dyn_cast<RankedTensorType>(operand.getType());
  return shape::ShapeOfOp::create(
      rewriter, loc,
      RankedTensorType::get(/*shape=*/{resultType.getRank()},
                            rewriter.getIndexType()),
      operand);
}

// Returns the mean of `input` over `reduce_dims` as a tensor of `result_type`.
Value computeMean(Location loc, Value input, ArrayRef<int64_t> reduce_dims,
                  RankedTensorType result_type, PatternRewriter& rewriter) {
  auto input_type = mlir::cast<RankedTensorType>(input.getType());
  Type element_type = result_type.getElementType();
  auto scalar_type = RankedTensorType::get(/*shape=*/{}, element_type);

  // Sum over `reduce_dims`.
  Value zero = stablehlo::ConstantOp::create(rewriter, loc,
                                             rewriter.getZeroAttr(scalar_type));
  auto reduce_op = stablehlo::ReduceOp::create(
      rewriter, loc, TypeRange{result_type}, ValueRange{input},
      ValueRange{zero}, rewriter.getDenseI64ArrayAttr(reduce_dims));
  {
    OpBuilder::InsertionGuard guard(rewriter);
    Block* body = rewriter.createBlock(&reduce_op.getBody(), {},
                                       {scalar_type, scalar_type}, {loc, loc});
    Value sum = stablehlo::AddOp::create(rewriter, loc, body->getArgument(0),
                                         body->getArgument(1));
    stablehlo::ReturnOp::create(rewriter, loc, ValueRange{sum});
  }
  Value sum = reduce_op.getResult(0);

  // Number of reduced elements.
  int64_t static_count = 1;
  Value count;
  for (int64_t dim : reduce_dims) {
    if (!input_type.isDynamicDim(dim)) {
      static_count *= input_type.getDimSize(dim);
      continue;
    }
    Value dim_size = stablehlo::ConvertOp::create(
        rewriter, loc, scalar_type,
        stablehlo::GetDimensionSizeOp::create(rewriter, loc, input, dim));
    count = count ? stablehlo::MulOp::create(rewriter, loc, count, dim_size)
                        .getResult()
                  : dim_size;
  }
  Attribute static_count_attr =
      rewriter.getFloatAttr(element_type, static_cast<double>(static_count));
  Value static_count_value = stablehlo::ConstantOp::create(
      rewriter, loc, DenseElementsAttr::get(scalar_type, {static_count_attr}));
  count =
      count ? stablehlo::MulOp::create(rewriter, loc, count, static_count_value)
                  .getResult()
            : static_count_value;

  // Broadcast the count to the shape of the sum and divide.
  auto dims = rewriter.getDenseI64ArrayAttr({});
  Value count_bcast;
  if (result_type.hasStaticShape()) {
    count_bcast = stablehlo::BroadcastInDimOp::create(rewriter, loc,
                                                      result_type, count, dims);
  } else {
    count_bcast = stablehlo::DynamicBroadcastInDimOp::create(
        rewriter, loc, result_type, count, getShapeValue(loc, sum, rewriter),
        dims);
  }
  return stablehlo::DivOp::create(rewriter, loc, sum, count_bcast);
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
  Value epsilon = stablehlo::ConstantOp::create(b, epsilon_tensor_attr);
  auto dims =
      DenseI64ArrayAttr::get(op->getContext(), SmallVector<int64_t, 1>{});
  if (broadcast_to_type.hasStaticShape()) {
    return stablehlo::BroadcastInDimOp::create(b, broadcast_to_type, epsilon,
                                               dims);
  }
  Value shape_value = getShapeValue(op->getLoc(), broadcast_to, rewriter);
  return b.createOrFold<stablehlo::DynamicBroadcastInDimOp>(
      broadcast_to_type, epsilon, shape_value, dims);
}

class UnfuseBatchNormTrainingPattern
    : public OpRewritePattern<stablehlo::BatchNormTrainingOp> {
 public:
  using OpRewritePattern<stablehlo::BatchNormTrainingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::BatchNormTrainingOp bn_op,
                                PatternRewriter &rewriter) const override {
    auto inputs = bn_op.getOperand();
    auto input_type = mlir::dyn_cast<RankedTensorType>(inputs.getType());
    if (!input_type) {
      return failure();
    }
    auto feature_index = bn_op.getFeatureIndex();

    // Compute mean
    ::mlir::SmallVector<int64_t> reduce_dims;
    for (int64_t i = 0; i < input_type.getRank(); ++i) {
      if (i != feature_index) reduce_dims.push_back(i);
    }
    auto mean_var_type =
        mlir::cast<RankedTensorType>(bn_op.getBatchMean().getType());
    Value mean = computeMean(bn_op.getLoc(), inputs, reduce_dims, mean_var_type,
                             rewriter);

    // Compute variance
    Value shape_value =
        getShapeValue(bn_op.getLoc(), bn_op.getOperand(), rewriter);
    auto broadcast_mean = broadcastToFeatureDim(
        bn_op.getLoc(), input_type, mean, shape_value, feature_index, rewriter);
    Value diff = stablehlo::SubtractOp::create(rewriter, bn_op.getLoc(), inputs,
                                               broadcast_mean);
    Value square_diff =
        stablehlo::MulOp::create(rewriter, bn_op.getLoc(), diff, diff);
    Value variance = computeMean(bn_op.getLoc(), square_diff, reduce_dims,
                                 mean_var_type, rewriter);

    // Invoke BatchNormInferenceOp
    ::mlir::FloatAttr epsilon = bn_op.getEpsilonAttr();
    ::mlir::Value batch_norm = stablehlo::BatchNormInferenceOp::create(
        rewriter, bn_op.getLoc(), inputs, bn_op.getScale(), bn_op.getOffset(),
        mean, variance, epsilon, rewriter.getI64IntegerAttr(feature_index));

    // Return normalized values, mean, variable.
    rewriter.replaceOp(bn_op, ::mlir::ValueRange{batch_norm, mean, variance});
    return success();
  }
};

class UnfuseBatchNormInferencePattern
    : public OpRewritePattern<stablehlo::BatchNormInferenceOp> {
 public:
  using OpRewritePattern<stablehlo::BatchNormInferenceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::BatchNormInferenceOp bn_op,
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
    Value multiplier = stablehlo::AddOp::create(rewriter, bn_op.getLoc(),
                                                bn_op.getVariance(), epsilon);
    multiplier =
        stablehlo::RsqrtOp::create(rewriter, bn_op.getLoc(), multiplier);
    multiplier = stablehlo::MulOp::create(rewriter, bn_op.getLoc(), multiplier,
                                          bn_op.getScale());

    // Compute rhs = offset - mean * multiplier
    Value rhs = stablehlo::MulOp::create(rewriter, bn_op.getLoc(), multiplier,
                                         bn_op.getMean());
    rhs = stablehlo::SubtractOp::create(rewriter, bn_op.getLoc(),
                                        bn_op.getOffset(), rhs);

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
    Value lhs = stablehlo::MulOp::create(
        rewriter, bn_op.getLoc(), bn_op.getOperand(), broadcast_multiplier);
    auto broadcast_rhs = broadcastToFeatureDim(
        bn_op.getLoc(), input_type, rhs, shape_value, feature_dim, rewriter);
    rewriter.replaceOpWithNewOp<stablehlo::AddOp>(bn_op, lhs, broadcast_rhs);

    return success();
  }
};

class StablehloUnfuseBatchNormPass
    : public impl::StablehloUnfuseBatchNormPassBase<
          StablehloUnfuseBatchNormPass> {
 public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<UnfuseBatchNormTrainingPattern>(&getContext());
    patterns.add<UnfuseBatchNormInferencePattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

static PassRegistration<StablehloUnfuseBatchNormPass> pass;

}  // namespace odml
}  // namespace mlir

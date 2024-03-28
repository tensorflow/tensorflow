/* Copyright 2019 The OpenXLA Authors.

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
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace mhlo {

namespace {

// Broadcasts the 1D value tensor 'value1d' to the shape of 'resultType'. If
// 'shapeValue' is initialized, creates a dynamic broadcast, otherwise creates
// a static broadcast.
Value broadcastToFeatureDim(Location loc, RankedTensorType resultType,
                            Value value1d, Value shapeValue, int64_t featureDim,
                            PatternRewriter& rewriter) {  // NOLINT
  auto dimsType = RankedTensorType::get({1}, rewriter.getIntegerType(64));
  auto dims = DenseIntElementsAttr::get(dimsType, {featureDim});
  if (shapeValue) {
    return rewriter.createOrFold<mhlo::DynamicBroadcastInDimOp>(
        loc, resultType, value1d, shapeValue, dims);
  }
  assert(resultType.hasStaticShape());
  return rewriter.create<mhlo::BroadcastInDimOp>(loc, resultType, value1d,
                                                 dims);
}

// Get the shape of operand, assuming it is a dynamic shape with static rank.
Value getShapeValue(Location loc, Value operand,
                    PatternRewriter &rewriter) {  // NOLINT
  RankedTensorType resultType = operand.getType().dyn_cast<RankedTensorType>();
  return rewriter.create<mlir::shape::ShapeOfOp>(
      loc,
      RankedTensorType::get({resultType.getRank()}, rewriter.getIndexType()),
      operand);
}

Value materializeEpsilon(Operation *op, FloatAttr epsilonAttr, FloatType fpType,
                         Value broadcastTo, RankedTensorType broadcastToType,
                         PatternRewriter &rewriter) {  // NOLINT
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  if (epsilonAttr.getType() != fpType) {
    // Need to convert.
    bool losesInfo;
    APFloat epsilonFloat = epsilonAttr.getValue();
    auto status = epsilonFloat.convert(
        fpType.getFloatSemantics(), APFloat::rmNearestTiesToEven, &losesInfo);
    if ((status & (~APFloat::opInexact)) != APFloat::opOK) {
      op->emitWarning() << "Could not convert batch_norm epsilon to target fp "
                           "type: opStatus = "
                        << static_cast<int>(status);
      return nullptr;
    }
    if (losesInfo) {
      op->emitWarning("Conversion of epsilon loses precision");
    }
    epsilonAttr = b.getFloatAttr(fpType, epsilonFloat);
  }

  auto scalarType = RankedTensorType::get({}, fpType);
  auto epsilonTensorAttr =
      DenseElementsAttr::get(scalarType, {epsilonAttr.cast<Attribute>()});
  Value epsilon = b.create<mhlo::ConstantOp>(epsilonTensorAttr);
  auto dimsType = RankedTensorType::get({0}, b.getIntegerType(64));
  auto dims = DenseIntElementsAttr::get(dimsType, SmallVector<int64_t, 1>{});
  if (broadcastToType.hasStaticShape()) {
    return b.create<mhlo::BroadcastInDimOp>(broadcastToType, epsilon,
                                            /*broadcast_dims=*/dims);
  }
  Value shapeValue = getShapeValue(op->getLoc(), broadcastTo, rewriter);
  return b.createOrFold<mhlo::DynamicBroadcastInDimOp>(broadcastToType, epsilon,
                                                       shapeValue,
                                                       /*broadcast_dims=*/dims);
}

class UnfuseBatchNormInferencePattern
    : public OpRewritePattern<mhlo::BatchNormInferenceOp> {
 public:
  using OpRewritePattern<mhlo::BatchNormInferenceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::BatchNormInferenceOp bnOp,
                                PatternRewriter &rewriter) const override {
    // Enforce type invariants.
    // Note that we deduce the actual element type from the variance,
    // which should not be subject to quantization at a higher level.
    auto inputType = bnOp.getOperand().getType().dyn_cast<RankedTensorType>();
    auto varianceType =
        bnOp.getVariance().getType().dyn_cast<RankedTensorType>();
    if (!inputType || !varianceType) {
      return failure();
    }
    auto fpType = varianceType.getElementType().dyn_cast<FloatType>();
    if (!fpType) {
      return failure();
    }

    // result = (x - mean) * scale / sqrt(variance + epsilon) + offset
    // Let multiplier = scale / sqrt(variance + epsilon), to compute
    // (x - mean) * scale / sqrt(variance + epsilon) + offset,
    // is then to compute (x * multiplier) + (offset - mean * multiplier).

    auto epsilon =
        materializeEpsilon(bnOp.getOperation(), bnOp.getEpsilonAttr(), fpType,
                           bnOp.getVariance(), varianceType, rewriter);
    if (!epsilon) {
      return failure();
    }

    // Compute multiplier = scale / sqrt(variance + epsilon)
    Value multiplier = rewriter.create<mhlo::AddOp>(
        bnOp.getLoc(), bnOp.getVariance(), epsilon);
    multiplier = rewriter.create<mhlo::RsqrtOp>(bnOp.getLoc(), multiplier);
    multiplier = rewriter.create<mhlo::MulOp>(bnOp.getLoc(), multiplier,
                                              bnOp.getScale());

    // Compute rhs = offset - mean * multiplier
    Value rhs =
        rewriter.create<mhlo::MulOp>(bnOp.getLoc(), multiplier, bnOp.getMean());
    rhs =
        rewriter.create<mhlo::SubtractOp>(bnOp.getLoc(), bnOp.getOffset(), rhs);

    // Broadcast `multiplier` and `rhs`
    Value shapeValue;
    if (!inputType.hasStaticShape()) {
      shapeValue = getShapeValue(bnOp.getLoc(), bnOp.getOperand(), rewriter);
    }
    int64_t featureDim = bnOp.getFeatureIndex();
    auto broadcastMultiplier = broadcastToFeatureDim(
        bnOp.getLoc(), inputType, multiplier, shapeValue, featureDim, rewriter);
    auto broadcastRhs = broadcastToFeatureDim(bnOp.getLoc(), inputType, rhs,
                                              shapeValue, featureDim, rewriter);

    // Computes x * multiplier + rhs
    Value lhs = rewriter.create<mhlo::MulOp>(bnOp.getLoc(), bnOp.getOperand(),
                                             broadcastMultiplier);
    rewriter.replaceOpWithNewOp<mhlo::AddOp>(bnOp, lhs, broadcastRhs);

    return success();
  }
};

// Create "mhlo.reduce", "operand" is reduce input and "zero" is init value,
// reduce sum from operand to operand[feature_index].
Value createReduce(Location loc, Value operand, Value zero,
                   SmallVector<int64_t>& reduceDims, int64_t featureIndex,
                   PatternRewriter& rewriter) {
  auto operandType = operand.getType().cast<RankedTensorType>();
  Type reduceResultType = RankedTensorType::get(
      {operandType.getDimSize(featureIndex)}, operandType.getElementType());
  mhlo::ReduceOp reduce =
      rewriter.create<mhlo::ReduceOp>(loc, reduceResultType, operand, zero,
                                      rewriter.getI64TensorAttr(reduceDims));

  // setup "mhlo.reduce"'s body
  Region &region = reduce.getBody();
  Block& block = region.emplaceBlock();
  RankedTensorType blockArgumentType =
      RankedTensorType::get({}, operandType.getElementType());
  block.addArgument(blockArgumentType, loc);
  block.addArgument(blockArgumentType, loc);
  auto* firstArgument = block.args_begin();
  auto secondArgument = block.args_rbegin();
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);
    Value addResult =
        rewriter.create<mhlo::AddOp>(loc, *firstArgument, *secondArgument);
    rewriter.create<mhlo::ReturnOp>(loc, addResult);
  }

  return reduce.getResult(0);
}

// Calculate total reduce size, assuming it is a dynamic shape with static rank.
// Reduce from operand to operand[feature_index]/scale
Value calculateReduceSize(Operation *op, Value operand,
                          RankedTensorType operandType, Value scale,
                          RankedTensorType scaleType, int64_t featureIndex,
                          PatternRewriter &rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  Type indexType = b.getIndexType();
  if (!operandType.hasStaticShape()) {
    // the "operand" has dynamic shape with static rank
    Value operandShape = getShapeValue(op->getLoc(), operand, rewriter);
    Value scaleShape = getShapeValue(op->getLoc(), scale, rewriter);
    Value operandTotalSize =
        b.create<shape::NumElementsOp>(indexType, operandShape);
    Value scaleTotalSize =
        b.create<shape::NumElementsOp>(indexType, scaleShape);
    Value reduceSize =
        b.create<shape::DivOp>(indexType, operandTotalSize, scaleTotalSize);
    reduceSize = b.create<arith::IndexCastOp>(b.getI64Type(), reduceSize);
    reduceSize = b.create<tensor::FromElementsOp>(reduceSize);
    reduceSize = b.create<mhlo::ConvertOp>(
        RankedTensorType::get({1}, operandType.getElementType()), reduceSize);
    reduceSize = b.create<mhlo::ReshapeOp>(
        RankedTensorType::get({}, operandType.getElementType()), reduceSize);
    return b.createOrFold<mhlo::DynamicBroadcastInDimOp>(
        scaleType, reduceSize, scaleShape, b.getI64TensorAttr({}));
  }

  // the "operand" has static shape
  int64_t reduceDimsSize = 1;
  for (int64_t i = 0, e = operandType.getRank(); i < e; i++) {
    if (i != featureIndex) {
      reduceDimsSize *= operandType.getDimSize(i);
    }
  }
  llvm::APFloat floatValue(static_cast<double>(reduceDimsSize));
  bool losesInfo;
  floatValue.convert(
      scaleType.getElementType().cast<FloatType>().getFloatSemantics(),
      APFloat::rmNearestTiesToEven, &losesInfo);
  if (losesInfo) {
    op->emitWarning("Conversion of reduce_dims_size loses precision");
  }
  Value reduceSize = b.create<mhlo::ConstantOp>(
      DenseFPElementsAttr::get(scaleType, floatValue));
  return reduceSize;
}

// BatchNormTraining(X, scale, offset) =
//    ((X - E[X]) / Sqrt(Var[X] + epsilon)) * scale + offset.
class UnfuseBatchNormTrainingPattern
    : public OpRewritePattern<mhlo::BatchNormTrainingOp> {
 public:
  using OpRewritePattern<mhlo::BatchNormTrainingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::BatchNormTrainingOp bnOp,
                                PatternRewriter& rewriter) const override {
    auto operandType = bnOp.getOperand().getType().dyn_cast<RankedTensorType>();
    auto scaleType = bnOp.getScale().getType().dyn_cast<RankedTensorType>();
    if (!operandType || !scaleType) {
      return failure();
    }
    auto fpType = operandType.getElementType().dyn_cast<FloatType>();
    if (!fpType) {
      return failure();
    }
    int64_t featureIndex = bnOp.getFeatureIndex();
    SmallVector<int64_t> dimensionsWithoutFeature;
    for (int64_t i = 0, e = operandType.getRank(); i < e; i++) {
      if (i != featureIndex) {
        dimensionsWithoutFeature.push_back(i);
      }
    }

    // zero constant
    Value constZero = rewriter.create<mhlo::ConstantOp>(
        bnOp.getLoc(),
        DenseFPElementsAttr::get(RankedTensorType::get({}, fpType),
                                 APFloat::getZero(fpType.getFloatSemantics())));
    // epsilon
    auto epsilon =
        materializeEpsilon(bnOp.getOperation(), bnOp.getEpsilonAttr(), fpType,
                           bnOp.getScale(), scaleType, rewriter);
    if (!epsilon) {
      return failure();
    }
    // reduce size constant
    Value reduceSize =
        calculateReduceSize(bnOp.getOperation(), bnOp.getOperand(), operandType,
                            bnOp.getScale(), scaleType, featureIndex, rewriter);
    if (!reduceSize) {
      return failure();
    }
    // Sum[X]
    Value sum = createReduce(bnOp.getLoc(), bnOp.getOperand(), constZero,
                             dimensionsWithoutFeature, featureIndex, rewriter);
    // X^2
    Value operandSquare = rewriter.create<mhlo::MulOp>(
        bnOp.getLoc(), bnOp.getOperand(), bnOp.getOperand());
    // Sum[X^2]
    Value squareSum =
        createReduce(bnOp.getLoc(), operandSquare, constZero,
                     dimensionsWithoutFeature, featureIndex, rewriter);
    // E[X]
    Value mean = rewriter.create<mhlo::DivOp>(bnOp.getLoc(), sum, reduceSize);
    // E[X^2]
    Value squareMean =
        rewriter.create<mhlo::DivOp>(bnOp.getLoc(), squareSum, reduceSize);
    // E^2[X]
    Value meanSquare = rewriter.create<mhlo::MulOp>(bnOp.getLoc(), mean, mean);
    // Var[X]
    Value var = rewriter.create<mhlo::SubtractOp>(bnOp.getLoc(), squareMean,
                                                  meanSquare);
    // Var[X] + epsilon
    Value varAddEpsilon =
        rewriter.create<mhlo::AddOp>(bnOp.getLoc(), var, epsilon);
    // Sqrt(Var[X] + epsilon)
    Value sqrtVar = rewriter.create<mhlo::SqrtOp>(bnOp.getLoc(), varAddEpsilon);

    Value shapeValue;
    if (!operandType.hasStaticShape()) {
      shapeValue = getShapeValue(bnOp.getLoc(), bnOp.getOperand(), rewriter);
    }
    // X - E[X]
    Value meanBroadcast = broadcastToFeatureDim(
        bnOp.getLoc(), operandType, mean, shapeValue, featureIndex, rewriter);
    Value operandMinusMean = rewriter.create<mhlo::SubtractOp>(
        bnOp.getLoc(), bnOp.getOperand(), meanBroadcast);
    // (X - E[X]) / Sqrt(Var[X] + epsilon)
    Value sqrtVarBroadcast =
        broadcastToFeatureDim(bnOp.getLoc(), operandType, sqrtVar, shapeValue,
                              featureIndex, rewriter);
    Value normalized = rewriter.create<mhlo::DivOp>(
        bnOp.getLoc(), operandMinusMean, sqrtVarBroadcast);

    // ((X - E[X]) / Sqrt(Var[X] + epsilon)) * scale
    Value scaleBroadcast =
        broadcastToFeatureDim(bnOp.getLoc(), operandType, bnOp.getScale(),
                              shapeValue, featureIndex, rewriter);
    Value scaledNormalized =
        rewriter.create<mhlo::MulOp>(bnOp.getLoc(), normalized, scaleBroadcast);
    // ((X - E[X]) / Sqrt(Var[X] + epsilon)) * scale + offset.
    Value offsetBroadcast =
        broadcastToFeatureDim(bnOp.getLoc(), operandType, bnOp.getOffset(),
                              shapeValue, featureIndex, rewriter);
    Value shiftedNormalized = rewriter.create<mhlo::AddOp>(
        bnOp.getLoc(), scaledNormalized, offsetBroadcast);

    // results
    SmallVector<Value> results = {shiftedNormalized, mean, var};
    rewriter.replaceOp(bnOp, results);

    return success();
  }
};

}  // namespace

// Populates conversion patterns to unfuse batch normalization operations.
// In combination with marking such ops as illegal, this allows backends that
// do not have special support for fused batchnorm to use simpler arithmetic
// primitives.
void populateUnfuseBatchNormInferencePattern(MLIRContext *context,
                                             RewritePatternSet *patterns) {
  patterns->add<UnfuseBatchNormInferencePattern>(context);
}

void populateUnfuseBatchNormTrainingPattern(MLIRContext *context,
                                            RewritePatternSet *patterns) {
  patterns->add<UnfuseBatchNormTrainingPattern>(context);
}

}  // namespace mhlo
}  // namespace mlir

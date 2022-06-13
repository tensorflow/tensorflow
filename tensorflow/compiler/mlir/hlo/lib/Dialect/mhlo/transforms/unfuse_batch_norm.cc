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
Value broadcastToFeatureDim(Location loc, RankedTensorType resultType,
                            Value value1d, Value shapeValue, int64_t featureDim,
                            PatternRewriter& rewriter) {  // NOLINT
  Builder b(rewriter.getContext());
  auto dimsType = RankedTensorType::get({1}, b.getIntegerType(64));
  auto dims = DenseIntElementsAttr::get(dimsType, {featureDim});
  if (shapeValue) {
    return rewriter.createOrFold<mhlo::DynamicBroadcastInDimOp>(
        loc, resultType, value1d, shapeValue, dims);
  }
  assert(resultType.hasStaticShape());
  return rewriter.create<mhlo::BroadcastInDimOp>(loc, resultType, value1d,
                                                 dims);
}

// Calculate the shape value of operand, assuming it is a dynamic shape with
// static rank.
Value calculateShapeValue(Location loc, Value operand,
                          PatternRewriter& rewriter) {  // NOLINT
  RankedTensorType resultType = operand.getType().dyn_cast<RankedTensorType>();
  llvm::SmallVector<Value, 4> shapeValues;
  int64_t rank = resultType.getRank();
  shapeValues.reserve(rank);
  for (int64_t i = 0; i < rank; ++i) {
    shapeValues.push_back(
        rewriter.create<mlir::tensor::DimOp>(loc, operand, i));
  }
  return rewriter.create<tensor::FromElementsOp>(loc, shapeValues);
}

Value materializeEpsilon(Operation* op, FloatAttr epsilonAttr, FloatType fpType,
                         Value broadcastTo, RankedTensorType broadcastToType,
                         PatternRewriter& rewriter) {  // NOLINT
  Builder b(rewriter.getContext());
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
  Value epsilon =
      rewriter.create<mhlo::ConstOp>(op->getLoc(), epsilonTensorAttr);
  auto dimsType = RankedTensorType::get({0}, b.getIntegerType(64));
  auto dims = DenseIntElementsAttr::get(dimsType, SmallVector<int64_t, 1>{});
  if (broadcastToType.hasStaticShape()) {
    return rewriter.create<mhlo::BroadcastInDimOp>(
        op->getLoc(), broadcastToType, epsilon, /*broadcast_dims=*/dims);
  }
  Value shapeValue = calculateShapeValue(op->getLoc(), broadcastTo, rewriter);
  return rewriter.createOrFold<mhlo::DynamicBroadcastInDimOp>(
      op->getLoc(), broadcastToType, epsilon, shapeValue,
      /*broadcast_dims=*/dims);
}

class UnfuseBatchNormInferencePattern
    : public OpRewritePattern<mhlo::BatchNormInferenceOp> {
 public:
  using OpRewritePattern<mhlo::BatchNormInferenceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::BatchNormInferenceOp bnOp,
                                PatternRewriter& rewriter) const override {
    // Enforce type invariants.
    // Note that we deduce the actual element type from the variance,
    // which should not be subject to quantization at a higher level.
    auto inputType = bnOp.operand().getType().dyn_cast<RankedTensorType>();
    auto varianceType = bnOp.variance().getType().dyn_cast<RankedTensorType>();
    if (!inputType || !varianceType) {
      return failure();
    }
    auto fpType = varianceType.getElementType().dyn_cast<FloatType>();
    if (!fpType) {
      return failure();
    }
    int64_t featureDim = bnOp.feature_index();

    // Add epsilon to the variance and sqrt to get stddev:
    // stddev = sqrt(variance + epsilon)
    auto epsilon =
        materializeEpsilon(bnOp.getOperation(), bnOp.epsilonAttr(), fpType,
                           bnOp.variance(), varianceType, rewriter);
    if (!epsilon) {
      return failure();
    }
    Value stddev =
        rewriter.create<mhlo::AddOp>(bnOp.getLoc(), bnOp.variance(), epsilon);
    stddev = rewriter.create<mhlo::SqrtOp>(bnOp.getLoc(), stddev);

    // Broadcast all terms.
    Value shapeValue;
    if (!inputType.hasStaticShape()) {
      shapeValue = calculateShapeValue(bnOp.getLoc(), bnOp.operand(), rewriter);
    }
    auto broadcastScale =
        broadcastToFeatureDim(bnOp.getLoc(), inputType, bnOp.scale(),
                              shapeValue, featureDim, rewriter);
    auto broadcastOffset =
        broadcastToFeatureDim(bnOp.getLoc(), inputType, bnOp.offset(),
                              shapeValue, featureDim, rewriter);
    auto broadcastMean =
        broadcastToFeatureDim(bnOp.getLoc(), inputType, bnOp.mean(), shapeValue,
                              featureDim, rewriter);
    auto broadcastStddev = broadcastToFeatureDim(
        bnOp.getLoc(), inputType, stddev, shapeValue, featureDim, rewriter);

    // Compute:
    // scale * (input - mean) / stddev + offset
    Value result = rewriter.create<mhlo::SubOp>(bnOp.getLoc(), bnOp.operand(),
                                                broadcastMean);
    result =
        rewriter.create<mhlo::MulOp>(bnOp.getLoc(), result, broadcastScale);
    result =
        rewriter.create<mhlo::DivOp>(bnOp.getLoc(), result, broadcastStddev);
    rewriter.replaceOpWithNewOp<mhlo::AddOp>(bnOp, result, broadcastOffset);

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
  Region& region = reduce.body();
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
// Reduce from operand to operand[feature_index]
Value calculateReduceSize(Operation* op, Value operand,
                          RankedTensorType operandType,
                          RankedTensorType scaleType, int64_t featureIndex,
                          PatternRewriter& rewriter) {
  Location loc = op->getLoc();
  if (!operandType.hasStaticShape()) {
    // the "operand" has dynamic shape with static rank
    llvm::SmallVector<Value, 4> reduceValues;
    for (int64_t i = 0, e = operandType.getRank(); i < e; i++) {
      if (i != featureIndex) {
        reduceValues.push_back(rewriter.create<tensor::DimOp>(loc, operand, i));
      }
    }
    assert(!reduceValues.empty());
    Value reduceSize = reduceValues[0];
    for (size_t i = 1, e = reduceValues.size(); i < e; i++) {
      reduceSize =
          rewriter.create<arith::MulIOp>(loc, reduceSize, reduceValues[i]);
    }
    reduceSize = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(),
                                                     reduceSize);
    reduceSize = rewriter.create<tensor::FromElementsOp>(loc, reduceSize);
    reduceSize = rewriter.create<mhlo::ConvertOp>(
        loc, RankedTensorType::get({1}, operandType.getElementType()),
        reduceSize);
    reduceSize = rewriter.create<mhlo::ReshapeOp>(
        loc, RankedTensorType::get({}, operandType.getElementType()),
        reduceSize);
    Value featureSize =
        rewriter.create<tensor::DimOp>(loc, operand, featureIndex);
    featureSize = rewriter.create<tensor::FromElementsOp>(loc, featureSize);

    return rewriter.createOrFold<mhlo::DynamicBroadcastInDimOp>(
        loc, scaleType, reduceSize, featureSize, rewriter.getI64TensorAttr({}));
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
  Value reduceSize = rewriter.create<mhlo::ConstOp>(
      loc, DenseFPElementsAttr::get(scaleType, floatValue));
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
    auto operandType = bnOp.operand().getType().dyn_cast<RankedTensorType>();
    auto scaleType = bnOp.scale().getType().dyn_cast<RankedTensorType>();
    if (!operandType || !scaleType) {
      return failure();
    }
    auto fpType = operandType.getElementType().dyn_cast<FloatType>();
    if (!fpType) {
      return failure();
    }
    int64_t featureIndex = bnOp.feature_index();
    SmallVector<int64_t> dimensionsWithoutFeature;
    for (int64_t i = 0, e = operandType.getRank(); i < e; i++) {
      if (i != featureIndex) {
        dimensionsWithoutFeature.push_back(i);
      }
    }

    // zero constant
    Value constZero = rewriter.create<mhlo::ConstOp>(
        bnOp.getLoc(),
        DenseFPElementsAttr::get(RankedTensorType::get({}, fpType),
                                 APFloat::getZero(fpType.getFloatSemantics())));
    // epsilon
    auto epsilon =
        materializeEpsilon(bnOp.getOperation(), bnOp.epsilonAttr(), fpType,
                           bnOp.scale(), scaleType, rewriter);
    if (!epsilon) {
      return failure();
    }
    // reduce size constant
    Value reduceSize =
        calculateReduceSize(bnOp.getOperation(), bnOp.operand(), operandType,
                            scaleType, featureIndex, rewriter);
    if (!reduceSize) {
      return failure();
    }
    // Sum[X]
    Value sum = createReduce(bnOp.getLoc(), bnOp.operand(), constZero,
                             dimensionsWithoutFeature, featureIndex, rewriter);
    // X^2
    Value operandSquare = rewriter.create<mhlo::MulOp>(
        bnOp.getLoc(), bnOp.operand(), bnOp.operand());
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
    Value var =
        rewriter.create<mhlo::SubOp>(bnOp.getLoc(), squareMean, meanSquare);
    // Var[X] + epsilon
    Value varAddEpsilon =
        rewriter.create<mhlo::AddOp>(bnOp.getLoc(), var, epsilon);
    // Sqrt(Var[X] + epsilon)
    Value sqrtVar = rewriter.create<mhlo::SqrtOp>(bnOp.getLoc(), varAddEpsilon);

    Value shapeValue;
    if (!operandType.hasStaticShape()) {
      shapeValue = calculateShapeValue(bnOp.getLoc(), bnOp.operand(), rewriter);
    }
    // X - E[X]
    Value meanBroadcast = broadcastToFeatureDim(
        bnOp.getLoc(), operandType, mean, shapeValue, featureIndex, rewriter);
    Value operandMinusMean = rewriter.create<mhlo::SubOp>(
        bnOp.getLoc(), bnOp.operand(), meanBroadcast);
    // (X - E[X]) / Sqrt(Var[X] + epsilon)
    Value sqrtVarBroadcast =
        broadcastToFeatureDim(bnOp.getLoc(), operandType, sqrtVar, shapeValue,
                              featureIndex, rewriter);
    Value normalized = rewriter.create<mhlo::DivOp>(
        bnOp.getLoc(), operandMinusMean, sqrtVarBroadcast);

    // ((X - E[X]) / Sqrt(Var[X] + epsilon)) * scale
    Value scaleBroadcast =
        broadcastToFeatureDim(bnOp.getLoc(), operandType, bnOp.scale(),
                              shapeValue, featureIndex, rewriter);
    Value scaledNormalized =
        rewriter.create<mhlo::MulOp>(bnOp.getLoc(), normalized, scaleBroadcast);
    // ((X - E[X]) / Sqrt(Var[X] + epsilon)) * scale + offset.
    Value offsetBroadcast =
        broadcastToFeatureDim(bnOp.getLoc(), operandType, bnOp.offset(),
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
void PopulateUnfuseBatchNormPatterns(MLIRContext* context,
                                     RewritePatternSet* patterns) {
  patterns->add<UnfuseBatchNormInferencePattern>(context);
  patterns->add<UnfuseBatchNormTrainingPattern>(context);
}

}  // namespace mhlo
}  // namespace mlir

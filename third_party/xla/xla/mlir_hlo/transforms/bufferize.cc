/* Copyright 2020 The OpenXLA Authors.

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

#include <optional>

// This file implements logic for translating mixed IR to buffer form.
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "transforms/rewriters.h"

namespace mlir {
namespace {

struct BufferizeConstantOp : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      arith::ConstantOp op, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter &rewriter) const final {
    // We only need to bufferize tensor constants.
    Location loc = op.getLoc();
    auto resultType = mlir::dyn_cast<RankedTensorType>(op.getType());
    int64_t resultRank = resultType.getRank();
    if (!resultType || !resultType.hasStaticShape() || resultRank > 1)
      return failure();

    auto elementType = resultType.getElementType();
    auto memrefType = MemRefType::get(resultType.getShape(), elementType);
    auto elementsAttr = mlir::cast<DenseElementsAttr>(op.getValue());

    // arith.constant doesn't handle scalar complex types.
    // TODO(kramerb): Should this use materializeConstant instead?
    auto makeConstant = [&](Attribute attr, Type type) -> Value {
      if (complex::ConstantOp::isBuildableWith(attr, type))
        return complex::ConstantOp::create(rewriter, loc, type,
                                           mlir::cast<ArrayAttr>(attr));
      return arith::ConstantOp::create(rewriter, loc, cast<TypedAttr>(attr));
    };

    if (resultRank == 0) {
      Value buffer = memref::AllocOp::create(rewriter, loc, memrefType);
      Value constant =
          makeConstant(elementsAttr.getValues<Attribute>()[0], elementType);
      memref::StoreOp::create(rewriter, loc, constant, buffer);
      rewriter.replaceOp(op, {buffer});
      return success();
    }

    Value buffer = memref::AllocaOp::create(rewriter, loc, memrefType);

    bool allSameElems = elementsAttr.isSplat();
    Value value;
    if (allSameElems)
      value = makeConstant(elementsAttr.getSplatValue<mlir::Attribute>(),
                           elementType);
    for (const auto &en :
         llvm::enumerate(elementsAttr.getValues<Attribute>())) {
      if (!allSameElems) value = makeConstant(en.value(), elementType);
      Value index = arith::ConstantIndexOp::create(rewriter, loc, en.index());
      memref::StoreOp::create(rewriter, loc, value, buffer, index);
    }
    rewriter.replaceOp(op, {buffer});
    return success();
  }
};

struct BufferizeAndConvertMinimumBroadcastShapesOp
    : public OpConversionPattern<mhlo::MinimumBroadcastShapesOp> {
  using OpConversionPattern<
      mhlo::MinimumBroadcastShapesOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::MinimumBroadcastShapesOp broadcastShapesOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = broadcastShapesOp.getLoc();
    ImplicitLocOpBuilder lb(loc, rewriter);
    Value zero = arith::ConstantIndexOp::create(lb, 0);
    SmallVector<Value> shapes = adaptor.getShapes();
    size_t k = shapes.size();
    SmallVector<Value> ranks;
    ranks.reserve(k);

    // Determine the maximum rank of the operands.
    Value maxRank;
    for (size_t i = 0; i < k; ++i) {
      Value rank = memref::DimOp::create(lb, loc, shapes[i], zero);
      ranks.push_back(rank);
      if (i) {
        Value rankIsGreater = arith::CmpIOp::create(
            lb, arith::CmpIPredicate::ugt, ranks[i], maxRank);
        maxRank = arith::SelectOp::create(lb, rankIsGreater, ranks[i], maxRank);
      } else {
        maxRank = ranks[0];
      }
    }

    // Allocate buffers for the return values and initialize them with 1's.
    SmallVector<Value> resultShapes;
    resultShapes.reserve(k);
    auto resultType =
        MemRefType::get({ShapedType::kDynamic}, lb.getIndexType());
    Value one = arith::ConstantIndexOp::create(lb, 1);
    for (size_t i = 0; i < k; ++i) {
      // We assume the buffer will be small, so we allocate it on the stack.
      // TODO(b/181654096): Replace AllocaOp with AllocOp.
      auto result = memref::AllocaOp::create(lb, resultType, ranks[i]);
      scf::ForOp::create(lb, zero, ranks[i], one, mlir::ValueRange(),
                         [&one, &result](OpBuilder& b, Location l, Value idx,
                                         ValueRange /*vr*/) {
                           memref::StoreOp::create(b, l, one, result, idx);
                           scf::YieldOp::create(b, l, mlir::ValueRange());
                         });
      resultShapes.push_back(result);
    }

    // Iterate through the dimensions and determine which adjacent dimensions
    // can be combined. Keep a running product of the dimensions that can be
    // combined as iteration variable (initialized to 1), and the current
    // dimension offset in the result shapes. We iterate through the shapes
    // backward, because the broadcasting semantics mean that the last
    // dimensions of each shape (the least significant ones) are matched
    // together.
    Value two = arith::ConstantIndexOp::create(lb, 2);
    Value maxRankPlusTwo = arith::AddIOp::create(lb, loc, maxRank, two);
    Value constantFalse =
        arith::ConstantOp::create(lb, lb.getI1Type(), lb.getBoolAttr(false));
    SmallVector<Value> initValues;
    initValues.reserve(k + 3);
    // Initially, all values are marked as not broadcasted.
    for (int64_t i = 0; i < static_cast<int64_t>(k); ++i) {
      initValues.push_back(constantFalse);
    }
    // The running product is initially 1.
    initValues.push_back(one);
    // The current dimension offset is initially 0.
    initValues.push_back(zero);
    // Whether the broadcasting is invalid.
    initValues.push_back(constantFalse);

    // Iterate from 1 to max_rank + 1 (inclusive). This iteration variable is
    // used as an offset from the end of each shape vector. We iterate until
    // max_rank + 1 to handle the case that we have a running_product > 1 left
    // when we have processed all dimensions of the largest shape.
    auto mainLoop = scf::ForOp::create(
        lb, one, maxRankPlusTwo, one, initValues,
        [&](OpBuilder& b, Location l, Value v, ValueRange vr) {
          // 'same_size' should track what the size of the dimension is to which
          // the 1-sized dimensions are broadcasted. If all of the dimensions
          // are 1, it will stay 1.
          Value sameSize = one;
          // 'result_dimensions' stores the current dimension with an offset of
          // 'leading_ones' to make it easier to check whether we are in-bounds
          // with respect to the "real" shape with leading 1's removed.
          SmallVector<Value> resultDimensions;
          resultDimensions.reserve(k);
          // 'no_broadcasting' stores boolean flags that encode whether the
          // corresponding shape does not need broadcasting at the current
          // position.
          SmallVector<Value> noBroadcasting;
          noBroadcasting.reserve(k + 3);
          // The first k loop carried values are the previous broadcasting
          // state.
          auto prevNoBroadcasting = vr.take_front(k);

          // This loop checks which shapes need broadcasting at the current
          // dimension. A shape needs broadcasting if it is indexed out of
          // bounds, or its current dimension size is 1.
          Value currentDimensionHasInvalidBroadcast = constantFalse;
          for (size_t i = 0; i < k; ++i) {
            // Determine the size of the current dimension. If the dimension is
            // out of bounds, we choose the value 'one'.
            Value isOutOfBounds = arith::CmpIOp::create(
                b, l, arith::CmpIPredicate::ult, ranks[i], v);
            Value dimension = arith::SubIOp::create(b, l, ranks[i], v);
            resultDimensions.push_back(dimension);
            Value currentSize = scf::IfOp::create(
                                    b, l, isOutOfBounds,
                                    [&](OpBuilder& b, Location l) {
                                      scf::YieldOp::create(b, l, one);
                                    },
                                    [&](OpBuilder& b, Location l) {
                                      // Using IfOp instead of SelectOp makes
                                      // sure that we don't try to load if the
                                      // dimension is out of bounds.
                                      Value size = memref::LoadOp::create(
                                          b, l, shapes[i], dimension);
                                      scf::YieldOp::create(b, l, size);
                                    })
                                    .getResult(0);
            // Compute whether the current dimension does require broadcasting.
            Value currentSizeIsNotOne = arith::CmpIOp::create(
                b, l, arith::CmpIPredicate::ne, currentSize, one);
            noBroadcasting.push_back(currentSizeIsNotOne);
            Value newSameSize = arith::SelectOp::create(
                b, l, currentSizeIsNotOne, currentSize, sameSize);
            Value sameSizeWasNotOne = arith::CmpIOp::create(
                b, l, arith::CmpIPredicate::ne, sameSize, one);
            Value isDifferentSize = arith::CmpIOp::create(
                b, l, arith::CmpIPredicate::ne, sameSize, newSameSize);
            // The broadcast is invalid if the size of the current dimension
            // is not equal to the expected size, unless the expected size was
            // still the initial value 1.
            Value isInvalid =
                arith::AndIOp::create(b, l, sameSizeWasNotOne, isDifferentSize);
            currentDimensionHasInvalidBroadcast = arith::OrIOp::create(
                b, l, currentDimensionHasInvalidBroadcast, isInvalid);
            sameSize = newSameSize;
          }

          // Check whether we have at least one shape that has a different
          // status regarding whether it needs broadcasting at the current
          // dimension versus whether it needs broadcasting at the previous
          // dimension.
          Value sameSizeIsOne = arith::CmpIOp::create(
              b, l, arith::CmpIPredicate::eq, sameSize, one);
          Value differentBroadcastingSet = constantFalse;
          for (size_t i = 0; i < k; ++i) {
            // If all dimensions are 1, we preserve the status whether a shape
            // needs broadcasting or not, because in that case the dimension can
            // just be ignored.
            noBroadcasting[i] = arith::SelectOp::create(
                b, l, sameSizeIsOne, prevNoBroadcasting[i], noBroadcasting[i]);
            // Compare whether the current shape changes its status regarding
            // whether it needs broadcasting at the current dimension.
            Value broadcastingIsDifferent =
                arith::CmpIOp::create(b, l, arith::CmpIPredicate::ne,
                                      prevNoBroadcasting[i], noBroadcasting[i]);
            differentBroadcastingSet = arith::OrIOp::create(
                b, l, differentBroadcastingSet, broadcastingIsDifferent);
          }
          Value runningProduct = vr[k];
          Value currentDimensionOffset = vr[k + 1];

          // We need to stop combining dimensions if the set of shapes which
          // need broadcasting at the current dimension changes compared to the
          // set of shapes needing broadcasting at the previous dimension.
          Value isLastIteration = arith::CmpIOp::create(
              b, l, arith::CmpIPredicate::sgt, v, maxRank);
          Value stopCombiningDimensions = arith::OrIOp::create(
              b, l, isLastIteration, differentBroadcastingSet);
          auto ifStopCombiningDimensions = scf::IfOp::create(
              b, l, stopCombiningDimensions,
              [&](OpBuilder& b, Location l) {
                // If the running product is not 1, add one dimension of size
                // 'running_product' to each shape that didn't need
                // broadcasting, otherwise add a 1 dimension if it was
                // previously indexed in-bounds.
                Value runningProductNotOne = arith::CmpIOp::create(
                    b, l, arith::CmpIPredicate::ne, runningProduct, one);
                Value newDimensionOffset =
                    scf::IfOp::create(
                        b, l, runningProductNotOne,
                        [&](OpBuilder& b, Location l) {
                          Value newDimensionOffset = arith::AddIOp::create(
                              b, l, currentDimensionOffset, one);
                          Value minusOne =
                              arith::ConstantIndexOp::create(lb, -1);
                          for (size_t i = 0; i < k; ++i) {
                            Value wasInBounds = arith::CmpIOp::create(
                                b, l, arith::CmpIPredicate::sge,
                                resultDimensions[i], minusOne);
                            Value shouldStoreDimension = arith::OrIOp::create(
                                b, l, wasInBounds, prevNoBroadcasting[i]);
                            scf::IfOp::create(
                                b, l, shouldStoreDimension,
                                [&](OpBuilder& b, Location l) {
                                  Value outputDimension = arith::SubIOp::create(
                                      b, l, ranks[i], newDimensionOffset);
                                  // If the shape needed broadcasting at the
                                  // previous dimension, we set the output size
                                  // to 1, otherwise to 'running_product'.
                                  Value outputSize = arith::SelectOp::create(
                                      b, l, prevNoBroadcasting[i],
                                      runningProduct, one);
                                  memref::StoreOp::create(b, l, outputSize,
                                                          resultShapes[i],
                                                          outputDimension);
                                  scf::YieldOp::create(b, l,
                                                       mlir::ValueRange());
                                });
                          }
                          scf::YieldOp::create(b, l, newDimensionOffset);
                        },
                        [&](OpBuilder& b, Location l) {
                          scf::YieldOp::create(b, l, currentDimensionOffset);
                        })
                        .getResult(0);
                scf::YieldOp::create(b, l,
                                     ValueRange{sameSize, newDimensionOffset});
              },
              [&](OpBuilder& b, Location l) {
                Value newRunningProduct =
                    arith::MulIOp::create(b, l, runningProduct, sameSize);
                scf::YieldOp::create(
                    b, l,
                    ValueRange{newRunningProduct, currentDimensionOffset});
              });
          // Add the remaining results.
          noBroadcasting.push_back(ifStopCombiningDimensions.getResult(0));
          noBroadcasting.push_back(ifStopCombiningDimensions.getResult(1));
          Value isInvalid = vr.back();
          isInvalid = arith::OrIOp::create(b, l, isInvalid,
                                           currentDimensionHasInvalidBroadcast);
          noBroadcasting.push_back(isInvalid);
          scf::YieldOp::create(b, l, noBroadcasting);
        });
    Value isInvalid = mainLoop.getResults().back();
    for (size_t i = 0; i < k; ++i) {
      resultShapes[i] =
          removeLeadingOnesFrom1DMemref(lb, resultShapes[i], ranks[i]);
      resultShapes[i] =
          arith::SelectOp::create(lb, isInvalid, shapes[i], resultShapes[i]);
    }
    rewriter.replaceOp(broadcastShapesOp, resultShapes);
    return success();
  }

 private:
  Value countLeadingOnes(ImplicitLocOpBuilder &lb, Value extentMemref,
                         Value rank) const {
    // Count leading 1's. Use two iteration variables for that: one with a
    // boolean flag for whether every size so far was 1, one with the number of
    // leading 1's.
    Value constantTrue =
        arith::ConstantOp::create(lb, lb.getI1Type(), lb.getBoolAttr(true));
    Value zero = arith::ConstantIndexOp::create(lb, 0);
    Value one = arith::ConstantIndexOp::create(lb, 1);
    auto leadingOnesLoop = scf::ForOp::create(
        lb, zero, rank, one, ValueRange{constantTrue, zero},
        [&](OpBuilder& b, Location l, Value idx, ValueRange vr) {
          auto size = memref::LoadOp::create(b, l, extentMemref, idx);
          auto isEqualToOne =
              arith::CmpIOp::create(b, l, arith::CmpIPredicate::eq, size, one);
          auto allOnes = arith::AndIOp::create(b, l, vr.front(), isEqualToOne);
          auto increasedValue = arith::AddIOp::create(b, l, vr.back(), one);
          auto numberOfLeadingOnes =
              arith::SelectOp::create(b, l, allOnes, increasedValue, vr.back());
          scf::YieldOp::create(b, l, ValueRange{allOnes, numberOfLeadingOnes});
        });
    return leadingOnesLoop.getResults()[1];
  }

  Value removeLeadingOnesFrom1DMemref(ImplicitLocOpBuilder &lb,
                                      Value extentMemref, Value rank) const {
    Value leadingOnes = countLeadingOnes(lb, extentMemref, rank);
    Value newRank = arith::SubIOp::create(lb, rank, leadingOnes);
    auto resultType =
        MemRefType::get({ShapedType::kDynamic}, lb.getIndexType());
    // We cannot use SubView here to return a MemRef with 'leading_ones' as
    // offset, because that also changes the size, so the result type would need
    // to have an affine map to change the layout. This is incompatible to our
    // other MemRef types without affine map. So instead we just allocate
    // another buffer of the desired size and copy the elements over. We assume
    // the buffer will be small, so we allocate it on the stack.
    // TODO(b/181654096): Replace AllocaOp with AllocOp.
    Value result = memref::AllocaOp::create(lb, resultType, newRank);
    Value zero = arith::ConstantIndexOp::create(lb, 0);
    Value one = arith::ConstantIndexOp::create(lb, 1);
    scf::ForOp::create(
        lb, zero, newRank, one, mlir::ValueRange(),
        [&](OpBuilder& b, Location l, Value idx, ValueRange /*vr*/) {
          Value idxWithOffset = arith::AddIOp::create(b, l, idx, leadingOnes);
          auto size = memref::LoadOp::create(b, l, extentMemref, idxWithOffset);
          memref::StoreOp::create(b, l, size, result, idx);
          scf::YieldOp::create(b, l, mlir::ValueRange());
        });
    return result;
  }
};

}  // namespace

void populateExtraBufferizePatterns(MLIRContext *context,
                                    TypeConverter *converter,
                                    RewritePatternSet *patterns) {
  // clang-format off
  patterns->add<
      BufferizeAndConvertMinimumBroadcastShapesOp,
      BufferizeConstantOp
  >(*converter, context);
  // clang-format on
}

}  // namespace mlir

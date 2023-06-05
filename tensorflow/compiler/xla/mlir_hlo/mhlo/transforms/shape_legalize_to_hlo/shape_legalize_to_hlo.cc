/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_SHAPELEGALIZETOHLOPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

namespace {

bool hasI32Style(Value value) {
  auto type = value.getType().dyn_cast<ShapedType>();
  return type && type.getElementType().isInteger(32);
}

// Cast from index-based shape representation used in the Shape dialect to the
// i32-based representation used in HLO:
//   * index => tensor<i32>.
//   * tensor<Nxindex> => tensor<Nxi32>.
//   * All i32-based types from above => themselves.
// There is no convenient op that can express this, so we're using
// unrealized_conversion_cast (with the idea that all these casts will
// annihilate at the end of the pass).
Value castToI32(PatternRewriter& rewriter, Location loc, Value value) {
  Type resultType;
  if (value.getType().isIndex())
    resultType = RankedTensorType::get({}, rewriter.getI32Type());
  if (auto valueType = value.getType().dyn_cast<ShapedType>()) {
    if (!valueType.hasStaticShape()) return {};
    if (valueType.getElementType().isInteger(32)) return value;
    if (valueType.getElementType().isIndex())
      resultType =
          RankedTensorType::get(valueType.getShape(), rewriter.getI32Type());
  }
  if (!resultType) return {};
  auto cast =
      rewriter.create<UnrealizedConversionCastOp>(loc, resultType, value);
  return cast.getResult(0);
}

bool hasIndexStyle(Value value) {
  if (value.getType().isIndex()) return true;
  auto type = value.getType().dyn_cast<ShapedType>();
  return type && type.getElementType().isIndex();
}

// Cast from the i32-based shape representation used in HLO to the index-based
// representation used in the Shape dialect:
//   * tensor<i32> => index.
//   * tensor<Nxi32> => tensor<Nxindex>.
//   * All index-based types from above => themselves.
// There is no convenient op that can express this, so we're using
// unrealized_conversion_cast (with the idea that all these casts will
// annihilate at the end of the pass).
Value castToIndex(PatternRewriter& rewriter, Location loc, Value value) {
  Type resultType;
  if (value.getType().isIndex()) return value;
  if (auto valueType = value.getType().dyn_cast<ShapedType>()) {
    if (!valueType.hasStaticShape()) return {};
    if (valueType.getElementType().isInteger(32)) {
      if (valueType.getRank() == 0) {
        resultType = rewriter.getIndexType();
      } else {
        resultType = RankedTensorType::get(valueType.getShape(),
                                           rewriter.getIndexType());
      }
    }
    if (valueType.getElementType().isIndex()) return value;
  }
  if (!resultType) return {};
  auto cast =
      rewriter.create<UnrealizedConversionCastOp>(loc, resultType, value);
  return cast.getResult(0);
}

struct ConvertComputeReshapeShapeOpPattern
    : public OpRewritePattern<ComputeReshapeShapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ComputeReshapeShapeOp op,
                                PatternRewriter& rewriter) const override {
    // Cast num_elements from index to tensor<i32>.
    // Cast dynamic_shape from tensor<Nxindex> to tensor<Nxi32> if needed.
    // (mhlo.compute_reshape_shape supports both index- and integer-based
    // dynamic_shape operands).
    // This cannot error out given how the operation is currently defined.
    auto numElementsI32 = castToI32(rewriter, op.getLoc(), op.getNumElements());
    auto dynamicShapeI32x1 =
        castToI32(rewriter, op.getLoc(), op.getDynamicShape());
    if (!numElementsI32 || !dynamicShapeI32x1)
      return rewriter.notifyMatchFailure(op, "cast to i32 failed");
    auto rank = dynamicShapeI32x1.getType().cast<ShapedType>().getNumElements();

    // Obtain individual input dimension sizes and also compute the product of
    // all these dimension sizes.
    auto i32Type = RankedTensorType::get({}, rewriter.getI32Type());
    Value dynamicNumElementsI32 = rewriter.create<ConstantOp>(
        op.getLoc(), DenseIntElementsAttr::get<int32_t>(i32Type, -1));
    SmallVector<Value> dynamicSizesI32;
    for (auto i = 0; i < rank; ++i) {
      auto dynamicSizeI32x1 = rewriter.create<SliceOp>(
          op.getLoc(), dynamicShapeI32x1, rewriter.getI64TensorAttr(i),
          rewriter.getI64TensorAttr(i + 1), rewriter.getI64TensorAttr(1));
      auto dynamicSizeI32 =
          rewriter.create<ReshapeOp>(op.getLoc(), i32Type, dynamicSizeI32x1);
      dynamicSizesI32.push_back(dynamicSizeI32);
      dynamicNumElementsI32 = rewriter.create<MulOp>(
          op.getLoc(), dynamicNumElementsI32, dynamicSizeI32);
    }

    // Compute the dimension size that corresponds to -1 in dynamic_shape.
    // If such a dimension doesn't exist, then this value doesn't matter.
    auto computedSizeI32 = rewriter.create<DivOp>(op.getLoc(), numElementsI32,
                                                  dynamicNumElementsI32);

    // Compute individual output dimension sizes, replacing a potential -1
    // with the value computed above.
    auto i32x1Type = RankedTensorType::get({1}, rewriter.getI32Type());
    Value minusOneI32 = rewriter.create<ConstantOp>(
        op.getLoc(), DenseIntElementsAttr::get<int32_t>(i32Type, -1));
    SmallVector<Value> resultSizesI32x1;
    for (auto i = 0; i < rank; ++i) {
      auto eqMinusOne =
          rewriter.create<CompareOp>(op.getLoc(), dynamicSizesI32[i],
                                     minusOneI32, ComparisonDirection::EQ);
      auto resultSizeI32 = rewriter.create<SelectOp>(
          op.getLoc(), eqMinusOne, computedSizeI32, dynamicSizesI32[i]);
      auto resultSizeI32x1 =
          rewriter.create<ReshapeOp>(op.getLoc(), i32x1Type, resultSizeI32);
      resultSizesI32x1.push_back(resultSizeI32x1);
    }
    auto resultI32 =
        rewriter.create<mhlo::ConcatenateOp>(op.getLoc(), resultSizesI32x1,
                                             /*dimension=*/0);

    // Cast the result to tensor<Nxindex> if needed.
    // (mhlo.compute_reshape_shape supports both index- and integer-based
    // results).
    // This cannot error out given how the operation is currently defined.
    auto resultIndex = hasI32Style(op.getResult())
                           ? resultI32
                           : castToIndex(rewriter, op.getLoc(), resultI32);
    if (!resultIndex || resultIndex.getType() != op.getResult().getType())
      return rewriter.notifyMatchFailure(op, "cast to index failed");
    rewriter.replaceOp(op, resultIndex);
    return success();
  }
};

struct ConvertNumElementsOpPattern
    : public OpRewritePattern<shape::NumElementsOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(shape::NumElementsOp op,
                                PatternRewriter& rewriter) const override {
    // Cast shape from tensor<Nxindex> to tensor<Nxi32>.
    // This will error out if shape is !shape.shape.
    auto shapeI32 = castToI32(rewriter, op.getLoc(), op.getShape());
    if (!shapeI32) return rewriter.notifyMatchFailure(op, "cast to i32 failed");
    auto rank = shapeI32.getType().cast<ShapedType>().getNumElements();

    // Compute the product of the individual dimension sizes.
    // Using this representation instead of mhlo::ReduceOp because it is more
    // amenable to optimizations. (Reduce can be folded only if the entire
    // shape is static, but individual multiplications can be folded if
    // individual dimensions are static).
    auto resultI32Type = RankedTensorType::get({}, rewriter.getI32Type());
    Value resultI32 = rewriter.create<ConstantOp>(
        op.getLoc(), DenseIntElementsAttr::get<int32_t>(resultI32Type, 1));
    for (auto i = 0; i < rank; ++i) {
      auto sizeI32x1 = rewriter.create<SliceOp>(
          op.getLoc(), shapeI32, rewriter.getI64TensorAttr(i),
          rewriter.getI64TensorAttr(i + 1), rewriter.getI64TensorAttr(1));
      auto sizeI32 =
          rewriter.create<ReshapeOp>(op.getLoc(), resultI32Type, sizeI32x1);
      resultI32 = rewriter.create<MulOp>(op.getLoc(), resultI32, sizeI32);
    }

    // Cast result from tensor<i32> to index.
    // This will error out if the result is !shape.size.
    auto resultIndex = castToIndex(rewriter, op.getLoc(), resultI32);
    if (!resultIndex || resultIndex.getType() != op.getResult().getType())
      return rewriter.notifyMatchFailure(op, "cast to index failed");
    rewriter.replaceOp(op, resultIndex);
    return success();
  }
};

struct ConvertShapeOfOpPattern : public OpRewritePattern<shape::ShapeOfOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(shape::ShapeOfOp op,
                                PatternRewriter& rewriter) const override {
    auto operandType = op.getArg().getType().dyn_cast<RankedTensorType>();
    if (!operandType)
      return rewriter.notifyMatchFailure(op, "expected ranked operand");

    // Produce an MHLO equivalent of this shape::ShapeOfOp.
    // This is a very laborious representation because MHLO is currently lacking
    // convenient tools to express this.
    SmallVector<Value> sizesI32x1;
    for (auto i = 0; i < operandType.getRank(); ++i) {
      auto sizeI32 =
          rewriter.create<GetDimensionSizeOp>(op.getLoc(), op.getArg(), i);
      auto sizeI32x1 = rewriter.create<ReshapeOp>(
          op.getLoc(), RankedTensorType::get({1}, rewriter.getI32Type()),
          sizeI32);
      sizesI32x1.push_back(sizeI32x1);
    }
    auto shapeI32 =
        rewriter.create<mhlo::ConcatenateOp>(op.getLoc(), sizesI32x1,
                                             /*dimension=*/0);

    // Cast result from tensor<Nxi32> to tensor<Nxindex>.
    // This will error out if the result is !shape.shape.
    auto shapeIndex = castToIndex(rewriter, op.getLoc(), shapeI32);
    if (!shapeIndex || shapeIndex.getType() != op.getResult().getType())
      return rewriter.notifyMatchFailure(op, "cast to index failed");
    rewriter.replaceOp(op, shapeIndex);
    return success();
  }
};

template <typename OpType>
struct CastOperandsPattern : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter& rewriter) const override {
    if (!llvm::any_of(op->getOperands(), hasIndexStyle))
      return rewriter.notifyMatchFailure(op, "no operands need a cast to i32");

    // If op has operands of type tensor<Nxindex>, cast them to tensor<Nxi32>.
    // If producers of these operands have been transformed into casts from
    // tensor<Nxi32> to tensor<Nxindex>, then these casts will annihilate with
    // each other upon canonicalization.
    SmallVector<Value> operandsI32;
    for (auto operand : op->getOperands()) {
      if (hasIndexStyle(operand)) {
        operandsI32.push_back(castToI32(rewriter, op.getLoc(), operand));
      } else {
        operandsI32.push_back(operand);
      }
    }

    rewriter.replaceOpWithNewOp<OpType>(op, op->getResultTypes(), operandsI32,
                                        op->getAttrs());
    return success();
  }
};

// TODO(b/264240901): Comprehensively support shape computations to the extent
// needed to support bounded dynamism in MHLO export.
struct ShapeLegalizeToHloPass
    : public impl::ShapeLegalizeToHloPassBase<ShapeLegalizeToHloPass> {
  void runOnOperation() override {
    // In order to make dynamic MHLO programs compatible with HLO,
    // we need to get rid of all non-MHLO ops as well as the two shape-related
    // MHLO ops: mhlo.compute_reshape_shape and mhlo.cstr_reshapable.
    //
    // As an example, a cursory inspection of the TF/XLA bridge, which provides
    // one data point of an MHLO producer that can generate dynamic MHLO
    // programs, reveals the following non-MHLO ops:
    //   * shape.broadcast
    //   * shape.concat
    //   * shape.cstr_broadcastable
    //   * shape.cstr_eq
    //   * shape.dim
    //   * shape.split_at
    //   * shape.to_extent_tensor
    //   * shape.assuming
    //   * shape.assuming_yield
    //   * tensor.dim
    //   * tensor.extract
    //   * tensor.from_elements
    //
    // Most of these ops are convertible to MHLO, although the representation is
    // going to be pretty laborious for many of them. Luckily, canonicalization
    // is able to remove unnecessary cruft. At the moment, this pass is a
    // work in progress, so now all of these ops are supported.
    //
    // The only problem (and a big problem at that) are the ops involved in
    // shape constraints: cstr* ops as well as shape.assuming*. Since HLO does
    // not support shape constraints, it is currently unclear what to do with
    // them, unless they can be removed by --symbolic-shape-optimization.
    // At the moment, this pass is a work in progress, so it does not provide
    // an answer to this problem yet.
    ConversionTarget target(getContext());
    target.addIllegalDialect<shape::ShapeDialect>();
    target.addIllegalDialect<tensor::TensorDialect>();
    target.addIllegalOp<mhlo::ComputeReshapeShapeOp>();
    target.addIllegalOp<mhlo::CstrReshapableOp>();
    target.addDynamicallyLegalDialect<mhlo::MhloDialect>([](Operation* op) {
      return !llvm::any_of(op->getOperands(), hasIndexStyle);
    });
    target.addLegalOp<tensor::CastOp>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    // The patterns do what one might expect, converting between MLIR-style
    // and HLO-style shape computations.
    //
    // The only complication is that MLIR style uses index/tensor<Nxindex>
    // whereas HLO style uses tensor<i32>/vararg of tensor<i32>. We bridge
    // this gap by producing unrealized_conversion_cast ops, which we expect
    // to ultimately annihilate with each other upon canonicalization if
    // everything went right.
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertComputeReshapeShapeOpPattern>(&getContext());
    patterns.add<ConvertNumElementsOpPattern>(&getContext());
    patterns.add<ConvertShapeOfOpPattern>(&getContext());
    patterns.add<CastOperandsPattern<DynamicBroadcastInDimOp>>(&getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<func::FuncOp>>
createShapeLegalizeToHloPass() {
  return std::make_unique<ShapeLegalizeToHloPass>();
}

}  // namespace mhlo
}  // namespace mlir

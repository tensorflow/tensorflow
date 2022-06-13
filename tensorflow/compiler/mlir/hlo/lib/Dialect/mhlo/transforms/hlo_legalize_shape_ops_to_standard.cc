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

// This file implements logic for lowering HLO/LHLO dialect to Linalg dialect.

#include <algorithm>
#include <numeric>
#include <string>
#include <utility>

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir-hlo/Dialect/mhlo/transforms/type_conversion.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace {

struct ComputeReshapeShapeConversion
    : public OpConversionPattern<mhlo::ComputeReshapeShapeOp> {
  using OpConversionPattern<mhlo::ComputeReshapeShapeOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::ComputeReshapeShapeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    auto* ctx = op->getContext();
    Value neg_one = rewriter.create<arith::ConstantIndexOp>(loc, -1);
    auto index_type = rewriter.getIndexType();
    auto numElements = adaptor.getOperands()[0];
    auto targetShapeType =
        adaptor.getOperands()[1].getType().cast<ShapedType>();
    auto extentType =
        shape::getExtentTensorType(ctx, targetShapeType.getDimSize(0));

    // Calculate the computed actual extent for a possible dynamic extent.
    auto newShape = targetShapeType.getElementType().isIndex()
                        ? adaptor.getOperands()[1]
                        : rewriter.create<arith::IndexCastOp>(
                              loc, extentType, adaptor.getOperands()[1]);
    Value newShapeRank =
        rewriter.create<shape::RankOp>(loc, index_type, newShape);
    // The product begins with a -1 seed which will cancel out a -1 extent in
    // the input shape if there is one. If there is not, this computed result
    // will never be used, so it's okay to compute a negative number of
    // elements.
    auto accountedNumEls =
        rewriter.create<shape::ReduceOp>(loc, newShape, neg_one);
    {
      PatternRewriter::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToEnd(accountedNumEls.getBody());
      Value lhs = accountedNumEls.getBody()->getArgument(1);
      Value rhs = accountedNumEls.getBody()->getArgument(2);
      rewriter.create<shape::YieldOp>(
          loc, rewriter.create<arith::MulIOp>(loc, lhs, rhs).getResult());
    }
    Value missing_dim_val = rewriter.create<arith::DivUIOp>(
        loc, numElements, accountedNumEls->getResult(0));

    // Create the final target shape with a possible dynamic extent replace with
    // the calculated extent.
    SmallVector<Value> dynamicExtent;
    if (!targetShapeType.hasStaticShape())
      dynamicExtent.push_back(newShapeRank);
    auto gen = rewriter.create<tensor::GenerateOp>(
        loc, targetShapeType, dynamicExtent,
        [&](OpBuilder& b, Location loc, ValueRange indices) {
          Value extent = b.create<shape::GetExtentOp>(loc, index_type, newShape,
                                                      indices[0]);
          Value useMissingDimVal = b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, extent, neg_one);
          Value dimVal = b.create<arith::SelectOp>(loc, useMissingDimVal,
                                                   missing_dim_val, extent);
          dimVal = targetShapeType.getElementType().isIndex()
                       ? dimVal
                       : b.create<arith::IndexCastOp>(
                             loc, targetShapeType.getElementType(), dimVal);
          b.create<tensor::YieldOp>(loc, dimVal);
        });
    rewriter.replaceOp(op, gen.result());

    return success();
  }
};

struct CstrReshapableConversion
    : public OpConversionPattern<mhlo::CstrReshapableOp> {
  using OpConversionPattern<mhlo::CstrReshapableOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::CstrReshapableOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    auto* ctx = op->getContext();
    Value negOne = rewriter.create<arith::ConstantIndexOp>(loc, -1);
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto numElements = adaptor.getOperands()[0];
    auto targetShapeType =
        adaptor.getOperands()[1].getType().cast<ShapedType>();
    auto extentType =
        shape::getExtentTensorType(ctx, targetShapeType.getDimSize(0));

    // Calculate the computed actual extent for a possible dynamic extent.
    auto newShape = targetShapeType.getElementType().isIndex()
                        ? adaptor.getOperands()[1]
                        : rewriter.create<arith::IndexCastOp>(
                              loc, extentType, adaptor.getOperands()[1]);
    auto reduction = rewriter.create<shape::ReduceOp>(
        loc, newShape, llvm::makeArrayRef({one, zero, zero}));
    {
      PatternRewriter::InsertionGuard g(rewriter);
      auto* body = reduction.getBody();
      rewriter.setInsertionPointToEnd(body);
      Value extent = body->getArgument(1);
      Value isDynamic = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, negOne, extent);
      Value isInvalid = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, extent, negOne);
      Value totalDynamic = rewriter.create<arith::AddIOp>(
          loc, rewriter.create<arith::SelectOp>(loc, isDynamic, one, zero),
          body->getArgument(3));
      Value totalInvalid = rewriter.create<arith::AddIOp>(
          loc, rewriter.create<arith::SelectOp>(loc, isInvalid, one, zero),
          body->getArgument(4));
      Value extentOrOne =
          rewriter.create<arith::SelectOp>(loc, isDynamic, one, extent);
      Value totalElements = rewriter.create<arith::MulIOp>(
          loc, extentOrOne, body->getArgument(2));
      rewriter.create<shape::YieldOp>(
          loc, llvm::makeArrayRef({totalElements, totalDynamic, totalInvalid}));
    }
    // Avoid division by zero.
    Value isZeroElements = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, reduction->getResult(0), zero);
    Value divisor = rewriter.create<arith::SelectOp>(loc, isZeroElements, one,
                                                     reduction->getResult(0));
    Value isDivisible = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, zero,
        rewriter.create<arith::RemSIOp>(loc, numElements, divisor));
    // Must have 0 or 1 dynamic dimensions.
    Value acceptablyDynamic = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ule, reduction->getResult(1), one);
    // Must have no invalid dimensions.
    Value noInvalid = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, reduction->getResult(2), zero);
    // If there is no dynamic dimension then the number of elements must match.
    Value hasOneDynamic = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, reduction->getResult(1), one);
    Value equalIfNotDynamic = rewriter.create<arith::OrIOp>(
        loc, hasOneDynamic,
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                       numElements, reduction->getResult(0)));

    Value allPassing = rewriter.create<arith::AndIOp>(
        loc, isDivisible,
        rewriter.create<arith::AndIOp>(
            loc, acceptablyDynamic,
            rewriter.create<arith::AndIOp>(loc, noInvalid, equalIfNotDynamic)));

    rewriter.replaceOpWithNewOp<shape::CstrRequireOp>(
        op, allPassing, "Required valid reshape shape input");

    return success();
  }
};

struct HloLegalizeShapeOpsToStandardPass
    : public mhlo::HloLegalizeShapeOpsToStandardPassBase<
          HloLegalizeShapeOpsToStandardPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<arith::ArithmeticDialect, shape::ShapeDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext& ctx = getContext();
    RewritePatternSet patterns(&ctx);
    ConversionTarget target(ctx);
    target.addLegalDialect<arith::ArithmeticDialect, tensor::TensorDialect,
                           shape::ShapeDialect>();

    target.addLegalOp<UnrealizedConversionCastOp>();

    auto func = getOperation();
    mhlo::RemoveSignTypeConverter typeConverter;
    mhlo::populateHloShapeOpsToStandardConversionPattern(&ctx, typeConverter,
                                                         &patterns);
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

namespace mhlo {

void populateHloShapeOpsToStandardConversionPattern(
    MLIRContext* context, TypeConverter& typeConverter,
    RewritePatternSet* patterns) {
  // clang-format off
  patterns->add<
      ComputeReshapeShapeConversion,
      CstrReshapableConversion>(typeConverter, context);
  // clang-format on
}

std::unique_ptr<OperationPass<func::FuncOp>>
createLegalizeHloShapeOpsToStandardPass() {
  return std::make_unique<HloLegalizeShapeOpsToStandardPass>();
}

}  // namespace mhlo
}  // namespace mlir

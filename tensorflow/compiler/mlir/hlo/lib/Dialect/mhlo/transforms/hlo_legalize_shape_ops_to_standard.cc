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
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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
    auto ctx = op->getContext();
    Value neg_one = rewriter.create<arith::ConstantIndexOp>(loc, -1);
    auto index_type = rewriter.getIndexType();
    auto num_elements = adaptor.getOperands()[0];
    auto target_shape_type =
        adaptor.getOperands()[1].getType().cast<ShapedType>();
    auto extent_type =
        shape::getExtentTensorType(ctx, target_shape_type.getDimSize(0));

    // Calculate the computed actual extent for a possible dynamic extent.
    auto new_shape = target_shape_type.getElementType().isIndex()
                         ? adaptor.getOperands()[1]
                         : rewriter.create<arith::IndexCastOp>(
                               loc, extent_type, adaptor.getOperands()[1]);
    Value new_shape_rank =
        rewriter.create<shape::RankOp>(loc, index_type, new_shape);
    // The product begins with a -1 seed which will cancel out a -1 extent in
    // the input shape if there is one. If there is not, this computed result
    // will never be used, so it's okay to compute a negative number of
    // elements.
    auto accounted_num_els =
        rewriter.create<shape::ReduceOp>(loc, new_shape, neg_one);
    {
      PatternRewriter::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToEnd(accounted_num_els.getBody());
      Value lhs = accounted_num_els.getBody()->getArgument(1);
      Value rhs = accounted_num_els.getBody()->getArgument(2);
      rewriter.create<shape::YieldOp>(
          loc, rewriter.create<arith::MulIOp>(loc, lhs, rhs).getResult());
    }
    Value missing_dim_val = rewriter.create<arith::DivUIOp>(
        loc, num_elements, accounted_num_els->getResult(0));

    // Create the final target shape with a possible dynamic extent replace with
    // the calculated extent.
    SmallVector<Value> dynamic_extent;
    if (!target_shape_type.hasStaticShape())
      dynamic_extent.push_back(new_shape_rank);
    auto gen = rewriter.create<tensor::GenerateOp>(
        loc, target_shape_type, dynamic_extent,
        [&](OpBuilder& b, Location loc, ValueRange indices) {
          Value extent = b.create<shape::GetExtentOp>(loc, index_type,
                                                      new_shape, indices[0]);
          Value use_missing_dim_val = b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, extent, neg_one);
          Value dim_val = b.create<SelectOp>(loc, use_missing_dim_val,
                                             missing_dim_val, extent);
          dim_val = target_shape_type.getElementType().isIndex()
                        ? dim_val
                        : b.create<arith::IndexCastOp>(
                              loc, target_shape_type.getElementType(), dim_val);
          b.create<tensor::YieldOp>(loc, dim_val);
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
    auto ctx = op->getContext();
    Value neg_one = rewriter.create<arith::ConstantIndexOp>(loc, -1);
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto num_elements = adaptor.getOperands()[0];
    auto target_shape_type =
        adaptor.getOperands()[1].getType().cast<ShapedType>();
    auto extent_type =
        shape::getExtentTensorType(ctx, target_shape_type.getDimSize(0));

    // Calculate the computed actual extent for a possible dynamic extent.
    auto new_shape = target_shape_type.getElementType().isIndex()
                         ? adaptor.getOperands()[1]
                         : rewriter.create<arith::IndexCastOp>(
                               loc, extent_type, adaptor.getOperands()[1]);
    auto reduction = rewriter.create<shape::ReduceOp>(
        loc, new_shape, llvm::makeArrayRef({one, zero, zero}));
    {
      PatternRewriter::InsertionGuard g(rewriter);
      auto body = reduction.getBody();
      rewriter.setInsertionPointToEnd(body);
      Value extent = body->getArgument(1);
      Value is_dynamic = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, neg_one, extent);
      Value is_invalid = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, extent, neg_one);
      Value total_dynamic = rewriter.create<arith::AddIOp>(
          loc, rewriter.create<SelectOp>(loc, is_dynamic, one, zero),
          body->getArgument(3));
      Value total_invalid = rewriter.create<arith::AddIOp>(
          loc, rewriter.create<SelectOp>(loc, is_invalid, one, zero),
          body->getArgument(4));
      Value extent_or_one =
          rewriter.create<SelectOp>(loc, is_dynamic, one, extent);
      Value total_elements = rewriter.create<arith::MulIOp>(
          loc, extent_or_one, body->getArgument(2));
      rewriter.create<shape::YieldOp>(
          loc,
          llvm::makeArrayRef({total_elements, total_dynamic, total_invalid}));
    }
    // Avoid division by zero.
    Value is_zero_elements = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, reduction->getResult(0), zero);
    Value divisor = rewriter.create<SelectOp>(loc, is_zero_elements, one,
                                              reduction->getResult(0));
    Value is_divisible = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, zero,
        rewriter.create<arith::RemSIOp>(loc, num_elements, divisor));
    // Must have 0 or 1 dynamic dimensions.
    Value acceptably_dynamic = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ule, reduction->getResult(1), one);
    // Must have no invalid dimensions.
    Value no_invalid = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, reduction->getResult(2), zero);
    // If the old shape has size zero, the new shape must have size zero too.
    // This can be a zero factor or a -1.
    Value has_one_dynamic = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, reduction->getResult(1), one);
    Value equal_if_empty = rewriter.create<arith::OrIOp>(
        loc, has_one_dynamic,
        rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, is_zero_elements,
            rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                           num_elements, zero)));

    Value all_passing = rewriter.create<arith::AndIOp>(
        loc, is_divisible,
        rewriter.create<arith::AndIOp>(
            loc, acceptably_dynamic,
            rewriter.create<arith::AndIOp>(loc, no_invalid, equal_if_empty)));

    rewriter.replaceOpWithNewOp<shape::CstrRequireOp>(
        op, all_passing, "Required valid reshape shape input");

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

  void runOnFunction() override {
    MLIRContext& ctx = getContext();
    OwningRewritePatternList patterns(&ctx);
    ConversionTarget target(ctx);
    target.addLegalDialect<arith::ArithmeticDialect, StandardOpsDialect,
                           tensor::TensorDialect, shape::ShapeDialect>();

    target.addLegalOp<UnrealizedConversionCastOp>();

    auto func = getFunction();
    mhlo::RemoveSignTypeConverter type_converter;
    mhlo::populateHLOShapeOpsToStandardConversionPattern(&ctx, type_converter,
                                                         &patterns);
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

namespace mhlo {

void populateHLOShapeOpsToStandardConversionPattern(
    MLIRContext* context, TypeConverter& type_converter,
    OwningRewritePatternList* patterns) {
  // clang-format off
  patterns->insert<
      ComputeReshapeShapeConversion,
      CstrReshapableConversion>(type_converter, context);
  // clang-format on
}

std::unique_ptr<OperationPass<FuncOp>>
createLegalizeHloShapeOpsToStandardPass() {
  return std::make_unique<HloLegalizeShapeOpsToStandardPass>();
}

}  // namespace mhlo
}  // namespace mlir

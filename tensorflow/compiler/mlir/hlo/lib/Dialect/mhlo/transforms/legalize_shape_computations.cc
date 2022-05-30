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

// This file implements logic for lowering HLO/LHLO dialect to scalar shape
// operations.

#include <algorithm>
#include <numeric>
#include <string>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSet.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_attrs.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/map_mhlo_to_scalar_op.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace {

// We assume that if one of the operands is a FromElements operation that means
// it is a shape computation.
bool OpIsShapeComputation(Operation *op) {
  bool found_from_elements = false;
  for (auto operand : op->getOperands()) {
    auto shaped_ty = operand.getType().template cast<ShapedType>();
    if (!shaped_ty.hasRank() || shaped_ty.getRank() > 1) return false;
    if (auto from_elements =
            operand.template getDefiningOp<tensor::FromElementsOp>()) {
      found_from_elements = true;
      continue;
    }
  }
  return found_from_elements;
}

template <typename OpTy>
class MhloElementwiseConverter : public OpRewritePattern<OpTy> {
 public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const final {
    if (!OpIsShapeComputation(op)) return failure();

    auto result_ty = op.getType().template cast<ShapedType>();

    Location loc = op.getLoc();
    SmallVector<Value> operands;
    for (int i = 0, s = result_ty.getNumElements(); i < s; i++) {
      SmallVector<Value> extracts;
      for (auto operand : op->getOperands()) {
        ShapedType operand_ty = operand.getType().template cast<ShapedType>();
        if (operand_ty.getRank() == 0) {
          Value extract =
              rewriter.create<tensor::ExtractOp>(loc, operand, ValueRange({}));
          extracts.push_back(extract);
        } else {
          Value idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
          Value extract = rewriter.create<tensor::ExtractOp>(loc, operand, idx);
          extracts.push_back(extract);
        }
      }

      Value scalar_op = mhlo::MhloOpToStdScalarOp::map<OpTy>(
          op, result_ty.getElementType(), extracts, &rewriter);
      operands.push_back(scalar_op);
    }

    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(op, result_ty,
                                                        operands);

    return success();
  }
};

class ConcatenateConverter : public OpRewritePattern<mhlo::ConcatenateOp> {
 public:
  using OpRewritePattern<mhlo::ConcatenateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ConcatenateOp op,
                                PatternRewriter &rewriter) const final {
    if (!OpIsShapeComputation(op)) return failure();

    Location loc = op.getLoc();
    auto result_ty = op.getType().cast<ShapedType>();
    llvm::SmallVector<Value> elements;
    elements.reserve(result_ty.getNumElements());

    for (auto operand : op->getOperands()) {
      ShapedType operand_ty = operand.getType().template cast<ShapedType>();
      if (operand_ty.getRank() == 0) {
        Value extract =
            rewriter.create<tensor::ExtractOp>(loc, operand, ValueRange({}));
        elements.push_back(extract);
      } else {
        for (int i = 0, s = operand_ty.getNumElements(); i < s; i++) {
          Value idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
          Value extract = rewriter.create<tensor::ExtractOp>(loc, operand, idx);
          elements.push_back(extract);
        }
      }
    }

    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(op, result_ty,
                                                        elements);
    return success();
  }
};

class GetDimSizeConverter : public OpRewritePattern<mhlo::GetDimensionSizeOp> {
 public:
  using OpRewritePattern<mhlo::GetDimensionSizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::GetDimensionSizeOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    auto result_ty = op.getType();
    auto element_ty = getElementTypeOrSelf(result_ty);
    auto dim_attr = rewriter.getIndexAttr(op.dimension());
    auto dim_const = rewriter.create<arith::ConstantOp>(loc, dim_attr);

    Value dim_op = rewriter.create<tensor::DimOp>(loc, rewriter.getIndexType(),
                                                  op.operand(), dim_const);

    // Cast to the correct element type and convert to a tensor.
    Value cast = rewriter.create<arith::IndexCastOp>(loc, element_ty, dim_op);
    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(op, result_ty, cast);
    return success();
  }
};

class ReshapeConverter : public OpRewritePattern<mhlo::ReshapeOp> {
 public:
  using OpRewritePattern<mhlo::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ReshapeOp op,
                                PatternRewriter &rewriter) const final {
    auto operand = op.operand();
    auto shaped_ty = operand.getType().template cast<ShapedType>();
    if (!shaped_ty.hasRank() || shaped_ty.getRank() > 1) return failure();

    auto result_ty = op.getType().cast<ShapedType>();

    auto from_elements = op.operand().getDefiningOp<tensor::FromElementsOp>();
    if (!from_elements) return failure();

    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(
        op, result_ty, from_elements.getOperands());
    return success();
  }
};

struct HloLegalizeShapeComputationsPass
    : public mhlo::HloLegalizeShapeComputationsPassBase<
          HloLegalizeShapeComputationsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithmeticDialect, math::MathDialect,
                    func::FuncDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    RewritePatternSet patterns(&ctx);

    auto func = getOperation();
    mhlo::populateShapeComputationPatterns(&ctx, &patterns);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

namespace mhlo {

void populateShapeComputationPatterns(MLIRContext *context,
                                      RewritePatternSet *patterns) {
  patterns->add<MhloElementwiseConverter<mhlo::AbsOp>,
                MhloElementwiseConverter<mhlo::AddOp>,
                MhloElementwiseConverter<mhlo::AndOp>,
                MhloElementwiseConverter<mhlo::CeilOp>,
                MhloElementwiseConverter<mhlo::ConvertOp>,
                MhloElementwiseConverter<mhlo::DivOp>,
                MhloElementwiseConverter<mhlo::FloorOp>,
                MhloElementwiseConverter<mhlo::MaxOp>,
                MhloElementwiseConverter<mhlo::MinOp>,
                MhloElementwiseConverter<mhlo::MulOp>,
                MhloElementwiseConverter<mhlo::NegOp>,
                MhloElementwiseConverter<mhlo::RoundOp>,
                MhloElementwiseConverter<mhlo::RsqrtOp>,
                MhloElementwiseConverter<mhlo::SqrtOp>,
                MhloElementwiseConverter<mhlo::SubOp>, ConcatenateConverter,
                GetDimSizeConverter, ReshapeConverter>(context);
}

std::unique_ptr<OperationPass<func::FuncOp>>
createLegalizeShapeComputationsPass() {
  return std::make_unique<HloLegalizeShapeComputationsPass>();
}

}  // namespace mhlo
}  // namespace mlir

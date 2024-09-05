/* Copyright 2022 The OpenXLA Authors.

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
#include <memory>
#include <numeric>
#include <string>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSet.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/map_mhlo_to_scalar_op.h"
#include "mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_HLOLEGALIZESHAPECOMPUTATIONSPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

namespace {

// We assume that if one of the operands is a FromElements operation that means
// it is a shape computation.
bool opIsShapeComputation(Operation *op) {
  bool foundFromElements = false;
  for (auto operand : op->getOperands()) {
    auto shapedTy = mlir::cast<ShapedType>(operand.getType());
    if (!shapedTy.hasRank() || shapedTy.getRank() > 1) return false;
    if (auto fromElements =
            operand.template getDefiningOp<tensor::FromElementsOp>()) {
      foundFromElements = true;
      continue;
    }
  }
  return foundFromElements;
}

template <typename OpTy>
class MhloElementwiseConverter : public OpRewritePattern<OpTy> {
 public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const final {
    if (!opIsShapeComputation(op)) return failure();

    auto resultTy = mlir::cast<ShapedType>(op.getType());

    Location loc = op.getLoc();
    SmallVector<Value> operands;
    for (int i = 0, s = resultTy.getNumElements(); i < s; i++) {
      SmallVector<Value> extracts;
      for (auto operand : op->getOperands()) {
        ShapedType operandTy = mlir::cast<ShapedType>(operand.getType());
        if (operandTy.getRank() == 0) {
          Value extract =
              rewriter.create<tensor::ExtractOp>(loc, operand, ValueRange({}));
          extracts.push_back(extract);
        } else {
          Value idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
          Value extract = rewriter.create<tensor::ExtractOp>(loc, operand, idx);
          extracts.push_back(extract);
        }
      }

      Value scalarOp = mhlo::MhloOpToStdScalarOp::mapOp(
          op, resultTy.getElementType(), extracts, &rewriter);
      operands.push_back(scalarOp);
    }

    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(op, resultTy, operands);

    return success();
  }
};

class ConcatenateConverter : public OpRewritePattern<mhlo::ConcatenateOp> {
 public:
  using OpRewritePattern<mhlo::ConcatenateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ConcatenateOp op,
                                PatternRewriter &rewriter) const final {
    if (!opIsShapeComputation(op)) return failure();

    Location loc = op.getLoc();
    auto resultTy = mlir::cast<ShapedType>(op.getType());
    llvm::SmallVector<Value> elements;
    elements.reserve(resultTy.getNumElements());

    for (auto operand : op->getOperands()) {
      ShapedType operandTy = mlir::cast<ShapedType>(operand.getType());
      if (operandTy.getRank() == 0) {
        Value extract =
            rewriter.create<tensor::ExtractOp>(loc, operand, ValueRange({}));
        elements.push_back(extract);
      } else {
        for (int i = 0, s = operandTy.getNumElements(); i < s; i++) {
          Value idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
          Value extract = rewriter.create<tensor::ExtractOp>(loc, operand, idx);
          elements.push_back(extract);
        }
      }
    }

    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(op, resultTy, elements);
    return success();
  }
};

class GetDimSizeConverter : public OpRewritePattern<mhlo::GetDimensionSizeOp> {
 public:
  using OpRewritePattern<mhlo::GetDimensionSizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::GetDimensionSizeOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    auto resultTy = op.getType();
    auto elementTy = getElementTypeOrSelf(resultTy);
    auto dimAttr = rewriter.getIndexAttr(op.getDimension());
    auto dimConst = rewriter.create<arith::ConstantOp>(loc, dimAttr);

    Value dimOp = rewriter.create<tensor::DimOp>(loc, rewriter.getIndexType(),
                                                 op.getOperand(), dimConst);

    // Cast to the correct element type and convert to a tensor.
    Value cast = rewriter.create<arith::IndexCastOp>(loc, elementTy, dimOp);
    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(op, resultTy, cast);
    return success();
  }
};

class ReshapeConverter : public OpRewritePattern<mhlo::ReshapeOp> {
 public:
  using OpRewritePattern<mhlo::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ReshapeOp op,
                                PatternRewriter &rewriter) const final {
    auto operand = op.getOperand();
    auto shapedTy = mlir::cast<ShapedType>(operand.getType());
    if (!shapedTy.hasRank() || shapedTy.getRank() > 1) return failure();

    auto resultTy = mlir::cast<ShapedType>(op.getType());

    auto fromElements = op.getOperand().getDefiningOp<tensor::FromElementsOp>();
    if (!fromElements) return failure();

    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(
        op, resultTy, fromElements.getOperands());
    return success();
  }
};

struct HloLegalizeShapeComputationsPass
    : public impl::HloLegalizeShapeComputationsPassBase<
          HloLegalizeShapeComputationsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, math::MathDialect, func::FuncDialect,
                    tensor::TensorDialect>();
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
                MhloElementwiseConverter<mhlo::SubtractOp>,
                ConcatenateConverter, GetDimSizeConverter, ReshapeConverter>(
      context);
}

std::unique_ptr<OperationPass<func::FuncOp>>
createLegalizeShapeComputationsPass() {
  return std::make_unique<HloLegalizeShapeComputationsPass>();
}

}  // namespace mhlo
}  // namespace mlir

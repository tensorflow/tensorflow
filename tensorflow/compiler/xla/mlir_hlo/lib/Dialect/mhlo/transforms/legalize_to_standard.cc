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

// This file implements logic for lowering MHLO dialect to Standard dialect.

#include <utility>

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace {
#include "generated_legalize_to_standard.inc"
}  // end anonymous namespace
namespace mhlo {
namespace {

class CompareIConvert : public OpRewritePattern<mhlo::CompareOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::CompareOp op,
                                PatternRewriter &rewriter) const override {
    auto lhs = op.lhs();
    auto rhs = op.rhs();
    auto lhsType = lhs.getType().cast<TensorType>();
    auto rhsType = rhs.getType().cast<TensorType>();

    // Broadcasting not supported by this rewrite.
    if (lhsType.getShape() != rhsType.getShape()) return failure();

    if (!lhsType.getElementType().isSignlessInteger() ||
        !rhsType.getElementType().isSignlessInteger())
      return failure();

    Optional<arith::CmpIPredicate> comparePredicate;
    switch (op.comparison_direction()) {
      case ComparisonDirection::EQ:
        comparePredicate = arith::CmpIPredicate::eq;
        break;
      case ComparisonDirection::NE:
        comparePredicate = arith::CmpIPredicate::ne;
        break;
      case ComparisonDirection::LT:
        comparePredicate = arith::CmpIPredicate::slt;
        break;
      case ComparisonDirection::LE:
        comparePredicate = arith::CmpIPredicate::sle;
        break;
      case ComparisonDirection::GT:
        comparePredicate = arith::CmpIPredicate::sgt;
        break;
      case ComparisonDirection::GE:
        comparePredicate = arith::CmpIPredicate::sge;
        break;
      default:
        comparePredicate = llvm::None;
    }

    if (!comparePredicate.has_value()) return failure();

    rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, comparePredicate.getValue(),
                                               lhs, rhs);
    return success();
  }
};

class CompareFConvert : public OpRewritePattern<mhlo::CompareOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::CompareOp op,
                                PatternRewriter &rewriter) const override {
    auto lhs = op.lhs();
    auto rhs = op.rhs();
    auto lhsType = lhs.getType().cast<TensorType>();
    auto rhsType = rhs.getType().cast<TensorType>();

    // Broadcasting not supported by this rewrite.
    if (lhsType.getShape() != rhsType.getShape()) return failure();

    if (!lhsType.getElementType().isa<FloatType>() ||
        !rhsType.getElementType().isa<FloatType>())
      return failure();

    Optional<arith::CmpFPredicate> comparePredicate;
    switch (op.comparison_direction()) {
      case ComparisonDirection::EQ:
        comparePredicate = arith::CmpFPredicate::OEQ;
        break;
      case ComparisonDirection::NE:
        comparePredicate = arith::CmpFPredicate::UNE;
        break;
      case ComparisonDirection::LT:
        comparePredicate = arith::CmpFPredicate::OLT;
        break;
      case ComparisonDirection::LE:
        comparePredicate = arith::CmpFPredicate::OLE;
        break;
      case ComparisonDirection::GT:
        comparePredicate = arith::CmpFPredicate::OGT;
        break;
      case ComparisonDirection::GE:
        comparePredicate = arith::CmpFPredicate::OGE;
        break;
      default:
        comparePredicate = llvm::None;
    }

    if (!comparePredicate.has_value()) return failure();

    rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, comparePredicate.getValue(),
                                               lhs, rhs);
    return success();
  }
};

// Replace IotaOp with an integer constant. A ConvertOp is added to
// convert the integer constant to iota result type. For complex types, the real
// part is replaced with the generated constant and the imaginary part is
// replaced with zero tensor.
class ConvertIotaOp : public OpRewritePattern<mhlo::IotaOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::IotaOp op,
                                PatternRewriter &rewriter) const override {
    auto outputType = op.getType().cast<ShapedType>();
    auto outputSize = outputType.getNumElements();
    auto dimension = op.iota_dimension();
    auto maxDimSize = outputType.getDimSize(dimension);

    auto elementType = outputType.getElementType();
    int bitwidth;

    auto complexTy = elementType.dyn_cast<ComplexType>();
    Type intOrFloatTy = elementType;
    if (complexTy) intOrFloatTy = complexTy.getElementType();

    bitwidth = intOrFloatTy.getIntOrFloatBitWidth();
    llvm::SmallVector<APInt, 10> values;
    values.reserve(outputSize);

    int64_t increaseStride = outputSize;
    for (uint64_t i = 0; i <= dimension; i++) {
      increaseStride /= outputType.getDimSize(i);
    }

    int64_t currentValue = 0;
    for (int i = 0; i < outputSize; i++) {
      int64_t value = (currentValue / increaseStride) % maxDimSize;
      values.push_back(APInt(bitwidth, value));
      ++currentValue;
    }

    auto intShapeType = RankedTensorType::get(
        outputType.getShape(),
        IntegerType::get(rewriter.getContext(), bitwidth));
    auto loc = op.getLoc();
    auto integerConst = rewriter.create<mlir::arith::ConstantOp>(
        loc, DenseIntElementsAttr::get(intShapeType, values));

    auto intOrFloatShapeTy =
        RankedTensorType::get(outputType.getShape(), intOrFloatTy);

    auto iotaConst =
        rewriter.create<ConvertOp>(loc, intOrFloatShapeTy, integerConst);

    // For int/float types we are done, replace op and return.
    if (!complexTy) {
      rewriter.replaceOp(op, iotaConst.getResult());
      return success();
    }

    // For complex types, generate a constant tensor of zeroes for the imaginary
    // part and use iota_const for real part.
    auto zeroes = rewriter.create<mlir::arith::ConstantOp>(
        loc, DenseIntElementsAttr::get(intShapeType, APInt(bitwidth, 0)));
    auto imagZeroes =
        rewriter.create<ConvertOp>(loc, intOrFloatShapeTy, zeroes);
    rewriter.replaceOpWithNewOp<mhlo::ComplexOp>(op, iotaConst, imagZeroes);
    return success();
  }
};

}  // end anonymous namespace

namespace {
struct LegalizeToStandardPass
    : public LegalizeToStandardPassBase<LegalizeToStandardPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithmeticDialect, math::MathDialect,
                    func::FuncDialect>();
  }

  /// Perform the lowering to Standard dialect.
  void runOnOperation() override;
};
}  // end anonymous namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createLegalizeToStdPass() {
  return std::make_unique<LegalizeToStandardPass>();
}

void populateMhloToStdPatterns(RewritePatternSet *patterns,
                               mlir::MLIRContext *ctx) {
  mlir::populateWithGenerated(*patterns);
  patterns->add<CompareFConvert, CompareIConvert, ConvertIotaOp>(ctx);
}

/// Perform the lowering to standard dialect.
void LegalizeToStandardPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  mlir::mhlo::populateMhloToStdPatterns(&patterns, &getContext());
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

}  // end namespace mhlo
}  // end namespace mlir

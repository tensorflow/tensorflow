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

#include "llvm/ADT/StringSwitch.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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
    auto lhs_type = lhs.getType().cast<TensorType>();
    auto rhs_type = rhs.getType().cast<TensorType>();

    // Broadcasting not supported by this rewrite.
    if (lhs_type.getShape() != rhs_type.getShape()) return failure();

    if (!lhs_type.getElementType().isSignlessInteger() ||
        !rhs_type.getElementType().isSignlessInteger())
      return failure();

    auto comparison_direction = op.comparison_direction();
    auto compare_predicate =
        llvm::StringSwitch<Optional<arith::CmpIPredicate>>(comparison_direction)
            .Case("EQ", arith::CmpIPredicate::eq)
            .Case("NE", arith::CmpIPredicate::ne)
            .Case("LT", arith::CmpIPredicate::slt)
            .Case("LE", arith::CmpIPredicate::sle)
            .Case("GT", arith::CmpIPredicate::sgt)
            .Case("GE", arith::CmpIPredicate::sge)
            .Default(llvm::None);

    if (!compare_predicate.hasValue()) return failure();

    rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, compare_predicate.getValue(),
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
    auto lhs_type = lhs.getType().cast<TensorType>();
    auto rhs_type = rhs.getType().cast<TensorType>();

    // Broadcasting not supported by this rewrite.
    if (lhs_type.getShape() != rhs_type.getShape()) return failure();

    if (!lhs_type.getElementType().isa<FloatType>() ||
        !rhs_type.getElementType().isa<FloatType>())
      return failure();

    auto comparison_direction = op.comparison_direction();
    auto compare_predicate =
        llvm::StringSwitch<Optional<arith::CmpFPredicate>>(comparison_direction)
            .Case("EQ", arith::CmpFPredicate::OEQ)
            .Case("NE", arith::CmpFPredicate::UNE)
            .Case("LT", arith::CmpFPredicate::OLT)
            .Case("LE", arith::CmpFPredicate::OLE)
            .Case("GT", arith::CmpFPredicate::OGT)
            .Case("GE", arith::CmpFPredicate::OGE)
            .Default(llvm::None);

    if (!compare_predicate.hasValue()) return failure();

    rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, compare_predicate.getValue(),
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
    auto output_type = op.getType().cast<ShapedType>();
    auto output_size = output_type.getNumElements();
    auto dimension = op.iota_dimension();
    auto max_dim_size = output_type.getDimSize(dimension);

    auto element_type = output_type.getElementType();
    int bitwidth;

    auto complex_ty = element_type.dyn_cast<ComplexType>();
    Type int_or_float_ty = element_type;
    if (complex_ty) int_or_float_ty = complex_ty.getElementType();

    bitwidth = int_or_float_ty.getIntOrFloatBitWidth();
    llvm::SmallVector<APInt, 10> values;
    values.reserve(output_size);

    int64_t increase_stride = output_size;
    for (uint64_t i = 0; i <= dimension; i++) {
      increase_stride /= output_type.getDimSize(i);
    }

    int64_t current_value = 0;
    for (int i = 0; i < output_size; i++) {
      int64_t value = (current_value / increase_stride) % max_dim_size;
      values.push_back(APInt(bitwidth, value));
      ++current_value;
    }

    auto int_shape_type = RankedTensorType::get(
        output_type.getShape(),
        IntegerType::get(rewriter.getContext(), bitwidth));
    auto loc = op.getLoc();
    auto integer_const = rewriter.create<mlir::arith::ConstantOp>(
        loc, DenseIntElementsAttr::get(int_shape_type, values));

    auto int_or_float_shape_ty =
        RankedTensorType::get(output_type.getShape(), int_or_float_ty);

    auto iota_const =
        rewriter.create<ConvertOp>(loc, int_or_float_shape_ty, integer_const);

    // For int/float types we are done, replace op and return.
    if (!complex_ty) {
      rewriter.replaceOp(op, iota_const.getResult());
      return success();
    }

    // For complex types, generate a constant tensor of zeroes for the imaginary
    // part and use iota_const for real part.
    auto zeroes = rewriter.create<mlir::arith::ConstantOp>(
        loc, DenseIntElementsAttr::get(int_shape_type, APInt(bitwidth, 0)));
    auto imag_zeroes =
        rewriter.create<ConvertOp>(loc, int_or_float_shape_ty, zeroes);
    rewriter.replaceOpWithNewOp<mhlo::ComplexOp>(op, iota_const, imag_zeroes);
    return success();
  }
};

}  // end anonymous namespace

namespace {
struct LegalizeToStandardPass
    : public LegalizeToStandardPassBase<LegalizeToStandardPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithmeticDialect, math::MathDialect,
                    StandardOpsDialect>();
  }

  /// Perform the lowering to Standard dialect.
  void runOnFunction() override;
};
}  // end anonymous namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> createLegalizeToStdPass() {
  return std::make_unique<LegalizeToStandardPass>();
}

void PopulateMhloToStdPatterns(OwningRewritePatternList *patterns,
                               mlir::MLIRContext *ctx) {
  mlir::populateWithGenerated(*patterns);
  patterns->insert<CompareFConvert, CompareIConvert, ConvertIotaOp>(ctx);
}

/// Perform the lowering to standard dialect.
void LegalizeToStandardPass::runOnFunction() {
  OwningRewritePatternList patterns(&getContext());
  mlir::mhlo::PopulateMhloToStdPatterns(&patterns, &getContext());
  (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
}

}  // end namespace mhlo
}  // end namespace mlir

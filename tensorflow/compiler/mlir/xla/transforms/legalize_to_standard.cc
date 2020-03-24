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

// This file implements logic for lowering XLA dialect to Standard dialect.

#include "llvm/ADT/StringSwitch.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/IR/Function.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"

namespace mlir {
namespace {
#include "tensorflow/compiler/mlir/xla/transforms/generated_legalize_to_standard.inc"
}  // end anonymous namespace
namespace xla_hlo {
namespace {

class CompareIConvert : public OpRewritePattern<xla_hlo::CompareOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(xla_hlo::CompareOp op,
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
        llvm::StringSwitch<Optional<CmpIPredicate>>(comparison_direction)
            .Case("EQ", CmpIPredicate::eq)
            .Case("NE", CmpIPredicate::ne)
            .Case("LT", CmpIPredicate::slt)
            .Case("LE", CmpIPredicate::sle)
            .Case("GT", CmpIPredicate::sgt)
            .Case("GE", CmpIPredicate::sge)
            .Default(llvm::None);

    if (!compare_predicate.hasValue()) return failure();

    rewriter.replaceOpWithNewOp<CmpIOp>(op, compare_predicate.getValue(), lhs,
                                        rhs);
    return success();
  }
};

class CompareFConvert : public OpRewritePattern<xla_hlo::CompareOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(xla_hlo::CompareOp op,
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
        llvm::StringSwitch<Optional<CmpFPredicate>>(comparison_direction)
            .Case("EQ", CmpFPredicate::OEQ)
            .Case("NE", CmpFPredicate::UNE)
            .Case("LT", CmpFPredicate::OLT)
            .Case("LE", CmpFPredicate::OLE)
            .Case("GT", CmpFPredicate::OGT)
            .Case("GE", CmpFPredicate::OGE)
            .Default(llvm::None);

    if (!compare_predicate.hasValue()) return failure();

    rewriter.replaceOpWithNewOp<CmpFOp>(op, compare_predicate.getValue(), lhs,
                                        rhs);
    return success();
  }
};

// Replace IotaOp with an integer constant. A ConvertOp is added to
// convert the integer constant to iota result type. For complex types, the real
// part is replaced with the generated constant and the imaginary part is
// replaced with zero tensor.
class ConvertIotaOp : public OpRewritePattern<xla_hlo::IotaOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(xla_hlo::IotaOp op,
                                PatternRewriter &rewriter) const override {
    auto output_type = op.getType().cast<ShapedType>();
    auto output_size = output_type.getNumElements();
    auto dimension = op.iota_dimension().getSExtValue();
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
    for (int i = 0; i <= dimension; i++) {
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
        IntegerType::get(bitwidth, rewriter.getContext()));
    auto loc = op.getLoc();
    auto integer_const = rewriter.create<mlir::ConstantOp>(
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
    auto zeroes = rewriter.create<mlir::ConstantOp>(
        loc, DenseIntElementsAttr::get(int_shape_type, APInt(bitwidth, 0)));
    auto imag_zeroes =
        rewriter.create<ConvertOp>(loc, int_or_float_shape_ty, zeroes);
    rewriter.replaceOpWithNewOp<xla_hlo::ComplexOp>(op, iota_const,
                                                    imag_zeroes);
    return success();
  }
};

}  // end anonymous namespace

namespace {
struct LegalizeToStandard : public FunctionPass<LegalizeToStandard> {
  /// Perform the lowering to Standard dialect.
  void runOnFunction() override;
};
}  // end anonymous namespace

std::unique_ptr<mlir::OpPassBase<mlir::FuncOp>> createLegalizeToStdPass() {
  return std::make_unique<LegalizeToStandard>();
}

void PopulateXlaToStdPatterns(OwningRewritePatternList *patterns,
                              mlir::MLIRContext *ctx) {
  mlir::populateWithGenerated(ctx, patterns);
  patterns->insert<CompareFConvert, CompareIConvert, ConvertIotaOp>(ctx);
}

/// Perform the lowering to standard dialect.
void LegalizeToStandard::runOnFunction() {
  OwningRewritePatternList patterns;
  mlir::xla_hlo::PopulateXlaToStdPatterns(&patterns, &getContext());
  applyPatternsGreedily(getFunction(), patterns);
}

static PassRegistration<LegalizeToStandard> legalize_pass(
    "xla-legalize-to-std", "Legalize from XLA dialect to standard dialect");

}  // end namespace xla_hlo
}  // end namespace mlir

/* Copyright 2025 The OpenXLA Authors.

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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::xla {

namespace ttir = ::mlir::triton;

#define GEN_PASS_DEF_ARITHFP8CONVERSIONTOTRITONPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

bool IsFp8TypeOrFp8TensorType(mlir::Type t) {
  auto element_type = mlir::getElementTypeOrSelf(t);
  return mlir::cast<mlir::FloatType>(element_type).getWidth() == 8;
}

bool IsFp8ToNonFp8Conversion(mlir::Type src_ty, mlir::Type dst_ty) {
  return IsFp8TypeOrFp8TensorType(src_ty) && !IsFp8TypeOrFp8TensorType(dst_ty);
}

bool IsNonFp8ToFp8Conversion(mlir::Type src_ty, mlir::Type dst_ty) {
  return !IsFp8TypeOrFp8TensorType(src_ty) && IsFp8TypeOrFp8TensorType(dst_ty);
}

class LowerExtFOp : public mlir::OpRewritePattern<mlir::arith::ExtFOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      mlir::arith::ExtFOp op, mlir::PatternRewriter& rewriter) const override {
    auto src_ty = op.getOperand().getType();
    auto dst_ty = op.getType();

    if (!IsFp8ToNonFp8Conversion(src_ty, dst_ty)) {
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          "ExtFOp will be lowered to FpToFpOp only if it converts from FP8 to "
          "FP16, BF16, FP32, FP64.");
    }
    rewriter.replaceOpWithNewOp<ttir::FpToFpOp>(op, op.getType(),
                                                op.getOperand());
    return mlir::success();
  }
};

class LowerTruncFOp : public mlir::OpRewritePattern<mlir::arith::TruncFOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      mlir::arith::TruncFOp op,
      mlir::PatternRewriter& rewriter) const override {
    auto src_ty = op.getOperand().getType();
    auto dst_ty = op.getType();

    if (!IsNonFp8ToFp8Conversion(src_ty, dst_ty)) {
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          "TruncFOp will be lowered to FpToFpOp only if it converts from FP16, "
          "BF16, FP32 or FP64 to FP8.");
    }

    // TruncFOp default rounding mode is to_nearest_even based on the code in
    // ArithOps.cpp
    auto rounding_mode = op.getRoundingmode().value_or(
        mlir::arith::RoundingMode::to_nearest_even);

    ttir::RoundingMode triton_rounding_mode;
    switch (rounding_mode) {
      case mlir::arith::RoundingMode::to_nearest_even:
        triton_rounding_mode = ttir::RoundingMode::RTNE;
        break;
      case mlir::arith::RoundingMode::toward_zero:
        triton_rounding_mode = ttir::RoundingMode::RTZ;
        break;
      default:
        return rewriter.notifyMatchFailure(
            op->getLoc(),
            "TruncFOp rounding mode attribute not supported by "
            "FpToFpOp.");
    }

    ttir::RoundingModeAttr triton_rounding_mode_attr =
        ttir::RoundingModeAttr::get(rewriter.getContext(),
                                    triton_rounding_mode);

    rewriter.replaceOpWithNewOp<ttir::FpToFpOp>(
        op, op.getType(), op.getOperand(), triton_rounding_mode_attr);
    return mlir::success();
  }
};

class ArithFP8ConversionToTritonPass
    : public impl::ArithFP8ConversionToTritonPassBase<
          ArithFP8ConversionToTritonPass> {
 public:
  void runOnOperation() override {
    mlir::MLIRContext* mlir_context = &getContext();
    mlir::RewritePatternSet patterns(mlir_context);
    patterns.add<LowerExtFOp, LowerTruncFOp>(mlir_context);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};
}  // namespace

// F8 <-> FP16, BF16, FP32, FP64 need to be handled via Triton's tt.fp_to_fp
// because LLVM doesn't support casts from/to FP8.
// TODO(b/413272992): Add better test coverage for FpToFpOp.
std::unique_ptr<Pass> CreateArithFP8ConversionToTritonPass() {
  return std::make_unique<ArithFP8ConversionToTritonPass>();
}

}  // namespace mlir::triton::xla

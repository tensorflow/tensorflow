
/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/pad.h"

#include <cstdint>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/op_util_common.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/pad_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir::odml {
namespace {

bool IsPadLegal(mhlo::PadOp op) {
  return AnyNegativePads(op) || !TrivialInterior(op);
}

bool IsPadValCstZero(mhlo::PadOp op) {
  if (matchPattern(op.getPaddingValue(), m_AnyZeroFloat())) {
    return true;
  }
  if (matchPattern(op.getPaddingValue(), m_Zero())) {
    return true;
  }
  return false;
}

DenseIntElementsAttr BuildTFLPaddingAttr(OpBuilder& b, mhlo::PadOp op) {
  auto lows = UnrollI64Splat(op.getEdgePaddingLow());
  auto highs = UnrollI64Splat(op.getEdgePaddingHigh());

  llvm::SmallVector<int64_t> res;
  for (auto [l, h] : llvm::zip(lows, highs)) {
    res.push_back(l);
    res.push_back(h);
  }

  const int64_t n_dims = res.size();
  auto tfl_padding_type =
      RankedTensorType::get({n_dims / 2, 2}, b.getI64Type());
  return DenseIntElementsAttr::get(tfl_padding_type, res);
}

//===------------------------------------------------------------------------===
// mhlo.pad -> tfl.pad
//===------------------------------------------------------------------------===

class LegalizePad : public OpConversionPattern<mhlo::PadOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::PadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final;
};

LogicalResult LegalizePad::matchAndRewrite(
    mhlo::PadOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  if (IsPadLegal(op)) {
    return rewriter.notifyMatchFailure(op, "Matching an already legal pad op.");
  }
  if (!IsPadValCstZero(op)) {
    return rewriter.notifyMatchFailure(
        op, "Legalizing to padv1 requires zero const padding values.");
  }

  auto tfl_paddings = BuildTFLPaddingAttr(rewriter, op);
  auto paddings_op =
      rewriter.create<arith::ConstantOp>(op->getLoc(), tfl_paddings);

  rewriter.replaceOpWithNewOp<TFL::PadOp>(op, op.getType(), op.getOperand(),
                                          paddings_op);
  return success();
}

//===------------------------------------------------------------------------===
// mhlo.pad -> tfl.padv2
//===------------------------------------------------------------------------===

class LegalizePadV2 : public OpConversionPattern<mhlo::PadOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::PadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final;
};

LogicalResult LegalizePadV2::matchAndRewrite(
    mhlo::PadOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  if (IsPadLegal(op)) {
    return rewriter.notifyMatchFailure(op, "Matching an already legal pad op.");
  }
  if (IsPadValCstZero(op)) {
    return rewriter.notifyMatchFailure(
        op, "Legalizing to padv2 requires non zero const padding values.");
  }

  auto tfl_paddings = BuildTFLPaddingAttr(rewriter, op);
  auto paddings_op =
      rewriter.create<arith::ConstantOp>(op->getLoc(), tfl_paddings);

  rewriter.replaceOpWithNewOp<TFL::PadV2Op>(op, op.getType(), op.getOperand(),
                                            paddings_op, op.getPaddingValue());
  return success();
}

}  // namespace

void PopulatePadPatterns(MLIRContext* ctx, RewritePatternSet& patterns,
                         ConversionTarget& target) {
  patterns.add<LegalizePad>(ctx);
  patterns.add<LegalizePadV2>(ctx);
  target.addDynamicallyLegalOp<mhlo::PadOp>(IsPadLegal);
}

}  // namespace mlir::odml

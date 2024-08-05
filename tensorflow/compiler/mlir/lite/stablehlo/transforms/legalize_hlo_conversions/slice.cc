

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
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/slice.h"

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  // IWYU pragma: keep
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir::odml {
namespace {

// Cast the value to i32.
Value BuildTFLCastOp(OpBuilder& b, Value value) {
  return b.create<TFL::CastOp>(
      value.getLoc(),
      RankedTensorType::get(llvm::cast<ShapedType>(value.getType()).getShape(),
                            b.getI32Type()),
      value);
}

class LegalizeSliceOp : public OpConversionPattern<mhlo::SliceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::SliceOp slice_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto begin = rewriter.create<arith::ConstantOp>(slice_op.getLoc(),
                                                    slice_op.getStartIndices());
    auto end = rewriter.create<arith::ConstantOp>(slice_op.getLoc(),
                                                  slice_op.getLimitIndices());
    auto strides = rewriter.create<arith::ConstantOp>(slice_op.getLoc(),
                                                      slice_op.getStrides());
    auto zero = rewriter.getIntegerAttr(rewriter.getI32Type(), 0);
    auto no_offset = rewriter.getBoolAttr(false);

    rewriter.replaceOpWithNewOp<TFL::StridedSliceOp>(
        slice_op, slice_op.getType(), slice_op.getOperand(),
        BuildTFLCastOp(rewriter, begin), BuildTFLCastOp(rewriter, end),
        BuildTFLCastOp(rewriter, strides), zero, zero, zero, zero, zero,
        no_offset);
    return success();
  }
};

}  // namespace

void PopulateSlicePatterns(MLIRContext* ctx, RewritePatternSet& patterns,
                           ConversionTarget& target) {
  patterns.add<LegalizeSliceOp>(ctx);
  target.addIllegalOp<mhlo::SliceOp>();
}

}  // namespace mlir::odml

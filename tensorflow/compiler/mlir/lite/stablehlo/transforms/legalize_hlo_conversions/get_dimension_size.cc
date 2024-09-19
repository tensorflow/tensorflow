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

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/get_dimension_size.h"

#include <cstdint>

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir::odml {
namespace {

// Converts a MHLO::GetDimensionSizeOP to TFL ops.
class LeagalizeDimensionSizeOp
    : public OpConversionPattern<mhlo::GetDimensionSizeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::GetDimensionSizeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    auto operand_type = llvm::cast<ShapedType>(op.getOperand().getType());

    auto shaped_op_type =
        RankedTensorType::get({operand_type.getRank()}, rewriter.getI64Type());
    Value shape_op = rewriter.create<TFL::ShapeOp>(op.getLoc(), shaped_op_type,
                                                   op.getOperand());

    Value size = BuildIntArrayConstOp<arith::ConstantOp>(builder, rewriter, {1},
                                                         rewriter.getI64Type());

    auto begin = BuildIntArrayConstOp<arith::ConstantOp>(
        builder, rewriter,
        llvm::SmallVector<int64_t>({static_cast<int64_t>(op.getDimension())}),
        rewriter.getI64Type());

    auto slice_type = RankedTensorType::get({1}, rewriter.getI64Type());
    Value slice = rewriter.create<TFL::SliceOp>(op.getLoc(), slice_type,
                                                shape_op, begin, size);

    auto op_el_type = llvm::cast<ShapedType>(op.getType()).getElementType();
    if (op_el_type != slice_type.getElementType()) {
      slice = rewriter.create<TFL::CastOp>(op->getLoc(),
                                           slice_type.clone(op_el_type), slice);
    }

    rewriter.replaceOpWithNewOp<TFL::SqueezeOp>(op, op.getType(), slice,
                                                rewriter.getI64ArrayAttr({0}));

    return success();
  }
};

}  // namespace

void PopulateGetDimensionSizePatterns(MLIRContext* ctx,
                                      RewritePatternSet& patterns,
                                      ConversionTarget& target) {
  target.addIllegalOp<mhlo::GetDimensionSizeOp>();
  patterns.add<LeagalizeDimensionSizeOp>(ctx);
}

}  // namespace mlir::odml

/* Copyright 2019 The OpenXLA Authors.

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

#include <numeric>

#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace mhlo {

namespace {

// Converts ClampOp with broadcast semantics. ClampOp requires "all three arrays
// must be the same shape. Alternatively, as a restricted form of broadcasting,
// min and/or max can be a scalar of type T."
struct ClampWithBroadcastConvert : public OpRewritePattern<ClampOp> {
  explicit ClampWithBroadcastConvert(MLIRContext *context)
      : OpRewritePattern<ClampOp>(context) {}

  LogicalResult matchAndRewrite(ClampOp op,
                                PatternRewriter &rewriter) const override {
    auto operandType =
        mlir::dyn_cast<RankedTensorType>(op.getOperand().getType());
    auto maxType = mlir::dyn_cast<RankedTensorType>(op.getMax().getType());
    auto minType = mlir::dyn_cast<RankedTensorType>(op.getMin().getType());
    // Unrancked types are not supported.
    if (!operandType || !maxType || !minType) return failure();
    // Does not support operand with dynamic dimensions for now.
    if (!operandType.hasStaticShape()) return failure();

    ArrayRef<int64_t> operandShape = operandType.getShape();

    Value maxValue = op.getMax();
    if (maxType != operandType) {
      assert(maxType.getRank() == 0);
      maxValue = rewriter.createOrFold<BroadcastOp>(
          op.getLoc(), operandType, maxValue,
          rewriter.getI64TensorAttr(operandShape));
    }

    Value minValue = op.getMin();
    if (minType != operandType) {
      assert(minType.getRank() == 0);
      minValue = rewriter.createOrFold<BroadcastOp>(
          op.getLoc(), operandType, minValue,
          rewriter.getI64TensorAttr(operandShape));
    }

    rewriter.replaceOpWithNewOp<ClampOp>(op, op.getType(), minValue,
                                         op.getOperand(), maxValue);
    return success();
  }
};

}  // namespace

void setupMaterializeBroadcastsLegality(MLIRContext * /*context*/,
                                        ConversionTarget *conversionTarget) {
  conversionTarget->addDynamicallyLegalOp<ClampOp>([](ClampOp op) {
    return op.getMax().getType() == op.getOperand().getType() &&
           op.getMin().getType() == op.getOperand().getType();
  });
}

void populateMaterializeBroadcastsPatterns(MLIRContext *context,
                                           RewritePatternSet *patterns) {
  // ClampOp. This op has a special case where it accepts either same-shaped
  // inputs or scalars (a restricted form of broadcasting). This makes the
  // broadcast explicit.
  patterns->add<ClampWithBroadcastConvert>(context);
}

}  // namespace mhlo
}  // namespace mlir

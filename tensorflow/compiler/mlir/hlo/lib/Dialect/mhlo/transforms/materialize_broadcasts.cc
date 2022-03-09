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

#include <numeric>

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
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
    auto operand_type = op.operand().getType().dyn_cast<RankedTensorType>();
    auto max_type = op.max().getType().dyn_cast<RankedTensorType>();
    auto min_type = op.min().getType().dyn_cast<RankedTensorType>();
    // Unrancked types are not supported.
    if (!operand_type || !max_type || !min_type) return failure();
    // Does not support operand with dynamic dimensions for now.
    if (!operand_type.hasStaticShape()) return failure();

    ArrayRef<int64_t> operand_shape = operand_type.getShape();

    Value max_value = op.max();
    if (max_type != operand_type) {
      assert(max_type.getRank() == 0);
      max_value = rewriter.createOrFold<BroadcastOp>(
          op.getLoc(), operand_type, max_value,
          rewriter.getI64TensorAttr(operand_shape));
    }

    Value min_value = op.min();
    if (min_type != operand_type) {
      assert(min_type.getRank() == 0);
      min_value = rewriter.createOrFold<BroadcastOp>(
          op.getLoc(), operand_type, min_value,
          rewriter.getI64TensorAttr(operand_shape));
    }

    rewriter.replaceOpWithNewOp<ClampOp>(op, op.getType(), min_value,
                                         op.operand(), max_value);
    return success();
  }
};

}  // namespace

void SetupMaterializeBroadcastsLegality(MLIRContext * /*context*/,
                                        ConversionTarget *conversionTarget) {
  conversionTarget->addDynamicallyLegalOp<ClampOp>([](ClampOp op) {
    return op.max().getType() == op.operand().getType() &&
           op.min().getType() == op.operand().getType();
  });
}

void PopulateMaterializeBroadcastsPatterns(MLIRContext *context,
                                           RewritePatternSet *patterns) {
  // ClampOp. This op has a special case where it accepts either same-shaped
  // inputs or scalars (a restricted form of broadcasting). This makes the
  // broadcast explicit.
  patterns->add<ClampWithBroadcastConvert>(context);
}

}  // namespace mhlo
}  // namespace mlir

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/ir/chlo_ops.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"

namespace mlir {
namespace xla_chlo {

namespace {

template <typename ChloOpTy, typename HloOpTy>
struct ConvertTrivialNonBroadcastBinaryOp
    : public OpConversionPattern<ChloOpTy> {
  using OpConversionPattern<ChloOpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ChloOpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Only rewrite for statically determinable non-broadcasting cases.
    auto lhs = operands[0].getType().dyn_cast<RankedTensorType>();
    auto rhs = operands[1].getType().dyn_cast<RankedTensorType>();
    if (!lhs || !rhs) return failure();

    // Requires rank broadcast.
    if (lhs.getRank() != rhs.getRank()) return failure();
    // Any dynamic dimension may require broadcasting and requires more
    // analysis.
    if (!lhs.hasStaticShape() || !rhs.hasStaticShape()) return failure();

    for (auto extents : llvm::zip(lhs.getShape(), rhs.getShape())) {
      auto lhs_extent = std::get<0>(extents);
      auto rhs_extent = std::get<1>(extents);
      if (lhs_extent != rhs_extent) {
        return failure();
      }
    }

    rewriter.replaceOp(op, rewriter.createOrFold<HloOpTy>(
                               op.getLoc(), operands[0], operands[1], nullptr));
    return success();
  }

  StringRef hlo_op_name_;
};

template <typename ChloOpTy, typename HloOpTy>
void PopulateForBinaryOp(MLIRContext *context,
                         OwningRewritePatternList *patterns) {
  patterns->insert<ConvertTrivialNonBroadcastBinaryOp<ChloOpTy, HloOpTy>>(
      context, 10);
}

}  // namespace

void PopulateLegalizeChloToHloPatterns(MLIRContext *context,
                                       OwningRewritePatternList *patterns) {
#define POPULATE_BCAST(ChloOp, HloOp) \
  PopulateForBinaryOp<ChloOp, xla_hlo::HloOp>(context, patterns);

  POPULATE_BCAST(BroadcastAddOp, AddOp);
  POPULATE_BCAST(BroadcastAtan2Op, Atan2Op);
  POPULATE_BCAST(BroadcastDivOp, DivOp);
  POPULATE_BCAST(BroadcastMaxOp, MaxOp);
  POPULATE_BCAST(BroadcastMinOp, MinOp);
  POPULATE_BCAST(BroadcastMulOp, MulOp);
  POPULATE_BCAST(BroadcastPowOp, PowOp);
  POPULATE_BCAST(BroadcastRemOp, RemOp);
  POPULATE_BCAST(BroadcastShiftLeftOp, ShiftLeftOp);
  POPULATE_BCAST(BroadcastShiftRightArithmeticOp, ShiftRightArithmeticOp);
  POPULATE_BCAST(BroadcastShiftRightLogicalOp, ShiftRightLogicalOp);
  POPULATE_BCAST(BroadcastSubOp, SubOp);
  POPULATE_BCAST(BroadcastAndOp, AndOp);
  POPULATE_BCAST(BroadcastOrOp, OrOp);
  POPULATE_BCAST(BroadcastXorOp, XorOp);
}

}  // namespace xla_chlo
}  // namespace mlir

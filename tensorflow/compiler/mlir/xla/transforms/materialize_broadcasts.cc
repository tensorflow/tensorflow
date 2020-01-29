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

#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project
#include "mlir/Transforms/DialectConversion.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"

namespace mlir {
namespace xla_hlo {

namespace {

// Returns a 1-d i64 elements attribute populated with numbers from start to
// end, excluding.
static DenseIntElementsAttr GetI64ElementsAttrForSeq(int start, int end,
                                                     Builder *builder) {
  int size = end - start;

  SmallVector<int64_t, 4> vals;
  vals.resize(size);
  std::iota(vals.begin(), vals.end(), start);

  TensorType ty = RankedTensorType::get({size}, builder->getIntegerType(64));
  return DenseIntElementsAttr::get(ty, vals);
}

// Helper function for OpRewritePattern classes to materialize broadcasts on
// LHS and RHS arguments to a binary op.
//
// Returns true and sets out_lhs and out_rhs to BroadcastInDimOps if successful,
// returns false otherwise.
template <typename SrcOp>
bool CreateBroadcastsForBinaryOp(SrcOp op, PatternRewriter *rewriter,
                                 Value *out_lhs, Value *out_rhs) {
  if (!op.broadcast_dimensions().hasValue()) {
    // Note: the op may still have an implicit broadcast on it, such as
    // for (tensor<1xf32>, tensor<4xf32>).
    return false;
  }

  // Insert BroadcastInDimOps for the left-hand-side and right-hand-side args,
  // replacing the original LHS and RHS args in the source op with the results
  // of the broadcasts.
  //
  // If the higher dimensional argument does not actually need the broadcast,
  // a canonicalization pass should be able to remove that op later.
  Value lhs = op.lhs();
  Value rhs = op.rhs();

  auto op_ranked_type = op.getType().template dyn_cast<RankedTensorType>();
  auto lhs_ranked_type = lhs.getType().dyn_cast<RankedTensorType>();
  auto rhs_ranked_type = rhs.getType().dyn_cast<RankedTensorType>();
  if (!op_ranked_type || !lhs_ranked_type || !rhs_ranked_type) {
    // Unranked, can't determine at this point how to perform the broadcast.
    return false;
  }

  if (!op_ranked_type.hasStaticShape()) {
    // Dynamic result shape, can't use BroadcastInDimOp.
    return false;
  }

  auto lhs_rank = lhs_ranked_type.getRank();
  auto rhs_rank = rhs_ranked_type.getRank();

  // Set broadcast_dimensions to [0, ..., rank] for the higher rank arg.
  // Use the original op.broadcast_dimensions for the lower rank arg.
  auto higher_rank_broadcast_dims =
      GetI64ElementsAttrForSeq(0, std::max(lhs_rank, rhs_rank), rewriter);
  DenseIntElementsAttr lhs_broadcast_dims;
  DenseIntElementsAttr rhs_broadcast_dims;
  if (lhs_rank > rhs_rank) {
    lhs_broadcast_dims = higher_rank_broadcast_dims;
    rhs_broadcast_dims = op.broadcast_dimensions().getValue();
  } else if (lhs_rank < rhs_rank) {
    lhs_broadcast_dims = op.broadcast_dimensions().getValue();
    rhs_broadcast_dims = higher_rank_broadcast_dims;
  } else {
    // This shouldn't happen for legal ops. If the broadcast_dimensions
    // attribute is set, the ranks should be different.
    // TODO(scotttodd): Add a custom verification for ops and assert here.
    return false;
  }

  // BroadcastInDimOp must have the same element type for operands and results,
  // so preserve the original output shape and the original input element type.
  // For example, `SrcOp (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xi1>`:
  //   broadcast_in_dim (tensor<1x4xf32>) -> tensor<1x4xf32>
  //   broadcast_in_dim (tensor<4xf32>) -> tensor<1x4xf32>
  //   SrcOp (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xi1>
  ArrayRef<int64_t> op_shape = op_ranked_type.getShape();
  auto lhs_type =
      RankedTensorType::get(op_shape, lhs_ranked_type.getElementType());
  auto rhs_type =
      RankedTensorType::get(op_shape, rhs_ranked_type.getElementType());

  *out_lhs = rewriter->createOrFold<BroadcastInDimOp>(op.getLoc(), lhs_type,
                                                      lhs, lhs_broadcast_dims);
  *out_rhs = rewriter->createOrFold<BroadcastInDimOp>(op.getLoc(), rhs_type,
                                                      rhs, rhs_broadcast_dims);
  return true;
}

template <typename SrcOp>
struct BinaryOpWithBroadcastConvert : public OpRewritePattern<SrcOp> {
  explicit BinaryOpWithBroadcastConvert(MLIRContext *context)
      : OpRewritePattern<SrcOp>(context) {}

  PatternMatchResult matchAndRewrite(SrcOp op,
                                     PatternRewriter &rewriter) const override {
    Value new_lhs;
    Value new_rhs;
    if (!CreateBroadcastsForBinaryOp(op, &rewriter, &new_lhs, &new_rhs)) {
      return this->matchFailure();
    }

    // Replace the original op with a new one that uses the new args.
    // New args are broadcasts, so no dims are needed on the replacement op.
    rewriter.replaceOpWithNewOp<SrcOp>(op, op.getType(), new_lhs, new_rhs,
                                       /*broadcast_dims=*/nullptr);
    return this->matchSuccess();
  }
};

// Specialized class for CompareOp, as it has an additional builder argument.
struct CompareWithBroadcastConvert : public OpRewritePattern<CompareOp> {
  explicit CompareWithBroadcastConvert(MLIRContext *context)
      : OpRewritePattern<CompareOp>(context) {}

  PatternMatchResult matchAndRewrite(CompareOp op,
                                     PatternRewriter &rewriter) const override {
    Value new_lhs;
    Value new_rhs;
    if (!CreateBroadcastsForBinaryOp(op, &rewriter, &new_lhs, &new_rhs)) {
      return this->matchFailure();
    }

    rewriter.replaceOpWithNewOp<CompareOp>(op, op.getType(), new_lhs, new_rhs,
                                           /*broadcast_dims=*/nullptr,
                                           op.comparison_direction());
    return this->matchSuccess();
  }
};

}  // namespace

void SetupMaterializeBroadcastsLegality(MLIRContext *context,
                                        ConversionTarget *conversionTarget) {
#define ADD_DYNAMICALLY_LEGAL_OP_WITH_BROADCAST(OpType) \
  conversionTarget->addDynamicallyLegalOp<OpType>(      \
      [](OpType op) { return !op.broadcast_dimensions().hasValue(); });
  // Binary elementwise ops.
  ADD_DYNAMICALLY_LEGAL_OP_WITH_BROADCAST(AddOp);
  ADD_DYNAMICALLY_LEGAL_OP_WITH_BROADCAST(Atan2Op);
  ADD_DYNAMICALLY_LEGAL_OP_WITH_BROADCAST(DivOp);
  ADD_DYNAMICALLY_LEGAL_OP_WITH_BROADCAST(MaxOp);
  ADD_DYNAMICALLY_LEGAL_OP_WITH_BROADCAST(MinOp);
  ADD_DYNAMICALLY_LEGAL_OP_WITH_BROADCAST(MulOp);
  ADD_DYNAMICALLY_LEGAL_OP_WITH_BROADCAST(PowOp);
  ADD_DYNAMICALLY_LEGAL_OP_WITH_BROADCAST(RemOp);
  ADD_DYNAMICALLY_LEGAL_OP_WITH_BROADCAST(ShiftLeftOp);
  ADD_DYNAMICALLY_LEGAL_OP_WITH_BROADCAST(ShiftRightArithmeticOp);
  ADD_DYNAMICALLY_LEGAL_OP_WITH_BROADCAST(ShiftRightLogicalOp);
  ADD_DYNAMICALLY_LEGAL_OP_WITH_BROADCAST(SubOp);

  // Binary logical elementwise ops.
  ADD_DYNAMICALLY_LEGAL_OP_WITH_BROADCAST(AndOp);
  ADD_DYNAMICALLY_LEGAL_OP_WITH_BROADCAST(OrOp);
  ADD_DYNAMICALLY_LEGAL_OP_WITH_BROADCAST(XorOp);

  // CompareOp.
  ADD_DYNAMICALLY_LEGAL_OP_WITH_BROADCAST(CompareOp);

#undef ADD_DYNAMICALLY_LEGAL_OP_WITH_BROADCAST
}

void PopulateMaterializeBroadcastsPatterns(MLIRContext *context,
                                           OwningRewritePatternList *patterns) {
  // Binary elementwise ops.
  patterns->insert<BinaryOpWithBroadcastConvert<AddOp>>(context);
  patterns->insert<BinaryOpWithBroadcastConvert<Atan2Op>>(context);
  patterns->insert<BinaryOpWithBroadcastConvert<DivOp>>(context);
  patterns->insert<BinaryOpWithBroadcastConvert<MaxOp>>(context);
  patterns->insert<BinaryOpWithBroadcastConvert<MinOp>>(context);
  patterns->insert<BinaryOpWithBroadcastConvert<MulOp>>(context);
  patterns->insert<BinaryOpWithBroadcastConvert<PowOp>>(context);
  patterns->insert<BinaryOpWithBroadcastConvert<RemOp>>(context);
  patterns->insert<BinaryOpWithBroadcastConvert<ShiftLeftOp>>(context);
  patterns->insert<BinaryOpWithBroadcastConvert<ShiftRightArithmeticOp>>(
      context);
  patterns->insert<BinaryOpWithBroadcastConvert<ShiftRightLogicalOp>>(context);
  patterns->insert<BinaryOpWithBroadcastConvert<SubOp>>(context);

  // Binary logical elementwise ops.
  patterns->insert<BinaryOpWithBroadcastConvert<AndOp>>(context);
  patterns->insert<BinaryOpWithBroadcastConvert<OrOp>>(context);
  patterns->insert<BinaryOpWithBroadcastConvert<XorOp>>(context);

  // CompareOp. Note the specialized class instead of using the template.
  patterns->insert<CompareWithBroadcastConvert>(context);
}

}  // namespace xla_hlo
}  // namespace mlir

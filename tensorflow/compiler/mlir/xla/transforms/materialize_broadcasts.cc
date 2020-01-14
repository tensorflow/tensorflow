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

template <typename SrcOp>
struct BinaryOpWithBroadcastConvert : public OpRewritePattern<SrcOp> {
  explicit BinaryOpWithBroadcastConvert(MLIRContext *context)
      : OpRewritePattern<SrcOp>(context) {}

  PatternMatchResult matchAndRewrite(SrcOp op,
                                     PatternRewriter &rewriter) const override {
    if (!op.broadcast_dimensions().hasValue()) {
      // Note: the op may still have an implicit broadcast on it, such as
      // for (tensor<1xf32>, tensor<4xf32>).
      return this->matchFailure();
    }

    auto result_type = op.getType();

    // Insert BroadcastInDimOps for the left-hand-side and right-hand-side args,
    // replacing the original LHS and RHS args in the source op with the results
    // of the broadcasts.
    //
    // If the higher dimensional argument does not actually need the broadcast,
    // a canonicalization pass should be able to remove that op later.
    Value lhs = op.lhs();
    Value rhs = op.rhs();

    auto lhs_ranked_type = lhs.getType().dyn_cast<RankedTensorType>();
    auto rhs_ranked_type = rhs.getType().dyn_cast<RankedTensorType>();
    if (!lhs_ranked_type || !rhs_ranked_type) {
      // Unranked, can't determine at this point how to perform the broadcast.
      return this->matchFailure();
    }

    auto lhs_rank = lhs_ranked_type.getRank();
    auto rhs_rank = rhs_ranked_type.getRank();

    // Set broadcast_dimensions to [0, ..., rank] for the higher rank arg.
    // Use the original op.broadcast_dimensions for the lower rank arg.
    auto higher_rank_broadcast_dims =
        GetI64ElementsAttrForSeq(0, std::max(lhs_rank, rhs_rank), &rewriter);
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
      return this->matchFailure();
    }
    lhs = rewriter.createOrFold<BroadcastInDimOp>(op.getLoc(), result_type, lhs,
                                                  lhs_broadcast_dims);
    rhs = rewriter.createOrFold<BroadcastInDimOp>(op.getLoc(), result_type, rhs,
                                                  rhs_broadcast_dims);

    // Replace the original op with a new one that uses the new args.
    // As the new args are broadcasts, no broadcast dimensions are needed on
    // the replacement op.
    rewriter.replaceOpWithNewOp<SrcOp>(op, result_type, lhs, rhs,
                                       /*broadcast_dims=*/nullptr);

    return this->matchSuccess();
  }
};

}  // namespace

void SetupMaterializeBroadcastsLegality(MLIRContext *context,
                                        ConversionTarget *conversionTarget) {
#define ADD_DYNAMICALLY_LEGAL_OP_WITH_BROADCAST(OpType) \
  conversionTarget->addDynamicallyLegalOp<OpType>(      \
      [](OpType op) { return !op.broadcast_dimensions().hasValue(); });
  ADD_DYNAMICALLY_LEGAL_OP_WITH_BROADCAST(AddOp);
  ADD_DYNAMICALLY_LEGAL_OP_WITH_BROADCAST(DivOp);
  ADD_DYNAMICALLY_LEGAL_OP_WITH_BROADCAST(MaxOp);
  ADD_DYNAMICALLY_LEGAL_OP_WITH_BROADCAST(MinOp);
  ADD_DYNAMICALLY_LEGAL_OP_WITH_BROADCAST(MulOp);
  ADD_DYNAMICALLY_LEGAL_OP_WITH_BROADCAST(PowOp);
  ADD_DYNAMICALLY_LEGAL_OP_WITH_BROADCAST(SubOp);
#undef ADD_DYNAMICALLY_LEGAL_OP_WITH_BROADCAST
}

void PopulateMaterializeBroadcastsPatterns(MLIRContext *context,
                                           OwningRewritePatternList *patterns) {
  patterns->insert<BinaryOpWithBroadcastConvert<AddOp>>(context);
  patterns->insert<BinaryOpWithBroadcastConvert<DivOp>>(context);
  patterns->insert<BinaryOpWithBroadcastConvert<MaxOp>>(context);
  patterns->insert<BinaryOpWithBroadcastConvert<MinOp>>(context);
  patterns->insert<BinaryOpWithBroadcastConvert<MulOp>>(context);
  patterns->insert<BinaryOpWithBroadcastConvert<PowOp>>(context);
  patterns->insert<BinaryOpWithBroadcastConvert<SubOp>>(context);
}

}  // namespace xla_hlo
}  // namespace mlir

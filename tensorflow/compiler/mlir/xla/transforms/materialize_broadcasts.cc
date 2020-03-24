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

#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
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

  // Dynamic result shape, can't use BroadcastInDimOp.
  assert(op_ranked_type.hasStaticShape() &&
         "dynamic shape requires DynamicBroadcastInDim");

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

// Helper template to generate code for computing the result shape of a
// broadcasted operation. This ultimately should be subsumed by functions
// from the shape dialect.
// Assumes that large and small are the operand values of `op` and that they
// have a ranked tensory type with rank(large) >= rank(small).
template <typename SrcOp>
std::vector<Value> ComputeBroadcastedShape(SrcOp op, Value small, Value large,
                                           PatternRewriter *rewriter) {
  auto loc = op.getLoc();
  auto larger_ranked_type = large.getType().cast<RankedTensorType>();
  auto output_rank = larger_ranked_type.getRank();

  constexpr int kExpandShape = -1;

  std::vector<Value> shape_values;
  shape_values.reserve(output_rank);
  std::vector<int> indexes(output_rank, kExpandShape);
  DenseIntElementsAttr broadcast_dimensions =
      op.broadcast_dimensions().getValue();
  // Compute a mapping from output dimensions to their corresponding input
  // dimensions in the smaller ranked operand.
  for (auto pair : llvm::enumerate(broadcast_dimensions.getIntValues())) {
    indexes.at(pair.value().getLimitedValue()) = pair.index();
  }

  // Compute the broadcasted shape of the result using numpy style broadcasting
  // semantics. The result shape at a position is the shape of the larger
  // operand at that position if the no dimension of the smaller operand is
  // mapped to it.
  // If both operands contribute to an output dimension, their shape has to
  // either be the same in that dimension or it can be 1, in which case the
  // shape of the other operand is used.
  for (int i = 0; i < output_rank; ++i) {
    Value index_value;
    if (indexes[i] == kExpandShape) {
      // The smaller shape gets expanded to the larger one in this case.
      index_value = rewriter->create<mlir::DimOp>(loc, large, i);
    } else {
      // Compute the result shape depending on whether the rank of smaller is 1.
      // This does not check that the broadcast operation actualy is correct.
      // In particular, we do not check that both shapes are the same if the
      // smaller ranked shape is not 1.
      ConstantOp one = rewriter->create<mlir::ConstantOp>(
          loc, rewriter->getIntegerAttr(rewriter->getIndexType(), 1));
      DimOp lrg_dim = rewriter->create<mlir::DimOp>(loc, large, i);
      DimOp sml_dim = rewriter->create<mlir::DimOp>(loc, small, indexes[i]);
      CmpIOp compare =
          rewriter->create<mlir::CmpIOp>(loc, CmpIPredicate::eq, lrg_dim, one);
      index_value =
          rewriter->create<mlir::SelectOp>(loc, compare, lrg_dim, sml_dim);
    }
    // Ideally, we would like to keep this on index but MLIR does not allow
    // this.
    shape_values.push_back(rewriter->create<mlir::IndexCastOp>(
        loc, index_value, rewriter->getIntegerType(32)));
  }

  return shape_values;
}

// Helper function for OpRewritePattern classes to materialize dynamic
// broadcasts on LHS and RHS arguments to a binary op.
//
// Returns true and set out_lhs and out_rhs for materialized dynamic broadcasts
// for LHS and RHS arguments, else returns false.
template <typename SrcOp>
bool CreateDynamicBroadcastsForBinaryOp(SrcOp op, PatternRewriter *rewriter,
                                        Value *out_lhs, Value *out_rhs) {
  if (!op.broadcast_dimensions().hasValue()) {
    // Note: the op may still have an implicit broadcast on it, such as
    // for (tensor<1xf32>, tensor<4xf32>).
    return false;
  }

  // Insert BroadcastInDimOps for the left-hand-side and right-hand-side args,
  // replacing the original LHS and RHS args in the source op with the results
  // of the broadcasts.
  Value lhs = op.lhs();
  Value rhs = op.rhs();

  auto lhs_ranked_type = lhs.getType().dyn_cast<RankedTensorType>();
  auto rhs_ranked_type = rhs.getType().dyn_cast<RankedTensorType>();
  if (!lhs_ranked_type || !rhs_ranked_type) {
    // Unranked, can't determine at this point how to perform the broadcast.
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
  std::vector<Value> shape_elements;
  if (lhs_rank > rhs_rank) {
    lhs_broadcast_dims = higher_rank_broadcast_dims;
    rhs_broadcast_dims = op.broadcast_dimensions().getValue();
    shape_elements = ComputeBroadcastedShape<SrcOp>(op, rhs, lhs, rewriter);
  } else if (lhs_rank < rhs_rank) {
    lhs_broadcast_dims = op.broadcast_dimensions().getValue();
    rhs_broadcast_dims = higher_rank_broadcast_dims;
    shape_elements = ComputeBroadcastedShape<SrcOp>(op, lhs, rhs, rewriter);
  } else {
    // This shouldn't happen for legal ops. If the broadcast_dimensions
    // attribute is set, the ranks should be different.
    // TODO(scotttodd): Add a custom verification for ops and assert here.
    return false;
  }

  // DynamicBroadcastInDimOp preserves the element type but produces a tensor
  // with unranked shape. The rank of the output is the length of the
  // output shape argument.
  SmallVector<int64_t, 4> op_shape(shape_elements.size(),
                                   RankedTensorType::kDynamicSize);
  auto lhs_type =
      RankedTensorType::get(op_shape, lhs_ranked_type.getElementType());
  auto rhs_type =
      RankedTensorType::get(op_shape, rhs_ranked_type.getElementType());

  // We need a way to turn a list of scalars into a vector. While Standard
  // dialect does not have one, use the XLA_HLO variant.
  int shape_size = shape_elements.size();
  Type shape_element_type = shape_elements.front().getType();
  Value shape_value = rewriter->create<ScalarsToDimensionTensorOp>(
      op.getLoc(), RankedTensorType::get({shape_size}, shape_element_type),
      shape_elements);

  *out_lhs = rewriter->createOrFold<DynamicBroadcastInDimOp>(
      op.getLoc(), lhs_type, lhs, shape_value, lhs_broadcast_dims);
  *out_rhs = rewriter->createOrFold<DynamicBroadcastInDimOp>(
      op.getLoc(), rhs_type, rhs, shape_value, rhs_broadcast_dims);
  return true;
}

template <typename SrcOp>
struct BinaryOpWithBroadcastConvert : public OpRewritePattern<SrcOp> {
  explicit BinaryOpWithBroadcastConvert(MLIRContext *context)
      : OpRewritePattern<SrcOp>(context) {}

  LogicalResult matchAndRewrite(SrcOp op,
                                PatternRewriter &rewriter) const override {
    Value new_lhs;
    Value new_rhs;

    auto op_ranked_type = op.getType().template dyn_cast<RankedTensorType>();
    if (!op_ranked_type) return failure();

    if (op_ranked_type.hasStaticShape()) {
      if (!CreateBroadcastsForBinaryOp(op, &rewriter, &new_lhs, &new_rhs)) {
        return failure();
      }
    } else {
      if (!CreateDynamicBroadcastsForBinaryOp(op, &rewriter, &new_lhs,
                                              &new_rhs)) {
        return failure();
      }
    }

    // Replace the original op with a new one that uses the new args.
    // New args are broadcasts, so no dims are needed on the replacement op.
    rewriter.replaceOpWithNewOp<SrcOp>(op, op.getType(), new_lhs, new_rhs,
                                       /*broadcast_dims=*/nullptr);
    return success();
  }
};

// Specialized class for CompareOp, as it has an additional builder argument.
struct CompareWithBroadcastConvert : public OpRewritePattern<CompareOp> {
  explicit CompareWithBroadcastConvert(MLIRContext *context)
      : OpRewritePattern<CompareOp>(context) {}

  LogicalResult matchAndRewrite(CompareOp op,
                                PatternRewriter &rewriter) const override {
    Value new_lhs;
    Value new_rhs;
    if (!CreateBroadcastsForBinaryOp(op, &rewriter, &new_lhs, &new_rhs)) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<CompareOp>(op, op.getType(), new_lhs, new_rhs,
                                           /*broadcast_dims=*/nullptr,
                                           op.comparison_direction());
    return success();
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

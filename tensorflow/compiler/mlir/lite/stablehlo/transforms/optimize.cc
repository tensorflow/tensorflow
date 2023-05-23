/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace odml {

// Convert mhlo.dot to mhlo.dot_general.
LogicalResult ConvertDotToDotGeneral(mhlo::DotOp op,
                                     PatternRewriter &rewriter) {
  auto lhs_type = op.getLhs().getType().cast<ShapedType>();
  auto rhs_type = op.getRhs().getType().cast<ShapedType>();
  if (!lhs_type.hasRank() || !rhs_type.hasRank()) {
    return rewriter.notifyMatchFailure(op, "unsupported unranked input type");
  }
  if (lhs_type.getRank() < 1 || 2 < lhs_type.getRank() ||
      rhs_type.getRank() < 1 || 2 < rhs_type.getRank()) {
    return rewriter.notifyMatchFailure(
        op,
        "unsupported dot operation type; operands must be vectors or "
        "matrices");
  }
  rewriter.replaceOpWithNewOp<mhlo::DotGeneralOp>(
      op, op.getType(), op.getLhs(), op.getRhs(),
      mhlo::DotDimensionNumbersAttr::get(
          op.getContext(),
          /*lhsBatchingDimensions=*/{},
          /*rhsBatchingDimensions=*/{},
          /*lhsContractingDimensions=*/{lhs_type.getRank() - 1},
          /*rhsContractingDimensions=*/{0}),
      op.getPrecisionConfigAttr());
  return success();
}

// Convert reshape(dot_general(reshape(%y), %z)) to
// dot_general(%y, %z) if possible.
LogicalResult RemoveReshapeAroundDotGeneral(mhlo::ReshapeOp reshape_after,
                                            PatternRewriter &rewriter) {
  auto dot = dyn_cast_or_null<mhlo::DotGeneralOp>(
      reshape_after.getOperand().getDefiningOp());
  if (!dot) return failure();

  auto reshape_before =
      dyn_cast_or_null<mhlo::ReshapeOp>(dot.getLhs().getDefiningOp());
  if (!reshape_before) return failure();

  if (!dot.getLhs().getType().hasStaticShape() ||
      !dot.getRhs().getType().hasStaticShape() ||
      !reshape_before.getOperand().getType().hasStaticShape() ||
      !dot.getType().hasStaticShape() ||
      !reshape_after.getType().hasStaticShape()) {
    return rewriter.notifyMatchFailure(reshape_after,
                                       "dynamic shapes not supported");
  }

  const auto range = [](int64_t begin, int n) {
    SmallVector<int64_t> result;
    result.reserve(n);
    for (int i = 0; i < n; ++i) {
      result.push_back(i + begin);
    }
    return result;
  };

  // We only support the mhlo.dot style input layouts, i.e.:
  //   lhs: [batch, other dims, contract dims]
  //   rhs: [batch, contract dims, other dims]
  auto dim_nums = dot.getDotDimensionNumbers();
  int batch_dims_count = dim_nums.getLhsBatchingDimensions().size();
  int contracting_dims_count = dim_nums.getLhsContractingDimensions().size();
  if (dim_nums.getLhsBatchingDimensions() !=
          ArrayRef<int64_t>(range(0, batch_dims_count)) ||
      dim_nums.getRhsBatchingDimensions() !=
          ArrayRef<int64_t>(range(0, batch_dims_count)) ||
      dim_nums.getLhsContractingDimensions() !=
          ArrayRef<int64_t>(
              range(dot.getLhs().getType().getRank() - contracting_dims_count,
                    contracting_dims_count)) ||
      dim_nums.getRhsContractingDimensions() !=
          ArrayRef<int64_t>(range(batch_dims_count, contracting_dims_count))) {
    return rewriter.notifyMatchFailure(reshape_after,
                                       "unsupported dot_general layout");
  }

  // (B = batch dims, C = contracting dims, Y/Z = other dims)
  //
  // This pattern converts:
  //   %before = "mhlo.reshape"(%lhs : BxY1xC) : BxY2xC
  //   %dot = "mhlo.dot_general"(%before, %rhs : BxCxZ) : BxY2xZ
  //   %after = "mhlo.reshape"(%dot) : BxY1xZ
  // to:
  //   %dot : "mhlo.dot_general"(%lhs : BxY1xC, %rhs : BxCxZ) : BxY1xZ

  // Extract B, Y1, C from %lhs.
  ArrayRef<int64_t> shape_lhs =
      reshape_before.getOperand().getType().getShape();
  ArrayRef<int64_t> shape_b = shape_lhs.take_front(batch_dims_count);
  ArrayRef<int64_t> shape_c = shape_lhs.take_back(contracting_dims_count);
  ArrayRef<int64_t> shape_y1 =
      shape_lhs.drop_front(shape_b.size()).drop_back(shape_c.size());

  // Check %before shape, and extract Y2 from it.
  ArrayRef<int64_t> shape_before = reshape_before.getType().getShape();
  if (shape_before.take_front(shape_b.size()) != shape_b ||
      shape_before.take_back(shape_c.size()) != shape_c) {
    return failure();
  }
  ArrayRef<int64_t> shape_y2 =
      shape_before.drop_front(shape_b.size()).drop_back(shape_c.size());

  // No need to check %dot; dot_general verifier ensures correct shapes.
  // Extract Z from %dot.
  ArrayRef<int64_t> shape_z =
      dot.getType().getShape().drop_front(shape_b.size() + shape_y2.size());

  // Check %after shape.
  if (reshape_after.getType().getShape() !=
      ArrayRef<int64_t>(llvm::to_vector(
          llvm::concat<const int64_t>(shape_b, shape_y1, shape_z)))) {
    return failure();
  }

  rewriter.replaceOpWithNewOp<mhlo::DotGeneralOp>(
      reshape_after, reshape_after.getType(), reshape_before.getOperand(),
      dot.getRhs(),
      mhlo::DotDimensionNumbersAttr::get(
          reshape_after.getContext(),
          /*lhsBatchingDimensions=*/range(0, batch_dims_count),
          /*rhsBatchingDimensions=*/range(0, batch_dims_count),
          /*lhsContractingDimensions=*/
          range(batch_dims_count + shape_y1.size(), contracting_dims_count),
          /*rhsContractingDimensions=*/
          range(batch_dims_count, contracting_dims_count)),
      dot.getPrecisionConfigAttr());
  return success();
}

// Convert:
//   %y0 = dot_general(%x0, %w)
//   %y1 = dot_general(%x1, %w)
//   ...
//   concatenate(%y0, %y1, ...)
// To:
//   %x = concatenate(%x0, %x1, ...)
//   dot_general(%x, %w)
LogicalResult LiftDotConcatLHS(mhlo::ConcatenateOp concat,
                               PatternRewriter &rewriter) {
  if (concat.getVal().size() < 2)
    return rewriter.notifyMatchFailure(
        concat, "Concatenate op should have at least two operands");

  auto first_dot = concat.getVal()[0].getDefiningOp<mhlo::DotGeneralOp>();
  if (!first_dot)
    return rewriter.notifyMatchFailure(concat, "Operand is not dot_general");
  if (!first_dot.getLhs().getType().hasStaticShape())
    return rewriter.notifyMatchFailure(
        first_dot, "All dot_general LHS must be statically shaped");
  if (!first_dot->hasOneUse())
    return rewriter.notifyMatchFailure(first_dot, "Op has multiple uses");

  SmallVector<Value> all_dot_lhs;
  all_dot_lhs.reserve(concat.getVal().size());
  all_dot_lhs.push_back(first_dot.getLhs());

  const uint64_t batch_dims_count =
      first_dot.getDotDimensionNumbers().getLhsBatchingDimensions().size();
  const uint64_t contracting_dims_count =
      first_dot.getDotDimensionNumbers().getLhsContractingDimensions().size();
  const uint64_t lhs_other_dims_count = first_dot.getLhs().getType().getRank() -
                                        batch_dims_count -
                                        contracting_dims_count;

  // This pattern only supports concating on LHS other dims (neither batch nor
  // contracting).
  if (concat.getDimension() < batch_dims_count ||
      concat.getDimension() >= batch_dims_count + lhs_other_dims_count) {
    return rewriter.notifyMatchFailure(concat,
                                       "Not concating on LHS other dims");
  }

  for (auto value : concat.getVal().drop_front()) {
    auto dot = value.getDefiningOp<mhlo::DotGeneralOp>();
    if (!dot)
      return rewriter.notifyMatchFailure(concat, "Operand is not dot_general");

    if (dot.getRhs() != first_dot.getRhs())
      return rewriter.notifyMatchFailure(
          dot, "dot_general ops have different rhs parameters");
    if (dot.getDotDimensionNumbers() != first_dot.getDotDimensionNumbers())
      return rewriter.notifyMatchFailure(
          dot, "dot_general ops have different dimension numbers");
    if (dot.getPrecisionConfig() != first_dot.getPrecisionConfig())
      return rewriter.notifyMatchFailure(
          dot, "dot_general ops have different precision configs");
    if (!dot.getLhs().getType().hasStaticShape())
      return rewriter.notifyMatchFailure(
          dot, "all dot_general LHS must be statically shaped");
    if (dot.getLhs().getType().getElementType() !=
            first_dot.getLhs().getType().getElementType() ||
        dot.getType().getElementType() != first_dot.getType().getElementType())
      return rewriter.notifyMatchFailure(
          dot, "all dot_general ops must have the same element type");
    if (!dot->hasOneUse())
      return rewriter.notifyMatchFailure(dot, "Op has multiple uses");

    all_dot_lhs.push_back(dot.getLhs());
  }

  const auto is_lhs_batch_or_contracting_dim = [&](uint64_t dim) {
    auto dim_nums = first_dot.getDotDimensionNumbers();
    return llvm::is_contained(dim_nums.getLhsBatchingDimensions(), dim) ||
           llvm::is_contained(dim_nums.getLhsContractingDimensions(), dim);
  };

  // dot_general outputs are always in the
  //   [batch dims, LHS other dims, RHS other dims]
  // layout, so the new concat dim is where the n-th (base-0 counting) LHS other
  // dim appears in the original LHS layout, where:
  //   n = old_concat_dim - batch_dims_count
  uint64_t n = concat.getDimension() - batch_dims_count;

  // Now try to answer where the n-th LHS other dim was originally placed.
  // This is the dimension we should now concat on.
  int new_concat_dim = -1;
  for (int i = 0; i < first_dot.getLhs().getType().getRank(); ++i) {
    if (!is_lhs_batch_or_contracting_dim(i) && n-- == 0) {
      new_concat_dim = i;
      break;
    }
  }

  // Now get the output shape of the lifted concat op.
  SmallVector<int64_t> new_concat_shape(
      first_dot.getLhs().getType().getShape());
  new_concat_shape[new_concat_dim] = 0;
  for (auto v : all_dot_lhs) {
    new_concat_shape[new_concat_dim] +=
        v.getType().dyn_cast<ShapedType>().getShape()[new_concat_dim];
  }

  auto new_concat = rewriter.create<mhlo::ConcatenateOp>(
      concat->getLoc(), concat.getType().clone(new_concat_shape), all_dot_lhs,
      rewriter.getI64IntegerAttr(new_concat_dim));
  rewriter.replaceOpWithNewOp<mhlo::DotGeneralOp>(
      concat, concat.getType(), new_concat, first_dot.getRhs(),
      first_dot.getDotDimensionNumbers(), first_dot.getPrecisionConfigAttr());
  return success();
}

// Convert:
//   %y0 = dot_general(%x0, %w0)
//   %y1 = dot_general(%x1, %w1)
//   ...
//   concatenate(%y0, %y1, ...)
// To:
//   %x = concatenate(%x0, %x1, ...)
//   %w = concatenate(%w0, %w1, ...)
//   dot_general(%x, %w)
//
// To simplify the implementation, we only handle the case where the final
// concat is on the only batching dim.
LogicalResult LiftDotConcatLHSAndRHS(mhlo::ConcatenateOp concat,
                                     PatternRewriter &rewriter) {
  if (concat.getVal().size() < 2)
    return rewriter.notifyMatchFailure(
        concat, "Concatenate op should have at least two operands");

  auto first_dot = concat.getVal()[0].getDefiningOp<mhlo::DotGeneralOp>();
  if (!first_dot)
    return rewriter.notifyMatchFailure(concat, "Operand is not dot_general");
  if (!first_dot.getLhs().getType().hasStaticShape())
    return rewriter.notifyMatchFailure(
        first_dot, "All dot_general LHS must be statically shaped");
  if (!first_dot->hasOneUse())
    return rewriter.notifyMatchFailure(first_dot, "Op has multiple uses");

  SmallVector<Value> all_dot_lhs;
  all_dot_lhs.reserve(concat.getVal().size());
  all_dot_lhs.push_back(first_dot.getLhs());
  SmallVector<Value> all_dot_rhs;
  all_dot_rhs.reserve(concat.getVal().size());
  all_dot_rhs.push_back(first_dot.getRhs());

  if (first_dot.getDotDimensionNumbers().getLhsBatchingDimensions().size() != 1)
    return rewriter.notifyMatchFailure(first_dot, "One batching dim required");
  if (concat.getDimension() != 0)
    return rewriter.notifyMatchFailure(
        concat, "Not concating on the first batching dim");

  for (auto value : concat.getVal().drop_front()) {
    auto dot = value.getDefiningOp<mhlo::DotGeneralOp>();
    if (!dot)
      return rewriter.notifyMatchFailure(concat, "Operand is not dot_general");

    if (dot.getDotDimensionNumbers() != first_dot.getDotDimensionNumbers())
      return rewriter.notifyMatchFailure(
          dot, "dot_general ops have different dimension numbers");
    if (dot.getPrecisionConfig() != first_dot.getPrecisionConfig())
      return rewriter.notifyMatchFailure(
          dot, "dot_general ops have different precision configs");
    if (!dot.getLhs().getType().hasStaticShape() ||
        !dot.getRhs().getType().hasStaticShape())
      return rewriter.notifyMatchFailure(
          dot, "all dot_general operands must be statically shaped");
    if (dot.getLhs().getType().getElementType() !=
            first_dot.getLhs().getType().getElementType() ||
        dot.getRhs().getType().getElementType() !=
            first_dot.getRhs().getType().getElementType() ||
        dot.getType().getElementType() != first_dot.getType().getElementType())
      return rewriter.notifyMatchFailure(
          dot, "all dot_general ops must have the same element type");
    if (!dot->hasOneUse())
      return rewriter.notifyMatchFailure(dot, "Op has multiple uses");

    all_dot_lhs.push_back(dot.getLhs());
    all_dot_rhs.push_back(dot.getRhs());
  }

  // Now get the output shapes of the lifted concat ops.
  const int64_t lhs_batch_dim =
      first_dot.getDotDimensionNumbers().getLhsBatchingDimensions()[0];
  SmallVector<int64_t> lhs_new_concat_shape(
      first_dot.getLhs().getType().getShape());
  lhs_new_concat_shape[lhs_batch_dim] = 0;
  for (auto v : all_dot_lhs) {
    lhs_new_concat_shape[lhs_batch_dim] +=
        v.getType().dyn_cast<ShapedType>().getShape()[lhs_batch_dim];
  }
  const int64_t rhs_batch_dim =
      first_dot.getDotDimensionNumbers().getRhsBatchingDimensions()[0];
  SmallVector<int64_t> rhs_new_concat_shape(
      first_dot.getRhs().getType().getShape());
  rhs_new_concat_shape[rhs_batch_dim] = 0;
  for (auto v : all_dot_rhs) {
    rhs_new_concat_shape[rhs_batch_dim] +=
        v.getType().dyn_cast<ShapedType>().getShape()[rhs_batch_dim];
  }

  auto lhs_new_concat = rewriter.create<mhlo::ConcatenateOp>(
      concat->getLoc(), concat.getType().clone(lhs_new_concat_shape),
      all_dot_lhs, rewriter.getI64IntegerAttr(lhs_batch_dim));
  auto rhs_new_concat = rewriter.create<mhlo::ConcatenateOp>(
      concat->getLoc(), concat.getType().clone(rhs_new_concat_shape),
      all_dot_rhs, rewriter.getI64IntegerAttr(rhs_batch_dim));
  rewriter.replaceOpWithNewOp<mhlo::DotGeneralOp>(
      concat, concat.getType(), lhs_new_concat, rhs_new_concat,
      first_dot.getDotDimensionNumbers(), first_dot.getPrecisionConfigAttr());
  return success();
}

// Convert:
//   %y0 = slice(%x, start=0, limit=2)
//   %y1 = slice(%x, start=2, limit=3)
//   concat(%y0, %y1, ...)
// To:
//   %y = slice(%x, start=0, limit=3)
//   concat(%y, ...)
LogicalResult FuseSliceConcat(mhlo::ConcatenateOp concat,
                              PatternRewriter &rewriter) {
  if (concat.getVal().size() < 2)
    return rewriter.notifyMatchFailure(
        concat, "Concatenate op should have at least two operands");

  auto first = concat.getVal()[0].getDefiningOp<mhlo::SliceOp>();
  auto second = concat.getVal()[1].getDefiningOp<mhlo::SliceOp>();
  if (!first || !second)
    return rewriter.notifyMatchFailure(concat, "operands are not slice ops");
  if (first.getOperand() != second.getOperand())
    return rewriter.notifyMatchFailure(concat, "slice not on the same input");
  if (!first.getStrides().isSplat() ||
      first.getStrides().getSplatValue<IntegerAttr>().getInt() != 1 ||
      first.getStrides() != second.getStrides())
    return rewriter.notifyMatchFailure(concat, "slice ops must have stride=1");
  if (!first->hasOneUse() || !second->hasOneUse())
    return rewriter.notifyMatchFailure(concat, "slice ops are used elsewhere");

  SmallVector<int64_t> new_start;
  SmallVector<int64_t> new_limit;
  SmallVector<int64_t> new_slice_shape;
  new_start.reserve(first.getStrides().size());
  new_limit.reserve(first.getStrides().size());
  new_slice_shape.reserve(first.getStrides().size());

  for (int i = 0; i < first.getStrides().size(); ++i) {
    const int64_t first_start =
        first.getStartIndicesAttr().getValues<IntegerAttr>()[i].getInt();
    const int64_t first_limit =
        first.getLimitIndicesAttr().getValues<IntegerAttr>()[i].getInt();
    const int64_t second_start =
        second.getStartIndicesAttr().getValues<IntegerAttr>()[i].getInt();
    const int64_t second_limit =
        second.getLimitIndicesAttr().getValues<IntegerAttr>()[i].getInt();

    if (i == concat.getDimension()) {
      if (first_limit != second_start)
        return rewriter.notifyMatchFailure(
            second, "slice is not continuous with previous slice");
    } else {
      if (first_start != second_start || first_limit != second_limit)
        return rewriter.notifyMatchFailure(
            second, "non-concat dims have mismatching slice bounds");
    }

    new_start.push_back(first_start);
    new_limit.push_back(second_limit);
    new_slice_shape.push_back(second_limit - first_start);
  }

  auto new_slice = rewriter.create<mhlo::SliceOp>(
      FusedLoc::get(first->getContext(), {first.getLoc(), second.getLoc()}),
      first.getType().clone(new_slice_shape), first.getOperand(),
      /*start_indices=*/rewriter.getI64TensorAttr(new_start),
      /*limit_indices=*/rewriter.getI64TensorAttr(new_limit),
      /*strides=*/first.getStrides());

  SmallVector<Value> new_concat_values;
  new_concat_values.reserve(concat.getVal().size() - 1);
  new_concat_values.push_back(new_slice);
  llvm::append_range(new_concat_values, concat.getVal().drop_front(2));

  rewriter.replaceOpWithNewOp<mhlo::ConcatenateOp>(
      concat, concat.getType(), new_concat_values, concat.getDimension());
  return success();
}

// Convert:
//   %input : 1xYxC
//   %1 = mhlo.reshape %param : (1xCxZ) -> CxZ
//   mhlo.dot_general %input, %1 {batch_dims = []}
// To:
//   mhlo.dot_general %input, %param {batch_dims = [0]}
//
// This usage will mostly come from tf-unroll-batch-matmul, so it's fine to only
// handle the case where batching dim is the leftmost dim.
LogicalResult ConvertReshapeDotRhsToBatchedDot(mhlo::DotGeneralOp dot,
                                               PatternRewriter &rewriter) {
  mhlo::DotDimensionNumbersAttr dim_nums = dot.getDotDimensionNumbers();
  if (!dim_nums.getLhsBatchingDimensions().empty()) return failure();

  auto reshape = dot.getRhs().getDefiningOp<mhlo::ReshapeOp>();
  if (!reshape) return failure();
  if (!reshape->hasOneUse())
    return rewriter.notifyMatchFailure(reshape, "reshape has multiple usages");
  if (!reshape.getType().hasStaticShape() ||
      !reshape.getOperand().getType().hasStaticShape() ||
      !dot.getLhs().getType().hasStaticShape()) {
    return rewriter.notifyMatchFailure(dot, "dynamic shaping not supported");
  }

  ArrayRef<int64_t> orig_param_shape =
      reshape.getOperand().getType().getShape();
  ArrayRef<int64_t> dot_param_shape = reshape.getType().getShape();
  if (orig_param_shape.size() != dot_param_shape.size() + 1 ||
      orig_param_shape.front() != 1) {
    return rewriter.notifyMatchFailure(reshape, "unsupported reshape pattern");
  }

  int lhs_first_other_dim = -1;
  for (int i = 0; i < dot.getLhs().getType().getRank(); ++i) {
    if (!llvm::is_contained(dim_nums.getLhsContractingDimensions(), i)) {
      lhs_first_other_dim = i;
      break;
    }
  }
  if (lhs_first_other_dim == -1 ||
      dot.getLhs().getType().getShape()[lhs_first_other_dim] != 1) {
    return rewriter.notifyMatchFailure(dot, "unsupported LHS shape");
  }

  SmallVector<int64_t, 4> new_rhs_contracting_dims;
  new_rhs_contracting_dims.reserve(
      dim_nums.getRhsContractingDimensions().size());
  for (int64_t d : dim_nums.getRhsContractingDimensions()) {
    new_rhs_contracting_dims.push_back(d + 1);
  }

  rewriter.replaceOpWithNewOp<mhlo::DotGeneralOp>(
      dot, dot.getType(), dot.getLhs(), reshape.getOperand(),
      mhlo::DotDimensionNumbersAttr::get(
          dot.getContext(),
          /*lhsBatchingDimensions=*/{lhs_first_other_dim},
          /*rhsBatchingDimensions=*/{0},
          /*lhsContractingDimensions=*/dim_nums.getLhsContractingDimensions(),
          /*rhsContractingDimensions=*/new_rhs_contracting_dims),
      dot.getPrecisionConfigAttr());
  return success();
}

class OptimizePass
    : public PassWrapper<OptimizePass, OperationPass<func::FuncOp>> {
 public:
  StringRef getArgument() const final { return "mhlo-optimize"; }
  StringRef getDescription() const final {
    return "Applies various optimizations on MHLO IR";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add(ConvertDotToDotGeneral);
    patterns.add(RemoveReshapeAroundDotGeneral);
    patterns.add(LiftDotConcatLHS);
    patterns.add(LiftDotConcatLHSAndRHS);
    patterns.add(FuseSliceConcat);
    patterns.add(ConvertReshapeDotRhsToBatchedDot);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createOptimizePass() {
  return std::make_unique<OptimizePass>();
}

static PassRegistration<OptimizePass> pass;

}  // namespace odml
}  // namespace mlir

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

#include <algorithm>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {
namespace {

LogicalResult TryLowerToCollapseShape(
    ReduceOp op, RankedTensorType arg_ty, Value arg,
    SmallVector<int64_t>& ordered_reduction_dims, PatternRewriter& rewriter) {
  // This only works for trivial reductions where all declared reduction
  // dimensiosn are of extent 1.
  if (!llvm::all_of(ordered_reduction_dims,
                    [&](int64_t i) { return arg_ty.getDimSize(i) == 1; })) {
    return failure();
  }

  int64_t arg_rank = arg_ty.getRank();
  int64_t num_reduction_dims = ordered_reduction_dims.size();

  int64_t j = 0;
  auto is_declared_as_reduction_dim = [&](int64_t i) {
    if (j < num_reduction_dims && ordered_reduction_dims[j] == i) {
      j++;
      return true;
    }
    return false;
  };

  // Build reassociation indices.
  SmallVector<ReassociationIndices, 4> reassociation;
  int64_t i_begin = 0;
  int64_t i = 0;
  while (i < arg_rank && is_declared_as_reduction_dim(i)) i++;
  while (i < arg_rank) {
    i++;
    while (i < arg_rank && is_declared_as_reduction_dim(i)) i++;
    reassociation.push_back(llvm::to_vector(llvm::seq(i_begin, i)));
    i_begin = i;
  }

  // Lower reduction op to collapse shape op.
  rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(op, arg, reassociation);
  return success();
}

enum class DimensionKind {
  kParallel,
  kReduction,
  kDegenerate,
};

struct DimensionGroup {
  DimensionKind kind;
  int64_t begin;
  int64_t end;
  int64_t size() { return end - begin; }
};

// Groups consecutive dimensions of a reduction argument by their kind, i.e. if
// they are reduction or parallel dimensions. Dimensions of size 1 can be
// considered as any kind.
void GroupDimensions(RankedTensorType arg_ty,
                     SmallVector<int64_t> ordered_reduction_dims,
                     SmallVector<DimensionGroup>& groups) {
  int64_t arg_rank = arg_ty.getRank();
  int64_t num_reduction_dims = ordered_reduction_dims.size();
  int64_t j = 0;
  for (int64_t i = 0; i < arg_rank; ++i) {
    // Check if the i-th dimension is one of the declared reduction dimensions.
    bool is_declared_as_reduction_dim = false;
    if (j < num_reduction_dims && i == ordered_reduction_dims[j]) {
      is_declared_as_reduction_dim = true;
      j++;
    }

    // Use the declared dimension kind unless the dimension is of extent 1, in
    // which case we can consider it either kind. We exploit this to form
    // maximal dimension groups.
    DimensionKind kind = is_declared_as_reduction_dim
                             ? DimensionKind::kReduction
                             : DimensionKind::kParallel;
    if (arg_ty.getDimSize(i) == 1) kind = DimensionKind::kDegenerate;

    // Start a new dimension group if the dimenion kind conflicts with the
    // trailing kind.
    if (groups.empty() || (groups.back().kind != kind &&
                           groups.back().kind != DimensionKind::kDegenerate &&
                           kind != DimensionKind::kDegenerate)) {
      groups.push_back({kind, i, i});
    }

    // Include dimension in trailing group and concretize dimension kind if
    // necessary.
    if (groups.back().kind == DimensionKind::kDegenerate)
      groups.back().kind = kind;
    groups.back().end++;
  }
}

LogicalResult TryLowerTo1DOr2DReduction(
    ReduceOp op, RankedTensorType arg_ty, Value arg,
    SmallVector<int64_t>& ordered_reduction_dims,
    bool prefer_columns_reductions, PatternRewriter& rewriter) {
  // Group the argument dimensions by their kind.
  SmallVector<DimensionGroup> dim_groups;
  GroupDimensions(arg_ty, ordered_reduction_dims, dim_groups);

  // Do not (re-)apply if the dimensions are already fully collapsed.
  if (dim_groups.size() <= 2 &&
      llvm::all_of(dim_groups, [](auto g) { return g.size() == 1; })) {
    return failure();
  }

  // Determine whether or not a dynamic reshape is needed for the final result.
  int64_t num_dyn_parallel_dims = 0;
  for (auto group : dim_groups) {
    if (group.kind != DimensionKind::kParallel) continue;
    for (int64_t i = group.begin; i < group.end; i++) {
      if (arg_ty.isDynamicDim(i)) num_dyn_parallel_dims++;
    }
  }
  bool requires_dynamic_reshape = num_dyn_parallel_dims > 1;

  // Reify the result shape early so that the pattern can fail without altering
  // the IR.
  Optional<Value> result_shape;
  if (requires_dynamic_reshape) {
    llvm::SmallVector<Value, 1> reified_shapes;
    if (failed(llvm::cast<InferShapedTypeOpInterface>(op.getOperation())
                   .reifyReturnTypeShapes(rewriter, op->getOperands(),
                                          reified_shapes))) {
      return failure();
    }
    assert(reified_shapes.size() == 1 && "expect exactly one shape");
    result_shape = reified_shapes.front();
  }

  // Collapse dimension groups so that all adjacent dimensions of the
  // intermediate result are of a different kind.
  Value interm_result = arg;
  auto loc = op.getLoc();
  bool requires_collapse =
      llvm::any_of(dim_groups, [&](auto g) { return g.size() > 1; });
  if (requires_collapse) {
    auto reassociation =
        llvm::to_vector(llvm::map_range(dim_groups, [&](auto g) {
          return llvm::to_vector<2>(llvm::seq<int64_t>(g.begin, g.end));
        }));
    interm_result = rewriter.create<tensor::CollapseShapeOp>(loc, interm_result,
                                                             reassociation);
  }

  // If required, transpose the intermediate result so that dimensions kinds
  // form two partitions, which can be collapsed to a 2D intermediate result.
  bool requires_transpose = dim_groups.size() > 2;
  if (requires_transpose) {
    // Materialize transpose.
    DimensionKind leading_dim_kind = prefer_columns_reductions
                                         ? DimensionKind::kReduction
                                         : DimensionKind::kParallel;
    DimensionKind trailing_dim_kind = prefer_columns_reductions
                                          ? DimensionKind::kParallel
                                          : DimensionKind::kReduction;
    SmallVector<int64_t> perm;
    for (int i = 0; i < dim_groups.size(); i++) {
      if (dim_groups[i].kind == leading_dim_kind) perm.push_back(i);
    }
    int64_t num_leading_dims = perm.size();
    for (int i = 0; i < dim_groups.size(); i++) {
      if (dim_groups[i].kind == trailing_dim_kind) perm.push_back(i);
    }
    auto perm_attr = rewriter.getI64TensorAttr(perm);
    interm_result = rewriter.create<TransposeOp>(loc, interm_result, perm_attr)
                        ->getResults()
                        .front();

    // Collapse intermediate result rank 2.
    SmallVector<ReassociationIndices, 2> reassociation = {
        llvm::to_vector<2>(llvm::seq<int64_t>(0, num_leading_dims)),
        llvm::to_vector<2>(llvm::seq<int64_t>(num_leading_dims, perm.size()))};
    interm_result = rewriter.create<tensor::CollapseShapeOp>(loc, interm_result,
                                                             reassociation);
  }

  // Materialize inner 1D or 2D reduction.
  bool leading_reduction =
      requires_transpose ? prefer_columns_reductions
                         : dim_groups.front().kind == DimensionKind::kReduction;
  int64_t reduction_dim = leading_reduction ? 0 : 1;
  auto reduction_dim_attr = rewriter.getI64VectorAttr({reduction_dim});
  Value init_val = op.init_values().front();
  auto reduction_op = rewriter.create<ReduceOp>(loc, interm_result, init_val,
                                                reduction_dim_attr);
  rewriter.inlineRegionBefore(op.body(), reduction_op.body(),
                              reduction_op.body().begin());
  interm_result = reduction_op->getResults().front();

  // Restore the expected shape by dynamic reshape, if required.
  auto result_ty = op->getResultTypes().front().cast<RankedTensorType>();
  if (requires_dynamic_reshape) {
    assert(result_shape && "expect to have reified the result shape");
    interm_result = rewriter.create<DynamicReshapeOp>(
        loc, result_ty, interm_result, *result_shape);
  }

  // Othwerise, restore the expected shape by shape expansion, if required.
  int64_t result_rank = result_ty.getRank();
  int64_t interm_result_rank =
      interm_result.getType().cast<RankedTensorType>().getRank();
  bool requires_expand =
      !requires_dynamic_reshape && result_rank != interm_result_rank;
  if (requires_expand) {
    assert(interm_result_rank <= 1 &&
           "expect intermediate result to be of rank 0 or 1 before expansion");
    SmallVector<ReassociationIndices, 1> reassociation;
    bool is_scalar_expansion = interm_result_rank == 0;
    if (!is_scalar_expansion)
      reassociation = {llvm::to_vector(llvm::seq<int64_t>(0, result_rank))};
    interm_result = rewriter.create<tensor::ExpandShapeOp>(
        loc, result_ty, interm_result, reassociation);
  }

  rewriter.replaceOp(op, interm_result);
  return success();
}

struct GroupReductionDimensionsPattern : public OpRewritePattern<ReduceOp> {
  GroupReductionDimensionsPattern(MLIRContext* ctx,
                                  bool prefer_columns_reductions)
      : OpRewritePattern<ReduceOp>(ctx, /*benefit=*/1),
        prefer_columns_reductions(prefer_columns_reductions) {}

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter& rewriter) const override {
    // Only apply to reduction of a unique argument.
    if (op.operands().size() != 1 || op.init_values().size() != 1)
      return failure();
    Value arg = op.operands().front();
    auto arg_ty = arg.getType().cast<RankedTensorType>();

    // Sort reduction dimensions, which is not an invariant of the op.
    SmallVector<int64_t> ordered_reduction_dims =
        llvm::to_vector<4>(llvm::map_range(op.dimensions(), [](auto d) {
          return static_cast<int64_t>(d.getLimitedValue());
        }));
    std::sort(ordered_reduction_dims.begin(), ordered_reduction_dims.end());

    // If all reduction dimensions are known to be of extent 1 then we can
    // express the reduction through an equivalent collapsing op.
    if (succeeded(TryLowerToCollapseShape(op, arg_ty, arg,
                                          ordered_reduction_dims, rewriter))) {
      return success();
    }

    // Otherwise, try lowering the reduction to an equivalent 1D or 2D
    // reduction, and insert transposes if needed.
    if (succeeded(
            TryLowerTo1DOr2DReduction(op, arg_ty, arg, ordered_reduction_dims,
                                      prefer_columns_reductions, rewriter))) {
      return success();
    }

    return failure();
  }

  bool prefer_columns_reductions;
};

struct GroupReductionDimensionsPass
    : public GroupReductionDimensionsPassBase<GroupReductionDimensionsPass> {
  explicit GroupReductionDimensionsPass(bool prefer_columns_reductions)
      : GroupReductionDimensionsPassBase<
            GroupReductionDimensionsPass>::GroupReductionDimensionsPassBase() {
    prefer_columns_reductions_ = prefer_columns_reductions;
  }

  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);
    populateGroupReductionDimensionsPatterns(ctx, &patterns,
                                             prefer_columns_reductions_);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

void populateGroupReductionDimensionsPatterns(MLIRContext* context,
                                              RewritePatternSet* patterns,
                                              bool prefer_columns_reductions) {
  patterns->add<GroupReductionDimensionsPattern>(context,
                                                 prefer_columns_reductions);
}

std::unique_ptr<OperationPass<func::FuncOp>> createGroupReductionDimensionsPass(
    bool prefer_columns_reductions) {
  return std::make_unique<GroupReductionDimensionsPass>(
      prefer_columns_reductions);
}

}  // namespace mhlo
}  // namespace mlir

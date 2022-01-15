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
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {
namespace {

enum class DimensionKind {
  kParallel,
  kReduction,
};

struct DimensionGroup {
  DimensionKind kind;
  int64_t begin;
  int64_t end;
};

// Groups consecutive dimensions of a reduction argument by their kind, i.e. if
// they are reduction or parallel dimensions.
void GroupDimensions(int64_t arg_rank, DenseIntElementsAttr reduction_dims,
                     SmallVector<DimensionGroup>& groups) {
  auto ordered_reduction_dims = llvm::to_vector<4>(llvm::map_range(
      reduction_dims, [](auto d) { return d.getLimitedValue(); }));
  std::sort(ordered_reduction_dims.begin(), ordered_reduction_dims.end());
  int j = 0;
  for (int64_t i = 0; i < arg_rank; ++i) {
    DimensionKind kind;
    if (j < ordered_reduction_dims.size() && i == ordered_reduction_dims[j]) {
      kind = DimensionKind::kReduction;
      j++;
    } else {
      kind = DimensionKind::kParallel;
    }
    if (groups.empty() || groups.back().kind != kind)
      groups.push_back({kind, i, i});
    groups.back().end++;
  }
}

struct GroupReductionDimensionsPattern : public OpRewritePattern<ReduceOp> {
  using OpRewritePattern<ReduceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter& rewriter) const override {
    // Only apply to reduction of a unique argument.
    if (op.inputs().size() != 1 || op.init_values().size() != 1)
      return failure();
    Value arg = op.inputs().front();
    auto arg_ty = arg.getType().cast<RankedTensorType>();
    int64_t arg_rank = arg_ty.getRank();

    // Group the argument dimensions by their kind.
    SmallVector<DimensionGroup> dim_groups;
    GroupDimensions(arg_rank, op.dimensions(), dim_groups);

    // Only apply this optimization if we can simplify the reduction operation
    // to a 1D or 2D reduction.
    if (dim_groups.size() > 2) return failure();

    // Do not (re-)apply if the dimensions are already fully collapsed.
    if (llvm::all_of(dim_groups, [](auto g) { return g.begin + 1 == g.end; })) {
      return failure();
    }

    // Accept at most one dynamic parallel dimension. For more dynamic
    // dimensions, we need a dynamic version of the `expand_shape` op (or use
    // `dynamic_reshape`) to restore the desired output shape.
    // TODO(frgossen): Implement result expansion for cases with more than one
    // dynamic dimension.
    int64_t num_dyn_parallel_dims = 0;
    for (auto group : dim_groups) {
      if (group.kind == DimensionKind::kReduction) continue;
      for (int64_t i = group.begin; i < group.end; i++) {
        if (arg_ty.isDynamicDim(i)) num_dyn_parallel_dims++;
      }
    }
    if (num_dyn_parallel_dims > 1) return failure();

    // Collapse argument to a 1D tensor for full a reduction and to a 2D tensor
    // for a partial reduction.
    SmallVector<ReassociationIndices, 2> collapsing_reassociation =
        llvm::to_vector<2>(llvm::map_range(dim_groups, [](auto g) {
          return llvm::to_vector<2>(llvm::seq<int64_t>(g.begin, g.end));
        }));
    auto loc = op.getLoc();
    auto collapsed_arg = rewriter.create<tensor::CollapseShapeOp>(
        loc, arg, collapsing_reassociation);

    // Materialize collapsed reduction op.
    int64_t collapsed_reduction_dim =
        dim_groups.front().kind == DimensionKind::kReduction ? 0 : 1;
    auto collapsed_reduction_dim_attr = DenseIntElementsAttr::get(
        RankedTensorType::get({1}, rewriter.getI64Type()),
        {collapsed_reduction_dim});
    auto collapsed_result = rewriter.create<ReduceOp>(
        loc, collapsed_arg->getResults().front(), op.init_values().front(),
        collapsed_reduction_dim_attr);
    rewriter.inlineRegionBefore(op.body(), collapsed_result.body(),
                                collapsed_result.body().begin());

    // For reductions to ranks <= 1, the collapsed reduction yields the exact
    // same result as the original one.
    auto result_ty = op->getResultTypes().front().cast<RankedTensorType>();
    int64_t result_rank = result_ty.getRank();
    if (result_rank <= 1) {
      rewriter.replaceOp(op, collapsed_result.getResults());
      return success();
    }

    // Othwerwise, we have to restore the desired shape by shape expansion.
    SmallVector<ReassociationIndices, 1> expanding_reassociation = {
        llvm::to_vector(llvm::seq<int64_t>(0, result_rank))};
    auto expanded_result = rewriter.create<tensor::ExpandShapeOp>(
        loc, result_ty, collapsed_result.getResults().front(),
        expanding_reassociation);
    rewriter.replaceOp(op, expanded_result->getResults());
    return success();
  }
};

struct GroupReductionDimensionsPass
    : public GroupReductionDimensionsPassBase<GroupReductionDimensionsPass> {
  void runOnFunction() override {
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);
    mhlo::populateGroupReductionDimensionsPatterns(ctx, &patterns);
    if (failed(
            applyPatternsAndFoldGreedily(getFunction(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

void populateGroupReductionDimensionsPatterns(
    MLIRContext* context, OwningRewritePatternList* patterns) {
  patterns->insert<GroupReductionDimensionsPattern>(context);
}

std::unique_ptr<FunctionPass> createGroupReductionDimensionsPass() {
  return std::make_unique<GroupReductionDimensionsPass>();
}

}  // namespace mhlo
}  // namespace mlir

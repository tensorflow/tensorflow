/* Copyright 2022 The OpenXLA Authors.

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
#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_GROUPREDUCTIONDIMENSIONSPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

namespace {

LogicalResult tryLowerToCollapseShape(
    ReduceOp op, RankedTensorType argTy, Value arg,
    SmallVector<int64_t>& orderedReductionDims, PatternRewriter& rewriter) {
  // This only works for trivial reductions where all declared reduction
  // dimensiosn are of extent 1.
  if (!llvm::all_of(orderedReductionDims,
                    [argTy](int64_t i) { return argTy.getDimSize(i) == 1; })) {
    return failure();
  }

  int64_t argRank = argTy.getRank();
  int64_t numReductionDims = orderedReductionDims.size();

  int64_t j = 0;
  auto isDeclaredAsReductionDim = [&](int64_t i) {
    if (j < numReductionDims && orderedReductionDims[j] == i) {
      j++;
      return true;
    }
    return false;
  };

  // Build reassociation indices.
  SmallVector<ReassociationIndices, 4> reassociation;
  int64_t iBegin = 0;
  int64_t i = 0;
  while (i < argRank && isDeclaredAsReductionDim(i)) i++;
  while (i < argRank) {
    i++;
    while (i < argRank && isDeclaredAsReductionDim(i)) i++;
    reassociation.push_back(llvm::to_vector(llvm::seq(iBegin, i)));
    iBegin = i;
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
void groupDimensions(RankedTensorType argTy,
                     SmallVector<int64_t> orderedReductionDims,
                     SmallVector<DimensionGroup>& groups) {
  int64_t argRank = argTy.getRank();
  int64_t numReductionDims = orderedReductionDims.size();
  int64_t j = 0;
  for (int64_t i = 0; i < argRank; ++i) {
    // Check if the i-th dimension is one of the declared reduction dimensions.
    bool isDeclaredAsReductionDim = false;
    if (j < numReductionDims && i == orderedReductionDims[j]) {
      isDeclaredAsReductionDim = true;
      j++;
    }

    // Use the declared dimension kind unless the dimension is of extent 1, in
    // which case we can consider it either kind. We exploit this to form
    // maximal dimension groups.
    DimensionKind kind = isDeclaredAsReductionDim ? DimensionKind::kReduction
                                                  : DimensionKind::kParallel;
    if (argTy.getDimSize(i) == 1) kind = DimensionKind::kDegenerate;

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

LogicalResult tryLowerTo1DOr2DReduction(
    ReduceOp op, RankedTensorType argTy, Value arg,
    SmallVector<int64_t>& orderedReductionDims, bool preferColumnsReductions,
    PatternRewriter& rewriter) {
  // Group the argument dimensions by their kind.
  SmallVector<DimensionGroup> dimGroups;
  groupDimensions(argTy, orderedReductionDims, dimGroups);

  // Do not (re-)apply if the dimensions are already fully collapsed.
  if (dimGroups.size() <= 2 &&
      llvm::all_of(dimGroups, [](auto g) { return g.size() == 1; })) {
    return failure();
  }

  // Determine whether or not a dynamic reshape is needed for the final result.
  int64_t numDynParallelDims = 0;
  for (auto group : dimGroups) {
    if (group.kind != DimensionKind::kParallel) continue;
    for (int64_t i = group.begin; i < group.end; i++) {
      if (argTy.isDynamicDim(i)) numDynParallelDims++;
    }
  }
  bool requiresDynamicReshape = numDynParallelDims > 1;

  // Reify the result shape early so that the pattern can fail without altering
  // the IR.
  std::optional<Value> resultShape;
  if (requiresDynamicReshape) {
    llvm::SmallVector<Value, 1> reifiedShapes;
    if (failed(llvm::cast<InferShapedTypeOpInterface>(op.getOperation())
                   .reifyReturnTypeShapes(rewriter, op->getOperands(),
                                          reifiedShapes))) {
      return failure();
    }
    assert(reifiedShapes.size() == 1 && "expect exactly one shape");
    resultShape = reifiedShapes.front();
  }

  // Collapse dimension groups so that all adjacent dimensions of the
  // intermediate result are of a different kind.
  Value intermResult = arg;
  auto loc = op.getLoc();
  bool requiresCollapse =
      llvm::any_of(dimGroups, [&](auto g) { return g.size() > 1; });
  if (requiresCollapse) {
    auto reassociation =
        llvm::to_vector(llvm::map_range(dimGroups, [&](auto g) {
          return llvm::to_vector<2>(llvm::seq<int64_t>(g.begin, g.end));
        }));
    intermResult = rewriter.create<tensor::CollapseShapeOp>(loc, intermResult,
                                                            reassociation);
  }

  // If required, transpose the intermediate result so that dimensions kinds
  // form two partitions, which can be collapsed to a 2D intermediate result.
  bool requiresTranspose = dimGroups.size() > 2;
  if (requiresTranspose) {
    // Materialize transpose.
    DimensionKind leadingDimKind = preferColumnsReductions
                                       ? DimensionKind::kReduction
                                       : DimensionKind::kParallel;
    DimensionKind trailingDimKind = preferColumnsReductions
                                        ? DimensionKind::kParallel
                                        : DimensionKind::kReduction;
    SmallVector<int64_t> perm;
    for (int64_t i = 0; i < static_cast<int64_t>(dimGroups.size()); i++) {
      if (dimGroups[i].kind == leadingDimKind) perm.push_back(i);
    }
    int64_t numLeadingDims = perm.size();
    for (int64_t i = 0; i < static_cast<int64_t>(dimGroups.size()); i++) {
      if (dimGroups[i].kind == trailingDimKind) perm.push_back(i);
    }
    auto permAttr = rewriter.getI64TensorAttr(perm);
    intermResult = rewriter.create<TransposeOp>(loc, intermResult, permAttr)
                       ->getResults()
                       .front();

    // Collapse intermediate result rank 2.
    SmallVector<ReassociationIndices, 2> reassociation = {
        llvm::to_vector<2>(llvm::seq<int64_t>(0, numLeadingDims)),
        llvm::to_vector<2>(llvm::seq<int64_t>(numLeadingDims, perm.size()))};
    intermResult = rewriter.create<tensor::CollapseShapeOp>(loc, intermResult,
                                                            reassociation);
  }

  // Materialize inner 1D or 2D reduction.
  bool leadingReduction =
      requiresTranspose ? preferColumnsReductions
                        : dimGroups.front().kind == DimensionKind::kReduction;
  int64_t reductionDim = leadingReduction ? 0 : 1;
  auto reductionDimAttr = rewriter.getI64VectorAttr({reductionDim});
  Value initVal = op.getInitValues().front();
  SmallVector<Type> elementTypes{llvm::map_range(
      op.getBody().front().getTerminator()->getOperands(),
      [](Value v) { return v.getType().cast<ShapedType>().getElementType(); })};
  auto reductionOp = rewriter.create<ReduceOp>(loc, intermResult, initVal,
                                               reductionDimAttr, elementTypes);
  rewriter.inlineRegionBefore(op.getBody(), reductionOp.getBody(),
                              reductionOp.getBody().begin());
  intermResult = reductionOp->getResults().front();

  // Restore the expected shape by dynamic reshape, if required.
  auto resultTy = op->getResultTypes().front().cast<RankedTensorType>();
  if (requiresDynamicReshape) {
    assert(resultShape && "expect to have reified the result shape");
    intermResult = rewriter.create<DynamicReshapeOp>(
        loc, resultTy, intermResult, *resultShape);
  }

  // Othwerise, restore the expected shape by shape expansion, if required.
  int64_t resultRank = resultTy.getRank();
  int64_t intermResultRank =
      intermResult.getType().cast<RankedTensorType>().getRank();
  bool requiresExpand =
      !requiresDynamicReshape && resultRank != intermResultRank;
  if (requiresExpand) {
    assert(intermResultRank <= 1 &&
           "expect intermediate result to be of rank 0 or 1 before expansion");
    SmallVector<ReassociationIndices, 1> reassociation;
    bool isScalarExpansion = intermResultRank == 0;
    if (!isScalarExpansion)
      reassociation = {llvm::to_vector(llvm::seq<int64_t>(0, resultRank))};
    intermResult = rewriter.create<tensor::ExpandShapeOp>(
        loc, resultTy, intermResult, reassociation);
  }

  rewriter.replaceOp(op, intermResult);
  return success();
}

struct GroupReductionDimensionsPattern : public OpRewritePattern<ReduceOp> {
  GroupReductionDimensionsPattern(MLIRContext* ctx,
                                  bool preferColumnsReductions)
      : OpRewritePattern<ReduceOp>(ctx, /*benefit=*/1),
        preferColumnsReductions(preferColumnsReductions) {}

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter& rewriter) const override {
    // Only apply to reduction of a unique argument.
    if (op.getInputs().size() != 1 || op.getInitValues().size() != 1)
      return failure();
    Value arg = op.getInputs().front();
    // Only apply to non-sparse tensors.
    if (auto rtp = arg.getType().cast<RankedTensorType>();
        rtp.getEncoding() != nullptr)
      return failure();

    auto argTy = arg.getType().cast<RankedTensorType>();

    // Sort reduction dimensions, which is not an invariant of the op.
    SmallVector<int64_t> orderedReductionDims =
        llvm::to_vector<4>(llvm::map_range(op.getDimensions(), [](auto d) {
          return static_cast<int64_t>(d.getLimitedValue());
        }));
    std::sort(orderedReductionDims.begin(), orderedReductionDims.end());

    // If all reduction dimensions are known to be of extent 1 then we can
    // express the reduction through an equivalent collapsing op.
    if (succeeded(tryLowerToCollapseShape(op, argTy, arg, orderedReductionDims,
                                          rewriter))) {
      return success();
    }

    // Otherwise, try lowering the reduction to an equivalent 1D or 2D
    // reduction, and insert transposes if needed.
    if (succeeded(
            tryLowerTo1DOr2DReduction(op, argTy, arg, orderedReductionDims,
                                      preferColumnsReductions, rewriter))) {
      return success();
    }

    return failure();
  }

  bool preferColumnsReductions;
};

struct GroupReductionDimensionsPass
    : public impl::GroupReductionDimensionsPassBase<
          GroupReductionDimensionsPass> {
  explicit GroupReductionDimensionsPass(bool preferColumnsReductions)
      : GroupReductionDimensionsPassBase<
            GroupReductionDimensionsPass>::GroupReductionDimensionsPassBase() {
    prefer_columns_reductions_ = preferColumnsReductions;
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
                                              bool preferColumnsReductions) {
  patterns->add<GroupReductionDimensionsPattern>(context,
                                                 preferColumnsReductions);
}

std::unique_ptr<OperationPass<func::FuncOp>> createGroupReductionDimensionsPass(
    bool preferColumnsReductions) {
  return std::make_unique<GroupReductionDimensionsPass>(
      preferColumnsReductions);
}

}  // namespace mhlo
}  // namespace mlir

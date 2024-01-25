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

// This file implements logic for simplifying HLO scatter.

#include <memory>
#include <numeric>
#include <optional>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "mhlo/utils/mhlo_scatter_gather_utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {
namespace {

#define GEN_PASS_DEF_HLOCANONICALIZESCATTERPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

DenseIntElementsAttr getI64ElementsAttr(ArrayRef<int64_t> values,
                                        Builder* builder) {
  auto ty = RankedTensorType::get({static_cast<int64_t>(values.size())},
                                  builder->getIntegerType(64));
  return DenseIntElementsAttr::get(ty, values);
}

bool isIdentityPermutation(ArrayRef<int64_t> permutation) {
  for (int64_t i = 0, e = permutation.size(); i < e; ++i)
    if (permutation[i] != i) return false;
  return true;
}

SmallVector<Value> transposeTensors(OpBuilder& b, Location loc,
                                    ValueRange tensors,
                                    ArrayRef<int64_t> permutation) {
  if (isIdentityPermutation(permutation)) return tensors;

  auto permutationAttr = getI64ElementsAttr(permutation, &b);
  SmallVector<Value> transposedTensors;
  for (Value tensor : tensors) {
    transposedTensors.push_back(
        b.create<TransposeOp>(loc, tensor, permutationAttr));
  }
  return transposedTensors;
}

// Transposes updates to align with the dims of operands.
SmallVector<Value> transposeUpdatesAccordingToScatterDimsMap(
    OpBuilder& b, Location loc, SmallVector<Value> updates,
    ArrayRef<int64_t> scatterDimsToOperandDims) {
  auto updatesType = updates.front().getType().cast<RankedTensorType>();
  int64_t updatesRank = updatesType.getRank();
  int64_t operandRank = updatesRank - 1;

  // For the updates, we need to add the scatter dimension to the permutation.
  SmallVector<int64_t> permutation{0};
  for (int64_t i : scatterDimsToOperandDims) {
    permutation.push_back(i + 1);
  }
  for (int64_t i = 0; i < operandRank; ++i) {
    if (!llvm::is_contained(scatterDimsToOperandDims, i))
      permutation.push_back(i + 1);
  }
  return transposeTensors(b, loc, updates, permutation);
}

// Makes window dimensions of `updates` the innermost ones.
SmallVector<Value> transposeUpdatesToMoveWindowDimensionsInside(
    OpBuilder& b, Location loc, SmallVector<Value> updates,
    ArrayRef<int64_t> updateWindowDims) {
  auto updatesType = updates.front().getType().cast<RankedTensorType>();
  int64_t updatesRank = updatesType.getRank();

  // Move update dimensions to the back
  SmallVector<int64_t> permutation;
  for (int i = 0; i < updatesRank; ++i) {
    if (!llvm::is_contained(updateWindowDims, i)) permutation.push_back(i);
  }
  permutation.append(updateWindowDims.begin(), updateWindowDims.end());
  return transposeTensors(b, loc, updates, permutation);
}

SmallVector<Value> reshapeUpdatesToEnsureSingleScatterDimension(
    OpBuilder& b, Location loc, ValueRange updates,
    ArrayRef<int64_t> updateWindowDims) {
  auto updatesType = updates.front().getType().cast<RankedTensorType>();
  int64_t updatesRank = updatesType.getRank();

  // Collapse scatter dimensions to 1D if there are more than 1 or prepend a
  // size-1 dimension if there are no explicit scatter dims.
  size_t numScatterDims = updatesRank - updateWindowDims.size();
  if (numScatterDims > 1) {
    SmallVector<ReassociationIndices> reassociation{
        llvm::to_vector<2>(llvm::seq<int64_t>(0, numScatterDims))};
    for (int i = numScatterDims, e = updatesRank; i < e; ++i)
      reassociation.push_back({i});

    return to_vector(llvm::map_range(updates, [&](Value update) -> Value {
      return b.create<tensor::CollapseShapeOp>(loc, update, reassociation);
    }));
  }
  if (numScatterDims == 0) {
    return to_vector(llvm::map_range(updates, [&](Value update) -> Value {
      return insertDegenerateDimensions(
          b, loc, cast<TypedValue<TensorType>>(update),
          {0});
    }));
  }
  return updates;
}

// Inserts size-1 dimensions to get rid of `insertedWindowDims` attribute.
SmallVector<Value> reshapeUpdatesToMatchOperandShape(
    OpBuilder& b, Location loc, SmallVector<Value> updates,
    ArrayRef<int64_t> insertedWindowDims) {
  size_t numScatterDims = insertedWindowDims.size();
  if (numScatterDims == 0) return updates;

  SmallVector<int64_t> shiftedScatterDimsToOperandDims;
  for (int64_t i : insertedWindowDims)
    shiftedScatterDimsToOperandDims.push_back(i + 1);

  return to_vector(map_range(updates, [&](Value update) -> Value {
    return insertDegenerateDimensions(
        b, loc, cast<TypedValue<TensorType>>(update),
        shiftedScatterDimsToOperandDims);
  }));
}

// Inserts transposes and reshapes to make window/slice dimensions become the
// innermost dimensions of updates. Also insert degenerate size-1 dimensions to
// match the shape of the slice and the shape of the operand.
SmallVector<Value> canonicalizeUpdates(
    OpBuilder& b, Location loc, SmallVector<Value> updates,
    ArrayRef<int64_t> scatterDimsToOperandDims,
    ArrayRef<int64_t> updateWindowDims, ArrayRef<int64_t> insertedWindowDims) {
  updates = transposeUpdatesToMoveWindowDimensionsInside(b, loc, updates,
                                                         updateWindowDims);
  updates = reshapeUpdatesToEnsureSingleScatterDimension(b, loc, updates,
                                                         updateWindowDims);
  updates =
      reshapeUpdatesToMatchOperandShape(b, loc, updates, insertedWindowDims);
  return transposeUpdatesAccordingToScatterDimsMap(b, loc, updates,
                                                   scatterDimsToOperandDims);
}

// This pattern rewrites scatter into a transposes, reshapes and a simpler
// scatter.
//
// It transposes and reshapes updates, scatterIndices and operands to get to
// the following characteristics:
//
// - scatter_indices is a two-dimensional tensor
// - index_vector_dim is 1
// - inserted_window_dims is []
// - update_window_dims is [1, 2, ...]
// - scatter_dims_to_operand_dims is [0, 1, ...]
struct CanonicalizeScatterPattern : public OpRewritePattern<ScatterOp> {
  using OpRewritePattern<ScatterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ScatterOp scatterOp,
                                PatternRewriter& rewriter) const override {
    if (isCanonicalScatter(scatterOp)) return failure();

    Location loc = scatterOp.getLoc();
    ScatterDimensionNumbersAttr dimsAttrs =
        scatterOp.getScatterDimensionNumbers();

    auto operandType =
        scatterOp.getInputs().front().getType().cast<RankedTensorType>();
    int64_t operandRank = operandType.getRank();
    auto [operandPermutation, operandPermutationInverse] =
        makeOperandStartIndexPermutations(
            dimsAttrs.getScatterDimsToOperandDims(), operandRank);

    Value canonicalIndices =
        canonicalizeStartIndices(rewriter, loc, scatterOp.getScatterIndices(),
                                 dimsAttrs.getIndexVectorDim());

    SmallVector<Value> canonicalOperands = transposeTensors(
        rewriter, loc, scatterOp.getInputs(), operandPermutation);

    SmallVector<Value> canonicalUpdates = canonicalizeUpdates(
        rewriter, loc, scatterOp.getUpdates(),
        dimsAttrs.getScatterDimsToOperandDims(),
        dimsAttrs.getUpdateWindowDims(), dimsAttrs.getInsertedWindowDims());

    int64_t scatterIndicesVectorSize =
        canonicalIndices.getType().cast<TensorType>().getDimSize(1);
    auto canonicalDimsAttrs = ScatterDimensionNumbersAttr::get(
        rewriter.getContext(),
        /*updateWindowDims=*/
        llvm::to_vector<4>(llvm::seq<int64_t>(1, operandRank + 1)),
        /*insertedWindowDims=*/std::nullopt,
        /*scatterDimsToOperandDims=*/
        llvm::to_vector<4>(llvm::seq<int64_t>(0, scatterIndicesVectorSize)),
        /*indexVectorDim=*/1);

    auto newScatterOp = rewriter.create<ScatterOp>(
        loc, TypeRange(ValueRange(canonicalOperands)), canonicalOperands,
        canonicalIndices, canonicalUpdates, canonicalDimsAttrs);
    Region& region = newScatterOp.getUpdateComputation();
    rewriter.inlineRegionBefore(scatterOp.getUpdateComputation(), region,
                                region.end());

    SmallVector<Value> transposedResults = transposeTensors(
        rewriter, loc, newScatterOp.getResults(), operandPermutationInverse);
    rewriter.replaceOp(scatterOp, transposedResults);
    return success();
  }
};

struct HloCanonicalizeScatterPass
    : impl::HloCanonicalizeScatterPassBase<HloCanonicalizeScatterPass> {
  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<CanonicalizeScatterPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createHloCanonicalizeScatterPass() {
  return std::make_unique<HloCanonicalizeScatterPass>();
}

}  // namespace mhlo
}  // namespace mlir

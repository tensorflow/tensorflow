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

// This files implements the logic for converting `scf.parallel` loops into
// tiled loops.

#include <cstdint>
#include <tuple>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "mlir-hlo/Transforms/PassDetail.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

using ::mlir::scf::ParallelOp;

namespace {

// This is the implementation of the TileLoops pass declared in
//  include/mlir-hlo/Transforms/passes.td
class TileLoopsPass : public TileLoopsPassBase<TileLoopsPass> {
 public:
  // Creates a TileLoopsPass with tiles sizes provided through `tile_sizes`
  // and unroll factors provided through `unroll_factors`.
  explicit TileLoopsPass(ArrayRef<int64_t> tileSizes,
                         ArrayRef<int64_t> unrollFactors) {
    tile_sizes_ = tileSizes;
    unroll_factors_ = unrollFactors;
  }

  void runOnOperation() override;
};

}  // namespace

// Returns whether the access pattern in `ploop` is "complex". That is, whether
// any memref.load op in its region uses indices that don't correspond to the
// loop induction variables.
static bool isComplexAccessPattern(ParallelOp ploop) {
  auto isComplex = [&](memref::LoadOp loadOp) {
    if (!loadOp.getMemRefType().getLayout().isIdentity()) return true;
    if (loadOp.getIndices().empty()) return false;
    return loadOp.getIndices() != ploop.getInductionVars();
  };
  return llvm::any_of(ploop.getBody()->getOps<memref::LoadOp>(), isComplex);
}

void TileLoopsPass::runOnOperation() {
  SmallVector<int64_t> unrolledTile;
  if (tile_sizes_.size() == unroll_factors_.size()) {
    unrolledTile.reserve(tile_sizes_.size());
    for (int i = 0; i < tile_sizes_.size(); ++i)
      unrolledTile.push_back(tile_sizes_[i] * unroll_factors_[i]);
  }

  SmallVector<ParallelOp, 2> ploops;
  getInnermostParallelLoops(this->getOperation().getOperation(), ploops);
  for (ParallelOp ploop : ploops) {
    // Do not unroll if the tiling and unrolling have different rank, or if
    // the access pattern is complex.
    if (unrolledTile.empty() || isComplexAccessPattern(ploop)) {
      tileParallelLoop(ploop, tile_sizes_, /*noMinMaxBounds=*/false);
      continue;
    }

    // Collect lower/upper bounds and step size, if they are constants.
    auto getConstDefOps = [](OperandRange operands) {
      return llvm::to_vector(llvm::map_range(operands, [&](Value value) {
        return value.getDefiningOp<arith::ConstantIndexOp>();
      }));
    };
    auto lower = getConstDefOps(ploop.getLowerBound());
    auto upper = getConstDefOps(ploop.getUpperBound());
    auto step = getConstDefOps(ploop.getStep());

    bool noMinMaxBounds = false;
    ploop = tileParallelLoop(ploop, unrolledTile, noMinMaxBounds).second;
    ploop = tileParallelLoop(ploop, unroll_factors_, noMinMaxBounds).second;

    // Use static upper bound on unrolled loop if possible. That is, if the
    // unroll factor evenly divides the iteration size of the outer ploop.
    OpBuilder builder(ploop);
    Location loc = ploop.getLoc();
    for (int i = 0; i < unrolledTile.size(); ++i) {
      if (!lower[i] || !upper[i] || !step[i]) continue;
      int64_t unrollFactor = unroll_factors_[i];
      int64_t difference = upper[i].value() - lower[i].value();
      if (difference % (step[i].value() * unrollFactor) != 0) continue;
      ploop.getUpperBoundMutable().slice(i, 1).assign(
          builder.create<arith::ConstantIndexOp>(loc, unrollFactor));
    }
  }

  // Apply arithmetic dialect canonicalizations so that
  // ParallelToGpuLaunchLowering can derive loop-invariant upper bound for
  // number of iterations.
  RewritePatternSet patterns(&getContext());
  getContext()
      .getOrLoadDialect<arith::ArithmeticDialect>()
      ->getCanonicalizationPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<OperationPass<func::FuncOp>> createTileLoopsPass(
    ArrayRef<int64_t> tileSizes, ArrayRef<int64_t> unrollFactors) {
  return std::make_unique<TileLoopsPass>(tileSizes, unrollFactors);
}

}  // namespace mlir

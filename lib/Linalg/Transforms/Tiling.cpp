//===- Tiling.cpp - Implementation of linalg Tiling -----------------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file implements the linalg dialect Tiling pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/EDSC/Helpers.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Linalg/IR/LinalgOps.h"
#include "mlir/Linalg/IR/LinalgTypes.h"
#include "mlir/Linalg/Passes.h"
#include "mlir/Linalg/Utils/Intrinsics.h"
#include "mlir/Linalg/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/STLExtras.h"
#include "mlir/Transforms/FoldUtils.h"

#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;
using namespace mlir::linalg::intrinsics;

#define DEBUG_TYPE "linalg-tiling"

static llvm::cl::OptionCategory clOptionsCategory(DEBUG_TYPE " options");
static llvm::cl::list<unsigned>
    clTileSizes("linalg-tile-sizes",
                llvm::cl::desc("Tile sizes by which to tile linalg operations"),
                llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated,
                llvm::cl::cat(clOptionsCategory));

static bool isZero(Value *v) {
  return isa_and_nonnull<ConstantIndexOp>(v->getDefiningOp()) &&
         cast<ConstantIndexOp>(v->getDefiningOp()).getValue() == 0;
}

// Creates a number of ranges equal to the number of non-zero in `tileSizes`.
// One for each loop of the LinalgOp that is tiled. The `tileSizes` argument has
// one entry per surrounding loop. It uses zero as the convention that a
// particular loop is not tiled. This convention simplifies implementations by
// avoiding affine map manipulations.
// The returned ranges correspond to the loop ranges, in the proper order, that
// are tiled and for which new loops will be created.
static SmallVector<Value *, 4>
makeTiledLoopRanges(OpBuilder &b, Location loc, AffineMap map,
                    ArrayRef<Value *> allViewSizes,
                    ArrayRef<Value *> allTileSizes, OperationFolder &state) {
  assert(allTileSizes.size() == map.getNumResults());
  // Apply `map` to get view sizes in loop order.
  auto viewSizes = applyMapToValues(b, loc, map, allViewSizes, state);
  SmallVector<Value *, 4> tileSizes(allTileSizes.begin(), allTileSizes.end());

  // Traverse the tile sizes, which are in loop order, erase zeros everywhere.
  for (int idx = tileSizes.size() - 1; idx >= 0; --idx) {
    if (isZero(tileSizes[idx])) {
      viewSizes.erase(viewSizes.begin() + idx);
      tileSizes.erase(tileSizes.begin() + idx);
    }
  }

  // Create a new range with the applied tile sizes.
  SmallVector<Value *, 4> res;
  for (unsigned idx = 0, e = tileSizes.size(); idx < e; ++idx) {
    res.push_back(b.create<RangeOp>(loc,
                                    state.create<ConstantIndexOp>(b, loc, 0),
                                    viewSizes[idx], tileSizes[idx]));
  }
  return res;
}

// E.g. for `A` in the expression:
//     `A(i, k) * B(k, j) -> C(i, j)`
// and for map:
//     `(i, j, k) -> (i, k)`
// and for `r` such that:
//     `r == 1` (i.e. result `k`)
// returns 2 (i.e. `k` on the map domain).
static unsigned getPosInDomain(LinalgOp op, unsigned viewIndex, unsigned dim) {
  auto map = loopToOperandRangesMaps(op)[viewIndex];
  return map.getResult(dim).cast<AffineDimExpr>().getPosition();
}

static bool isTiledView(LinalgOp linalgOp, unsigned viewIndex,
                        ArrayRef<Value *> tileSizes) {
  auto viewIteratorBegin = linalgOp.getInputsAndOutputs().begin();
  Value *view = *(viewIteratorBegin + viewIndex);
  unsigned viewRank = view->getType().cast<ViewType>().getRank();
  for (unsigned r = 0; r < viewRank; ++r) {
    // Loop position for the range dimension.
    auto pos = getPosInDomain(linalgOp, viewIndex, r);
    auto tileSize = tileSizes[pos];
    if (!isZero(tileSize))
      return true;
  }
  return false;
}

static SmallVector<Value *, 4> makeTiledViews(OpBuilder &b, Location loc,
                                              LinalgOp linalgOp,
                                              ArrayRef<Value *> ivs,
                                              ArrayRef<Value *> tileSizes,
                                              OperationFolder &state) {
  assert(ivs.size() == static_cast<size_t>(llvm::count_if(
                           llvm::make_range(tileSizes.begin(), tileSizes.end()),
                           [](Value *v) { return !isZero(v); })) &&
         "expected as many ivs as non-zero sizes");

  using edsc::intrinsics::select;
  using edsc::op::operator+;
  using edsc::op::operator<;

  auto *op = linalgOp.getOperation();

  SmallVector<Value *, 4> res;
  res.reserve(op->getNumOperands());
  auto viewIteratorBegin = linalgOp.getInputsAndOutputs().begin();
  for (unsigned viewIndex = 0; viewIndex < linalgOp.getNumInputsAndOutputs();
       ++viewIndex) {
    Value *view = *(viewIteratorBegin + viewIndex);
    unsigned viewRank = view->getType().cast<ViewType>().getRank();
    // Early exit in the untiled case.
    if (!isTiledView(linalgOp, viewIndex, tileSizes)) {
      res.push_back(view);
      continue;
    }

    // If not a scalar, then construct a new subview for the tile.
    SmallVector<SubViewOp::Range, 4> subViewOperands;
    subViewOperands.reserve(viewRank * 3);
    for (unsigned r = 0; r < viewRank; ++r) {
      // Loop position for the range dimension.
      auto pos = getPosInDomain(linalgOp, viewIndex, r);
      auto tileSize = tileSizes[pos];
      if (isZero(tileSize)) {
        subViewOperands.push_back(
            SubViewOp::Range{state.create<ConstantIndexOp>(b, loc, 0),
                             linalg::intrinsics::dim(view, r),
                             state.create<ConstantIndexOp>(b, loc, 1)});
        continue;
      }

      // `tileSizes` of `0` don't have an induction variable counterpart. So
      // we count the number of zeros ot align the index in `ivs` to pos.
      auto count = llvm::count_if(
          llvm::make_range(tileSizes.begin(), tileSizes.begin() + pos),
          [](Value *v) { return isZero(v); });
      auto iv = ivs[pos - count];

      ScopedContext scope(b, loc);
      // TODO(ntv): lb = iv is a poor man's folding of max(0, i) == i which is
      // generally wrong but correct in the specific case of tiling linalg ops.
      // Tie this loose end in the future.
      ValueHandle lb(iv);
      ValueHandle step(tileSize);
      ValueHandle steppedLb = lb + step;
      // Tiling creates a new slice at the proper index, the slice step is 1
      // (i.e. the slice view does not subsample, stepping occurs in the loop).
      subViewOperands.push_back(SubViewOp::Range{
          iv, steppedLb, state.create<ConstantIndexOp>(b, loc, 1)});
    }
    res.push_back(b.create<SubViewOp>(loc, view, subViewOperands));
  }
  return res;
}

llvm::Optional<TiledLinalgOp>
mlir::linalg::tileLinalgOp(LinalgOp op, ArrayRef<Value *> tileSizes,
                           OperationFolder &state) {
  // Enforce the convention that "tiling by zero" skips tiling a particular
  // dimension. This convention is significantly simpler to handle instead of
  // adjusting affine maps to account for missing dimensions.
  assert(op.getNumParallelLoops() + op.getNumReductionLoops() +
                 op.getNumWindowLoops() ==
             tileSizes.size() &&
         "expected matching number of tile sizes and loops");

  OpBuilder builder(op.getOperation());
  ScopedContext scope(builder, op.getLoc());
  auto loopRanges = makeTiledLoopRanges(
      scope.getBuilder(), scope.getLocation(),
      // The flattened loopToOperandRangesMaps is expected to be an invertible
      // permutation map (which is asserted in the inverse calculation).
      inversePermutation(concatAffineMaps(loopToOperandRangesMaps(op))),
      getViewSizes(op), tileSizes, state);

  LinalgOp res = op;
  SmallVector<IndexHandle, 4> ivs(loopRanges.size());
  auto pivs = IndexHandle::makeIndexHandlePointers(ivs);
  LoopNestRangeBuilder(pivs, loopRanges)([&op, &tileSizes, &ivs, &res, &state] {
    auto b = ScopedContext::getBuilder();
    auto loc = ScopedContext::getLocation();
    SmallVector<Value *, 4> ivValues(ivs.begin(), ivs.end());
    // If/when the assertion below becomes false, we will have to templatize
    // `makeTiledViews`.
    assert(op.getNumInputsAndOutputs() == op.getOperation()->getNumOperands());
    auto views = makeTiledViews(b, loc, op, ivValues, tileSizes, state);
    res = op.create(b, loc, views);
  });

  SmallVector<ForOp, 8> loops;
  loops.reserve(ivs.size());
  for (auto iv : ivs)
    loops.push_back(linalg::getForInductionVarOwner(iv));
  return TiledLinalgOp{res, loops};
}

llvm::Optional<TiledLinalgOp>
mlir::linalg::tileLinalgOp(LinalgOp op, ArrayRef<int64_t> tileSizes,
                           OperationFolder &state) {
  if (tileSizes.empty())
    return llvm::None;

  // The following uses the convention that "tiling by zero" skips tiling a
  // particular dimension. This convention is significantly simpler to handle
  // instead of adjusting affine maps to account for missing dimensions.
  auto nLoops = op.getNumParallelLoops() + op.getNumReductionLoops() +
                op.getNumWindowLoops();
  tileSizes = tileSizes.take_front(nLoops);
  // If only 0 tilings are left, then return.
  if (llvm::all_of(tileSizes, [](int64_t v) { return v == 0; }))
    return llvm::None;

  // Create a builder for tile size constants.
  OpBuilder builder(op);
  auto loc = op.getLoc();

  // Materialize concrete tile size values to pass the generic tiling function.
  SmallVector<Value *, 8> tileSizeValues;
  tileSizeValues.reserve(tileSizes.size());
  for (auto ts : tileSizes)
    tileSizeValues.push_back(state.create<ConstantIndexOp>(builder, loc, ts));
  // Pad tile sizes with zero values to enforce our convention.
  if (tileSizeValues.size() < nLoops) {
    for (unsigned i = tileSizeValues.size(); i < nLoops; ++i)
      tileSizeValues.push_back(state.create<ConstantIndexOp>(builder, loc, 0));
  }

  return tileLinalgOp(op, tileSizeValues, state);
}

static void tileLinalgOps(Function f, ArrayRef<int64_t> tileSizes) {
  OperationFolder state;
  f.walk<LinalgOp>([tileSizes, &state](LinalgOp op) {
    auto opLoopsPair = tileLinalgOp(op, tileSizes, state);
    // If tiling occurred successfully, erase old op.
    if (opLoopsPair)
      op.erase();
  });
}

namespace {
struct LinalgTilingPass : public FunctionPass<LinalgTilingPass> {
  LinalgTilingPass();
  LinalgTilingPass(ArrayRef<int64_t> sizes);

  void runOnFunction() { tileLinalgOps(getFunction(), tileSizes); }

  SmallVector<int64_t, 8> tileSizes;
};
} // namespace

LinalgTilingPass::LinalgTilingPass()
    : tileSizes(clTileSizes.begin(), clTileSizes.end()) {}

LinalgTilingPass::LinalgTilingPass(ArrayRef<int64_t> sizes)
    : LinalgTilingPass() {
  if (!sizes.empty())
    this->tileSizes.assign(sizes.begin(), sizes.end());
}

FunctionPassBase *
mlir::linalg::createLinalgTilingPass(ArrayRef<int64_t> tileSizes) {
  return new LinalgTilingPass(tileSizes);
}

static PassRegistration<LinalgTilingPass>
    pass("linalg-tile", "Tile operations in the linalg dialect");

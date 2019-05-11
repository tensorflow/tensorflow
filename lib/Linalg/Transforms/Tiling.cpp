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
#include "mlir/Linalg/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/STLExtras.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;
using namespace llvm;

static llvm::cl::OptionCategory clOptionsCategory("linalg options");
static llvm::cl::list<unsigned>
    clTileSizes("linalg-tile-sizes",
                llvm::cl::desc("Tile sizes by which to tile linalg operations"),
                llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated,
                llvm::cl::cat(clOptionsCategory));

namespace {
class PerFunctionState {
public:
  PerFunctionState(Function &f) : f(f) {}

  Value *getOrCreate(int64_t v) {
    auto it = map.find(v);
    if (it != map.end())
      return it->second;
    edsc::ScopedContext s(&f);
    return map.insert(std::make_pair(v, edsc::intrinsics::constant_index(v)))
        .first->getSecond();
  }

private:
  Function &f;
  SmallDenseMap<int64_t, Value *> map;
};
} // namespace

// Folding eagerly is necessary to abide by affine.for static step requirement.
// We must propagate constants on the steps as aggressively as possible.
// Returns nullptr if folding is not trivially feasible.
static Value *tryFold(AffineMap map, ArrayRef<Value *> operands,
                      PerFunctionState &state) {
  assert(map.getNumResults() == 1 && "single result map expected");
  auto expr = map.getResult(0);
  if (auto dim = expr.dyn_cast<AffineDimExpr>())
    return operands[dim.getPosition()];
  if (auto sym = expr.dyn_cast<AffineSymbolExpr>())
    return operands[map.getNumDims() + sym.getPosition()];
  if (auto cst = expr.dyn_cast<AffineConstantExpr>())
    return state.getOrCreate(cst.getValue());
  return nullptr;
}

static Value *emitOrFoldComposedAffineApply(FuncBuilder *b, Location loc,
                                            AffineMap map,
                                            ArrayRef<Value *> operandsRef,
                                            PerFunctionState &state) {
  SmallVector<Value *, 4> operands(operandsRef.begin(), operandsRef.end());
  fullyComposeAffineMapAndOperands(&map, &operands);
  if (auto *v = tryFold(map, operands, state))
    return v;
  return b->create<AffineApplyOp>(loc, map, operands);
}

static SmallVector<Value *, 4> applyMapToRangePart(FuncBuilder *b, Location loc,
                                                   AffineMap map,
                                                   ArrayRef<Value *> ranges,
                                                   RangePart part,
                                                   PerFunctionState &state) {
  SmallVector<Value *, 4> rangeParts(ranges.size());
  transform(llvm::make_range(ranges.begin(), ranges.end()), rangeParts.begin(),
            [&](Value *range) { return extractRangePart(range, part); });

  SmallVector<Value *, 4> res;
  res.reserve(map.getNumResults());
  unsigned numDims = map.getNumDims();
  for (auto expr : map.getResults()) {
    AffineMap map = AffineMap::get(numDims, 0, expr, {});
    res.push_back(
        emitOrFoldComposedAffineApply(b, loc, map, rangeParts, state));
  }
  return res;
}

static bool isZero(Value *v) {
  return v->getDefiningOp() && v->getDefiningOp()->isa<ConstantIndexOp>() &&
         v->getDefiningOp()->cast<ConstantIndexOp>().getValue() == 0;
}

/// Returns a map that can be used to filter the zero values out of tileSizes.
/// For example, if tileSizes contains `{v1, 0, v2}`, the returned map is:
///
/// ```{.mlir}
///    (d0, d1, d2) -> (d0, d2)
/// ```
static AffineMap nonZeroMap(ArrayRef<Value *> tileSizes) {
  SmallVector<AffineExpr, 4> exprs;
  for (auto en : llvm::enumerate(tileSizes))
    if (!isZero(en.value()))
      exprs.push_back(getAffineDimExpr(en.index(), en.value()->getContext()));
  assert(!exprs.empty() &&
         "unexpected zero-only tile sizes, should have been handled earlier");
  return AffineMap::get(tileSizes.size(), 0, exprs, {});
}

// Creates a number of ranges equal to the number of non-zero in `tileSizes`.
// One for each loop of the LinalgOp that is tiled. The `tileSizes` argument has
// one entry per surrounding loop. It uses zero as the convention that a
// particular loop is not tiled. This convention simplifies implementations by
// avoiding affine map manipulations.
// The returned ranges correspond to the loop ranges, in the proper order, that
// are tiled and for which new loops will be created.
static SmallVector<Value *, 4>
makeTiledLoopRanges(FuncBuilder *b, Location loc, AffineMap map,
                    ArrayRef<Value *> allOpRanges, ArrayRef<Value *> tileSizes,
                    PerFunctionState &state) {
  assert(tileSizes.size() == map.getNumResults());
  // Tile sizes are in loop order by construction, apply `map` to
  // get mins/maxes/steps in loop order.
  auto mins =
      applyMapToRangePart(b, loc, map, allOpRanges, RangePart::Min, state);
  auto maxes =
      applyMapToRangePart(b, loc, map, allOpRanges, RangePart::Max, state);
  auto steps =
      applyMapToRangePart(b, loc, map, allOpRanges, RangePart::Step, state);
  SmallVector<Value *, 4> sizes(tileSizes.begin(), tileSizes.end());

  // Traverse the tile sizes, which are in loop order, erase zeros everywhere.
  for (int idx = mins.size() - 1; idx >= 0; --idx) {
    if (isZero(tileSizes[idx])) {
      mins.erase(mins.begin() + idx);
      maxes.erase(maxes.begin() + idx);
      steps.erase(steps.begin() + idx);
      sizes.erase(sizes.begin() + idx);
    }
  }

  // Create a new range with the applied tile sizes.
  SmallVector<Value *, 4> res;
  for (unsigned idx = 0, e = steps.size(); idx < e; ++idx) {
    auto *step = steps[idx];
    auto *tileSize = sizes[idx];
    // clang-format off
    // Steps must be constant for now to abide by affine.for semantics.
    auto *newStep =
        state.getOrCreate(
            step->getDefiningOp()->cast<ConstantIndexOp>().getValue() *
            tileSize->getDefiningOp()->cast<ConstantIndexOp>().getValue());
    res.push_back(b->create<RangeOp>(loc, mins[idx], maxes[idx], newStep));
    // clang-format on
  }
  return res;
}

static SmallVector<Value *, 4> makeTiledViews(FuncBuilder *b, Location loc,
                                              Operation *op,
                                              ArrayRef<Value *> ivs,
                                              ArrayRef<Value *> tileSizes,
                                              PerFunctionState &state) {
  assert(ivs.size() == llvm::count_if(
                           llvm::make_range(tileSizes.begin(), tileSizes.end()),
                           [](Value *v) { return !isZero(v); }) &&
         "expected as many ivs as non-zero sizes");
  auto *context = op->getContext();

  SmallVector<Value *, 4> res;
  res.reserve(op->getNumOperands());
  for (unsigned i = 0, ei = op->getNumOperands(); i < ei; ++i) {
    auto *viewDefiningOp = op->getOperand(i)->getDefiningOp();
    assert(viewDefiningOp && "Need operations to extract ranges from views");
    auto ranges = getRanges(viewDefiningOp);
    // E.g. for A in A(i, k) * B(k, j) -> C(i, j) returns the map:
    //   (i, j, k) -> (i, k)
    auto map = loopToOperandRangesMaps(op)[i];
    if (!map) {
      assert(ranges.empty() && "scalar should have empty ranges");
      res.push_back(op->getOperand(i));
      continue;
    }
    assert(ranges.size() == map.getNumResults());
    // E.g. for {0, 0, v2} returns the map:
    //   (i, j, k) -> (k)
    auto nzMap = nonZeroMap(tileSizes);

    SmallVector<Value *, 4> newRanges;
    newRanges.reserve(ranges.size());
    for (unsigned j = 0, ej = ranges.size(); j < ej; ++j) {
      // Loop position for the range dimension.
      // E.g. for A in A(i, k) * B(k, j) -> C(i, j) and map: (i, j, k) -> (i, k)
      //   and for j == 1 (i.e. result `k`)
      //   returns loopPos = 2 (i.e. `k` on the map domain).
      auto pos = map.getResult(j).template cast<AffineDimExpr>().getPosition();
      if (isZero(tileSizes[pos])) {
        newRanges.push_back(ranges[j]);
        continue;
      }
      auto it = llvm::find_if(nzMap.getResults(), [pos, context](AffineExpr e) {
        return e == getAffineDimExpr(pos, context);
      });
      assert(it != nzMap.getResults().end() &&
             "position does not correspond to a valid induction variable");
      unsigned pos2 = it - nzMap.getResults().begin();
      using edsc::op::operator+;
      using range = ValueBuilder<RangeOp>;
      ScopedContext scope(*b, loc);
      ValueHandle iv(ivs[pos2]), step(tileSizes[pos]);
      auto min = ValueHandle(extractRangePart(ranges[j], RangePart::Min));
      // zero case is important enough to fold away by special-casing.
      auto newMin = isZero(min) ? iv : min + iv;
      // TODO(ntv): intersect with current range once the operation exists.
      Value *r = range(newMin, newMin + step, step);
      newRanges.push_back(r);
    }
    res.push_back(createOrReturnView(b, loc, viewDefiningOp, newRanges));
  }
  return res;
}

template <class LinalgOp>
static LogicalResult tileLinalgOp(LinalgOp &op, ArrayRef<Value *> tileSizes,
                                  PerFunctionState &state) {
  // Enforce the convention that "tiling by zero" skips tiling a particular
  // dimension. This convention is significantly simpler to handle instead of
  // adjusting affine maps to account for missing dimensions.
  assert(op.getNumParallelLoops() + op.getNumReductionLoops() +
                 op.getNumWindowLoops() ==
             tileSizes.size() &&
         "expected matching number of tile sizes and loops");

  ScopedContext scope(FuncBuilder(op.getOperation()), op.getLoc());
  auto loopRanges = makeTiledLoopRanges(
      scope.getBuilder(), scope.getLocation(),
      // The flattened loopToOperandRangesMaps is expected to be an invertible
      // permutation map (which is asserted in the inverse calculation).
      inversePermutation(concatAffineMaps(loopToOperandRangesMaps(op))),
      getRanges(op.getOperation()), tileSizes, state);

  SmallVector<IndexHandle, 4> ivs(loopRanges.size());
  auto pivs = IndexHandle::makeIndexHandlePointers(ivs);
  LoopNestRangeBuilder(pivs, loopRanges)({[&op, &tileSizes, &ivs, &state]() {
    auto *b = ScopedContext::getBuilder();
    auto loc = ScopedContext::getLocation();
    SmallVector<Value *, 4> ivValues(ivs.begin(), ivs.end());
    // If/when the assertion below becomes false, we will have to templatize
    // `makeTiledViews`.
    assert(op.getNumInputsAndOutputs() == op.getOperation()->getNumOperands());
    auto views =
        makeTiledViews(b, loc, op.getOperation(), ivValues, tileSizes, state);
    b->create<LinalgOp>(loc, views);
    /// NestedBuilders expect handles, we thus return an IndexHandle.
    return IndexHandle();
  }()});

  return success();
}

template <class LinalgOp>
static LogicalResult tileLinalgOp(LinalgOp &op, ArrayRef<int64_t> tileSizes,
                                  PerFunctionState &state) {
  if (tileSizes.empty())
    return failure();

  // The following uses the convention that "tiling by zero" skips tiling a
  // particular dimension. This convention is significantly simpler to handle
  // instead of adjusting affine maps to account for missing dimensions.
  auto nLoops = op.getNumParallelLoops() + op.getNumReductionLoops() +
                op.getNumWindowLoops();
  tileSizes = tileSizes.take_front(nLoops);
  // If only 0 tilings are left, then return.
  if (llvm::all_of(tileSizes, [](int64_t v) { return v == 0; }))
    return failure();

  // Materialize concrete tile size values to pass the generic tiling function.
  SmallVector<Value *, 8> tileSizeValues;
  tileSizeValues.reserve(tileSizes.size());
  for (auto ts : tileSizes)
    tileSizeValues.push_back(state.getOrCreate(ts));
  // Pad tile sizes with zero values to enforce our convention.
  if (tileSizeValues.size() < nLoops) {
    for (unsigned i = tileSizeValues.size(); i < nLoops; ++i)
      tileSizeValues.push_back(state.getOrCreate(0));
  }

  return tileLinalgOp(op, tileSizeValues, state);
}

// TODO(ntv) expose as a primitive for other passes.
static LogicalResult tileLinalgOp(Operation *op, ArrayRef<int64_t> tileSizes,
                                  PerFunctionState &state) {
  if (auto matmulOp = dyn_cast<MatmulOp>(op)) {
    return tileLinalgOp(matmulOp, tileSizes, state);
  } else if (auto matvecOp = dyn_cast<MatvecOp>(op)) {
    return tileLinalgOp(matvecOp, tileSizes, state);
  } else if (auto dotOp = dyn_cast<DotOp>(op)) {
    return tileLinalgOp(dotOp, tileSizes, state);
  }
  return failure();
}

static void tileLinalgOps(Function &f, ArrayRef<int64_t> tileSizes) {
  PerFunctionState state(f);
  f.walk([tileSizes, &state](Operation *op) {
    if (succeeded(tileLinalgOp(op, tileSizes, state)))
      op->erase();
  });
}

namespace {
struct LinalgTilingPass : public ModulePass<LinalgTilingPass> {
  LinalgTilingPass();
  LinalgTilingPass(ArrayRef<int64_t> sizes);

  void runOnModule() {
    for (auto &f : getModule())
      tileLinalgOps(f, tileSizes);
  }

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

ModulePassBase *mlir::createLinalgTilingPass(ArrayRef<int64_t> tileSizes) {
  return new LinalgTilingPass(tileSizes);
}

static PassRegistration<LinalgTilingPass>
    pass("linalg-tile", "Tile operations in the linalg dialect");

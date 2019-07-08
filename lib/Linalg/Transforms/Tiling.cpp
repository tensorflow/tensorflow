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
#include "mlir/IR/AffineExprVisitor.h"
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
static llvm::cl::opt<bool> clPromoteFullTileViews(
    "linalg-tile-promote-full-tile-views",
    llvm::cl::desc("Create scoped local buffers for tiled views "),
    llvm::cl::init(false), llvm::cl::cat(clOptionsCategory));

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
static SmallVector<SubViewOp::Range, 4>
makeTiledLoopRanges(OpBuilder &b, Location loc, AffineMap map,
                    ArrayRef<Value *> allViewSizes,
                    ArrayRef<Value *> allTileSizes, OperationFolder &folder) {
  assert(allTileSizes.size() == map.getNumResults());
  // Apply `map` to get view sizes in loop order.
  auto viewSizes = applyMapToValues(b, loc, map, allViewSizes, folder);
  SmallVector<Value *, 4> tileSizes(allTileSizes.begin(), allTileSizes.end());

  // Traverse the tile sizes, which are in loop order, erase zeros everywhere.
  for (int idx = tileSizes.size() - 1; idx >= 0; --idx) {
    if (isZero(tileSizes[idx])) {
      viewSizes.erase(viewSizes.begin() + idx);
      tileSizes.erase(tileSizes.begin() + idx);
    }
  }

  // Create a new range with the applied tile sizes.
  SmallVector<SubViewOp::Range, 4> res;
  for (unsigned idx = 0, e = tileSizes.size(); idx < e; ++idx) {
    res.push_back(SubViewOp::Range{constant_index(folder, 0), viewSizes[idx],
                                   tileSizes[idx]});
  }
  return res;
}

namespace {
// Helper visitor to determine whether an AffineExpr is tiled.
// This is achieved by traversing every AffineDimExpr with position `pos` and
// checking whether the corresponding `tileSizes[pos]` is non-zero.
// This also enforces only positive coefficients occur in multiplications.
//
// Example:
//   `d0 + 2 * d1 + d3` is tiled by [0, 0, 0, 2] but not by [0, 0, 2, 0]
//
struct TileCheck : public AffineExprVisitor<TileCheck> {
  TileCheck(ArrayRef<Value *> tileSizes)
      : isTiled(false), tileSizes(tileSizes) {}

  void visitDimExpr(AffineDimExpr expr) {
    isTiled |= !isZero(tileSizes[expr.getPosition()]);
  }
  void visitAffineBinaryOpExpr(AffineBinaryOpExpr expr) {
    visit(expr.getLHS());
    visit(expr.getRHS());
    if (expr.getKind() == mlir::AffineExprKind::Mul)
      assert(expr.getRHS().cast<AffineConstantExpr>().getValue() > 0 &&
             "nonpositive multipliying coefficient");
  }
  bool isTiled;
  ArrayRef<Value *> tileSizes;
};
} // namespace

static bool isTiled(AffineExpr expr, ArrayRef<Value *> tileSizes) {
  if (!expr)
    return false;
  TileCheck t(tileSizes);
  t.visit(expr);
  return t.isTiled;
}

// Checks whether the view with index `viewIndex` within `linalgOp` varies with
// respect to a non-zero `tileSize`.
static bool isTiled(AffineMap map, ArrayRef<Value *> tileSizes) {
  if (!map)
    return false;
  for (unsigned r = 0; r < map.getNumResults(); ++r)
    if (isTiled(map.getResult(r), tileSizes))
      return true;
  return false;
}

static SmallVector<Value *, 4>
makeTiledViews(OpBuilder &b, Location loc, LinalgOp linalgOp,
               ArrayRef<Value *> ivs, ArrayRef<Value *> tileSizes,
               ArrayRef<Value *> viewSizes, OperationFolder &folder) {
  assert(ivs.size() == static_cast<size_t>(llvm::count_if(
                           llvm::make_range(tileSizes.begin(), tileSizes.end()),
                           [](Value *v) { return !isZero(v); })) &&
         "expected as many ivs as non-zero sizes");

  using edsc::intrinsics::select;
  using edsc::op::operator+;
  using edsc::op::operator<;

  // Construct (potentially temporary) mins and maxes on which to apply maps
  // that define tile subviews.
  SmallVector<Value *, 8> mins, maxes;
  for (unsigned idx = 0, idxIvs = 0, e = tileSizes.size(); idx < e; ++idx) {
    if (isZero(tileSizes[idx])) {
      mins.push_back(constant_index(folder, 0));
      maxes.push_back(viewSizes[idx]);
    } else {
      ValueHandle lb(ivs[idxIvs++]), step(tileSizes[idx]);
      mins.push_back(lb);
      maxes.push_back(lb + step);
    }
  }

  auto *op = linalgOp.getOperation();

  SmallVector<Value *, 4> res;
  res.reserve(op->getNumOperands());
  auto viewIteratorBegin = linalgOp.getInputsAndOutputs().begin();
  for (unsigned viewIndex = 0; viewIndex < linalgOp.getNumInputsAndOutputs();
       ++viewIndex) {
    Value *view = *(viewIteratorBegin + viewIndex);
    unsigned viewRank = view->getType().cast<ViewType>().getRank();
    auto map = loopToOperandRangesMaps(linalgOp)[viewIndex];
    // If the view is not tiled, we can use it as is.
    if (!isTiled(map, tileSizes)) {
      res.push_back(view);
      continue;
    }

    // Construct a new subview for the tile.
    SmallVector<SubViewOp::Range, 4> subViewOperands;
    subViewOperands.reserve(viewRank * 3);
    for (unsigned r = 0; r < viewRank; ++r) {
      if (!isTiled(map.getSubMap({r}), tileSizes)) {
        subViewOperands.push_back(SubViewOp::Range{
            constant_index(folder, 0), linalg::intrinsics::dim(view, r),
            constant_index(folder, 1)});
        continue;
      }

      auto m = map.getSubMap({r});
      auto *min = applyMapToValues(b, loc, m, mins, folder).front();
      auto *max = applyMapToValues(b, loc, m, maxes, folder).front();
      // Tiling creates a new slice at the proper index, the slice step is 1
      // (i.e. the slice view does not subsample, stepping occurs in the loop).
      subViewOperands.push_back(
          SubViewOp::Range{min, max, constant_index(folder, 1)});
    }
    res.push_back(b.create<SubViewOp>(loc, view, subViewOperands));
  }

  // Traverse the mins/maxes and erase those that don't have uses left.
  mins.append(maxes.begin(), maxes.end());
  for (auto *v : mins)
    if (v->use_empty())
      v->getDefiningOp()->erase();

  return res;
}

static AffineMap getAffineDifferenceMap(MLIRContext *context) {
  AffineExpr d0(getAffineDimExpr(0, context)), d1(getAffineDimExpr(1, context));
  return AffineMap::get(2, 0, {d0 - d1});
}

static Value *allocBuffer(Type elementType, Value *size) {
  if (auto cst = dyn_cast_or_null<ConstantIndexOp>(size->getDefiningOp()))
    return buffer_alloc(
        BufferType::get(size->getContext(), elementType, cst.getValue()));
  return buffer_alloc(BufferType::get(size->getContext(), elementType), size);
}

// Performs promotion of a `subView` into a local buffer of the size of the
// *ranges* of the `subView`. This produces a buffer whose size may be bigger
// than the actual size of the `subView` at the boundaries.
// This is related to the full/partial tile problem.
// Returns a PromotionInfo containing a `buffer`, `fullLocalView` and
// `partialLocalView` such that:
//   * `buffer` is always the size of the full tile.
//   * `fullLocalView` is a dense contiguous view into that buffer.
//   * `partialLocalView` is a dense non-contiguous slice of `fullLocalView`
//     that corresponds to the size of `subView` and accounting for boundary
//     effects.
// The point of the full tile buffer is that constant static tile sizes are
// folded and result in a buffer type with statically known size and alignment
// properties.
// To account for general boundary effects, padding must be performed on the
// boundary tiles. For now this is done with an unconditional `fill` op followed
// by a partial `copy` op.
static PromotionInfo promoteFullTileBuffer(OpBuilder &b, Location loc,
                                           SubViewOp subView,
                                           OperationFolder &folder) {
  auto zero = constant_index(folder, 0);
  auto one = constant_index(folder, 1);

  auto viewType = subView.getViewType();
  auto rank = viewType.getRank();
  Value *allocSize = one;
  SmallVector<Value *, 8> fullRanges, partialRanges;
  fullRanges.reserve(rank);
  partialRanges.reserve(rank);
  for (auto en : llvm::enumerate(subView.getRanges())) {
    auto rank = en.index();
    auto rangeValue = en.value();
    Value *d =
        isa<linalg::DimOp>(rangeValue.max->getDefiningOp())
            ? rangeValue.max
            : applyMapToValues(b, loc, getAffineDifferenceMap(b.getContext()),
                               {rangeValue.max, rangeValue.min}, folder)
                  .front();
    allocSize = muli(folder, allocSize, d).getValue();
    fullRanges.push_back(range(folder, zero, d, one));
    partialRanges.push_back(
        range(folder, zero, linalg::intrinsics::dim(subView, rank), one));
  }
  auto *buffer = allocBuffer(viewType.getElementType(), allocSize);
  auto fullLocalView = view(buffer, fullRanges);
  auto partialLocalView = slice(fullLocalView, partialRanges);
  return PromotionInfo{buffer, fullLocalView, partialLocalView};
}

// Performs promotion of a view `v` into a local buffer of the size of the
// view. This produces a buffer whose size is exactky the size of `v`.
// Returns a PromotionInfo containing a `buffer`, `fullLocalView` and
// `partialLocalView` such that:
//   * `buffer` is always the size of the view.
//   * `partialLocalView` is a dense contiguous view into that buffer.
//   * `fullLocalView` is equal to `partialLocalView`.
// The point of the full tile buffer is that constant static tile sizes are
// folded and result in a buffer type with statically known size and alignment
// properties.
static PromotionInfo promotePartialTileBuffer(OpBuilder &b, Location loc,
                                              Value *v,
                                              OperationFolder &folder) {
  auto zero = constant_index(folder, 0);
  auto one = constant_index(folder, 1);

  auto viewType = v->getType().cast<ViewType>();
  auto rank = viewType.getRank();
  Value *allocSize = one;
  SmallVector<Value *, 8> partialRanges;
  partialRanges.reserve(rank);
  for (unsigned r = 0; r < rank; ++r) {
    Value *d = linalg::intrinsics::dim(v, r);
    allocSize = muli(folder, allocSize, d).getValue();
    partialRanges.push_back(range(folder, zero, d, one));
  }
  auto *buffer = allocBuffer(viewType.getElementType(), allocSize);
  auto partialLocalView = view(folder, buffer, partialRanges);
  return PromotionInfo{buffer, partialLocalView, partialLocalView};
}

SmallVector<PromotionInfo, 8>
mlir::linalg::promoteLinalgViews(OpBuilder &b, Location loc,
                                 ArrayRef<Value *> views,
                                 OperationFolder &folder) {
  if (views.empty())
    return {};

  ScopedContext scope(b, loc);
  SmallVector<PromotionInfo, 8> res;
  res.reserve(views.size());
  DenseMap<Value *, PromotionInfo> promotionInfo;
  for (auto *v : views) {
    PromotionInfo pi;
    if (auto subView = dyn_cast<SubViewOp>(v->getDefiningOp()))
      pi = promoteFullTileBuffer(b, loc, subView, folder);
    else
      pi = promotePartialTileBuffer(b, loc, v, folder);
    promotionInfo.insert(std::make_pair(v, pi));
    res.push_back(pi);
  }

  for (auto *v : views) {
    auto info = promotionInfo.find(v);
    if (info == promotionInfo.end())
      continue;
    auto viewType = v->getType().cast<ViewType>();
    // TODO(ntv): value to fill with should be related to the operation.
    // For now, just use APFloat(0.0f).
    auto t = viewType.getElementType().cast<FloatType>();
    Value *fillVal = constant_float(folder, APFloat(0.0f), t);
    // TODO(ntv): fill is only necessary if `promotionInfo` has a full local
    // view that is different from the partial local view and we are on the
    // boundary.
    fill(info->second.fullLocalView, fillVal);
  }

  for (auto *v : views) {
    auto info = promotionInfo.find(v);
    if (info == promotionInfo.end())
      continue;
    copy(v, info->second.partialLocalView);
  }
  return res;
}

llvm::Optional<TiledLinalgOp>
mlir::linalg::tileLinalgOp(LinalgOp op, ArrayRef<Value *> tileSizes,
                           OperationFolder &folder,
                           ArrayRef<bool> viewsToPromote) {
  // 1. Enforce the convention that "tiling by zero" skips tiling a particular
  // dimension. This convention is significantly simpler to handle instead of
  // adjusting affine maps to account for missing dimensions.
  assert(op.getNumParallelLoops() + op.getNumReductionLoops() +
                 op.getNumWindowLoops() ==
             tileSizes.size() &&
         "expected matching number of tile sizes and loops");

  OpBuilder builder(op.getOperation());
  ScopedContext scope(builder, op.getLoc());
  // 2. Build the tiled loop ranges.
  auto viewSizes = getViewSizes(op);
  // The flattened loopToOperandRangesMaps is expected to be an invertible
  // permutation map (asserted in the inverse calculation).
  auto viewSizesToLoopsMap =
      inversePermutation(concatAffineMaps(loopToOperandRangesMaps(op)));
  auto loopRanges =
      makeTiledLoopRanges(scope.getBuilder(), scope.getLocation(),
                          viewSizesToLoopsMap, viewSizes, tileSizes, folder);

  // 3. Create the tiled loops.
  LinalgOp res = op;
  SmallVector<IndexHandle, 4> ivs(loopRanges.size());
  auto pivs = IndexHandle::makeIndexHandlePointers(ivs);
  LoopNestRangeBuilder(pivs, loopRanges)([&] {
    auto b = ScopedContext::getBuilder();
    auto loc = ScopedContext::getLocation();
    SmallVector<Value *, 4> ivValues(ivs.begin(), ivs.end());
    // If/when the assertion below becomes false, templatize `makeTiledViews`.
    assert(op.getNumInputsAndOutputs() == op.getOperation()->getNumOperands());
    auto views =
        makeTiledViews(b, loc, op, ivValues, tileSizes, viewSizes, folder);

    // If no promotion, we are done.
    auto promote = !viewsToPromote.empty() &&
                   llvm::any_of(llvm::make_range(viewsToPromote.begin(),
                                                 viewsToPromote.end()),
                                [](bool b) { return b; });
    if (!promote) {
      res = op.create(b, loc, views);
      return;
    }

    // 4. Filter the subset of views that need to be promoted.
    SmallVector<Value *, 8> filteredViews;
    filteredViews.reserve(views.size());
    assert(
        viewsToPromote.empty() ||
        views.size() == viewsToPromote.size() &&
            "expected viewsToPromote to be empty or of the same size as view");
    for (auto it : llvm::zip(views, viewsToPromote)) {
      if (!std::get<1>(it))
        continue;
      filteredViews.push_back(std::get<0>(it));
    }

    // 5. Promote the specified views and use them in the new op.
    auto promotedBufferAndViews =
        promoteLinalgViews(b, loc, filteredViews, folder);
    SmallVector<Value *, 8> opViews(views.size(), nullptr);
    SmallVector<Value *, 8> writebackViews(views.size(), nullptr);
    for (unsigned i = 0, promotedIdx = 0, e = opViews.size(); i < e; ++i) {
      if (viewsToPromote[i]) {
        opViews[i] = promotedBufferAndViews[promotedIdx].fullLocalView;
        writebackViews[i] =
            promotedBufferAndViews[promotedIdx].partialLocalView;
        promotedIdx++;
      } else {
        opViews[i] = views[i];
      }
    }
    res = op.create(b, loc, opViews);

    // 6. Emit write-back for the promoted output views: copy the partial view.
    for (unsigned i = 0, e = writebackViews.size(); i < e; ++i) {
      bool isOutput = res.getIndexOfOutput(opViews[i]).hasValue();
      if (writebackViews[i] && isOutput)
        copy(writebackViews[i], views[i]);
    }

    // 7. Dealloc local buffers.
    for (const auto &pi : promotedBufferAndViews)
      buffer_dealloc(pi.buffer);
  });

  // 8. Gather the newly created loops and return them with the new op.
  SmallVector<ForOp, 8> loops;
  loops.reserve(ivs.size());
  for (auto iv : ivs)
    loops.push_back(linalg::getForInductionVarOwner(iv));

  return TiledLinalgOp{res, loops};
}

llvm::Optional<TiledLinalgOp>
mlir::linalg::tileLinalgOp(LinalgOp op, ArrayRef<int64_t> tileSizes,
                           OperationFolder &folder,
                           ArrayRef<bool> viewsToPromote) {
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
  ScopedContext scope(builder, op.getLoc());

  // Materialize concrete tile size values to pass the generic tiling function.
  SmallVector<Value *, 8> tileSizeValues;
  tileSizeValues.reserve(tileSizes.size());
  for (auto ts : tileSizes)
    tileSizeValues.push_back(constant_index(folder, ts));
  // Pad tile sizes with zero values to enforce our convention.
  if (tileSizeValues.size() < nLoops) {
    for (unsigned i = tileSizeValues.size(); i < nLoops; ++i)
      tileSizeValues.push_back(constant_index(folder, 0));
  }

  return tileLinalgOp(op, tileSizeValues, folder, viewsToPromote);
}

static void tileLinalgOps(Function f, ArrayRef<int64_t> tileSizes,
                          bool promoteViews) {
  OperationFolder folder;
  f.walk<LinalgOp>([promoteViews, tileSizes, &folder](LinalgOp op) {
    // TODO(ntv) some heuristic here to decide what to promote. Atm it is all or
    // nothing.
    SmallVector<bool, 8> viewsToPromote(op.getNumInputsAndOutputs(),
                                        promoteViews);
    auto opLoopsPair = tileLinalgOp(op, tileSizes, folder, viewsToPromote);
    // If tiling occurred successfully, erase old op.
    if (opLoopsPair)
      op.erase();
  });
  f.walk<LinalgOp>([](LinalgOp op) {
    if (!op.getOperation()->hasNoSideEffect())
      return;
    if (op.getOperation()->use_empty())
      op.erase();
  });
}

namespace {
struct LinalgTilingPass : public FunctionPass<LinalgTilingPass> {
  LinalgTilingPass(ArrayRef<int64_t> sizes, bool promoteViews);

  void runOnFunction() {
    tileLinalgOps(getFunction(), tileSizes, promoteViews);
  }

  SmallVector<int64_t, 8> tileSizes;
  bool promoteViews;

protected:
  LinalgTilingPass() {}
};

struct LinalgTilingPassCLI : public LinalgTilingPass {
  LinalgTilingPassCLI();
};
} // namespace

LinalgTilingPass::LinalgTilingPass(ArrayRef<int64_t> sizes, bool promoteViews) {
  this->tileSizes.assign(sizes.begin(), sizes.end());
  this->promoteViews = promoteViews;
}

LinalgTilingPassCLI::LinalgTilingPassCLI() : LinalgTilingPass() {
  this->tileSizes.assign(clTileSizes.begin(), clTileSizes.end());
  this->promoteViews = clPromoteFullTileViews;
}

FunctionPassBase *
mlir::linalg::createLinalgTilingPass(ArrayRef<int64_t> tileSizes,
                                     bool promoteViews) {
  return new LinalgTilingPass(tileSizes, promoteViews);
}

static PassRegistration<LinalgTilingPassCLI>
    pass("linalg-tile", "Tile operations in the linalg dialect");

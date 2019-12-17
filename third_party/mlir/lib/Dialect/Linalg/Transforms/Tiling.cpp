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

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Utils/Intrinsics.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/EDSC/Helpers.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OpImplementation.h"
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
using namespace mlir::loop;

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

using LoopIndexToRangeIndexMap = DenseMap<int, int>;

// Creates a number of ranges equal to the number of non-zero in `tileSizes`.
// One for each loop of the LinalgOp that is tiled. The `tileSizes` argument has
// one entry per surrounding loop. It uses zero as the convention that a
// particular loop is not tiled. This convention simplifies implementations by
// avoiding affine map manipulations.
// The returned ranges correspond to the loop ranges, in the proper order, that
// are tiled and for which new loops will be created. Also the function returns
// a map from loop indices of the LinalgOp to the corresponding non-empty range
// indices of newly created loops.
static std::tuple<SmallVector<SubViewOp::Range, 4>, LoopIndexToRangeIndexMap>
makeTiledLoopRanges(OpBuilder &b, Location loc, AffineMap map,
                    ArrayRef<Value *> allViewSizes,
                    ArrayRef<Value *> allTileSizes, OperationFolder *folder) {
  assert(allTileSizes.size() == map.getNumResults());
  // Apply `map` to get view sizes in loop order.
  auto viewSizes = applyMapToValues(b, loc, map, allViewSizes, folder);
  SmallVector<Value *, 4> tileSizes(allTileSizes.begin(), allTileSizes.end());

  // Traverse the tile sizes, which are in loop order, erase zeros everywhere.
  LoopIndexToRangeIndexMap loopIndexToRangeIndex;
  for (int idx = 0, e = tileSizes.size(), zerosCount = 0; idx < e; ++idx) {
    if (isZero(tileSizes[idx - zerosCount])) {
      viewSizes.erase(viewSizes.begin() + idx - zerosCount);
      tileSizes.erase(tileSizes.begin() + idx - zerosCount);
      ++zerosCount;
      continue;
    }
    loopIndexToRangeIndex[idx] = idx - zerosCount;
  }

  // Create a new range with the applied tile sizes.
  SmallVector<SubViewOp::Range, 4> res;
  for (unsigned idx = 0, e = tileSizes.size(); idx < e; ++idx) {
    res.push_back(SubViewOp::Range{constant_index(folder, 0), viewSizes[idx],
                                   tileSizes[idx]});
  }
  return std::make_tuple(res, loopIndexToRangeIndex);
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
             "nonpositive multiplying coefficient");
  }
  bool isTiled;
  ArrayRef<Value *> tileSizes;
};

} // namespace

// IndexedGenericOp explicitly uses induction variables in the loop body. The
// values of the indices that are used in the loop body for any given access of
// input/output memref before `subview` op was applied should be invariant with
// respect to tiling.
//
// Therefore, if the operation is tiled, we have to transform the indices
// accordingly, i.e. offset them by the values of the corresponding induction
// variables that are captured implicitly in the body of the op.
//
// Example. `linalg.indexed_generic` before tiling:
//
// #id_2d = (i, j) -> (i, j)
// #pointwise_2d_trait = {
//   indexing_maps = [#id_2d, #id_2d],
//   iterator_types = ["parallel", "parallel"],
//   n_views = [1, 1]
// }
// linalg.indexed_generic #pointwise_2d_trait %operand, %result {
//   ^bb0(%i: index, %j: index, %operand_in: f32, %result_in: f32):
//     <some operations that use %i, %j>
// }: memref<50x100xf32>, memref<50x100xf32>
//
// After tiling pass with tiles sizes 10 and 25:
//
// #strided = (i, j)[s0, s1, s2] -> (i * s1 + s0 + j * s2)
//
// %c1 = constant 1 : index
// %c0 = constant 0 : index
// %c25 = constant 25 : index
// %c10 = constant 10 : index
// operand_dim_0 = dim %operand, 0 : memref<50x100xf32>
// operand_dim_1 = dim %operand, 1 : memref<50x100xf32>
// loop.for %k = %c0 to operand_dim_0 step %c10 {
//   loop.for %l = %c0 to operand_dim_1 step %c25 {
//     %4 = std.subview %operand[%k, %l][%c10, %c25][%c1, %c1]
//       : memref<50x100xf32> to memref<?x?xf32, #strided>
//     %5 = std.subview %result[%k, %l][%c10, %c25][%c1, %c1]
//       : memref<50x100xf32> to memref<?x?xf32, #strided>
//     linalg.indexed_generic pointwise_2d_trait %4, %5 {
//     ^bb0(%i: index, %j: index, %operand_in: f32, %result_in: f32):
//       // Indices `k` and `l` are implicitly captured in the body.
//       %transformed_i = addi %i, %k : index // index `i` is offset by %k
//       %transformed_j = addi %j, %l : index // index `j` is offset by %l
//       // Every use of %i, %j is replaced with %transformed_i, %transformed_j
//       <some operations that use %transformed_i, %transformed_j>
//     }: memref<?x?xf32, #strided>, memref<?x?xf32, #strided>
//   }
// }
//
// TODO(pifon, ntv): Investigate whether mixing implicit and explicit indices
// does not lead to losing information.
void transformIndexedGenericOpIndices(
    OpBuilder &b, LinalgOp op, ArrayRef<ValueHandle *> pivs,
    const LoopIndexToRangeIndexMap &loopIndexToRangeIndex) {
  auto indexedGenericOp = dyn_cast<IndexedGenericOp>(op.getOperation());
  if (!indexedGenericOp)
    return;

  // `linalg.indexed_generic` comes in two flavours. One has a region with a
  // single block that defines the loop body. The other has a `fun` attribute
  // that refers to an existing function symbol. The `fun` function call will be
  // inserted in the loop body in that case.
  //
  // TODO(pifon): Add support for `linalg.indexed_generic` with `fun` attribute.
  auto &region = indexedGenericOp.region();
  if (region.empty()) {
    indexedGenericOp.emitError("op expected a region");
    return;
  }
  auto &block = region.getBlocks().front();

  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToStart(&block);
  for (unsigned i = 0; i < indexedGenericOp.getNumLoops(); ++i) {
    auto rangeIndex = loopIndexToRangeIndex.find(i);
    if (rangeIndex == loopIndexToRangeIndex.end())
      continue;
    Value *oldIndex = block.getArgument(i);
    // Offset the index argument `i` by the value of the corresponding induction
    // variable and replace all uses of the previous value.
    Value *newIndex = b.create<AddIOp>(indexedGenericOp.getLoc(), oldIndex,
                                       pivs[rangeIndex->second]->getValue());
    for (auto &use : oldIndex->getUses()) {
      if (use.getOwner() == newIndex->getDefiningOp())
        continue;
      use.set(newIndex);
    }
  }
}

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
               ArrayRef<Value *> viewSizes, OperationFolder *folder) {
  assert(ivs.size() == static_cast<size_t>(llvm::count_if(
                           llvm::make_range(tileSizes.begin(), tileSizes.end()),
                           [](Value *v) { return !isZero(v); })) &&
         "expected as many ivs as non-zero sizes");

  using edsc::intrinsics::select;
  using edsc::op::operator+;
  using edsc::op::operator<;

  // Construct (potentially temporary) mins and maxes on which to apply maps
  // that define tile subviews.
  SmallVector<Value *, 8> lbs, subViewSizes;
  for (unsigned idx = 0, idxIvs = 0, e = tileSizes.size(); idx < e; ++idx) {
    bool isTiled = !isZero(tileSizes[idx]);
    lbs.push_back(isTiled ? ivs[idxIvs++] : (Value *)constant_index(folder, 0));
    subViewSizes.push_back(isTiled ? tileSizes[idx] : viewSizes[idx]);
  }

  auto *op = linalgOp.getOperation();

  SmallVector<Value *, 4> res;
  res.reserve(op->getNumOperands());
  auto viewIteratorBegin = linalgOp.getInputsAndOutputs().begin();
  for (unsigned viewIndex = 0; viewIndex < linalgOp.getNumInputsAndOutputs();
       ++viewIndex) {
    Value *view = *(viewIteratorBegin + viewIndex);
    unsigned rank = view->getType().cast<MemRefType>().getRank();
    auto map = loopToOperandRangesMaps(linalgOp)[viewIndex];
    // If the view is not tiled, we can use it as is.
    if (!isTiled(map, tileSizes)) {
      res.push_back(view);
      continue;
    }

    // Construct a new subview for the tile.
    SmallVector<Value *, 4> offsets, sizes, strides;
    offsets.reserve(rank);
    sizes.reserve(rank);
    strides.reserve(rank);
    for (unsigned r = 0; r < rank; ++r) {
      if (!isTiled(map.getSubMap({r}), tileSizes)) {
        offsets.push_back(constant_index(folder, 0));
        sizes.push_back(dim(view, r));
        strides.push_back(constant_index(folder, 1));
        continue;
      }

      // Tiling creates a new slice at the proper index, the slice step is 1
      // (i.e. the slice view does not subsample, stepping occurs in the loop).
      auto m = map.getSubMap({r});
      auto *offset = applyMapToValues(b, loc, m, lbs, folder).front();
      offsets.push_back(offset);
      auto *size = applyMapToValues(b, loc, m, subViewSizes, folder).front();
      sizes.push_back(size);
      strides.push_back(constant_index(folder, 1));
    }
    // TODO(b/144419024) Atm std.subview is not guaranteed in-bounds. Depending
    // on the semantics we attach to it, we may need to use min(size, dim) here
    // and canonicalize later.
    res.push_back(b.create<SubViewOp>(loc, view, offsets, sizes, strides));
  }

  // Traverse the mins/maxes and erase those that don't have uses left.
  // This is a special type of folding that we only apply when `folder` is
  // defined.
  if (folder)
    for (auto *v : llvm::concat<Value *>(lbs, subViewSizes))
      if (v->use_empty())
        v->getDefiningOp()->erase();

  return res;
}

llvm::Optional<TiledLinalgOp> mlir::linalg::tileLinalgOp(
    OpBuilder &b, LinalgOp op, ArrayRef<Value *> tileSizes,
    ArrayRef<unsigned> permutation, OperationFolder *folder) {
  // 1. Enforce the convention that "tiling by zero" skips tiling a particular
  // dimension. This convention is significantly simpler to handle instead of
  // adjusting affine maps to account for missing dimensions.
  assert(op.getNumParallelLoops() + op.getNumReductionLoops() +
                 op.getNumWindowLoops() ==
             tileSizes.size() &&
         "expected matching number of tile sizes and loops");

  // If permutation is empty, use the identity. Build the permutation map
  // otherwise.
  auto invPermutationMap = AffineMap::getMultiDimIdentityMap(
      tileSizes.size(), ScopedContext::getContext());
  if (!permutation.empty())
    invPermutationMap = inversePermutation(
        AffineMap::getPermutationMap(permutation, ScopedContext::getContext()));

  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  ScopedContext scope(b, op.getLoc());
  // 2. Build the tiled loop ranges.
  auto viewSizes = getViewSizes(op);
  // The flattened loopToOperandRangesMaps is expected to be an invertible
  // permutation map (asserted in the inverse calculation).
  auto viewSizesToLoopsMap =
      inversePermutation(concatAffineMaps(loopToOperandRangesMaps(op)));
  assert(viewSizesToLoopsMap && "expected invertible map");

  SmallVector<SubViewOp::Range, 4> loopRanges;
  LoopIndexToRangeIndexMap loopIndexToRangeIndex;
  std::tie(loopRanges, loopIndexToRangeIndex) =
      makeTiledLoopRanges(b, scope.getLocation(), viewSizesToLoopsMap,
                          viewSizes, tileSizes, folder);
  if (!permutation.empty())
    applyPermutationToVector(loopRanges, permutation);

  // 3. Create the tiled loops.
  LinalgOp res = op;
  SmallVector<IndexHandle, 4> ivs(loopRanges.size());
  auto pivs = makeHandlePointers(MutableArrayRef<IndexHandle>(ivs));
  LoopNestRangeBuilder(pivs, loopRanges)([&] {
    auto b = ScopedContext::getBuilder();
    auto loc = ScopedContext::getLocation();
    SmallVector<Value *, 4> ivValues(ivs.begin(), ivs.end());

    // If we have to apply a permutation to the tiled loop nest, we have to
    // reorder the induction variables This permutation is the right one
    // assuming that loopRanges have previously been permuted by
    // (i,j,k)->(k,i,j) So this permutation should be the inversePermutation of
    // that one: (d0,d1,d2)->(d2,d0,d1)
    if (!permutation.empty())
      ivValues = applyMapToValues(b, loc, invPermutationMap, ivValues, folder);

    auto views =
        makeTiledViews(b, loc, op, ivValues, tileSizes, viewSizes, folder);
    auto operands = getAssumedNonViewOperands(op);
    views.append(operands.begin(), operands.end());
    res = op.clone(b, loc, views);
  });

  // 4. Transforms index arguments of `linalg.generic` w.r.t. to the tiling.
  transformIndexedGenericOpIndices(b, res, pivs, loopIndexToRangeIndex);

  // 5. Gather the newly created loops and return them with the new op.
  SmallVector<ForOp, 8> loops;
  loops.reserve(ivs.size());
  for (auto iv : ivs)
    loops.push_back(loop::getForInductionVarOwner(iv));

  return TiledLinalgOp{res, loops};
}

llvm::Optional<TiledLinalgOp> mlir::linalg::tileLinalgOp(
    OpBuilder &b, LinalgOp op, ArrayRef<int64_t> tileSizes,
    ArrayRef<unsigned> permutation, OperationFolder *folder) {
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
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  ScopedContext scope(b, op.getLoc());

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

  return tileLinalgOp(b, op, tileSizeValues, permutation, folder);
}

static void tileLinalgOps(FuncOp f, ArrayRef<int64_t> tileSizes) {
  OpBuilder b(f);
  OperationFolder folder(f.getContext());
  f.walk([tileSizes, &b, &folder](LinalgOp op) {
    auto opLoopsPair =
        tileLinalgOp(b, op, tileSizes, /*permutation=*/{}, &folder);
    // If tiling occurred successfully, erase old op.
    if (opLoopsPair)
      op.erase();
  });
  f.walk([](LinalgOp op) {
    if (!op.getOperation()->hasNoSideEffect())
      return;
    if (op.getOperation()->use_empty())
      op.erase();
  });
}

namespace {
struct LinalgTilingPass : public FunctionPass<LinalgTilingPass> {
  LinalgTilingPass() = default;
  LinalgTilingPass(ArrayRef<int64_t> sizes);

  void runOnFunction() override { tileLinalgOps(getFunction(), tileSizes); }

  SmallVector<int64_t, 8> tileSizes;
};
} // namespace

LinalgTilingPass::LinalgTilingPass(ArrayRef<int64_t> sizes) {
  this->tileSizes.assign(sizes.begin(), sizes.end());
}

std::unique_ptr<OpPassBase<FuncOp>>
mlir::linalg::createLinalgTilingPass(ArrayRef<int64_t> tileSizes) {
  return std::make_unique<LinalgTilingPass>(tileSizes);
}

static PassRegistration<LinalgTilingPass>
    pass("linalg-tile", "Tile operations in the linalg dialect", [] {
      auto pass = std::make_unique<LinalgTilingPass>();
      pass->tileSizes.assign(clTileSizes.begin(), clTileSizes.end());
      return pass;
    });

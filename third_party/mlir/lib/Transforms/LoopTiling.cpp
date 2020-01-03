//===- LoopTiling.cpp --- Loop tiling pass ------------------------------*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to tile loop nests.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
using namespace mlir;

#define DEBUG_TYPE "affine-loop-tile"

static llvm::cl::OptionCategory clOptionsCategory(DEBUG_TYPE " options");

static llvm::cl::opt<unsigned long long>
    clCacheSizeKiB("tile-cache-size",
                   llvm::cl::desc("Set size of cache to tile for in KiB"),
                   llvm::cl::cat(clOptionsCategory));

// Tile size to use for all loops (overrides -tile-sizes if provided).
static llvm::cl::opt<unsigned>
    clTileSize("tile-size", llvm::cl::desc("Use this tile size for all loops"),
               llvm::cl::cat(clOptionsCategory));

// List of tile sizes. If any of them aren't provided, they are filled with
// clTileSize / kDefaultTileSize.
static llvm::cl::list<unsigned> clTileSizes(
    "tile-sizes",
    llvm::cl::desc(
        "List of tile sizes for each perfect nest (overridden by -tile-size)"),
    llvm::cl::ZeroOrMore, llvm::cl::cat(clOptionsCategory));

namespace {

/// A pass to perform loop tiling on all suitable loop nests of a Function.
struct LoopTiling : public FunctionPass<LoopTiling> {
  explicit LoopTiling(uint64_t cacheSizeBytes = kDefaultCacheMemCapacity,
                      bool avoidMaxMinBounds = true)
      : cacheSizeBytes(cacheSizeBytes), avoidMaxMinBounds(avoidMaxMinBounds) {}

  void runOnFunction() override;
  void getTileSizes(ArrayRef<AffineForOp> band,
                    SmallVectorImpl<unsigned> *tileSizes);

  // Default tile size if nothing is provided.
  constexpr static unsigned kDefaultTileSize = 4;
  constexpr static uint64_t kDefaultCacheMemCapacity = 512 * 1024UL;

  // Capacity of the cache to tile for.
  uint64_t cacheSizeBytes;
  // If true, tile sizes are set to avoid max/min in bounds if possible.
  bool avoidMaxMinBounds;
};

} // end anonymous namespace

/// Creates a pass to perform loop tiling on all suitable loop nests of a
/// Function.
std::unique_ptr<OpPassBase<FuncOp>>
mlir::createLoopTilingPass(uint64_t cacheSizeBytes) {
  return std::make_unique<LoopTiling>(cacheSizeBytes);
}

// Move the loop body of AffineForOp 'src' from 'src' into the specified
// location in destination's body, ignoring the terminator.
static inline void moveLoopBody(AffineForOp src, AffineForOp dest,
                                Block::iterator loc) {
  auto &insts = src.getBody()->getOperations();
  dest.getBody()->getOperations().splice(loc, insts, insts.begin(),
                                         std::prev(insts.end()));
}

// Move the loop body of AffineForOp 'src' from 'src' to the start of dest's
// body.
static inline void moveLoopBody(AffineForOp src, AffineForOp dest) {
  moveLoopBody(src, dest, dest.getBody()->begin());
}

/// Constructs and sets new loop bounds after tiling for the case of
/// hyper-rectangular index sets, where the bounds of one dimension do not
/// depend on other dimensions. Bounds of each dimension can thus be treated
/// independently, and deriving the new bounds is much simpler and faster
/// than for the case of tiling arbitrary polyhedral shapes.
static void
constructTiledIndexSetHyperRect(MutableArrayRef<AffineForOp> origLoops,
                                MutableArrayRef<AffineForOp> newLoops,
                                ArrayRef<unsigned> tileSizes) {
  assert(!origLoops.empty());
  assert(origLoops.size() == tileSizes.size());

  OpBuilder b(origLoops[0].getOperation());
  unsigned width = origLoops.size();

  // Bounds for tile space loops.
  for (unsigned i = 0; i < width; i++) {
    auto lbOperands = origLoops[i].getLowerBoundOperands();
    auto ubOperands = origLoops[i].getUpperBoundOperands();
    SmallVector<Value, 4> newLbOperands(lbOperands);
    SmallVector<Value, 4> newUbOperands(ubOperands);
    newLoops[i].setLowerBound(newLbOperands, origLoops[i].getLowerBoundMap());
    newLoops[i].setUpperBound(newUbOperands, origLoops[i].getUpperBoundMap());
    newLoops[i].setStep(tileSizes[i]);
  }
  // Bounds for intra-tile loops.
  for (unsigned i = 0; i < width; i++) {
    int64_t largestDiv = getLargestDivisorOfTripCount(origLoops[i]);
    auto mayBeConstantCount = getConstantTripCount(origLoops[i]);
    // The lower bound is just the tile-space loop.
    AffineMap lbMap = b.getDimIdentityMap();
    newLoops[width + i].setLowerBound(
        /*operands=*/newLoops[i].getInductionVar(), lbMap);

    // Set the upper bound.
    if (mayBeConstantCount.hasValue() &&
        mayBeConstantCount.getValue() < tileSizes[i]) {
      // Trip count is less than tile size; upper bound is the trip count.
      auto ubMap = b.getConstantAffineMap(mayBeConstantCount.getValue());
      newLoops[width + i].setUpperBoundMap(ubMap);
    } else if (largestDiv % tileSizes[i] != 0) {
      // Intra-tile loop ii goes from i to min(i + tileSize, ub_i).
      // Construct the upper bound map; the operands are the original operands
      // with 'i' (tile-space loop) appended to it. The new upper bound map is
      // the original one with an additional expression i + tileSize appended.
      auto ub = origLoops[i].getUpperBound();
      SmallVector<Value, 4> ubOperands;
      ubOperands.reserve(ub.getNumOperands() + 1);
      auto origUbMap = ub.getMap();
      // Add dim operands from original upper bound.
      for (unsigned j = 0, e = origUbMap.getNumDims(); j < e; ++j) {
        ubOperands.push_back(ub.getOperand(j));
      }
      // Add dim operand for new loop upper bound.
      ubOperands.push_back(newLoops[i].getInductionVar());
      // Add symbol operands from original upper bound.
      for (unsigned j = 0, e = origUbMap.getNumSymbols(); j < e; ++j) {
        ubOperands.push_back(ub.getOperand(origUbMap.getNumDims() + j));
      }
      SmallVector<AffineExpr, 4> boundExprs;
      boundExprs.reserve(1 + origUbMap.getNumResults());
      auto dim = b.getAffineDimExpr(origUbMap.getNumDims());
      // The new upper bound map is the original one with an additional
      // expression i + tileSize appended.
      boundExprs.push_back(dim + tileSizes[i]);
      boundExprs.append(origUbMap.getResults().begin(),
                        origUbMap.getResults().end());
      auto ubMap = AffineMap::get(origUbMap.getNumDims() + 1,
                                  origUbMap.getNumSymbols(), boundExprs);
      newLoops[width + i].setUpperBound(/*operands=*/ubOperands, ubMap);
    } else {
      // No need of the min expression.
      auto dim = b.getAffineDimExpr(0);
      auto ubMap = AffineMap::get(1, 0, dim + tileSizes[i]);
      newLoops[width + i].setUpperBound(newLoops[i].getInductionVar(), ubMap);
    }
  }
}

/// Tiles the specified band of perfectly nested loops creating tile-space loops
/// and intra-tile loops. A band is a contiguous set of loops.
//  TODO(bondhugula): handle non hyper-rectangular spaces.
LogicalResult mlir::tileCodeGen(MutableArrayRef<AffineForOp> band,
                                ArrayRef<unsigned> tileSizes) {
  assert(!band.empty());
  assert(band.size() == tileSizes.size() && "Incorrect number of tile sizes");

  // Check if the supplied for op's are all successively nested.
  for (unsigned i = 1, e = band.size(); i < e; i++) {
    assert(band[i].getParentOp() == band[i - 1].getOperation());
  }

  auto origLoops = band;

  AffineForOp rootAffineForOp = origLoops[0];
  auto loc = rootAffineForOp.getLoc();
  // Note that width is at least one since band isn't empty.
  unsigned width = band.size();

  SmallVector<AffineForOp, 12> newLoops(2 * width);
  AffineForOp innermostPointLoop;

  // The outermost among the loops as we add more..
  auto *topLoop = rootAffineForOp.getOperation();

  // Add intra-tile (or point) loops.
  for (unsigned i = 0; i < width; i++) {
    OpBuilder b(topLoop);
    // Loop bounds will be set later.
    auto pointLoop = b.create<AffineForOp>(loc, 0, 0);
    pointLoop.getBody()->getOperations().splice(
        pointLoop.getBody()->begin(), topLoop->getBlock()->getOperations(),
        topLoop);
    newLoops[2 * width - 1 - i] = pointLoop;
    topLoop = pointLoop.getOperation();
    if (i == 0)
      innermostPointLoop = pointLoop;
  }

  // Add tile space loops;
  for (unsigned i = width; i < 2 * width; i++) {
    OpBuilder b(topLoop);
    // Loop bounds will be set later.
    auto tileSpaceLoop = b.create<AffineForOp>(loc, 0, 0);
    tileSpaceLoop.getBody()->getOperations().splice(
        tileSpaceLoop.getBody()->begin(), topLoop->getBlock()->getOperations(),
        topLoop);
    newLoops[2 * width - i - 1] = tileSpaceLoop;
    topLoop = tileSpaceLoop.getOperation();
  }

  // Move the loop body of the original nest to the new one.
  moveLoopBody(origLoops[origLoops.size() - 1], innermostPointLoop);

  SmallVector<Value, 8> origLoopIVs;
  extractForInductionVars(band, &origLoopIVs);
  SmallVector<Optional<Value>, 6> ids(origLoopIVs.begin(), origLoopIVs.end());
  FlatAffineConstraints cst;
  getIndexSet(band, &cst);

  if (!cst.isHyperRectangular(0, width)) {
    rootAffineForOp.emitError("tiled code generation unimplemented for the "
                              "non-hyperrectangular case");
    return failure();
  }

  constructTiledIndexSetHyperRect(origLoops, newLoops, tileSizes);
  // In this case, the point loop IVs just replace the original ones.
  for (unsigned i = 0; i < width; i++) {
    origLoopIVs[i]->replaceAllUsesWith(newLoops[i + width].getInductionVar());
  }

  // Erase the old loop nest.
  rootAffineForOp.erase();

  return success();
}

// Identify valid and profitable bands of loops to tile. This is currently just
// a temporary placeholder to test the mechanics of tiled code generation.
// Returns all maximal outermost perfect loop nests to tile.
static void getTileableBands(FuncOp f,
                             std::vector<SmallVector<AffineForOp, 6>> *bands) {
  // Get maximal perfect nest of 'affine.for' insts starting from root
  // (inclusive).
  auto getMaximalPerfectLoopNest = [&](AffineForOp root) {
    SmallVector<AffineForOp, 6> band;
    getPerfectlyNestedLoops(band, root);
    bands->push_back(band);
  };

  for (auto &block : f)
    for (auto &op : block)
      if (auto forOp = dyn_cast<AffineForOp>(op))
        getMaximalPerfectLoopNest(forOp);
}

// Reduce each tile size to the largest divisor of the corresponding trip count
// (if the trip count is known).
static void adjustToDivisorsOfTripCounts(ArrayRef<AffineForOp> band,
                                         SmallVectorImpl<unsigned> *tileSizes) {
  assert(band.size() == tileSizes->size() && "invalid tile size count");
  for (unsigned i = 0, e = band.size(); i < e; i++) {
    unsigned &tSizeAdjusted = (*tileSizes)[i];
    auto mayConst = getConstantTripCount(band[i]);
    if (!mayConst.hasValue())
      continue;
    // Adjust the tile size to largest factor of the trip count less than
    // tSize.
    uint64_t constTripCount = mayConst.getValue();
    if (constTripCount > 1 && tSizeAdjusted > constTripCount / 2)
      tSizeAdjusted = constTripCount / 2;
    while (constTripCount % tSizeAdjusted != 0)
      tSizeAdjusted--;
  }
}

// Returns tile sizes to use. Checks CL options; if none are specified, sets it
// based on a simple model that looks at the memory footprint and determines
// tile sizes assuming identity accesses / 1:1 tile size proportional footprint
// along each of the dimensions being tiled.
// TODO(mlir-team): evolve this model. Tile size determination is a large area
// to play with in general.
void LoopTiling::getTileSizes(ArrayRef<AffineForOp> band,
                              SmallVectorImpl<unsigned> *tileSizes) {
  if (band.empty())
    return;

  tileSizes->resize(band.size());

  // Use clTileSize for all loops if specified.
  if (clTileSize.getNumOccurrences() > 0) {
    std::fill(tileSizes->begin(), tileSizes->end(), clTileSize);
    return;
  }

  // Use clTileSizes and fill them with default tile size if it's short.
  if (!clTileSizes.empty()) {
    std::fill(tileSizes->begin(), tileSizes->end(),
              LoopTiling::kDefaultTileSize);
    std::copy(clTileSizes.begin(),
              clTileSizes.begin() + std::min(clTileSizes.size(), band.size()),
              tileSizes->begin());
    return;
  }

  // The first loop in the band.
  auto rootForOp = band[0];
  (void)rootForOp;

  // Obtain memory footprint and set tile sizes so that a tile fits in
  // the cache size. This is an approximation with the assumption that the
  // footprint increases with the tile size linearly in that dimension (i.e.,
  // assumes one-to-one access function).
  auto fp = getMemoryFootprintBytes(band[0], 0);
  if (!fp.hasValue()) {
    // Fill with default tile sizes if footprint is unknown.
    std::fill(tileSizes->begin(), tileSizes->end(),
              LoopTiling::kDefaultTileSize);
    if (avoidMaxMinBounds)
      adjustToDivisorsOfTripCounts(band, tileSizes);
    LLVM_DEBUG(
        rootForOp.emitWarning("memory footprint unknown: using default tile "
                              "sizes adjusted to trip count divisors"));
    return;
  }

  // Check how many times larger the cache size is when compared to footprint.
  uint64_t excessFactor = llvm::divideCeil(fp.getValue(), cacheSizeBytes);
  if (excessFactor <= 1) {
    // No need of any tiling - set tile size to 1.
    std::fill(tileSizes->begin(), tileSizes->end(), 1);
    return;
  }

  // Divide all loops equally in an attempt to reduce footprint.
  // TODO(bondhugula): this is approximate. Ideally, obtain reuse factor /
  // profitability along each dimension and weight tile sizes based on that as
  // one possible approach. Or compute a polynomial in tile sizes and solve for
  // it.

  // For an n-d tileable band, compute n^th root of the excess.
  unsigned tSize =
      static_cast<unsigned>(floorl(std::pow(excessFactor, 1.0 / band.size())));
  // We'll keep a running product to determine the last tile size better.
  unsigned cumulProductOfTileSizes = 1;
  for (unsigned i = 0, e = band.size(); i < e; i++) {
    if (i < e - 1)
      (*tileSizes)[i] = tSize;
    else
      // Set last tile size to cover the balance.
      (*tileSizes)[i] = std::max(
          1U, static_cast<unsigned>(excessFactor / cumulProductOfTileSizes));
    cumulProductOfTileSizes *= (*tileSizes)[i];
  }
  if (avoidMaxMinBounds)
    adjustToDivisorsOfTripCounts(band, tileSizes);
}

void LoopTiling::runOnFunction() {
  // Override cache size if provided on command line.
  if (clCacheSizeKiB.getNumOccurrences() > 0)
    cacheSizeBytes = clCacheSizeKiB * 1024;

  // Bands of loops to tile.
  std::vector<SmallVector<AffineForOp, 6>> bands;
  getTileableBands(getFunction(), &bands);

  for (auto &band : bands) {
    // Set up tile sizes; fill missing tile sizes at the end with default tile
    // size or clTileSize if one was provided.
    SmallVector<unsigned, 6> tileSizes;
    getTileSizes(band, &tileSizes);
    if (llvm::DebugFlag) {
      auto diag = band[0].emitRemark("using tile sizes [");
      for (auto tSize : tileSizes)
        diag << tSize << " ";
      diag << "]\n";
    }
    if (failed(tileCodeGen(band, tileSizes)))
      return signalPassFailure();
  }
}

constexpr unsigned LoopTiling::kDefaultTileSize;
constexpr uint64_t LoopTiling::kDefaultCacheMemCapacity;

static PassRegistration<LoopTiling> pass("affine-loop-tile", "Tile loop nests");

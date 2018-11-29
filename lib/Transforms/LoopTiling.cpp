//===- LoopTiling.cpp --- Loop tiling pass ------------------------------*-===//
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
// This file implements a pass to tile loop nests.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

// Tile size for all loops.
static llvm::cl::opt<unsigned>
    clTileSize("tile-size", llvm::cl::Hidden,
               llvm::cl::desc("Use this tile size for all loops"));

namespace {

/// A pass to perform loop tiling on all suitable loop nests of an MLFunction.
struct LoopTiling : public FunctionPass {
  LoopTiling() : FunctionPass(&LoopTiling::passID) {}
  PassResult runOnMLFunction(MLFunction *f) override;
  constexpr static unsigned kDefaultTileSize = 4;

  static char passID;
};

} // end anonymous namespace

char LoopTiling::passID = 0;

/// Creates a pass to perform loop tiling on all suitable loop nests of an
/// MLFunction.
FunctionPass *mlir::createLoopTilingPass() { return new LoopTiling(); }

// Move the loop body of ForStmt 'src' from 'src' into the specified location in
// destination's body.
static inline void moveLoopBody(ForStmt *src, ForStmt *dest,
                                StmtBlock::iterator loc) {
  dest->getStatements().splice(loc, src->getStatements());
}

// Move the loop body of ForStmt 'src' from 'src' to the start of dest's body.
static inline void moveLoopBody(ForStmt *src, ForStmt *dest) {
  moveLoopBody(src, dest, dest->begin());
}

/// Constructs/sets new loop bounds after tiling for the case of
/// hyper-rectangular index sets, where the bounds of one dimension do not
/// depend on other dimensions. Bounds of each dimension can thus be treated
/// independently, and deriving the new bounds is much simpler and faster
/// than for the case of tiling arbitrary polyhedral shapes.
static bool setTiledIndexSetHyperRect(ArrayRef<ForStmt *> origLoops,
                                      ArrayRef<ForStmt *> newLoops,
                                      ArrayRef<unsigned> tileSizes) {
  assert(!origLoops.empty());
  assert(origLoops.size() == tileSizes.size());

  MLFuncBuilder b(origLoops[0]);
  unsigned width = origLoops.size();

  // Bounds for tile space loops.
  for (unsigned i = 0; i < width; i++) {
    auto lbOperands = origLoops[i]->getLowerBoundOperands();
    auto ubOperands = origLoops[i]->getUpperBoundOperands();
    SmallVector<MLValue *, 4> newLbOperands(lbOperands.begin(),
                                            lbOperands.end());
    SmallVector<MLValue *, 4> newUbOperands(ubOperands.begin(),
                                            ubOperands.end());
    newLoops[i]->setLowerBound(newLbOperands, origLoops[i]->getLowerBoundMap());
    newLoops[i]->setUpperBound(newUbOperands, origLoops[i]->getUpperBoundMap());
    newLoops[i]->setStep(tileSizes[i]);
  }
  // Bounds for intra-tile loops.
  for (unsigned i = 0; i < width; i++) {
    // TODO(bondhugula): Keep it simple for now - constant upper bound.
    if (!origLoops[i]->hasConstantUpperBound())
      return false;

    int64_t largestDiv = getLargestDivisorOfTripCount(*origLoops[i]);
    auto mayBeConstantCount = getConstantTripCount(*origLoops[i]);
    AffineMap lbMap, ubMap;
    auto dim = b.getAffineDimExpr(0);
    lbMap = b.getAffineMap(1, 0, dim, {});
    newLoops[width + i]->setLowerBound(newLoops[i], lbMap);

    // Set the upper bound.
    if (mayBeConstantCount.hasValue() &&
        mayBeConstantCount.getValue() < tileSizes[i]) {
      // Trip count is less than tile size; upper bound is the trip count.
      ubMap = b.getConstantAffineMap(mayBeConstantCount.getValue());
      newLoops[width + i]->setUpperBoundMap(ubMap);
    } else if (largestDiv % tileSizes[i] != 0) {
      // Intra-tile loop ii goes from i to min(i + tileSize, ub_i).
      auto ubMax =
          b.getAffineConstantExpr(origLoops[i]->getConstantUpperBound());
      ubMap = b.getAffineMap(1, 0, {dim + tileSizes[i], ubMax}, {});
      newLoops[width + i]->setUpperBound(newLoops[i], ubMap);
    } else {
      // No need of the min expression.
      ubMap = b.getAffineMap(1, 0, dim + tileSizes[i], {});
      newLoops[width + i]->setUpperBound(newLoops[i], ubMap);
    }
  }
  return true;
}

/// Tiles the specified band of perfectly nested loops creating tile-space loops
/// and intra-tile loops. A band is a contiguous set of loops.
//  TODO(bondhugula): handle non-constant bounds.
//  TODO(bondhugula): handle non hyper-rectangular spaces.
UtilResult mlir::tileCodeGen(ArrayRef<ForStmt *> band,
                             ArrayRef<unsigned> tileSizes) {
  assert(!band.empty());
  assert(band.size() == tileSizes.size());
  // Check if the supplied for stmt's are all successively nested.
  for (unsigned i = 1, e = band.size(); i < e; i++) {
    assert(band[i]->getParentStmt() == band[i - 1]);
  }

  auto origLoops = band;

  ForStmt *rootForStmt = origLoops[0];
  auto loc = rootForStmt->getLoc();
  // Note that width is at least one since band isn't empty.
  unsigned width = band.size();

  SmallVector<ForStmt *, 12> newLoops(2 * width);
  ForStmt *innermostPointLoop;

  // The outermost among the loops as we add more..
  auto *topLoop = rootForStmt;

  // Add intra-tile (or point) loops.
  for (unsigned i = 0; i < width; i++) {
    MLFuncBuilder b(topLoop);
    // Loop bounds will be set later.
    auto *pointLoop = b.createFor(loc, 0, 0);
    pointLoop->getStatements().splice(
        pointLoop->begin(), topLoop->getBlock()->getStatements(), topLoop);
    newLoops[2 * width - 1 - i] = pointLoop;
    topLoop = pointLoop;
    if (i == 0)
      innermostPointLoop = pointLoop;
  }

  // Add tile space loops;
  for (unsigned i = width; i < 2 * width; i++) {
    MLFuncBuilder b(topLoop);
    // Loop bounds will be set later.
    auto *tileSpaceLoop = b.createFor(loc, 0, 0);
    tileSpaceLoop->getStatements().splice(
        tileSpaceLoop->begin(), topLoop->getBlock()->getStatements(), topLoop);
    newLoops[2 * width - i - 1] = tileSpaceLoop;
    topLoop = tileSpaceLoop;
  }

  // Move the loop body of the original nest to the new one.
  moveLoopBody(origLoops[origLoops.size() - 1], innermostPointLoop);

  SmallVector<MLValue *, 6> origLoopIVs(band.begin(), band.end());

  FlatAffineConstraints cst(width, 0);
  addIndexSet(origLoopIVs, &cst);
  if (cst.isHyperRectangular(0, width)) {
    if (!setTiledIndexSetHyperRect(origLoops, newLoops, tileSizes)) {
      rootForStmt->emitError(
          "tiled code generation unimplemented for this case");
      return UtilResult::Failure;
    }
    // In this case, the point loop IVs just replace the original ones.
    for (unsigned i = 0; i < width; i++) {
      origLoopIVs[i]->replaceAllUsesWith(newLoops[i + width]);
    }
  } else {
    rootForStmt->emitError("tiled code generation unimplemented for this case");
    return UtilResult::Failure;
  }

  // Erase the old loop nest.
  rootForStmt->erase();

  return UtilResult::Success;
}

// Identify valid and profitable bands of loops to tile. This is currently just
// a temporary placeholder to test the mechanics of tiled code generation.
// Returns all maximal outermost perfect loop nests to tile.
static void getTileableBands(MLFunction *f,
                             std::vector<SmallVector<ForStmt *, 6>> *bands) {
  // Get maximal perfect nest of 'for' stmts starting from root (inclusive).
  auto getMaximalPerfectLoopNest = [&](ForStmt *root) {
    SmallVector<ForStmt *, 6> band;
    ForStmt *currStmt = root;
    do {
      band.push_back(currStmt);
    } while (currStmt->getStatements().size() == 1 &&
             (currStmt = dyn_cast<ForStmt>(&*currStmt->begin())));
    bands->push_back(band);
  };

  for (auto &stmt : *f) {
    ForStmt *forStmt = dyn_cast<ForStmt>(&stmt);
    if (!forStmt)
      continue;
    getMaximalPerfectLoopNest(forStmt);
  }
}

PassResult LoopTiling::runOnMLFunction(MLFunction *f) {
  std::vector<SmallVector<ForStmt *, 6>> bands;
  getTileableBands(f, &bands);

  // Temporary tile sizes.
  unsigned tileSize =
      clTileSize.getNumOccurrences() > 0 ? clTileSize : kDefaultTileSize;

  for (const auto &band : bands) {
    SmallVector<unsigned, 6> tileSizes(band.size(), tileSize);
    if (tileCodeGen(band, tileSizes)) {
      return failure();
    }
  }
  return success();
}

static PassRegistration<LoopTiling> pass("loop-tile", "Tile loop nests");

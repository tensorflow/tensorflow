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

#include "mlir/AffineOps/AffineOps.h"
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

#define DEBUG_TYPE "loop-tile"

static llvm::cl::OptionCategory clOptionsCategory(DEBUG_TYPE " options");

// Tile size for all loops.
static llvm::cl::opt<unsigned>
    clTileSize("tile-size", llvm::cl::Hidden,
               llvm::cl::desc("Use this tile size for all loops"),
               llvm::cl::cat(clOptionsCategory));

namespace {

/// A pass to perform loop tiling on all suitable loop nests of a Function.
struct LoopTiling : public FunctionPass {
  LoopTiling() : FunctionPass(&LoopTiling::passID) {}
  PassResult runOnFunction(Function *f) override;
  constexpr static unsigned kDefaultTileSize = 4;

  static char passID;
};

} // end anonymous namespace

char LoopTiling::passID = 0;

/// Creates a pass to perform loop tiling on all suitable loop nests of an
/// Function.
FunctionPass *mlir::createLoopTilingPass() { return new LoopTiling(); }

// Move the loop body of AffineForOp 'src' from 'src' into the specified
// location in destination's body.
static inline void moveLoopBody(AffineForOp *src, AffineForOp *dest,
                                Block::iterator loc) {
  dest->getBody()->getInstructions().splice(loc,
                                            src->getBody()->getInstructions());
}

// Move the loop body of AffineForOp 'src' from 'src' to the start of dest's
// body.
static inline void moveLoopBody(AffineForOp *src, AffineForOp *dest) {
  moveLoopBody(src, dest, dest->getBody()->begin());
}

/// Constructs and sets new loop bounds after tiling for the case of
/// hyper-rectangular index sets, where the bounds of one dimension do not
/// depend on other dimensions. Bounds of each dimension can thus be treated
/// independently, and deriving the new bounds is much simpler and faster
/// than for the case of tiling arbitrary polyhedral shapes.
static void constructTiledIndexSetHyperRect(
    MutableArrayRef<OpPointer<AffineForOp>> origLoops,
    MutableArrayRef<OpPointer<AffineForOp>> newLoops,
    ArrayRef<unsigned> tileSizes) {
  assert(!origLoops.empty());
  assert(origLoops.size() == tileSizes.size());

  FuncBuilder b(origLoops[0]->getInstruction());
  unsigned width = origLoops.size();

  // Bounds for tile space loops.
  for (unsigned i = 0; i < width; i++) {
    auto lbOperands = origLoops[i]->getLowerBoundOperands();
    auto ubOperands = origLoops[i]->getUpperBoundOperands();
    SmallVector<Value *, 4> newLbOperands(lbOperands);
    SmallVector<Value *, 4> newUbOperands(ubOperands);
    newLoops[i]->setLowerBound(newLbOperands, origLoops[i]->getLowerBoundMap());
    newLoops[i]->setUpperBound(newUbOperands, origLoops[i]->getUpperBoundMap());
    newLoops[i]->setStep(tileSizes[i]);
  }
  // Bounds for intra-tile loops.
  for (unsigned i = 0; i < width; i++) {
    int64_t largestDiv = getLargestDivisorOfTripCount(origLoops[i]);
    auto mayBeConstantCount = getConstantTripCount(origLoops[i]);
    // The lower bound is just the tile-space loop.
    AffineMap lbMap = b.getDimIdentityMap();
    newLoops[width + i]->setLowerBound(
        /*operands=*/newLoops[i]->getInductionVar(), lbMap);

    // Set the upper bound.
    if (mayBeConstantCount.hasValue() &&
        mayBeConstantCount.getValue() < tileSizes[i]) {
      // Trip count is less than tile size; upper bound is the trip count.
      auto ubMap = b.getConstantAffineMap(mayBeConstantCount.getValue());
      newLoops[width + i]->setUpperBoundMap(ubMap);
    } else if (largestDiv % tileSizes[i] != 0) {
      // Intra-tile loop ii goes from i to min(i + tileSize, ub_i).
      // Construct the upper bound map; the operands are the original operands
      // with 'i' (tile-space loop) appended to it. The new upper bound map is
      // the original one with an additional expression i + tileSize appended.
      SmallVector<Value *, 4> ubOperands(origLoops[i]->getUpperBoundOperands());
      ubOperands.push_back(newLoops[i]->getInductionVar());

      auto origUbMap = origLoops[i]->getUpperBoundMap();
      SmallVector<AffineExpr, 4> boundExprs;
      boundExprs.reserve(1 + origUbMap.getNumResults());
      auto dim = b.getAffineDimExpr(origUbMap.getNumInputs());
      // The new upper bound map is the original one with an additional
      // expression i + tileSize appended.
      boundExprs.push_back(dim + tileSizes[i]);
      boundExprs.append(origUbMap.getResults().begin(),
                        origUbMap.getResults().end());
      auto ubMap =
          b.getAffineMap(origUbMap.getNumInputs() + 1, 0, boundExprs, {});
      newLoops[width + i]->setUpperBound(/*operands=*/ubOperands, ubMap);
    } else {
      // No need of the min expression.
      auto dim = b.getAffineDimExpr(0);
      auto ubMap = b.getAffineMap(1, 0, dim + tileSizes[i], {});
      newLoops[width + i]->setUpperBound(newLoops[i]->getInductionVar(), ubMap);
    }
  }
}

/// Tiles the specified band of perfectly nested loops creating tile-space loops
/// and intra-tile loops. A band is a contiguous set of loops.
//  TODO(bondhugula): handle non hyper-rectangular spaces.
UtilResult mlir::tileCodeGen(MutableArrayRef<OpPointer<AffineForOp>> band,
                             ArrayRef<unsigned> tileSizes) {
  assert(!band.empty());
  assert(band.size() == tileSizes.size());
  // Check if the supplied for inst's are all successively nested.
  for (unsigned i = 1, e = band.size(); i < e; i++) {
    assert(band[i]->getInstruction()->getParentInst() ==
           band[i - 1]->getInstruction());
  }

  auto origLoops = band;

  OpPointer<AffineForOp> rootAffineForOp = origLoops[0];
  auto loc = rootAffineForOp->getLoc();
  // Note that width is at least one since band isn't empty.
  unsigned width = band.size();

  SmallVector<OpPointer<AffineForOp>, 12> newLoops(2 * width);
  OpPointer<AffineForOp> innermostPointLoop;

  // The outermost among the loops as we add more..
  auto *topLoop = rootAffineForOp->getInstruction();

  // Add intra-tile (or point) loops.
  for (unsigned i = 0; i < width; i++) {
    FuncBuilder b(topLoop);
    // Loop bounds will be set later.
    auto pointLoop = b.create<AffineForOp>(loc, 0, 0);
    pointLoop->createBody();
    pointLoop->getBody()->getInstructions().splice(
        pointLoop->getBody()->begin(), topLoop->getBlock()->getInstructions(),
        topLoop);
    newLoops[2 * width - 1 - i] = pointLoop;
    topLoop = pointLoop->getInstruction();
    if (i == 0)
      innermostPointLoop = pointLoop;
  }

  // Add tile space loops;
  for (unsigned i = width; i < 2 * width; i++) {
    FuncBuilder b(topLoop);
    // Loop bounds will be set later.
    auto tileSpaceLoop = b.create<AffineForOp>(loc, 0, 0);
    tileSpaceLoop->createBody();
    tileSpaceLoop->getBody()->getInstructions().splice(
        tileSpaceLoop->getBody()->begin(),
        topLoop->getBlock()->getInstructions(), topLoop);
    newLoops[2 * width - i - 1] = tileSpaceLoop;
    topLoop = tileSpaceLoop->getInstruction();
  }

  // Move the loop body of the original nest to the new one.
  moveLoopBody(origLoops[origLoops.size() - 1], innermostPointLoop);

  SmallVector<Value *, 8> origLoopIVs = extractForInductionVars(band);
  SmallVector<Optional<Value *>, 6> ids(origLoopIVs.begin(), origLoopIVs.end());
  FlatAffineConstraints cst;
  getIndexSet(band, &cst);

  if (!cst.isHyperRectangular(0, width)) {
    rootAffineForOp->emitError("tiled code generation unimplemented for the"
                               "non-hyperrectangular case");
    return UtilResult::Failure;
  }

  constructTiledIndexSetHyperRect(origLoops, newLoops, tileSizes);
  // In this case, the point loop IVs just replace the original ones.
  for (unsigned i = 0; i < width; i++) {
    origLoopIVs[i]->replaceAllUsesWith(newLoops[i + width]->getInductionVar());
  }

  // Erase the old loop nest.
  rootAffineForOp->erase();

  return UtilResult::Success;
}

// Identify valid and profitable bands of loops to tile. This is currently just
// a temporary placeholder to test the mechanics of tiled code generation.
// Returns all maximal outermost perfect loop nests to tile.
static void
getTileableBands(Function *f,
                 std::vector<SmallVector<OpPointer<AffineForOp>, 6>> *bands) {
  // Get maximal perfect nest of 'for' insts starting from root (inclusive).
  auto getMaximalPerfectLoopNest = [&](OpPointer<AffineForOp> root) {
    SmallVector<OpPointer<AffineForOp>, 6> band;
    OpPointer<AffineForOp> currInst = root;
    do {
      band.push_back(currInst);
    } while (currInst->getBody()->getInstructions().size() == 1 &&
             (currInst = cast<OperationInst>(currInst->getBody()->front())
                             .dyn_cast<AffineForOp>()));
    bands->push_back(band);
  };

  for (auto &block : *f)
    for (auto &inst : block)
      if (auto forOp = cast<OperationInst>(inst).dyn_cast<AffineForOp>())
        getMaximalPerfectLoopNest(forOp);
}

PassResult LoopTiling::runOnFunction(Function *f) {
  std::vector<SmallVector<OpPointer<AffineForOp>, 6>> bands;
  getTileableBands(f, &bands);

  // Temporary tile sizes.
  unsigned tileSize =
      clTileSize.getNumOccurrences() > 0 ? clTileSize : kDefaultTileSize;

  for (auto &band : bands) {
    SmallVector<unsigned, 6> tileSizes(band.size(), tileSize);
    if (tileCodeGen(band, tileSizes)) {
      return failure();
    }
  }
  return success();
}

static PassRegistration<LoopTiling> pass("loop-tile", "Tile loop nests");

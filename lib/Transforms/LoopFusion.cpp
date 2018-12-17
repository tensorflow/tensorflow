//===- LoopFusion.cpp - Code to perform loop fusion -----------------------===//
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
// This file implements loop fusion.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/StmtVisitor.h"
#include "mlir/Pass.h"
#include "mlir/StandardOps/StandardOps.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"

using llvm::SetVector;

using namespace mlir;

namespace {

/// Loop fusion pass. This pass currently supports a greedy fusion policy,
/// which fuses loop nests with single-writer/single-reader memref dependences
/// with the goal of improving locality.

// TODO(andydavis) Support fusion of source loop nests which write to multiple
// memrefs, where each memref can have multiple users (if profitable).
// TODO(andydavis) Extend this pass to check for fusion preventing dependences,
// and add support for more general loop fusion algorithms.

struct LoopFusion : public FunctionPass {
  LoopFusion() : FunctionPass(&LoopFusion::passID) {}

  PassResult runOnMLFunction(MLFunction *f) override;
  static char passID;
};

} // end anonymous namespace

char LoopFusion::passID = 0;

FunctionPass *mlir::createLoopFusionPass() { return new LoopFusion; }

static void getSingleMemRefAccess(OperationStmt *loadOrStoreOpStmt,
                                  MemRefAccess *access) {
  if (auto loadOp = loadOrStoreOpStmt->dyn_cast<LoadOp>()) {
    access->memref = cast<MLValue>(loadOp->getMemRef());
    access->opStmt = loadOrStoreOpStmt;
    auto loadMemrefType = loadOp->getMemRefType();
    access->indices.reserve(loadMemrefType.getRank());
    for (auto *index : loadOp->getIndices()) {
      access->indices.push_back(cast<MLValue>(index));
    }
  } else {
    assert(loadOrStoreOpStmt->isa<StoreOp>());
    auto storeOp = loadOrStoreOpStmt->dyn_cast<StoreOp>();
    access->opStmt = loadOrStoreOpStmt;
    access->memref = cast<MLValue>(storeOp->getMemRef());
    auto storeMemrefType = storeOp->getMemRefType();
    access->indices.reserve(storeMemrefType.getRank());
    for (auto *index : storeOp->getIndices()) {
      access->indices.push_back(cast<MLValue>(index));
    }
  }
}

// FusionCandidate encapsulates source and destination memref access within
// loop nests which are candidates for loop fusion.
struct FusionCandidate {
  // Load or store access within src loop nest to be fused into dst loop nest.
  MemRefAccess srcAccess;
  // Load or store access within dst loop nest.
  MemRefAccess dstAccess;
};

static FusionCandidate buildFusionCandidate(OperationStmt *srcStoreOpStmt,
                                            OperationStmt *dstLoadOpStmt) {
  FusionCandidate candidate;
  // Get store access for src loop nest.
  getSingleMemRefAccess(srcStoreOpStmt, &candidate.srcAccess);
  // Get load access for dst loop nest.
  getSingleMemRefAccess(dstLoadOpStmt, &candidate.dstAccess);
  return candidate;
}

namespace {

// LoopNestStateCollector walks loop nests and collects load and store
// operations, and whether or not an IfStmt was encountered in the loop nest.
class LoopNestStateCollector : public StmtWalker<LoopNestStateCollector> {
public:
  SmallVector<ForStmt *, 4> forStmts;
  SmallVector<OperationStmt *, 4> loadOpStmts;
  SmallVector<OperationStmt *, 4> storeOpStmts;
  bool hasIfStmt = false;

  void visitForStmt(ForStmt *forStmt) { forStmts.push_back(forStmt); }

  void visitIfStmt(IfStmt *ifStmt) { hasIfStmt = true; }

  void visitOperationStmt(OperationStmt *opStmt) {
    if (opStmt->isa<LoadOp>())
      loadOpStmts.push_back(opStmt);
    if (opStmt->isa<StoreOp>())
      storeOpStmts.push_back(opStmt);
  }
};

// GreedyFusionPolicy greedily fuses loop nests which have a producer/consumer
// relationship on a memref, with the goal of improving locality. Currently,
// this the producer/consumer relationship is required to be unique in the
// MLFunction (there are TODOs to relax this constraint in the future).
//
// The steps of the algorithm are as follows:
//
// *) Initialize. While visiting each statement in the MLFunction do:
//   *) Assign each top-level ForStmt a 'position' which is its initial
//      position in the MLFunction's StmtBlock at the start of the pass.
//   *) Gather memref load/store state aggregated by top-level statement. For
//      example, all loads and stores contained in a loop nest are aggregated
//      under the loop nest's top-level ForStmt.
//   *) Add each top-level ForStmt to a worklist.
//
// *) Run. The algorithm processes the worklist with the following steps:
//   *) The worklist is processed in reverse order (starting from the last
//      top-level ForStmt in the MLFunction).
//   *) Pop a ForStmt of the worklist. This 'dstForStmt' will be a candidate
//      destination ForStmt into which fusion will be attempted.
//   *) Add each LoadOp currently in 'dstForStmt' into list 'dstLoadOps'.
//   *) For each LoadOp in 'dstLoadOps' do:
//      *) Lookup dependent loop nests at earlier positions in the MLFunction
//         which have a single store op to the same memref.
//      *) Check if dependences would be violated by the fusion. For example,
//         the src loop nest may load from memrefs which are different than
//         the producer-consumer memref between src and dest loop nests.
//      *) Get a computation slice of 'srcLoopNest', which adjust its loop
//         bounds to be functions of 'dstLoopNest' IVs and symbols.
//      *) Fuse the 'srcLoopNest' computation slice into the 'dstLoopNest',
//         just before the dst load op user.
//      *) Add the newly fused load/store operation statements to the state,
//         and also add newly fuse load ops to 'dstLoopOps' to be considered
//         as fusion dst load ops in another iteration.
//      *) Remove old src loop nest and its associated state.
//
// Given a graph where top-level statements are vertices in the set 'V' and
// edges in the set 'E' are dependences between vertices, this algorithm
// takes O(V) time for initialization, and has runtime O(V * E).
// TODO(andydavis) Reduce this time complexity to O(V + E).
//
// This greedy algorithm is not 'maximally' but there is a TODO to fix this.
//
// TODO(andydavis) Experiment with other fusion policies.
struct GreedyFusionPolicy {
  // Convenience wrapper with information about 'stmt' ready to access.
  struct StmtInfo {
    Statement *stmt;
    bool isOrContainsIfStmt = false;
  };
  // The worklist of top-level loop nest positions.
  SmallVector<unsigned, 4> worklist;
  // Mapping from top-level position to StmtInfo.
  DenseMap<unsigned, StmtInfo> posToStmtInfo;
  // Mapping from memref MLValue to set of top-level positions of loop nests
  // which contain load ops on that memref.
  DenseMap<MLValue *, DenseSet<unsigned>> memrefToLoadPosSet;
  // Mapping from memref MLValue to set of top-level positions of loop nests
  // which contain store ops on that memref.
  DenseMap<MLValue *, DenseSet<unsigned>> memrefToStorePosSet;
  // Mapping from top-level loop nest to the set of load ops it contains.
  DenseMap<ForStmt *, SetVector<OperationStmt *>> forStmtToLoadOps;
  // Mapping from top-level loop nest to the set of store ops it contains.
  DenseMap<ForStmt *, SetVector<OperationStmt *>> forStmtToStoreOps;

  GreedyFusionPolicy(MLFunction *f) { init(f); }

  void run() {
    if (hasIfStmts())
      return;

    while (!worklist.empty()) {
      // Pop the position of a loop nest into which fusion will be attempted.
      unsigned dstPos = worklist.back();
      worklist.pop_back();
      // Skip if 'dstPos' is not tracked (was fused into another loop nest).
      if (posToStmtInfo.count(dstPos) == 0)
        continue;
      // Get the top-level ForStmt at 'dstPos'.
      auto *dstForStmt = getForStmtAtPos(dstPos);
      // Skip if this ForStmt contains no load ops.
      if (forStmtToLoadOps.count(dstForStmt) == 0)
        continue;

      // Greedy Policy: iterate through load ops in 'dstForStmt', greedily
      // fusing in src loop nests which have a single store op on the same
      // memref, until a fixed point is reached where there is nothing left to
      // fuse.
      SetVector<OperationStmt *> dstLoadOps = forStmtToLoadOps[dstForStmt];
      while (!dstLoadOps.empty()) {
        auto *dstLoadOpStmt = dstLoadOps.pop_back_val();

        auto dstLoadOp = dstLoadOpStmt->cast<LoadOp>();
        auto *memref = cast<MLValue>(dstLoadOp->getMemRef());
        // Skip if not single src store / dst load pair on 'memref'.
        if (memrefToLoadPosSet[memref].size() != 1 ||
            memrefToStorePosSet[memref].size() != 1)
          continue;
        unsigned srcPos = *memrefToStorePosSet[memref].begin();
        if (srcPos >= dstPos)
          continue;
        auto *srcForStmt = getForStmtAtPos(srcPos);
        // Skip if 'srcForStmt' has more than one store op.
        if (forStmtToStoreOps[srcForStmt].size() > 1)
          continue;
        // Skip if fusion would violated dependences between 'memref' access
        // for loop nests between 'srcPos' and 'dstPos':
        //  For each src load op: check for store ops in range (srcPos, dstPos).
        //  For each src store op: check for load ops in range (srcPos, dstPos).
        if (moveWouldViolateDependences(srcPos, dstPos))
          continue;
        auto *srcStoreOpStmt = forStmtToStoreOps[srcForStmt].front();
        // Build fusion candidate out of 'srcStoreOpStmt' and 'dstLoadOpStmt'.
        FusionCandidate candidate =
            buildFusionCandidate(srcStoreOpStmt, dstLoadOpStmt);
        // Fuse computation slice of 'srcLoopNest' into 'dstLoopNest'.
        auto *sliceLoopNest = mlir::insertBackwardComputationSlice(
            &candidate.srcAccess, &candidate.dstAccess);
        if (sliceLoopNest != nullptr) {
          // Remove 'srcPos' mappings from 'state'.
          moveAccessesAndRemovePos(srcPos, dstPos);
          // Record all load/store accesses in 'sliceLoopNest' at 'dstPos'.
          LoopNestStateCollector collector;
          collector.walkForStmt(sliceLoopNest);
          // Record mappings for loads and stores from 'collector'.
          for (auto *opStmt : collector.loadOpStmts) {
            addLoadOpStmtAt(dstPos, opStmt, dstForStmt);
            // Add newly fused load ops to 'dstLoadOps' to be considered for
            // fusion on subsequent iterations.
            dstLoadOps.insert(opStmt);
          }
          for (auto *opStmt : collector.storeOpStmts) {
            addStoreOpStmtAt(dstPos, opStmt, dstForStmt);
          }
          for (auto *forStmt : collector.forStmts) {
            promoteIfSingleIteration(forStmt);
          }
          // Remove old src loop nest.
          srcForStmt->erase();
        }
      }
    }
  }

  // Walk MLFunction 'f' assigning each top-level statement a position, and
  // gathering state on load and store ops.
  void init(MLFunction *f) {
    unsigned pos = 0;
    for (auto &stmt : *f) {
      if (auto *forStmt = dyn_cast<ForStmt>(&stmt)) {
        // Record all loads and store accesses in 'forStmt' at 'pos'.
        LoopNestStateCollector collector;
        collector.walkForStmt(forStmt);
        // Create StmtInfo for 'forStmt' for top-level loop nests.
        addStmtInfoAt(pos, forStmt, collector.hasIfStmt);
        // Record mappings for loads and stores from 'collector'.
        for (auto *opStmt : collector.loadOpStmts) {
          addLoadOpStmtAt(pos, opStmt, forStmt);
        }
        for (auto *opStmt : collector.storeOpStmts) {
          addStoreOpStmtAt(pos, opStmt, forStmt);
        }
        // Add 'pos' associated with 'forStmt' to worklist.
        worklist.push_back(pos);
      }
      if (auto *opStmt = dyn_cast<OperationStmt>(&stmt)) {
        if (auto loadOp = opStmt->dyn_cast<LoadOp>()) {
          // Create StmtInfo for top-level load op.
          addStmtInfoAt(pos, &stmt, /*hasIfStmt=*/false);
          addLoadOpStmtAt(pos, opStmt, /*containingForStmt=*/nullptr);
        }
        if (auto storeOp = opStmt->dyn_cast<StoreOp>()) {
          // Create StmtInfo for top-level store op.
          addStmtInfoAt(pos, &stmt, /*hasIfStmt=*/false);
          addStoreOpStmtAt(pos, opStmt, /*containingForStmt=*/nullptr);
        }
      }
      if (auto *ifStmt = dyn_cast<IfStmt>(&stmt)) {
        addStmtInfoAt(pos, &stmt, /*hasIfStmt=*/true);
      }
      ++pos;
    }
  }

  // Check if fusing loop nest at 'srcPos' into the loop nest at 'dstPos'
  // would violated any dependences w.r.t other loop nests in that range.
  bool moveWouldViolateDependences(unsigned srcPos, unsigned dstPos) {
    // Lookup src ForStmt at 'srcPos'.
    auto *srcForStmt = getForStmtAtPos(srcPos);
    // For each src load op: check for store ops in range (srcPos, dstPos).
    if (forStmtToLoadOps.count(srcForStmt) > 0) {
      for (auto *opStmt : forStmtToLoadOps[srcForStmt]) {
        auto loadOp = opStmt->cast<LoadOp>();
        auto *memref = cast<MLValue>(loadOp->getMemRef());
        for (unsigned pos = srcPos + 1; pos < dstPos; ++pos) {
          if (memrefToStorePosSet.count(memref) > 0 &&
              memrefToStorePosSet[memref].count(pos) > 0)
            return true;
        }
      }
    }
    // For each src store op: check for load ops in range (srcPos, dstPos).
    if (forStmtToStoreOps.count(srcForStmt) > 0) {
      for (auto *opStmt : forStmtToStoreOps[srcForStmt]) {
        auto storeOp = opStmt->cast<StoreOp>();
        auto *memref = cast<MLValue>(storeOp->getMemRef());
        for (unsigned pos = srcPos + 1; pos < dstPos; ++pos) {
          if (memrefToLoadPosSet.count(memref) > 0 &&
              memrefToLoadPosSet[memref].count(pos) > 0)
            return true;
        }
      }
    }
    return false;
  }

  // Update mappings of memref loads and stores at 'srcPos' to 'dstPos'.
  void moveAccessesAndRemovePos(unsigned srcPos, unsigned dstPos) {
    // Lookup ForStmt at 'srcPos'.
    auto *srcForStmt = getForStmtAtPos(srcPos);
    // Move load op accesses from src to dst.
    if (forStmtToLoadOps.count(srcForStmt) > 0) {
      for (auto *opStmt : forStmtToLoadOps[srcForStmt]) {
        auto loadOp = opStmt->cast<LoadOp>();
        auto *memref = cast<MLValue>(loadOp->getMemRef());
        // Remove 'memref' to 'srcPos' mapping.
        memrefToLoadPosSet[memref].erase(srcPos);
      }
    }
    // Move store op accesses from src to dst.
    if (forStmtToStoreOps.count(srcForStmt) > 0) {
      for (auto *opStmt : forStmtToStoreOps[srcForStmt]) {
        auto storeOp = opStmt->cast<StoreOp>();
        auto *memref = cast<MLValue>(storeOp->getMemRef());
        // Remove 'memref' to 'srcPos' mapping.
        memrefToStorePosSet[memref].erase(srcPos);
      }
    }
    // Remove old state.
    forStmtToLoadOps.erase(srcForStmt);
    forStmtToStoreOps.erase(srcForStmt);
    posToStmtInfo.erase(srcPos);
  }

  ForStmt *getForStmtAtPos(unsigned pos) {
    assert(posToStmtInfo.count(pos) > 0);
    assert(isa<ForStmt>(posToStmtInfo[pos].stmt));
    return cast<ForStmt>(posToStmtInfo[pos].stmt);
  }

  void addStmtInfoAt(unsigned pos, Statement *stmt, bool hasIfStmt) {
    StmtInfo stmtInfo;
    stmtInfo.stmt = stmt;
    stmtInfo.isOrContainsIfStmt = hasIfStmt;
    // Add mapping from 'pos' to StmtInfo for 'forStmt'.
    posToStmtInfo[pos] = stmtInfo;
  }

  // Adds the following mappings:
  // *) 'containingForStmt' to load 'opStmt'
  // *) 'memref' of load 'opStmt' to 'topLevelPos'.
  void addLoadOpStmtAt(unsigned topLevelPos, OperationStmt *opStmt,
                       ForStmt *containingForStmt) {
    if (containingForStmt != nullptr) {
      // Add mapping from 'containingForStmt' to 'opStmt' for load op.
      forStmtToLoadOps[containingForStmt].insert(opStmt);
    }
    auto loadOp = opStmt->cast<LoadOp>();
    auto *memref = cast<MLValue>(loadOp->getMemRef());
    // Add mapping from 'memref' to 'topLevelPos' for load.
    memrefToLoadPosSet[memref].insert(topLevelPos);
  }

  // Adds the following mappings:
  // *) 'containingForStmt' to store 'opStmt'
  // *) 'memref' of store 'opStmt' to 'topLevelPos'.
  void addStoreOpStmtAt(unsigned topLevelPos, OperationStmt *opStmt,
                        ForStmt *containingForStmt) {
    if (containingForStmt != nullptr) {
      // Add mapping from 'forStmt' to 'opStmt' for store op.
      forStmtToStoreOps[containingForStmt].insert(opStmt);
    }
    auto storeOp = opStmt->cast<StoreOp>();
    auto *memref = cast<MLValue>(storeOp->getMemRef());
    // Add mapping from 'memref' to 'topLevelPos' for store.
    memrefToStorePosSet[memref].insert(topLevelPos);
  }

  bool hasIfStmts() {
    for (auto &pair : posToStmtInfo)
      if (pair.second.isOrContainsIfStmt)
        return true;
    return false;
  }
};

} // end anonymous namespace

PassResult LoopFusion::runOnMLFunction(MLFunction *f) {
  GreedyFusionPolicy(f).run();
  return success();
}

static PassRegistration<LoopFusion> pass("loop-fusion", "Fuse loop nests");

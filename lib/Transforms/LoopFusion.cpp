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
#include "mlir/Analysis/LoopAnalysis.h"
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

using namespace mlir;

namespace {

/// Loop fusion pass. This pass fuses adjacent loops in MLFunctions which
/// access the same memref with no dependences.
// See MatchTestPattern for details on candidate loop selection.
// TODO(andydavis) Extend this pass to check for fusion preventing dependences,
// and add support for more general loop fusion algorithms.
struct LoopFusion : public FunctionPass {
  LoopFusion() {}

  PassResult runOnMLFunction(MLFunction *f) override;
  static char passID;
};

// LoopCollector walks the statements in an MLFunction and builds a map from
// StmtBlocks to a list of loops within the StmtBlock, and a map from ForStmts
// to the list of loads and stores with its StmtBlock.
class LoopCollector : public StmtWalker<LoopCollector> {
public:
  DenseMap<StmtBlock *, SmallVector<ForStmt *, 2>> loopMap;
  DenseMap<ForStmt *, SmallVector<OperationStmt *, 2>> loadsAndStoresMap;
  bool hasIfStmt = false;

  void visitForStmt(ForStmt *forStmt) {
    loopMap[forStmt->getBlock()].push_back(forStmt);
  }

  void visitIfStmt(IfStmt *ifStmt) { hasIfStmt = true; }

  void visitOperationStmt(OperationStmt *opStmt) {
    if (auto *parentStmt = opStmt->getParentStmt()) {
      if (auto *parentForStmt = dyn_cast<ForStmt>(parentStmt)) {
        if (opStmt->isa<LoadOp>() || opStmt->isa<StoreOp>()) {
          loadsAndStoresMap[parentForStmt].push_back(opStmt);
        }
      }
    }
  }
};

} // end anonymous namespace

char LoopFusion::passID = 0;

FunctionPass *mlir::createLoopFusionPass() { return new LoopFusion; }

// TODO(andydavis) Remove the following test code when more general loop
// fusion is supported.
struct FusionCandidate {
  // Loop nest of ForStmts with 'accessA' in the inner-most loop.
  SmallVector<ForStmt *, 2> forStmtsA;
  // Load or store operation within loop nest 'forStmtsA'.
  MemRefAccess accessA;
  // Loop nest of ForStmts with 'accessB' in the inner-most loop.
  SmallVector<ForStmt *, 2> forStmtsB;
  // Load or store operation within loop nest 'forStmtsB'.
  MemRefAccess accessB;
};

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

// Checks if 'forStmtA' and 'forStmtB' match specific test criterion:
// constant loop bounds, no nested loops, single StoreOp in 'forStmtA' and
// a single LoadOp in 'forStmtB'.
// Returns true if the test pattern matches, false otherwise.
static bool MatchTestPatternLoopPair(LoopCollector *lc,
                                     FusionCandidate *candidate,
                                     ForStmt *forStmtA, ForStmt *forStmtB) {
  if (forStmtA == nullptr || forStmtB == nullptr)
    return false;
  // Return if 'forStmtA' and 'forStmtB' do not have matching constant
  // bounds and step.
  if (!forStmtA->hasConstantBounds() || !forStmtB->hasConstantBounds() ||
      forStmtA->getConstantLowerBound() != forStmtB->getConstantLowerBound() ||
      forStmtA->getConstantUpperBound() != forStmtB->getConstantUpperBound() ||
      forStmtA->getStep() != forStmtB->getStep())
    return false;

  // Return if 'forStmtA' or 'forStmtB' have nested loops.
  if (lc->loopMap.count(forStmtA) > 0 || lc->loopMap.count(forStmtB))
    return false;

  // Return if 'forStmtA' or 'forStmtB' do not have exactly one load or store.
  if (lc->loadsAndStoresMap[forStmtA].size() != 1 ||
      lc->loadsAndStoresMap[forStmtB].size() != 1)
    return false;

  // Get load/store access for forStmtA.
  getSingleMemRefAccess(lc->loadsAndStoresMap[forStmtA][0],
                        &candidate->accessA);
  // Return if 'accessA' is not a store.
  if (!candidate->accessA.opStmt->isa<StoreOp>())
    return false;

  // Get load/store access for forStmtB.
  getSingleMemRefAccess(lc->loadsAndStoresMap[forStmtB][0],
                        &candidate->accessB);

  // Return if accesses do not access the same memref.
  if (candidate->accessA.memref != candidate->accessB.memref)
    return false;

  candidate->forStmtsA.push_back(forStmtA);
  candidate->forStmtsB.push_back(forStmtB);
  return true;
}

// Returns the child ForStmt of 'parent' if unique, returns false otherwise.
ForStmt *getSingleForStmtChild(ForStmt *parent) {
  if (parent->getStatements().size() == 1 && isa<ForStmt>(parent->front()))
    return dyn_cast<ForStmt>(&parent->front());
  return nullptr;
}

// Checks for a specific ForStmt/OpStatment test pattern in 'f', returns true
// on success and resturns fusion candidate in 'candidate'. Returns false
// otherwise.
// Currently supported test patterns:
// *) Adjacent loops with a StoreOp the only op in first loop, and a LoadOp the
//    only op in the second loop (both load/store accessing the same memref).
// *) As above, but with one level of perfect loop nesting.
//
// TODO(andydavis) Look into using ntv@ pattern matcher here.
static bool MatchTestPattern(MLFunction *f, FusionCandidate *candidate) {
  LoopCollector lc;
  lc.walk(f);
  // Return if an IfStmt was found or if less than two ForStmts were found.
  if (lc.hasIfStmt || lc.loopMap.count(f) == 0 || lc.loopMap[f].size() < 2)
    return false;
  auto *forStmtA = lc.loopMap[f][0];
  auto *forStmtB = lc.loopMap[f][1];
  if (!MatchTestPatternLoopPair(&lc, candidate, forStmtA, forStmtB)) {
    // Check for one level of loop nesting.
    candidate->forStmtsA.push_back(forStmtA);
    candidate->forStmtsB.push_back(forStmtB);
    return MatchTestPatternLoopPair(&lc, candidate,
                                    getSingleForStmtChild(forStmtA),
                                    getSingleForStmtChild(forStmtB));
  }
  return true;
}

// FuseLoops implements the code generation mechanics of loop fusion.
// Fuses the operations statments from the inner-most loop in 'c.forStmtsB',
// by cloning them into the inner-most loop in 'c.forStmtsA', then erasing
// old statements and loops.
static void fuseLoops(const FusionCandidate &c) {
  MLFuncBuilder builder(c.forStmtsA.back(),
                        StmtBlock::iterator(c.forStmtsA.back()->end()));
  DenseMap<const MLValue *, MLValue *> operandMap;
  assert(c.forStmtsA.size() == c.forStmtsB.size());
  for (unsigned i = 0, e = c.forStmtsA.size(); i < e; i++) {
    // Map loop IVs to 'forStmtB[i]' to loop IV for 'forStmtA[i]'.
    operandMap[c.forStmtsB[i]] = c.forStmtsA[i];
  }
  // Clone the body of inner-most loop in 'forStmtsB', into the body of
  // inner-most loop in 'forStmtsA'.
  SmallVector<Statement *, 2> stmtsToErase;
  auto *innerForStmtB = c.forStmtsB.back();
  for (auto &stmt : *innerForStmtB) {
    builder.clone(stmt, operandMap);
    stmtsToErase.push_back(&stmt);
  }
  // Erase 'forStmtB' and its statement list.
  for (auto it = stmtsToErase.rbegin(); it != stmtsToErase.rend(); ++it)
    (*it)->erase();
  // Erase 'forStmtsB' loop nest.
  for (int i = static_cast<int>(c.forStmtsB.size()) - 1; i >= 0; --i)
    c.forStmtsB[i]->erase();
}

PassResult LoopFusion::runOnMLFunction(MLFunction *f) {
  FusionCandidate candidate;
  if (!MatchTestPattern(f, &candidate))
    return failure();

  // TODO(andydavis) Add checks for fusion-preventing dependences and ordering
  // constraints which would prevent fusion.
  // TODO(andydavis) This check if overly conservative for now. Support fusing
  // statements with compatible dependences (i.e. statements where the
  // dependence between the statements does not reverse direction when the
  // statements are fused into the same loop).
  if (!checkMemrefAccessDependence(candidate.accessA, candidate.accessB)) {
    // Current conservatinve test policy: No dependence exists between accesses
    // in different loop nests -> fuse loops.
    fuseLoops(candidate);
  }

  return success();
}

static PassRegistration<LoopFusion> pass("loop-fusion", "Fuse loop nests");

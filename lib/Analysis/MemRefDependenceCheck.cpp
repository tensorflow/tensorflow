//===- MemRefDependenceCheck.cpp - MemRef DependenceCheck Class -*- C++ -*-===//
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
// This file implements a pass to run pair-wise memref access dependence checks.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Passes.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/StmtVisitor.h"
#include "mlir/Pass.h"
#include "mlir/StandardOps/StandardOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "memref-dependence-check"

using namespace mlir;

namespace {

// TODO(andydavis) Add common surrounding loop depth-wise dependence checks.
/// Checks dependences between all pairs of memref accesses in an MLFunction.
struct MemRefDependenceCheck : public FunctionPass,
                               StmtWalker<MemRefDependenceCheck> {
  SmallVector<OperationStmt *, 4> loadsAndStores;
  explicit MemRefDependenceCheck()
      : FunctionPass(&MemRefDependenceCheck::passID) {}

  PassResult runOnMLFunction(MLFunction *f) override;
  // Not applicable to CFG functions.
  PassResult runOnCFGFunction(CFGFunction *f) override { return success(); }

  void visitOperationStmt(OperationStmt *opStmt) {
    if (opStmt->isa<LoadOp>() || opStmt->isa<StoreOp>()) {
      loadsAndStores.push_back(opStmt);
    }
  }
  static char passID;
};

} // end anonymous namespace

char MemRefDependenceCheck::passID = 0;

FunctionPass *mlir::createMemRefDependenceCheckPass() {
  return new MemRefDependenceCheck();
}

// Adds memref access indices 'opIndices' from 'memrefType' to 'access'.
static void addMemRefAccessIndices(
    llvm::iterator_range<Operation::const_operand_iterator> opIndices,
    MemRefType memrefType, MemRefAccess *access) {
  access->indices.reserve(memrefType.getRank());
  for (auto *index : opIndices) {
    access->indices.push_back(cast<MLValue>(const_cast<SSAValue *>(index)));
  }
}

// Populates 'access' with memref, indices and opstmt from 'loadOrStoreOpStmt'.
static void getMemRefAccess(const OperationStmt *loadOrStoreOpStmt,
                            MemRefAccess *access) {
  access->opStmt = loadOrStoreOpStmt;
  if (auto loadOp = loadOrStoreOpStmt->dyn_cast<LoadOp>()) {
    access->memref = cast<MLValue>(loadOp->getMemRef());
    addMemRefAccessIndices(loadOp->getIndices(), loadOp->getMemRefType(),
                           access);
  } else {
    assert(loadOrStoreOpStmt->isa<StoreOp>());
    auto storeOp = loadOrStoreOpStmt->dyn_cast<StoreOp>();
    access->memref = cast<MLValue>(storeOp->getMemRef());
    addMemRefAccessIndices(storeOp->getIndices(), storeOp->getMemRefType(),
                           access);
  }
}

// Returns the number of surrounding loops common to 'loopsA' and 'loopsB',
// where each lists loops from outer-most to inner-most in loop nest.
static unsigned getNumCommonSurroundingLoops(ArrayRef<const ForStmt *> loopsA,
                                             ArrayRef<const ForStmt *> loopsB) {
  unsigned minNumLoops = std::min(loopsA.size(), loopsB.size());
  unsigned numCommonLoops = 0;
  for (unsigned i = 0; i < minNumLoops; ++i) {
    if (loopsA[i] != loopsB[i])
      break;
    ++numCommonLoops;
  }
  return numCommonLoops;
}

// Returns a result string which represents the direction vector (if there was
// a dependence), returns the string "false" otherwise.
static string
getDirectionVectorStr(bool ret, unsigned numCommonLoops, unsigned loopNestDepth,
                      ArrayRef<DependenceComponent> dependenceComponents) {
  if (!ret)
    return "false";
  if (dependenceComponents.empty() || loopNestDepth > numCommonLoops)
    return "true";
  string result;
  for (unsigned i = 0, e = dependenceComponents.size(); i < e; ++i) {
    string lbStr = dependenceComponents[i].lb.hasValue()
                       ? std::to_string(dependenceComponents[i].lb.getValue())
                       : "-inf";
    string ubStr = dependenceComponents[i].ub.hasValue()
                       ? std::to_string(dependenceComponents[i].ub.getValue())
                       : "+inf";
    result += "[" + lbStr + ", " + ubStr + "]";
  }
  return result;
}

// For each access in 'loadsAndStores', runs a depence check between this
// "source" access and all subsequent "destination" accesses in
// 'loadsAndStores'. Emits the result of the dependence check as a note with
// the source access.
static void checkDependences(ArrayRef<OperationStmt *> loadsAndStores) {
  for (unsigned i = 0, e = loadsAndStores.size(); i < e; ++i) {
    auto *srcOpStmt = loadsAndStores[i];
    MemRefAccess srcAccess;
    getMemRefAccess(srcOpStmt, &srcAccess);
    SmallVector<ForStmt *, 4> srcLoops;
    getLoopIVs(*srcOpStmt, &srcLoops);
    for (unsigned j = 0; j < e; ++j) {
      auto *dstOpStmt = loadsAndStores[j];
      MemRefAccess dstAccess;
      getMemRefAccess(dstOpStmt, &dstAccess);

      SmallVector<ForStmt *, 4> dstLoops;
      getLoopIVs(*dstOpStmt, &dstLoops);
      unsigned numCommonLoops =
          getNumCommonSurroundingLoops(srcLoops, dstLoops);
      for (unsigned d = 1; d <= numCommonLoops + 1; ++d) {
        FlatAffineConstraints dependenceConstraints;
        llvm::SmallVector<DependenceComponent, 2> dependenceComponents;
        bool ret = checkMemrefAccessDependence(srcAccess, dstAccess, d,
                                               &dependenceConstraints,
                                               &dependenceComponents);
        // TODO(andydavis) Print dependence type (i.e. RAW, etc) and print
        // distance vectors as: ([2, 3], [0, 10]). Also, shorten distance
        // vectors from ([1, 1], [3, 3]) to (1, 3).
        srcOpStmt->emitNote(
            "dependence from " + Twine(i) + " to " + Twine(j) + " at depth " +
            Twine(d) + " = " +
            getDirectionVectorStr(ret, numCommonLoops, d, dependenceComponents)
                .c_str());
      }
    }
  }
}

// Walks the MLFunction 'f' adding load and store ops to 'loadsAndStores'.
// Runs pair-wise dependence checks.
PassResult MemRefDependenceCheck::runOnMLFunction(MLFunction *f) {
  loadsAndStores.clear();
  walk(f);
  checkDependences(loadsAndStores);
  return success();
}

static PassRegistration<MemRefDependenceCheck>
    pass("memref-dependence-check",
         "Checks dependences between all pairs of memref accesses.");

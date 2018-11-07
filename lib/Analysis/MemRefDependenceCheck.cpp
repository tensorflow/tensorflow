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
  explicit MemRefDependenceCheck() {}

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

// For each access in 'loadsAndStores', runs a depence check between this
// "source" access and all subsequent "destination" accesses in
// 'loadsAndStores'. Emits the result of the dependence check as a note with
// the source access.
// TODO(andydavis) Clarify expected-note logs. In particular we may want to
// drop the 'i' from the note string, tag dependence destination accesses
// with a note with their 'j' index. In addition, we may want a schedme that
// first assigned unique ids to each access, then emits a note for each access
// with its id, and emits a note for each dependence check with a pair of ids.
// For example, given this code:
//
//   memref_access0
//   // emit note: "this op is memref access 0'
//   // emit note: "dependence from memref access 0 to access 1 = false"
//   // emit note: "dependence from memref access 0 to access 2 = true"
//   memref_access1
//   // emit note: "this op is memref access 1'
//   // emit note: "dependence from memref access 1 to access 2 = false"
//   memref_access2
//   // emit note: "this op is memref access 2'
//
static void checkDependences(ArrayRef<OperationStmt *> loadsAndStores) {
  for (unsigned i = 0, e = loadsAndStores.size(); i < e; ++i) {
    auto *srcOpStmt = loadsAndStores[i];
    MemRefAccess srcAccess;
    getMemRefAccess(srcOpStmt, &srcAccess);
    for (unsigned j = i + 1; j < e; ++j) {
      auto *dstOpStmt = loadsAndStores[j];
      MemRefAccess dstAccess;
      getMemRefAccess(dstOpStmt, &dstAccess);
      bool ret = checkMemrefAccessDependence(srcAccess, dstAccess);
      srcOpStmt->emitNote("dependence from memref access " + Twine(i) +
                          " to access " + Twine(j) + " = " +
                          (ret ? "true" : "false"));
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

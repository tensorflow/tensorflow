//===- MemRefBoundCheck.cpp - MLIR Affine Structures Class-----*- C++ -*-===//
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
// This file implements a pass to check memref accessses for out of bound
// accesses.
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

#define DEBUG_TYPE "memref-bound-check"

using namespace mlir;

namespace {

/// Checks for out of bound memef access subscripts..
struct MemRefBoundCheck : public FunctionPass, StmtWalker<MemRefBoundCheck> {
  explicit MemRefBoundCheck() {}

  PassResult runOnMLFunction(MLFunction *f) override;
  // Not applicable to CFG functions.
  PassResult runOnCFGFunction(CFGFunction *f) override { return success(); }

  void visitOperationStmt(OperationStmt *opStmt);
};

} // end anonymous namespace

FunctionPass *mlir::createMemRefBoundCheckPass() {
  return new MemRefBoundCheck();
}

/// Returns the memory region accessed by this memref.
// TODO(bondhugula): extend this to store's and other memref dereferencing ops.
bool getMemoryRegion(OpPointer<LoadOp> loadOp, FlatAffineConstraints *region) {
  OperationStmt *opStmt = dyn_cast<OperationStmt>(loadOp->getOperation());
  // Only in MLFunctions.
  if (!opStmt)
    return false;

  unsigned rank = loadOp->getMemRefType().getRank();
  MLFuncBuilder b(opStmt);
  auto idMap = b.getMultiDimIdentityMap(rank);

  SmallVector<MLValue *, 4> indices;
  for (auto *index : loadOp->getIndices()) {
    indices.push_back(cast<MLValue>(index));
  }

  // Initialize 'srcValueMap' and compose with reachable AffineApplyOps.
  AffineValueMap srcValueMap(idMap, indices);
  forwardSubstituteReachableOps(&srcValueMap);
  AffineMap srcMap = srcValueMap.getAffineMap();

  region->reset(8, 8, srcMap.getNumInputs() + 1, srcMap.getNumDims(),
                srcMap.getNumSymbols());

  // Add equality constraints.
  AffineMap map = srcValueMap.getAffineMap();
  unsigned numDims = map.getNumDims();
  unsigned numSymbols = map.getNumSymbols();
  // Add inEqualties for loop lower/upper bounds.
  for (unsigned i = 0; i < numDims + numSymbols; ++i) {
    if (auto *loop = dyn_cast<ForStmt>(srcValueMap.getOperand(i))) {
      if (!loop->hasConstantBounds())
        return false;
      // Add lower bound and upper bounds.
      region->addConstantLowerBound(i, loop->getConstantLowerBound());
      region->addConstantUpperBound(i, loop->getConstantUpperBound());
    } else {
      // Has to be a valid symbol.
      auto *symbol = cast<MLValue>(srcValueMap.getOperand(i));
      assert(symbol->isValidSymbol());
      // Check if the symbols is a constant.
      if (auto *opStmt = symbol->getDefiningStmt()) {
        if (auto constOp = opStmt->dyn_cast<ConstantIndexOp>()) {
          region->setIdToConstant(i, constOp->getValue());
        }
      }
    }
  }

  // Add access function equalities to connect loop IVs to data dimensions.
  region->addDimsForMap(0, srcValueMap.getAffineMap());

  // Eliminate the loop IVs.
  for (unsigned i = 0, e = srcValueMap.getNumOperands(); i < e; i++) {
    region->FourierMotzkinEliminate(srcMap.getNumResults());
  }
  assert(region->getNumDimIds() == rank);
  return true;
}

void MemRefBoundCheck::visitOperationStmt(OperationStmt *opStmt) {
  // TODO(bondhugula): extend this to store's and other memref dereferencing
  // op's.
  if (auto loadOp = opStmt->dyn_cast<LoadOp>()) {
    FlatAffineConstraints memoryRegion;
    if (!getMemoryRegion(loadOp, &memoryRegion))
      return;
    unsigned rank = loadOp->getMemRefType().getRank();
    // For each dimension, check for out of bounds.
    for (unsigned r = 0; r < rank; r++) {
      FlatAffineConstraints ucst(memoryRegion);
      // Intersect memory region with constraint capturing out of bounds,
      // and check if the constraint system is feasible. If it is, there is at
      // least one point out of bounds.
      SmallVector<int64_t, 4> ineq(rank + 1, 0);
      // d_i >= memref dim size.
      ucst.addConstantLowerBound(r, loadOp->getMemRefType().getDimSize(r));
      LLVM_DEBUG(llvm::dbgs() << "System to check for overflow:\n");
      LLVM_DEBUG(ucst.dump());
      //
      if (!ucst.isEmpty()) {
        loadOp->emitOpError(
            "memref out of upper bound access along dimension #" +
            Twine(r + 1));
      }
      // Check for less than negative index.
      FlatAffineConstraints lcst(memoryRegion);
      std::fill(ineq.begin(), ineq.end(), 0);
      // d_i <= -1;
      lcst.addConstantUpperBound(r, -1);
      LLVM_DEBUG(llvm::dbgs() << "System to check for underflow:\n");
      LLVM_DEBUG(lcst.dump());
      if (!lcst.isEmpty()) {
        loadOp->emitOpError(
            "memref out of lower bound access along dimension #" +
            Twine(r + 1));
      }
    }
  }
}

PassResult MemRefBoundCheck::runOnMLFunction(MLFunction *f) {
  return walk(f), success();
}

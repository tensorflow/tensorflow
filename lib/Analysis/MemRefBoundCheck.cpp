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
#include "mlir/Analysis/Utils.h"
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
  explicit MemRefBoundCheck() : FunctionPass(&MemRefBoundCheck::passID) {}

  PassResult runOnMLFunction(MLFunction *f) override;
  // Not applicable to CFG functions.
  PassResult runOnCFGFunction(CFGFunction *f) override { return success(); }

  void visitOperationStmt(OperationStmt *opStmt);

  static char passID;
};

} // end anonymous namespace

char MemRefBoundCheck::passID = 0;

FunctionPass *mlir::createMemRefBoundCheckPass() {
  return new MemRefBoundCheck();
}

void MemRefBoundCheck::visitOperationStmt(OperationStmt *opStmt) {
  if (auto loadOp = opStmt->dyn_cast<LoadOp>()) {
    boundCheckLoadOrStoreOp(loadOp);
  } else if (auto storeOp = opStmt->dyn_cast<StoreOp>()) {
    boundCheckLoadOrStoreOp(storeOp);
  }
  // TODO(bondhugula): do this for DMA ops as well.
}

PassResult MemRefBoundCheck::runOnMLFunction(MLFunction *f) {
  return walk(f), success();
}

static PassRegistration<MemRefBoundCheck>
    memRefBoundCheck("memref-bound-check",
                     "Check memref access bounds in an MLFunction");

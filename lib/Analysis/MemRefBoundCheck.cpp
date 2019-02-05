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
#include "mlir/Pass.h"
#include "mlir/StandardOps/StandardOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "memref-bound-check"

using namespace mlir;

namespace {

/// Checks for out of bound memef access subscripts..
struct MemRefBoundCheck : public FunctionPass {
  explicit MemRefBoundCheck() : FunctionPass(&MemRefBoundCheck::passID) {}

  PassResult runOnFunction(Function *f) override;

  static char passID;
};

} // end anonymous namespace

char MemRefBoundCheck::passID = 0;

FunctionPass *mlir::createMemRefBoundCheckPass() {
  return new MemRefBoundCheck();
}

PassResult MemRefBoundCheck::runOnFunction(Function *f) {
  f->walk([](Instruction *opInst) {
    if (auto loadOp = opInst->dyn_cast<LoadOp>()) {
      boundCheckLoadOrStoreOp(loadOp);
    } else if (auto storeOp = opInst->dyn_cast<StoreOp>()) {
      boundCheckLoadOrStoreOp(storeOp);
    }
    // TODO(bondhugula): do this for DMA ops as well.
  });
  return success();
}

static PassRegistration<MemRefBoundCheck>
    memRefBoundCheck("memref-bound-check",
                     "Check memref access bounds in a Function");

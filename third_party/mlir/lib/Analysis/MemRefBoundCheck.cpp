//===- MemRefBoundCheck.cpp - MLIR Affine Structures Class ----------------===//
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
// This file implements a pass to check memref accesses for out of bound
// accesses.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Passes.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "memref-bound-check"

using namespace mlir;

namespace {

/// Checks for out of bound memef access subscripts..
struct MemRefBoundCheck : public FunctionPass<MemRefBoundCheck> {
  void runOnFunction() override;
};

} // end anonymous namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createMemRefBoundCheckPass() {
  return std::make_unique<MemRefBoundCheck>();
}

void MemRefBoundCheck::runOnFunction() {
  getFunction().walk([](Operation *opInst) {
    if (auto loadOp = dyn_cast<AffineLoadOp>(opInst)) {
      boundCheckLoadOrStoreOp(loadOp);
    } else if (auto storeOp = dyn_cast<AffineStoreOp>(opInst)) {
      boundCheckLoadOrStoreOp(storeOp);
    }
    // TODO(bondhugula): do this for DMA ops as well.
  });
}

static PassRegistration<MemRefBoundCheck>
    memRefBoundCheck("memref-bound-check",
                     "Check memref access bounds in a Function");

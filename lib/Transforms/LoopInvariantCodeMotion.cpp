//===- LoopInvariantCodeMotion.cpp - Code to perform loop fusion-----------===//
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
// This file implements loop invariant code motion.
//
//===----------------------------------------------------------------------===//

#include <iomanip>
#include <sstream>

#include "mlir/AffineOps/AffineOps.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/StandardOps/Ops.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "licm"

using llvm::SetVector;

using namespace mlir;

namespace {

/// Loop invariant code motion (LICM) pass.
/// TODO(asabne) : The pass is missing zero-trip tests.
/// TODO(asabne) : Check for the presence of side effects before hoisting.
struct LoopInvariantCodeMotion : public FunctionPass<LoopInvariantCodeMotion> {
  void runOnFunction() override;
  void runOnAffineForOp(AffineForOp forOp);
  std::vector<AffineForOp> forOps;
};
} // end anonymous namespace

FunctionPassBase *mlir::createLoopInvariantCodeMotionPass() {
  return new LoopInvariantCodeMotion();
}

void LoopInvariantCodeMotion::runOnAffineForOp(AffineForOp forOp) {
  auto *loopBody = forOp.getBody();

  // This is the place where hoisted instructions would reside.
  FuncBuilder b(forOp.getOperation());

  // This vector is used to place loop invariant operations.
  SmallVector<Operation *, 8> opsToMove;

  SetVector<Operation *> loopDefinedOps;
  // Generate forward slice which contains ops that fall under the transitive
  // definition closure following the loop induction variable.
  getForwardSlice(forOp, &loopDefinedOps);

  LLVM_DEBUG(for (auto i
                  : loopDefinedOps) {
    (i->print(llvm::dbgs() << "\nLoop-dependent op\n"));
  });

  for (auto &op : *loopBody) {
    // If the operation is loop invariant, insert it into opsToMove.
    if (!op.isa<AffineForOp>() && !op.isa<AffineTerminatorOp>() &&
        loopDefinedOps.count(&op) != 1) {
      LLVM_DEBUG(op.print(llvm::dbgs() << "\nLICM'ing op\n"));
      opsToMove.push_back(&op);
    }
  }

  // For all instructions that we found to be invariant, place them sequentially
  // right before the for loop.
  for (auto *op : opsToMove) {
    op->moveBefore(forOp);
  }

  LLVM_DEBUG(forOp.getOperation()->print(llvm::dbgs() << "\nModified loop\n"));

  // If the for loop body has a single operation (the terminator), erase it.
  if (forOp.getBody()->getOperations().size() == 1) {
    assert(forOp.getBody()->getOperations().front().isa<AffineTerminatorOp>());
    forOp.erase();
  }
}

void LoopInvariantCodeMotion::runOnFunction() {
  forOps.clear();

  // Gather all loops in a function, and order them in innermost-loop-first
  // order. This way, we first LICM from the inner loop, and place the ops in
  // the outer loop, which in turn can be further LICM'ed. This saves iterating
  // on the inner loop operations while LICMing through the outer loop.
  getFunction().walk<AffineForOp>(
      [&](AffineForOp forOp) { forOps.push_back(forOp); });
  // We gather loops first, and then go over them later because we don't want to
  // mess the iterators up.
  for (auto op : forOps) {
    LLVM_DEBUG(op.getOperation()->print(llvm::dbgs() << "\nOriginal loop\n"));
    runOnAffineForOp(op);
  }
}

static PassRegistration<LoopInvariantCodeMotion>
    pass("affine-loop-invariant-code-motion",
         "Hoist loop invariant instructions outside of the loop");

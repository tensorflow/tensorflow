//===- TestLoopFusion.cpp - Test loop fusion ------------------------------===//
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
// This file implements a pass to test various loop fusion utility functions.
//
//===----------------------------------------------------------------------===//

#include "mlir/AffineOps/AffineOps.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Passes.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/StandardOps/Ops.h"
#include "mlir/Transforms/LoopFusionUtils.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "test-loop-fusion"

using namespace mlir;

static llvm::cl::OptionCategory clOptionsCategory(DEBUG_TYPE " options");

static llvm::cl::opt<bool> clTestDependenceCheck(
    "test-loop-fusion-dependence-check",
    llvm::cl::desc("Enable testing of loop fusion dependence check"),
    llvm::cl::cat(clOptionsCategory));

namespace {

struct TestLoopFusion : public FunctionPass<TestLoopFusion> {
  void runOnFunction() override;
};

} // end anonymous namespace

FunctionPassBase *mlir::createTestLoopFusionPass() {
  return new TestLoopFusion;
}

// Gathers all AffineForOps in 'block' at 'currLoopDepth' in 'depthToLoops'.
static void
gatherLoops(Block *block, unsigned currLoopDepth,
            DenseMap<unsigned, SmallVector<AffineForOp, 2>> &depthToLoops) {
  auto &loopsAtDepth = depthToLoops[currLoopDepth];
  for (auto &op : *block) {
    if (auto forOp = dyn_cast<AffineForOp>(op)) {
      loopsAtDepth.push_back(forOp);
      gatherLoops(forOp.getBody(), currLoopDepth + 1, depthToLoops);
    }
  }
}

// Run fusion dependence check on 'loops[i]' and 'loops[j]' at 'loopDepth'.
// Emits a remark on 'loops[i]' if a fusion-preventing dependence exists.
static void testDependenceCheck(SmallVector<AffineForOp, 2> &loops, unsigned i,
                                unsigned j, unsigned loopDepth) {
  AffineForOp srcForOp = loops[i];
  AffineForOp dstForOp = loops[j];
  mlir::ComputationSliceState sliceUnion;
  // TODO(andydavis) Test at deeper loop depths current loop depth + 1.
  FusionResult result =
      mlir::canFuseLoops(srcForOp, dstForOp, loopDepth + 1, &sliceUnion);
  if (result.value == FusionResult::FailBlockDependence) {
    srcForOp.getOperation()->emitRemark("block-level dependence preventing"
                                        " fusion of loop nest ")
        << i << " into loop nest " << j << " at depth " << loopDepth;
  }
}

void TestLoopFusion::runOnFunction() {
  // Gather all AffineForOps by loop depth.
  DenseMap<unsigned, SmallVector<AffineForOp, 2>> depthToLoops;
  for (auto &block : getFunction()) {
    gatherLoops(&block, /*currLoopDepth=*/0, depthToLoops);
  }

  // Run tests on all combinations of src/dst loop nests in 'depthToLoops'.
  for (auto &depthAndLoops : depthToLoops) {
    unsigned loopDepth = depthAndLoops.first;
    auto &loops = depthAndLoops.second;
    unsigned numLoops = loops.size();
    for (unsigned j = 0; j < numLoops; ++j) {
      for (unsigned k = 0; k < numLoops; ++k) {
        if (j == k)
          continue;
        if (clTestDependenceCheck)
          testDependenceCheck(loops, j, k, loopDepth);
      }
    }
  }
}

static PassRegistration<TestLoopFusion>
    pass("test-loop-fusion", "Tests loop fusion utility functions.");

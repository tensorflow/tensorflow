//===- TestLoopFusion.cpp - Test loop fusion ------------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to test various loop fusion utility functions.
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

static llvm::cl::opt<bool> clTestSliceComputation(
    "test-loop-fusion-slice-computation",
    llvm::cl::desc("Enable testing of loop fusion slice computation"),
    llvm::cl::cat(clOptionsCategory));

namespace {

struct TestLoopFusion : public FunctionPass<TestLoopFusion> {
  void runOnFunction() override;
};

} // end anonymous namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createTestLoopFusionPass() {
  return std::make_unique<TestLoopFusion>();
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

// Run fusion dependence check on 'loops[i]' and 'loops[j]' at loop depths
// in range ['loopDepth' + 1, 'maxLoopDepth'].
// Emits a remark on 'loops[i]' if a fusion-preventing dependence exists.
static void testDependenceCheck(SmallVector<AffineForOp, 2> &loops, unsigned i,
                                unsigned j, unsigned loopDepth,
                                unsigned maxLoopDepth) {
  AffineForOp srcForOp = loops[i];
  AffineForOp dstForOp = loops[j];
  mlir::ComputationSliceState sliceUnion;
  for (unsigned d = loopDepth + 1; d <= maxLoopDepth; ++d) {
    FusionResult result =
        mlir::canFuseLoops(srcForOp, dstForOp, d, &sliceUnion);
    if (result.value == FusionResult::FailBlockDependence) {
      srcForOp.getOperation()->emitRemark("block-level dependence preventing"
                                          " fusion of loop nest ")
          << i << " into loop nest " << j << " at depth " << loopDepth;
    }
  }
}

// Returns the index of 'op' in its block.
static unsigned getBlockIndex(Operation &op) {
  unsigned index = 0;
  for (auto &opX : *op.getBlock()) {
    if (&op == &opX)
      break;
    ++index;
  }
  return index;
}

// Returns a string representation of 'sliceUnion'.
static std::string getSliceStr(const mlir::ComputationSliceState &sliceUnion) {
  std::string result;
  llvm::raw_string_ostream os(result);
  // Slice insertion point format [loop-depth, operation-block-index]
  unsigned ipd = getNestingDepth(*sliceUnion.insertPoint);
  unsigned ipb = getBlockIndex(*sliceUnion.insertPoint);
  os << "insert point: (" << std::to_string(ipd) << ", " << std::to_string(ipb)
     << ")";
  assert(sliceUnion.lbs.size() == sliceUnion.ubs.size());
  os << " loop bounds: ";
  for (unsigned k = 0, e = sliceUnion.lbs.size(); k < e; ++k) {
    os << '[';
    sliceUnion.lbs[k].print(os);
    os << ", ";
    sliceUnion.ubs[k].print(os);
    os << "] ";
  }
  return os.str();
}

// Computes fusion slice union on 'loops[i]' and 'loops[j]' at loop depths
// in range ['loopDepth' + 1, 'maxLoopDepth'].
// Emits a string representation of the slice union as a remark on 'loops[j]'.
static void testSliceComputation(SmallVector<AffineForOp, 2> &loops, unsigned i,
                                 unsigned j, unsigned loopDepth,
                                 unsigned maxLoopDepth) {
  AffineForOp forOpA = loops[i];
  AffineForOp forOpB = loops[j];
  for (unsigned d = loopDepth + 1; d <= maxLoopDepth; ++d) {
    mlir::ComputationSliceState sliceUnion;
    FusionResult result = mlir::canFuseLoops(forOpA, forOpB, d, &sliceUnion);
    if (result.value == FusionResult::Success) {
      forOpB.getOperation()->emitRemark("slice (")
          << " src loop: " << i << ", dst loop: " << j << ", depth: " << d
          << " : " << getSliceStr(sliceUnion) << ")";
    }
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
          testDependenceCheck(loops, j, k, loopDepth, depthToLoops.size());
        if (clTestSliceComputation)
          testSliceComputation(loops, j, k, loopDepth, depthToLoops.size());
      }
    }
  }
}

static PassRegistration<TestLoopFusion>
    pass("test-loop-fusion", "Tests loop fusion utility functions.");

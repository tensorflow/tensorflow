//===- LoopParametricTiling.cpp --- Parametric loop tiling pass -----------===//
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
// This file implements a pass to parametrically tile nests of standard loops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SetVector.h"

using namespace mlir;
using llvm::SetVector;

static llvm::cl::list<int> clOuterLoopSizes(
    "outer-loop-sizes", llvm::cl::MiscFlags::CommaSeparated,
    llvm::cl::desc(
        "fixed number of iterations that the outer loops should have"));

// Hoist the ops within `outer` that appear before `inner`.
// Such ops include the ops that have been introduced by parametric tiling.
// Ops that come from triangular loops (i.e. that belong to the program slice
// rooted at `outer`) and ops that have side effects cannot be hoisted.
// Returns failure when any op fails to hoist.
static LogicalResult hoistOpsBetween(loop::ForOp outer, loop::ForOp inner) {
  SetVector<Operation *> forwardSlice;
  getForwardSlice(outer.getOperation(), &forwardSlice, [&inner](Operation *op) {
    return op != inner.getOperation();
  });
  LogicalResult status = success();
  SmallVector<Operation *, 8> toHoist;
  for (auto &op : outer.getBody()->getOperations()) {
    // Stop when encountering the inner loop.
    if (&op == inner.getOperation())
      break;
    // Skip over non-hoistable ops.
    if (forwardSlice.count(&op) > 0) {
      status = failure();
      continue;
    }
    // Skip loop::ForOp, these are not considered a failure.
    if (op.getNumRegions() > 0)
      continue;
    // Skip other ops with regions.
    if (op.getNumRegions() > 0) {
      status = failure();
      continue;
    }
    // Skip if op has side effects.
    // TODO(ntv): loads to immutable memory regions are ok.
    if (!op.hasNoSideEffect()) {
      status = failure();
      continue;
    }
    toHoist.push_back(&op);
  }
  auto *outerForOp = outer.getOperation();
  for (auto *op : toHoist)
    op->moveBefore(outerForOp);
  return status;
}

// Traverse the interTile and intraTile loops and tries to hoist ops such that
// bands of perfectly nested loops are isolated.
// Returns failure if either perfect interTile or perfect intraTile bands cannot
// be formed.
static LogicalResult tryIsolateBands(const SmallVector<TileLoops, 8> &loops) {
  LogicalResult status = success();
  for (auto &tl : loops) {
    auto &interTile = tl.first;
    auto &intraTile = tl.second;
    auto size = interTile.size();
    assert(size == intraTile.size());
    if (size <= 1)
      continue;
    for (unsigned s = 1; s < size; ++s)
      status = succeeded(status) ? hoistOpsBetween(intraTile[0], intraTile[s])
                                 : failure();
    for (unsigned s = 1; s < size; ++s)
      status = succeeded(status) ? hoistOpsBetween(interTile[0], interTile[s])
                                 : failure();
  }
  return status;
}

namespace {
// Extracts fixed-range loops for top-level loop nests with ranges defined in
// the pass constructor.  Assumes loops are permutable.
class SimpleParametricLoopTilingPass
    : public FunctionPass<SimpleParametricLoopTilingPass> {
public:
  explicit SimpleParametricLoopTilingPass(ArrayRef<int64_t> outerLoopSizes)
      : sizes(outerLoopSizes.begin(), outerLoopSizes.end()) {}

  void runOnFunction() override {
    FuncOp func = getFunction();

    SmallVector<TileLoops, 8> loops;
    func.walk<loop::ForOp>([this, &loops](loop::ForOp op) {
      // Ignore nested loops.
      if (op.getContainingRegion()->getParentOfType<loop::ForOp>())
        return;
      loops.push_back(extractFixedOuterLoops(op, sizes));
    });

    // TODO(ntv, zinenko) for now we just ignore the result of band isolation.
    // In the future, mapping decisions may be impacted by the ability to
    // isolate perfectly nested bands.
    tryIsolateBands(loops);
  }

  SmallVector<int64_t, 4> sizes;
};
} // end namespace

FunctionPassBase *
mlir::createSimpleParametricTilingPass(ArrayRef<int64_t> outerLoopSizes) {
  return new SimpleParametricLoopTilingPass(outerLoopSizes);
}

static PassRegistration<SimpleParametricLoopTilingPass>
    reg("extract-fixed-outer-loops",
        "apply parametric tiling to the outer loops so that the ranges of "
        "outer loops become static",
        [] {
          auto *pass = new SimpleParametricLoopTilingPass({});
          pass->sizes.assign(clOuterLoopSizes.begin(), clOuterLoopSizes.end());
          return pass;
        });

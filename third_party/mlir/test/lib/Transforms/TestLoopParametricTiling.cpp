//===- TestLoopParametricTiling.cpp --- Parametric loop tiling pass -------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to parametrically tile nests of standard loops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

// Extracts fixed-range loops for top-level loop nests with ranges defined in
// the pass constructor.  Assumes loops are permutable.
class SimpleParametricLoopTilingPass
    : public FunctionPass<SimpleParametricLoopTilingPass> {
public:
  struct Options : public PassOptions<Options> {
    List<int> clOuterLoopSizes{
        *this, "test-outer-loop-sizes", llvm::cl::MiscFlags::CommaSeparated,
        llvm::cl::desc(
            "fixed number of iterations that the outer loops should have")};
  };

  explicit SimpleParametricLoopTilingPass(ArrayRef<int64_t> outerLoopSizes)
      : sizes(outerLoopSizes.begin(), outerLoopSizes.end()) {}
  explicit SimpleParametricLoopTilingPass(const Options &options) {
    sizes.assign(options.clOuterLoopSizes.begin(),
                 options.clOuterLoopSizes.end());
  }

  void runOnFunction() override {
    FuncOp func = getFunction();
    func.walk([this](loop::ForOp op) {
      // Ignore nested loops.
      if (op.getParentRegion()->getParentOfType<loop::ForOp>())
        return;
      extractFixedOuterLoops(op, sizes);
    });
  }

  SmallVector<int64_t, 4> sizes;
};
} // end namespace

std::unique_ptr<OpPassBase<FuncOp>>
mlir::createSimpleParametricTilingPass(ArrayRef<int64_t> outerLoopSizes) {
  return std::make_unique<SimpleParametricLoopTilingPass>(outerLoopSizes);
}

static PassRegistration<SimpleParametricLoopTilingPass,
                        SimpleParametricLoopTilingPass::Options>
    reg("test-extract-fixed-outer-loops",
        "test application of parametric tiling to the outer loops so that the "
        "ranges of outer loops become static");

//===- TestLoopParametricTiling.cpp --- Parametric loop tiling pass -------===//
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

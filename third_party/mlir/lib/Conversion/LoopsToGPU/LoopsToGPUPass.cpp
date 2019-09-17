//===- LoopsToGPUPass.cpp - Convert a loop nest to a GPU kernel -----------===//
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

#include "mlir/Conversion/LoopsToGPU/LoopsToGPUPass.h"
#include "mlir/Conversion/LoopsToGPU/LoopsToGPU.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/CommandLine.h"

#define PASS_NAME "convert-loops-to-gpu"

using namespace mlir;
using namespace mlir::loop;

static llvm::cl::OptionCategory clOptionsCategory(PASS_NAME " options");
static llvm::cl::opt<unsigned>
    clNumBlockDims("gpu-block-dims",
                   llvm::cl::desc("Number of GPU block dimensions for mapping"),
                   llvm::cl::cat(clOptionsCategory), llvm::cl::init(1u));
static llvm::cl::opt<unsigned> clNumThreadDims(
    "gpu-thread-dims",
    llvm::cl::desc("Number of GPU thread dimensions for mapping"),
    llvm::cl::cat(clOptionsCategory), llvm::cl::init(1u));

namespace {
// A pass that traverses top-level loops in the function and converts them to
// GPU launch operations.  Nested launches are not allowed, so this does not
// walk the function recursively to avoid considering nested loops.
struct ForLoopMapper : public FunctionPass<ForLoopMapper> {
  ForLoopMapper(unsigned numBlockDims, unsigned numThreadDims)
      : numBlockDims(numBlockDims), numThreadDims(numThreadDims) {}

  void runOnFunction() override {
    for (Block &block : getFunction())
      for (Operation &op : llvm::make_early_inc_range(block)) {
        if (auto forOp = dyn_cast<AffineForOp>(&op)) {
          if (failed(convertAffineLoopNestToGPULaunch(forOp, numBlockDims,
                                                      numThreadDims)))
            signalPassFailure();
        } else if (auto forOp = dyn_cast<ForOp>(&op)) {
          if (failed(convertLoopNestToGPULaunch(forOp, numBlockDims,
                                                numThreadDims)))
            signalPassFailure();
        }
      }
  }

  unsigned numBlockDims;
  unsigned numThreadDims;
};
} // namespace

std::unique_ptr<OpPassBase<FuncOp>>
mlir::createSimpleLoopsToGPUPass(unsigned numBlockDims,
                                 unsigned numThreadDims) {
  return std::make_unique<ForLoopMapper>(numBlockDims, numThreadDims);
}

static PassRegistration<ForLoopMapper>
    registration(PASS_NAME, "Convert top-level loops to GPU kernels", [] {
      return std::make_unique<ForLoopMapper>(clNumBlockDims.getValue(),
                                             clNumThreadDims.getValue());
    });

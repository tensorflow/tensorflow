/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This files implements the logic for converting multidimensional
// `scf.parallel` loops into 1D loops.

#include <memory>
#include <numeric>
#include <vector>

#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"

using ::mlir::scf::ParallelOp;

namespace mlir {

#define GEN_PASS_DEF_COLLAPSEPARALLELLOOPSTO1DPASS
#include "mlir-hlo/Transforms/passes.h.inc"

namespace {

// This is the implementation of the CollapseParallelLoopsTo1D pass declared in
//  include/mlir-hlo/Transforms/passes.td
struct CollapseParallelLoopsTo1D
    : public impl::CollapseParallelLoopsTo1DPassBase<
          CollapseParallelLoopsTo1D> {
  void runOnOperation() override;
};

}  // namespace
}  // namespace mlir

using namespace mlir;

void mlir::CollapseParallelLoopsTo1D::runOnOperation() {
  getOperation()->walk([&](ParallelOp op) {
    unsigned numLoops = op.getNumLoops();
    if (numLoops == 1) return;
    std::vector<unsigned> combinedLoops(numLoops);
    std::iota(combinedLoops.begin(), combinedLoops.end(), 0u);
    mlir::collapseParallelLoops(op, {combinedLoops});
  });
}

std::unique_ptr<OperationPass<>> mlir::createCollapseParallelLoopsTo1DPass() {
  return std::make_unique<CollapseParallelLoopsTo1D>();
}

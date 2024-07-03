/* Copyright 2022 The OpenXLA Authors.

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

// This files implements a pass that partially bufferized IR.

#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Value.h"
#include "transforms/passes.h"

namespace mlir {

#define GEN_PASS_DEF_UNROLLLOOPSPASS
#include "transforms/passes.h.inc"

namespace {
class UnrollLoopsPass : public impl::UnrollLoopsPassBase<UnrollLoopsPass> {
 public:
  using UnrollLoopsPassBase<UnrollLoopsPass>::UnrollLoopsPassBase;

 private:
  void runOnOperation() override;
};
}  // namespace

// Returns constant or negative value.
static int64_t getConstant(Value value) {
  auto op = value.getDefiningOp<arith::ConstantIndexOp>();
  return op ? op.value() : -1;
}

void UnrollLoopsPass::runOnOperation() {
  getOperation()->walk([&](scf::ForOp op) {
    int64_t lower = getConstant(op.getLowerBound());
    int64_t upper = getConstant(op.getUpperBound());
    int64_t step = getConstant(op.getStep());
    if (lower < 0 || upper < 0 || step <= 0 || lower >= upper) return;
    int64_t count = (upper - lower + step - 1) / step;
    if (count > 8) return;
    (void)loopUnrollByFactor(op, count);
  });
}

std::unique_ptr<Pass> hlo::createUnrollLoopsPass() {
  return std::make_unique<UnrollLoopsPass>();
}

}  // namespace mlir

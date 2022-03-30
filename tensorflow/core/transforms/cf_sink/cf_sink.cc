/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/transforms/cf_sink/cf_sink.h"

#include <memory>

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Dominance.h"  // from @llvm-project
#include "mlir/Interfaces/ControlFlowInterfaces.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/ControlFlowSinkUtils.h"  // from @llvm-project
#include "tensorflow/core/ir/interfaces.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/transforms/pass_detail.h"

namespace mlir {
namespace tfg {

namespace {
struct ControlFlowSinkPass : public ControlFlowSinkBase<ControlFlowSinkPass> {
  void runOnOperation() override;
};
}  // namespace

static bool IsStateless(Operation *op) {
  if (auto registry = dyn_cast<TensorFlowRegistryInterface>(op))
    return !registry.isStateful();
  return false;
}

void ControlFlowSinkPass::runOnOperation() {
  auto &domInfo = getAnalysis<DominanceInfo>();
  getOperation()->walk([&](RegionBranchOpInterface branch) {
    SmallVector<Region *> regions;
    getSinglyExecutedRegionsToSink(branch, regions);
    controlFlowSink(
        regions, domInfo,
        [&](Operation *op, Region *) { return IsStateless(op); },
        [](Operation *op, Region *region) {
          op->moveBefore(&region->front(), region->front().begin());
        });
  });
}

std::unique_ptr<Pass> CreateControlFlowSinkPass() {
  return std::make_unique<ControlFlowSinkPass>();
}

}  // namespace tfg
}  // namespace mlir

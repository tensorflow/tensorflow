//===- TestLoopMapping.cpp --- Parametric loop mapping pass ---------------===//
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
// This file implements a pass to parametrically map loop.for loops to virtual
// processing element dimensions.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace {
class TestLoopMappingPass : public FunctionPass<TestLoopMappingPass> {
public:
  explicit TestLoopMappingPass() {}

  void runOnFunction() override {
    FuncOp func = getFunction();

    // SSA values for the transformation are created out of thin air by
    // unregistered "new_processor_id_and_range" operations. This is enough to
    // emulate mapping conditions.
    SmallVector<Value, 8> processorIds, numProcessors;
    func.walk([&processorIds, &numProcessors](Operation *op) {
      if (op->getName().getStringRef() != "new_processor_id_and_range")
        return;
      processorIds.push_back(op->getResult(0));
      numProcessors.push_back(op->getResult(1));
    });

    func.walk([&processorIds, &numProcessors](loop::ForOp op) {
      // Ignore nested loops.
      if (op.getParentRegion()->getParentOfType<loop::ForOp>())
        return;
      mapLoopToProcessorIds(op, processorIds, numProcessors);
    });
  }
};
} // end namespace

static PassRegistration<TestLoopMappingPass>
    reg("test-mapping-to-processing-elements",
        "test mapping a single loop on a virtual processor grid",
        [] { return std::make_unique<TestLoopMappingPass>(); });

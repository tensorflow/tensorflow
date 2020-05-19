/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

namespace mlir {
namespace xla_hlo {

namespace {

// A pass that sinks constants implicitly captured in control flow regions. This
// is necessary to export to XLA.
class SinkConstantsToControlFlow
    : public mlir::PassWrapper<SinkConstantsToControlFlow, FunctionPass> {
  void runOnFunction() override {
    getFunction().walk([](Operation* op) {
      if (auto while_op = llvm::dyn_cast<WhileOp>(op)) {
        SinkToRegion(&while_op.body());
        SinkToRegion(&while_op.cond());
      } else if (auto cond_op = llvm::dyn_cast<ConditionalOp>(op)) {
        SinkToRegion(&cond_op.true_branch());
        SinkToRegion(&cond_op.false_branch());
      }
    });
  }

 private:
  // Performs constant sinking into a region.
  static void SinkToRegion(Region* region) {
    llvm::DenseMap<Value, ConstOp> sunk_constant;
    visitUsedValuesDefinedAbove({*region}, [&](OpOperand* use) {
      Value constant = use->get();
      auto const_op = dyn_cast_or_null<ConstOp>(constant.getDefiningOp());
      if (!const_op) return;
      auto map_entry = sunk_constant.try_emplace(constant, nullptr);
      if (!map_entry.second) {
        // This constant has already been cloned into the region, reuse it.
        use->set(map_entry.first->getSecond().getResult());
        if (constant.use_empty()) const_op.erase();
        return;
      }
      if (constant.hasOneUse()) {
        const_op.getOperation()->moveBefore(&region->front().front());
        return;
      }
      map_entry.first->getSecond() = const_op.clone();
      region->front().getOperations().insert(region->front().begin(),
                                             map_entry.first->getSecond());
      use->set(map_entry.first->getSecond().getResult());
    });
  }
};

static mlir::PassRegistration<SinkConstantsToControlFlow> pass(
    "xla-hlo-sink-constants-to-control-flow",
    "Sink constants implicitly captured in control flow regions. This is "
    "necessary to export to XLA.");

}  // anonymous namespace

std::unique_ptr<OperationPass<FuncOp>> createSinkConstantsToControlFlowPass() {
  return std::make_unique<SinkConstantsToControlFlow>();
}

}  // namespace xla_hlo
}  // namespace mlir

/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>

#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFTPU {

namespace {

#define GEN_PASS_DEF_TPUCOLOCATESPLITSPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

constexpr char kDeviceAttr[] = "device";
// Attribute of colocation classes.
constexpr char kClassAttr[] = "_class";

bool HasDevice(Operation* op) {
  auto attr = op->getAttrOfType<StringAttr>(kDeviceAttr);
  if (!attr) return false;
  return !attr.getValue().empty();
}

// Returns the predecessors of `op` when `op`'s predecessors are wrapped by
// islands.
llvm::SmallVector<Operation*> IslandPredecessors(Operation* op) {
  llvm::SmallVector<Operation*> predecessors;
  for (Value operand : op->getOperands()) {
    if (Operation* pred = operand.getDefiningOp()) {
      int result_number = llvm::cast<OpResult>(operand).getResultNumber();
      if (auto pred_island = llvm::dyn_cast<tf_executor::IslandOp>(pred)) {
        Value yield_operand = pred_island.GetYield().getOperand(result_number);
        predecessors.push_back(yield_operand.getDefiningOp());
      }
    }
  }
  return predecessors;
}

struct TPUColocateSplits
    : public impl::TPUColocateSplitsPassBase<TPUColocateSplits> {
  void runOnOperation() override;
};

void TPUColocateSplits::runOnOperation() {
  getOperation().walk([&](Operation* op) {
    if (auto split = llvm::dyn_cast<TF::SplitOp>(op)) {
      if (HasDevice(split) || split->getAttrOfType<ArrayAttr>(kClassAttr))
        return WalkResult::advance();
      for (Operation* pred : IslandPredecessors(split)) {
        if (auto colocation_classes =
                pred->getAttrOfType<ArrayAttr>(kClassAttr)) {
          split->setAttr(kClassAttr, colocation_classes);
          return WalkResult::advance();
        }
      }
    }
    return WalkResult::advance();
  });
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateTPUColocateSplitsPass() {
  return std::make_unique<TPUColocateSplits>();
}

}  // namespace TFTPU
}  // namespace mlir

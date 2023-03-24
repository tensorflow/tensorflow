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

#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TF {

namespace {

constexpr char kShapeInvariantAttr[] = "shape_invariant";

#define GEN_PASS_DEF_DROPWHILESHAPEINVARIANTPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

class DropWhileShapeInvariantPass
    : public impl::DropWhileShapeInvariantPassBase<
          DropWhileShapeInvariantPass> {
  void runOnOperation() override;
};

#define GEN_PASS_DEF_DROPWHILESHAPEINVARIANTINDEVICECLUSTERPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

class DropWhileShapeInvariantInDeviceClusterPass
    : public impl::DropWhileShapeInvariantInDeviceClusterPassBase<
          DropWhileShapeInvariantInDeviceClusterPass> {
  void runOnOperation() override;
};

void DropWhileShapeInvariantAttr(Operation* op) {
  if (llvm::isa<WhileOp, WhileRegionOp>(op))
    op->removeAttr(kShapeInvariantAttr);
}
void DropWhileShapeInvariantPass::runOnOperation() {
  getOperation().walk([](Operation* op) { DropWhileShapeInvariantAttr(op); });
}

void DropWhileShapeInvariantInDeviceClusterPass::runOnOperation() {
  getOperation().walk([](tf_device::ClusterOp cluster) {
    cluster.walk([](Operation* op) { DropWhileShapeInvariantAttr(op); });
  });
}
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateDropWhileShapeInvariantPass() {
  return std::make_unique<DropWhileShapeInvariantPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
CreateDropWhileShapeInvariantInDeviceClusterPass() {
  return std::make_unique<DropWhileShapeInvariantInDeviceClusterPass>();
}

}  // namespace TF
}  // namespace mlir

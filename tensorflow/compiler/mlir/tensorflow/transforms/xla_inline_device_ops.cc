/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This pass remove Cluster ops by inlining Cluster ops.

#include "llvm/ADT/SmallVector.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_device_passes_detail.h"

namespace mlir {

namespace {
// XlaInlineDeviceOps Pass will inline cluster op based in the parent region.
struct XlaInlineDeviceOpsPass
    : public TFDevice::XlaInlineDeviceOpsPassBase<XlaInlineDeviceOpsPass> {
  void runOnOperation() override;
};

void InlineDeviceOp(tf_device::ClusterOp cluster_op) {
  auto& block = cluster_op.GetBody();

  llvm::SmallVector<Operation*, 4> non_terminator_ops;
  for (Operation& op : block.without_terminator()) {
    non_terminator_ops.push_back(&op);
  }
  for (auto op : non_terminator_ops) {
    op->moveBefore(cluster_op);
  }

  auto& return_op = cluster_op.GetBody().front();
  // This is the last op, should be tf_device::ReturnOp.
  assert(mlir::isa<mlir::tf_device::ReturnOp>(return_op));

  Value old_val, new_val;
  for (auto it : llvm::zip(cluster_op.getResults(), return_op.getOperands())) {
    std::tie(old_val, new_val) = it;
    old_val.replaceAllUsesExcept(new_val, &return_op);
  }

  cluster_op.erase();
}

void XlaInlineDeviceOpsPass::runOnOperation() {
  ModuleOp module = getOperation();

  llvm::SmallVector<tf_device::ClusterOp, 4> ops;
  module.walk(
      [&](tf_device::ClusterOp cluster_op) { ops.push_back(cluster_op); });

  for (auto cluster_op : ops) {
    InlineDeviceOp(cluster_op);
  }
}

}  // namespace

namespace TFDevice {
std::unique_ptr<OperationPass<ModuleOp>> CreateXlaInlineDeviceOpsPass() {
  return std::make_unique<XlaInlineDeviceOpsPass>();
}
}  // namespace TFDevice

}  // namespace mlir

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

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

// This pass eliminate `_tpu_replicate` and `device` attribute on operations
// that are contained in a tf_device.cluster op.

namespace mlir {
namespace TFTPU {

namespace {

constexpr char kTPUReplicateAttr[] = "_tpu_replicate";
constexpr char kDeviceAttr[] = "device";

class TPUCleanupClusterAttributesPass
    : public PassWrapper<TPUCleanupClusterAttributesPass,
                         OperationPass<ModuleOp>> {
 public:
  void runOnOperation() override {
    getOperation().walk([](tf_device::ClusterOp cluster) {
      cluster.walk([](Operation *op) {
        if (isa<tf_device::ClusterOp>(op)) return;
        for (StringRef attr : {kTPUReplicateAttr, kDeviceAttr})
          op->removeAttr(attr);
      });
    });
  }
};

PassRegistration<TPUCleanupClusterAttributesPass> pass(
    "tf-tpu-cleanup-cluster-attributes",
    "Eliminate _tpu_replicate and other attributes from ops in a cluster");

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateTPUClusterCleanupAttributesPass() {
  return std::make_unique<TPUCleanupClusterAttributesPass>();
}

}  // namespace TFTPU
}  // namespace mlir

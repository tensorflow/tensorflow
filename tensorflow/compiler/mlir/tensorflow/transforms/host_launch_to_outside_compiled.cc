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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Analysis/CallGraph.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_device_passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_cluster_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"

namespace mlir {
namespace TFDevice {

namespace {

constexpr char kDeviceAttr[] = "device";
constexpr char kXlaOutsideCompilationAttr[] = "_xla_outside_compilation";

struct HostLaunchToOutsideCompiledPass
    : public HostLaunchToOutsideCompiledPassBase<
          HostLaunchToOutsideCompiledPass> {
  void runOnOperation() override;
};

// Assign all ops in region with _xla_outside_compilation attribute.
void MarkOutsideCompiledInRegion(Region& region) {
  region.walk([&](Operation* op) {
    op->setAttr(kXlaOutsideCompilationAttr,
                StringAttr::get(op->getContext(), "from_launch"));
  });
}

void HoistOpsAndAnnotateWithOutsideCompilation(tf_device::LaunchOp launch) {
  // Forward launch inner op results to launch op results.
  launch.replaceAllUsesWith(launch.GetBody().getTerminator()->getOperands());

  // For all inner ops, assign the launch device as a `device` attribute.
  MarkOutsideCompiledInRegion(launch.body());

  // Move all inner ops of the launch to the block containing the launch.
  auto body = launch.GetBody().without_terminator();
  Operation* launch_op = launch.getOperation();
  launch_op->getBlock()->getOperations().splice(
      launch_op->getIterator(), launch.GetBody().getOperations(), body.begin(),
      body.end());

  launch.erase();
}

void HostLaunchToOutsideCompiledPass::runOnOperation() {
  auto traverse_op = [&](Operation* op, tf_device::ClusterOp tpu_cluster,
                         std::optional<std::string> host_device) {
    // Hoist launch.
    if (tf_device::LaunchOp launch = dyn_cast<tf_device::LaunchOp>(op)) {
      StringAttr device_attr = launch->getAttrOfType<StringAttr>(kDeviceAttr);
      if (host_device && device_attr &&
          device_attr.getValue().equals(*host_device))
        HoistOpsAndAnnotateWithOutsideCompilation(launch);
    }
    return WalkResult::advance();
  };

  ModuleOp module = getOperation();
  // If there is model parallelism, we return early since
  // GetHostDeviceOutsideComputation will fail and an error should have been
  // returned in an earlier pass.
  // TODO(b/186420116): Remove this check once outside compilation and model
  // parallelism work together.
  auto result = module.walk([&](tf_device::ClusterOp tpu_cluster) {
    if (tensorflow::HasModelParallelism(tpu_cluster)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) return;

  if (failed(TFTPU::WalkReachableFromTpuCluster(module, traverse_op)))
    return signalPassFailure();
}

}  // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateHostLaunchToOutsideCompiledPass() {
  return std::make_unique<HostLaunchToOutsideCompiledPass>();
}

}  // namespace TFDevice
}  // namespace mlir

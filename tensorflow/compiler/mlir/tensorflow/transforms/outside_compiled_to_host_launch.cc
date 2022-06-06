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

#include "llvm/ADT/SmallVector.h"
#include "mlir/Analysis/CallGraph.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_cluster_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"

namespace mlir {
namespace TFTPU {

namespace {

constexpr char kDeviceAttr[] = "device";
constexpr char kXlaOutsideCompilationAttr[] = "_xla_outside_compilation";

struct OutsideCompiledToHostLaunchPass
    : public TF::OutsideCompiledToHostLaunchPassBase<
          OutsideCompiledToHostLaunchPass> {
  void runOnOperation() override;
};

void WrapOpInLaunch(Operation* host_op, llvm::StringRef host_device) {
  OpBuilder builder(host_op);

  auto launch_op = builder.create<tf_device::LaunchOp>(
      host_op->getLoc(), builder.getStringAttr(host_device),
      /*result_types=*/host_op->getResultTypes());
  host_op->replaceAllUsesWith(launch_op);

  launch_op.body().push_back(new Block);
  builder.setInsertionPointToEnd(&launch_op.GetBody());
  auto* return_op =
      builder
          .create<tf_device::ReturnOp>(host_op->getLoc(), host_op->getResults())
          .getOperation();
  MLIRContext* context = launch_op.getContext();
  host_op->removeAttr(StringAttr::get(context, kXlaOutsideCompilationAttr));
  host_op->removeAttr(StringAttr::get(context, kDeviceAttr));
  host_op->moveBefore(return_op);
}

void OutsideCompiledToHostLaunchPass::runOnOperation() {
  // traverse_op is applied to each op reachable from each tf_device::ClusterOp
  // in the module returned by getOperation().
  auto traverse_op = [&](Operation* op, tf_device::ClusterOp tpu_cluster,
                         std::optional<std::string> host_device) {
    // Apply WrapOpInLaunch when the op has _xla_outside_compilation.
    if (op->hasAttrOfType<StringAttr>(kXlaOutsideCompilationAttr)) {
      if (!host_device) {
        tpu_cluster.emitOpError(
            "outside compilation is not supported with model parallelism.");
        return WalkResult::interrupt();
      }
      WrapOpInLaunch(op, *host_device);
    }
    return WalkResult::advance();
  };
  if (failed(WalkReachableFromTpuCluster(getOperation(), traverse_op)))
    return signalPassFailure();
}

}  // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateOutsideCompiledToHostLaunchPass() {
  return std::make_unique<OutsideCompiledToHostLaunchPass>();
}

}  // namespace TFTPU
}  // namespace mlir

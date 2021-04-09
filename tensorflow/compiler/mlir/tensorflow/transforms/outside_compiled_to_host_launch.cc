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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"

namespace mlir {
namespace TFTPU {

namespace {

constexpr char kDeviceAttr[] = "device";
constexpr char kXlaOutsideCompilationAttr[] = "_xla_outside_compilation";

// This pass wraps ops with the same `_xla_outside_compilation`
// attribute value in a tf_device.launch op with host device assignment.
//
// A simple example:
//   "tf_device.cluster"() ( {
//     "tf.A"()
//     "tf.B"() {_xla_outside_compilation = "cluster1"}
//     "tf.C"()
//     tf_device.return
//   }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []}
//
// Would become the following ops (unimportant attribute, type are omitted):
//   "tf_device.cluster"() ( {
//     "tf.A"()
//     "tf_device.launch"() {
//       "tf.B"() {_xla_outside_compilation = "cluster1"}
//       tf_device.return
//     } {device = "TPU_REPLICATED_HOST"} : () -> ()
//     "tf.C"()
//     tf_device.return
//   }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []}
//

struct OutsideCompiledToHostLaunch
    : public PassWrapper<OutsideCompiledToHostLaunch, OperationPass<ModuleOp>> {
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
  host_op->removeAttr(Identifier::get(kXlaOutsideCompilationAttr, context));
  host_op->removeAttr(Identifier::get(kDeviceAttr, context));
  host_op->moveBefore(return_op);
}

void OutsideCompiledToHostLaunch::runOnOperation() {
  // Get runtime devices information from the closest parent module.
  auto module = getOperation();
  mlir::TF::RuntimeDevices devices;
  if (failed(tensorflow::GetDevicesFromOp(module, &devices)))
    return signalPassFailure();

  auto result = module.walk([&](tf_device::ClusterOp tpu_cluster) {
    std::string host_device;
    (void)tensorflow::GetHostDeviceOutsideComputation(devices, tpu_cluster,
                                                      &host_device);
    tpu_cluster.walk([&](Operation* op) {
      if (op->hasAttrOfType<StringAttr>(kXlaOutsideCompilationAttr))
        WrapOpInLaunch(op, host_device);
    });
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) return signalPassFailure();
}

}  // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateOutsideCompiledToHostLaunchPass() {
  return std::make_unique<OutsideCompiledToHostLaunch>();
}

static PassRegistration<OutsideCompiledToHostLaunch> pass(
    "tf-outside-compiled-to-host-launch",
    "Wraps each op with the _xla_outside_compiled attribute in "
    "a separate tf_device.launch on replicated host device.");

}  // namespace TFTPU
}  // namespace mlir

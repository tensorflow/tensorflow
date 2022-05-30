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
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_device_passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"

namespace mlir {
namespace TFDevice {

namespace {

constexpr char kDeviceAttr[] = "device";

struct DeviceAttributeToLaunch
    : public DeviceAttributeToLaunchPassBase<DeviceAttributeToLaunch> {
  void runOnOperation() override;
};

void WrapOpInLaunch(Operation* op, llvm::StringRef device) {
  OpBuilder builder(op);

  auto launch_op = builder.create<tf_device::LaunchOp>(
      op->getLoc(), builder.getStringAttr(device),
      /*result_types=*/op->getResultTypes());
  op->replaceAllUsesWith(launch_op);

  launch_op.body().push_back(new Block);
  builder.setInsertionPointToEnd(&launch_op.GetBody());
  auto* return_op =
      builder.create<tf_device::ReturnOp>(op->getLoc(), op->getResults())
          .getOperation();
  MLIRContext* context = launch_op.getContext();
  op->removeAttr(StringAttr::get(context, kDeviceAttr));
  op->moveBefore(return_op);
}

void DeviceAttributeToLaunch::runOnOperation() {
  const Dialect* tf_dialect = getContext().getLoadedDialect("tf");

  getOperation().walk([&](Operation* op) {
    if (op->getDialect() != tf_dialect) return WalkResult::advance();
    if (auto device = op->getAttrOfType<StringAttr>(kDeviceAttr)) {
      if (!device.getValue().empty()) WrapOpInLaunch(op, device.getValue());
    }
    return WalkResult::advance();
  });
}

}  // anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateDeviceAttributeToLaunchPass() {
  return std::make_unique<DeviceAttributeToLaunch>();
}

}  // namespace TFDevice
}  // namespace mlir

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

// This pass sets the device ordinal attribute of the required op using
// the replica id attribute.

#include <memory>

#include "llvm/Support/Casting.h"
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"

namespace mlir {
namespace TFDevice {
namespace {
constexpr char kReplicaIdAttr[] = "_xla_replica_id";
constexpr char kDeviceOrdinalAttr[] = "device_ordinal";

#define GEN_PASS_DEF_REPLICAIDTODEVICEORDINALPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

struct ReplicaIDToDeviceOrdinalPass
    : public impl::ReplicaIDToDeviceOrdinalPassBase<
          ReplicaIDToDeviceOrdinalPass> {
  void runOnOperation() override;
};

// Returns whether op requires `device_ordinal` attribute.
bool RequiresDeviceOrdinalAttribute(Operation* op) {
  return (llvm::isa<TF::EnqueueTPUEmbeddingSparseTensorBatchOp,
                    TF::EnqueueTPUEmbeddingRaggedTensorBatchOp,
                    TF::EnqueueTPUEmbeddingArbitraryTensorBatchOp>(op) &&
          op->hasAttr(kDeviceOrdinalAttr) && op->hasAttr(kReplicaIdAttr));
}

void ReplicaIDToDeviceOrdinalPass::runOnOperation() {
  const Dialect* tf_dialect = getContext().getLoadedDialect("tf");
  if (!tf_dialect) {
    getOperation().emitError() << "'tf' dialect is not registered";
    return signalPassFailure();
  }

  // Get the number of devices per host.
  int device_num = 0;
  mlir::TF::RuntimeDevices devices;
  if (failed(tensorflow::GetDevicesFromOp(
          getOperation()->getParentOfType<ModuleOp>(), &devices)))
    return signalPassFailure();
  for (const auto& device_name : devices.device_names()) {
    if (device_name.has_type && device_name.type == "TPU") ++device_num;
  }

  if (device_num == 0) return;

  llvm::SmallVector<Operation*, 4> require_device_ordinal_ops;
  getOperation().walk([&](Operation* op) {
    if (RequiresDeviceOrdinalAttribute(op)) {
      require_device_ordinal_ops.push_back(op);
    }
  });

  if (require_device_ordinal_ops.size() == 1) {
    // If there is only one op which requires the device ordinal being set,
    // set the device ordinal to 0. Note: This is for single device use case
    // (eg. pf megacore) for which `_xla_replica_id` isn't set via the
    // replicate_to_islands pass.
    Operation* op = require_device_ordinal_ops.front();
    if (op->getAttrOfType<IntegerAttr>(kDeviceOrdinalAttr).getInt() == -1) {
      OpBuilder builder(op);
      op->setAttr(kDeviceOrdinalAttr, builder.getI64IntegerAttr(0));
    }
  } else {
    // If the device ordinal attribute is -1, set it with the replica id
    // attribute modulo the number of TPU cores in the system.
    for (auto op : require_device_ordinal_ops) {
      if (op->getAttrOfType<IntegerAttr>(kDeviceOrdinalAttr).getInt() == -1) {
        OpBuilder builder(op);
        int device_ordinal =
            op->getAttrOfType<IntegerAttr>(kReplicaIdAttr).getInt() %
            device_num;
        op->setAttr(kDeviceOrdinalAttr,
                    builder.getI64IntegerAttr(device_ordinal));
      }
    }
  }
}
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateReplicaIDToDeviceOrdinalPass() {
  return std::make_unique<ReplicaIDToDeviceOrdinalPass>();
}

}  // namespace TFDevice
}  // namespace mlir

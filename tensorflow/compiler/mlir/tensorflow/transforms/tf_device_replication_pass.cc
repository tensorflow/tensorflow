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

// This pass hoists a `tf_device.replicate` body and replicates each TensorFlow
// dialect op in the body based on its `device` attribute and the `devices`
// attribute on the `tf_device.replicate`.

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFDevice {
namespace {

constexpr char kDeviceAttr[] = "device";

class TFDeviceReplicationPass
    : public PassWrapper<TFDeviceReplicationPass, OperationPass<ModuleOp>> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TF::TensorFlowDialect>();
  }

  StringRef getArgument() const final { return "tf-device-replication"; }

  StringRef getDescription() const final {
    return "Hoists and replicates the tf_device.replicate inner ops once for "
           "each associated device.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    const Dialect *tf_dialect = getContext().getLoadedDialect("tf");
    module.walk([&](tf_device::ReplicateOp replicate_op) {
      OpBuilder builder(replicate_op);
      // Map from the existing operation in ReplicateOp's region to a list of
      // its replicated operations.
      llvm::DenseMap<Operation *, llvm::SmallVector<Operation *, 4>>
          operation_map;
      llvm::Optional<DictionaryAttr> devices = replicate_op.devices();
      const int replicate_num = replicate_op.n();

      // Replicates every operation in the region of the ReplicateOp to match
      // the number of devices.
      for (int i : llvm::seq<int>(0, replicate_num)) {
        // Gets the mapping from the packed and replicated block arguments to
        // the actual value. This mapping is used to replace the arguments used
        // by the cloned operations.
        BlockAndValueMapping mapping;
        for (BlockArgument &arg : replicate_op.GetBody().getArguments()) {
          Value new_arg =
              replicate_op.GetReplicaOperandForBlockArgument(arg, i);
          mapping.map(arg, new_arg);
        }
        for (Operation &op : replicate_op.GetBody().without_terminator()) {
          // Clones the operation and places it outside the replicate_op's body.
          llvm::SmallVector<Operation *, 4> &new_ops = operation_map[&op];
          Operation *new_op = builder.clone(op, mapping);
          new_ops.push_back(new_op);
          // If the op is a TF op, it has a string-valued device attribute and
          // the replicate_op has a list of devices corresponding to this device
          // attribute's value, updates the device attribute for this op.
          if (!devices) continue;

          if (op.getDialect() != tf_dialect) continue;

          StringAttr device_alias =
              new_op->getAttrOfType<StringAttr>(kDeviceAttr);
          if (!device_alias) continue;

          Attribute new_devices = devices->get(device_alias.getValue());
          if (!new_devices) continue;

          ArrayAttr new_devices_array = new_devices.cast<ArrayAttr>();
          new_op->setAttr(kDeviceAttr, new_devices_array[i].cast<StringAttr>());
        }
      }
      // Replaces usages of the existing results of the tf_device.replicate
      // op with the results of the newly replicated operations.
      llvm::SmallVector<Value, 4> new_results;
      for (Value v : replicate_op.GetBody().getTerminator()->getOperands()) {
        OpResult result = v.dyn_cast<OpResult>();
        // Uses the original value if the value is not an OpResult.
        if (!result) {
          for (int i = 0; i < replicate_num; ++i) new_results.push_back(v);
          continue;
        }
        // Uses the original value if the value is defined by an op outside the
        // tf_device.replicate's body.
        Operation *op = result.getDefiningOp();
        if (operation_map.find(op) == operation_map.end()) {
          for (int i = 0; i < replicate_num; ++i) new_results.push_back(v);
          continue;
        }
        // Uses the values defined by the newly replicated operations.
        int result_num = result.getResultNumber();
        for (Operation *new_op : operation_map[op]) {
          new_results.push_back(new_op->getResult(result_num));
        }
      }
      replicate_op.replaceAllUsesWith(new_results);
      replicate_op.erase();
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateTFDeviceReplicationPass() {
  return std::make_unique<TFDeviceReplicationPass>();
}

static PassRegistration<TFDeviceReplicationPass> pass;

}  // namespace TFDevice
}  // namespace mlir

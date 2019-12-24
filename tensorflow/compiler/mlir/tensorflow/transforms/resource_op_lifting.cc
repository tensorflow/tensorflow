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

// This pass lifts resource variable operations outside of device computation.

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BlockAndValueMapping.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Diagnostics.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Transforms/RegionUtils.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"

namespace mlir {
namespace TFDevice {

namespace {

constexpr char kDTypeAttr[] = "dtype";

// This pass lifts resource variable operations outside of device computation.
// This is useful because a lot of accelerator devices can not interact with
// resource variables directly..
//
// Here is a simple example in TensorFlow where a device doubles the value of a
// TensorFlow resource variable and returns new value:
//
// %resource_handle = "tf.VarHandleOp"()
// %1 = "tf_device.launch"() ( {
//   %init_value = "tf.ReadVariableOp"(%resource_handle)
//   "tf.AssignAddVariableOp"(%resource_handle, %init_value)
//   %new_value = "tf.ReadVariableOp"(%resource_handle)
//   tf_device.return %new_value
// })
//
// After this pass, the computation would become:
//
// %resource_handle = "tf.VarHandleOp"()
// %init_value = "tf.ReadVariableOp"(%resource_handle)
// %1:2 = "tf_device.launch"() ( {
//   %new_value = "tf.AddV2"(%init_value, %init_value)
//   tf_device.return %new_value, %new_value
// })
// "tf.AssignVariableOp"(%resource_handle, %1#1)
//
// You can see that there are a few main changes applied:
// 1) All the resource variable reads and writes are now outside of
//    tf_device.launch op.
// 2) Instead of taking resource handles as input, this device computation now
//    takes snapshotted values of that device.
// 3) Some resource load operations are eliminated with store-load forwarding.
// 4) Updated values to resource are appended to `tf_device.return` and used by
//    external resource store operations so that resources are still updated
//    after the computation.
struct ResourceOpLiftingPass : public FunctionPass<ResourceOpLiftingPass> {
  void runOnFunction() override;
};

// Performs store-load forwarding. This effectively removes
// 1) Any resource loads after a store to that same resource is done
// 2) Any resource stores except the last one.
// TODO(ycao): Store-load forwarding implemented here is only correct when
// computation is purely sequential (no concurrency). Need to support concurrent
// computation as well.
void ForwardStoreToLoad(tf_device::LaunchOp launch_op) {
  // resource_handle_to_last_store_op keeps track of the most recent (last)
  // store to each resource. Non-existent entry indicates that a resource has
  // not been stored to yet.
  llvm::SmallDenseMap<Value, TF::AssignVariableOp>
      resource_handle_to_last_store_op;

  // Only iterate through ops directly in launch_op's body as we can't handle
  // ops nested deeper in regions.
  for (Operation& op : llvm::make_early_inc_range(launch_op.GetBody())) {
    if (auto read_variable_op = dyn_cast<TF::ReadVariableOp>(&op)) {
      Value resource = read_variable_op.resource();
      auto last_store = resource_handle_to_last_store_op[resource];
      if (!last_store) continue;

      // Use stored value in last_store to replace all uses of current resource
      // load's result, then erase this resource load.
      read_variable_op.value()->replaceAllUsesWith(last_store.value());
      read_variable_op.erase();
      continue;
    }

    if (auto assign_variable_op = dyn_cast<TF::AssignVariableOp>(&op)) {
      Value resource = assign_variable_op.resource();
      auto last_store = resource_handle_to_last_store_op[resource];
      // Previous store ops to same resource can be erased.
      if (last_store) last_store.erase();

      resource_handle_to_last_store_op[resource] = assign_variable_op;
    }
  }
}

// Moves resource load operations to before launch_op. This assumes load-store
// forwarding has been performed on this launch_op such that all loads of same
// resource are on its initial values.
void HoistResourceLoads(tf_device::LaunchOp launch_op) {
  llvm::SmallDenseMap<Value, TF::ReadVariableOp> resource_to_read_ops;

  // Only iterate through ops directly in launch_op's body as we can't handle
  // ops nested deeper in regions.
  for (Operation& op : llvm::make_early_inc_range(launch_op.GetBody())) {
    auto read_variable_op = dyn_cast<TF::ReadVariableOp>(&op);
    if (!read_variable_op) continue;
    Value resource = read_variable_op.resource();

    // Skip resources created inside of launch_op.
    if (resource->getParentRegion() == &launch_op.body()) continue;

    auto p = resource_to_read_ops.insert({resource, read_variable_op});
    if (p.second) {
      op.moveBefore(launch_op);
      continue;
    }

    // Getting here means a load operation of this resource has been hoisted out
    // before. Use hoisted load result to replace all uses of current op result
    // and erase op.
    op.replaceAllUsesWith(p.first->second);
    op.erase();
  }
}

// If there are any stores to resource defined outside of launch_op's body
// region, the stored values must be returned by launch_op and its return op so
// that new values can be used by sunk resource stores.
// Returns true if any resource variable stored values are appended, otherwise
// false.
bool AppendResourceStoreValueToReturn(tf_device::LaunchOp launch_op) {
  bool has_resource_store = false;
  Block* body = &launch_op.GetBody();
  auto old_return = body->getTerminator();

  llvm::SmallVector<Value, 4> new_return_operands(old_return->getOperands());

  // Only iterate through ops directly in launch_op's body as we can't handle
  // ops nested deeper in regions.
  for (Operation& op : launch_op.GetBody()) {
    auto assign_variable_op = dyn_cast<TF::AssignVariableOp>(&op);
    if (!assign_variable_op) continue;
    Value resource = assign_variable_op.resource();
    if (!resource) continue;

    // Skip resources created inside of launch_op.
    if (resource->getParentRegion() == &launch_op.body()) continue;

    // TODO(ycao): Prevent same value from being returned multiple times.
    // TODO(ycao): Do not return resource store value if it is defined outside
    // of launch_op.
    new_return_operands.push_back(assign_variable_op.value());
    has_resource_store = true;
  }

  // If no resource stores are found, no need to update return op.
  if (!has_resource_store) return false;

  OpBuilder builder(old_return);
  builder.create<tf_device::ReturnOp>(old_return->getLoc(),
                                      new_return_operands);
  old_return->erase();
  return true;
}

// Moves resource store operations to after launch_op. This assumes load-store
// forwarding has been performed on this launch_op such that there is at most
// one resource store operation carrying its final value.
void SinkResourceStores(tf_device::LaunchOp launch_op, OpBuilder* builder) {
  // Update ReturnOp inside launch_op's body to output final values of updated
  // external resources.
  bool has_resource_store = AppendResourceStoreValueToReturn(launch_op);
  if (!has_resource_store) return;

  auto new_return_op = launch_op.GetBody().getTerminator();
  llvm::SmallVector<Type, 4> new_launch_return_types(
      new_return_op->getOperandTypes());

  builder->setInsertionPoint(launch_op);
  auto new_launch_op = builder->create<tf_device::LaunchOp>(
      launch_op.getLoc(), new_launch_return_types,
      /*operands=*/llvm::SmallVector<Value, 4>(), launch_op.getAttrs());
  new_launch_op.body().takeBody(launch_op.body());

  // Replace uses of old launch_op results with those of new_launch_op.
  for (auto p : llvm::zip(launch_op.getResults(), new_launch_op.getResults())) {
    std::get<0>(p)->replaceAllUsesWith(std::get<1>(p));
  }

  // Create a mapping from operands of new_return_op operands to new_launch_op
  // results.
  BlockAndValueMapping mapper;
  for (auto p :
       llvm::zip(new_return_op->getOperands(), new_launch_op.getResults())) {
    mapper.map(std::get<0>(p), std::get<1>(p));
  }

  // Clone all resource store ops and map their operands to values returned from
  // new_launch_op.
  for (Operation& op : llvm::make_early_inc_range(new_launch_op.GetBody())) {
    if (dyn_cast<TF::AssignVariableOp>(&op)) {
      builder->clone(op, mapper);
      op.erase();
    }
  }

  launch_op.erase();
}

// Hoists resource variable loads and sinks stores from launch_op.
void HoistResourceOpsFromLaunchOp(tf_device::LaunchOp launch_op) {
  ModuleOp m = launch_op.getParentOfType<ModuleOp>();
  OpBuilder builder(m);

  // Perform store-load forwarding. So that each resource is only loaded with
  // its initial value and is only stored with its final value.
  ForwardStoreToLoad(launch_op);

  // Move loads of external resources, if any, to before launch_op.
  HoistResourceLoads(launch_op);

  // Move stores of external resources, if any, to after launch_op.
  SinkResourceStores(launch_op, &builder);
}

}  // namespace

// Lifts resource operation from tf_device.launch_func ops nested in `op`
// outside.
void LiftResourceOps(Operation* op) {
  op->walk([](tf_device::LaunchOp launch_op) {
    HoistResourceOpsFromLaunchOp(launch_op);
  });
}

void ResourceOpLiftingPass::runOnFunction() { LiftResourceOps(getFunction()); }

std::unique_ptr<OpPassBase<FuncOp>> CreateResourceOpLiftingPass() {
  return std::make_unique<ResourceOpLiftingPass>();
}

static PassRegistration<ResourceOpLiftingPass> pass(
    "tf-resource-op-lifting",
    "Lifting resource operations out of device computation");

}  // namespace TFDevice
}  // namespace mlir

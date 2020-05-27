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

#include <cstdint>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"

namespace mlir {

namespace {

// This pass lifts resource variable operations outside of device computation.
// This is useful because a lot of accelerator devices can not interact with
// resource variables directly..
//
// Here is a simple example in TensorFlow where a device doubles the value of a
// TensorFlow resource variable and returns new value:
//
// %resource_handle = "tf.VarHandleOp"()
// %1 = "tf_device.cluster"() ( {
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
// %1:2 = "tf_device.cluster"() ( {
//   %new_value = "tf.AddV2"(%init_value, %init_value)
//   tf_device.return %new_value, %new_value
// })
// "tf.AssignVariableOp"(%resource_handle, %1#1)
//
// You can see that there are a few main changes applied:
// 1) All the resource variable reads and writes are now outside of
//    tf_device.cluster op.
// 2) Instead of taking resource handles as input, this device computation now
//    takes snapshotted values of that device.
// 3) Some resource load operations are eliminated with store-load forwarding.
// 4) Updated values to resource are appended to `tf_device.return` and used by
//    external resource store operations so that resources are still updated
//    after the computation.
//
// If the cluster body contains functional control flow, the pass first lifts
// the loads/stores in the body/cond/branch functions to the cluster body, then
// performs the above lifting. E.g.,
//
// func @cluster_with_loop() -> () {
//   %0 = "tf.VarHandleOp"() ...
//   "tf_device.cluster"() ( {
//      %1 = "tf.While"(%0) {body = @while_body, cond = @while_cond}
//      tf_device.return
//   })
//   return
// }
// func @while_body(%arg0: tensor<*x!tf.resource<tensor<f32>>>) {
//  %constant = "tf.Const"() ...
//  "tf.AssignVariableOp"(%arg0, %constant)
//  return %arg0
// }
// func @while_cond(%arg0: tensor<*x!tf.resource<tensor<f32>>>) {
//   %read = "tf.ReadVariableOp"(%arg0)
//   return %read
// }
//
// will be be transformed to:
//
// func @cluster_with_loop() {
//   %0 = "tf.VarHandleOp"() ...
//   %1 = "tf.ReadVariableOp"(%0)
//   %2 = "tf_device.cluster"() ( {
//     %3 = "tf.While"(%1) {body = @while_body, cond = @while_cond}
//     tf_device.return %3 : tensor<f32>
//   }) : () -> tensor<f32>
//   "tf.AssignVariableOp"(%0, %2)
//   return
// }
// func @while_body(%arg0: tensor<f32>) {
//   %0 = "tf.Const"() ...
//   return %0 : tensor<f32>
// }
// func @while_cond(%arg0: tensor<f32>) {
//   return %arg0
// }
//
struct ResourceOpLiftingPass
    : public PassWrapper<ResourceOpLiftingPass, OperationPass<ModuleOp>> {
  void runOnOperation() override;
};

// Removes identity nodes in the block. The device computation does not need
// such nodes to carry information.
void RemoveIdentity(Block* block) {
  for (auto& op : llvm::make_early_inc_range(*block)) {
    if (isa<TF::IdentityOp>(&op) || isa<TF::IdentityNOp>(&op)) {
      op.replaceAllUsesWith(op.getOperands());
      op.erase();
    }
  }
}

// Performs store-load forwarding. This effectively removes
// 1) Any resource loads after a store to that same resource is done
// 2) Any resource stores except the last one.
// TODO(ycao): Store-load forwarding implemented here is only correct when
// computation is purely sequential (no concurrency). Need to support concurrent
// computation as well.
void ForwardStoreToLoad(Block* block) {
  // resource_handle_to_last_store_op keeps track of the most recent (last)
  // store to each resource. Non-existent entry indicates that a resource has
  // not been stored to yet.
  llvm::SmallDenseMap<Value, TF::AssignVariableOp>
      resource_handle_to_last_store_op;

  // Only iterate through ops directly in the block as we can't handle ops
  // nested deeper in regions.
  for (Operation& op : llvm::make_early_inc_range(*block)) {
    if (auto read_variable_op = dyn_cast<TF::ReadVariableOp>(&op)) {
      Value resource = read_variable_op.resource();
      auto last_store = resource_handle_to_last_store_op[resource];
      if (!last_store) continue;

      // Use stored value in last_store to replace all uses of current resource
      // load's result, then erase this resource load.
      read_variable_op.value().replaceAllUsesWith(last_store.value());
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

// Moves resource load operations with the provided `move_load` function. This
// assumes load-store forwarding has been performed on this block such that
// all loads of same resource are on its initial values. A `skip_load` functions
// is used to indicate whether a load should be skipped. If there are multiple
// loads on the same resource, only the first one will be moved, and the later
// ones will be removed and replaced with the first one.
void HoistResourceLoads(
    Block* block, llvm::function_ref<bool(TF::ReadVariableOp)> skip_load,
    llvm::function_ref<void(TF::ReadVariableOp)> move_load) {
  llvm::SmallDenseMap<Value, TF::ReadVariableOp> resource_to_read_ops;

  // Only iterate through ops directly in the body as we can't handle
  // ops nested deeper in regions.
  for (Operation& op : llvm::make_early_inc_range(*block)) {
    auto read_variable_op = dyn_cast<TF::ReadVariableOp>(&op);
    if (!read_variable_op) continue;
    if (skip_load(read_variable_op)) continue;

    Value resource = read_variable_op.resource();
    auto p = resource_to_read_ops.insert({resource, read_variable_op});
    if (p.second) {
      move_load(read_variable_op);
      continue;
    }

    // Getting here means a load operation of this resource has been hoisted out
    // before. Use hoisted load result to replace all uses of current op result
    // and erase op.
    op.replaceAllUsesWith(p.first->second);
    op.erase();
  }
}

// If there are any stores to resource defined outside of the block then the
// stored values must be returned so that new values can be used by sunk
// resource stores.
// Returns true if any resource variable stored values are appended, otherwise
// false.
bool AppendResourceStoreValueToReturn(Block* body) {
  bool has_resource_store = false;
  auto old_return = body->getTerminator();

  llvm::SmallVector<Value, 4> new_return_operands(old_return->getOperands());

  // Only iterate through ops directly in the body as we can't handle ops nested
  // deeper in regions.
  for (auto assign_variable_op : body->getOps<TF::AssignVariableOp>()) {
    Value resource = assign_variable_op.resource();
    if (!resource) continue;

    // Skip resources created inside of the body.
    if (resource.getParentRegion() == body->getParent()) continue;

    // TODO(ycao): Prevent same value from being returned multiple times.
    // TODO(ycao): Do not return resource store value if it is defined outside
    // of cluster.
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

// Moves resource store operations to after cluster. This assumes load-store
// forwarding has been performed on this cluster such that there is at most one
// resource store operation carrying its final value.
tf_device::ClusterOp SinkResourceStores(tf_device::ClusterOp cluster,
                                        OpBuilder* builder) {
  // Update ReturnOp inside cluster's body to output final values of updated
  // external resources.
  if (!AppendResourceStoreValueToReturn(&cluster.GetBody())) return cluster;

  auto new_return_op = cluster.GetBody().getTerminator();
  llvm::SmallVector<Type, 4> new_return_types(new_return_op->getOperandTypes());

  builder->setInsertionPoint(cluster);
  auto new_cluster = builder->create<tf_device::ClusterOp>(
      cluster.getLoc(), new_return_types,
      /*operands=*/llvm::SmallVector<Value, 4>(), cluster.getAttrs());
  new_cluster.body().takeBody(cluster.body());

  // Replace uses of old cluster results with those of new_cluster.
  for (auto result : llvm::zip(cluster.getResults(), new_cluster.getResults()))
    std::get<0>(result).replaceAllUsesWith(std::get<1>(result));

  // Create a mapping from operands of new_return_op operands to new_cluster
  // results.
  BlockAndValueMapping mapper;
  for (auto operand_result :
       llvm::zip(new_return_op->getOperands(), new_cluster.getResults()))
    mapper.map(std::get<0>(operand_result), std::get<1>(operand_result));

  // Clone all resource store ops and map their operands to values returned from
  // new_cluster.
  for (Operation& op : llvm::make_early_inc_range(new_cluster.GetBody())) {
    if (isa<TF::AssignVariableOp>(op)) {
      builder->clone(op, mapper);
      op.erase();
    }
  }

  cluster.erase();
  return new_cluster;
}

// Hoists resource variable loads and sinks stores from cluster.
LogicalResult HoistResourceOpsFromCluster(tf_device::ClusterOp cluster,
                                          ModuleOp module) {
  OpBuilder builder(module);

  // Remove identity nodes to avoid aliasing.
  RemoveIdentity(&cluster.GetBody());

  // Perform store-load forwarding. So that each resource is only loaded with
  // its initial value and is only stored with its final value.
  ForwardStoreToLoad(&cluster.GetBody());

  // Move loads of external resources, if any, to before cluster.
  // (Skipping resources created inside of cluster.)
  HoistResourceLoads(
      &cluster.GetBody(),
      /*skip_load=*/
      [&](TF::ReadVariableOp read) {
        return read.resource().getParentRegion() == &cluster.body();
      },
      /*move_load=*/
      [&](TF::ReadVariableOp read) {
        read.getOperation()->moveBefore(cluster);
      });

  // Move stores of external resources, if any, to after cluster.
  auto new_cluster = SinkResourceStores(cluster, &builder);

  llvm::SetVector<Value> captured_values;
  getUsedValuesDefinedAbove(new_cluster.body(), new_cluster.body(),
                            captured_values);

  for (Value v : captured_values) {
    auto tensor_type = v.getType().dyn_cast<TensorType>();
    if (!tensor_type) continue;
    if (!tensor_type.getElementType().isa<TF::ResourceType>()) continue;

    return new_cluster.emitOpError()
           << "has remaining resource inputs that can not be lifted";
  }

  return success();
}

// Holds information about a function's use of a resource argument.
struct ResourceArgUseInfo {
  Type data_type;
  bool updated;
  bool used;
};

// Finds the ResourceArgUseInfo for each resource argument. Forwarding to the
// output (i.e., the argument is an operand of the return op) is not considered
// as a use. This doesn't support nesting of ops, so before calling this, nested
// ops/functions need to be already resource-lifted.
LogicalResult FindResourceArgUseInfo(
    FuncOp func_op, llvm::SmallDenseMap<int64_t, ResourceArgUseInfo>* result) {
  auto return_op = func_op.front().getTerminator();
  for (auto arg : func_op.getArguments()) {
    if (!getElementTypeOrSelf(arg.getType()).isa<TF::ResourceType>()) continue;
    ResourceArgUseInfo info;
    info.used = false;
    info.updated = false;
    bool do_not_touch = false;
    for (auto user : arg.getUsers()) {
      if (user == return_op) continue;
      if (auto read = llvm::dyn_cast<TF::ReadVariableOp>(user)) {
        info.used = true;
        info.data_type = read.getType();
        continue;
      }
      if (auto assign = llvm::dyn_cast<TF::AssignVariableOp>(user)) {
        info.used = true;
        info.updated = true;
        info.data_type = assign.value().getType();
        continue;
      }
      if (isa<TF::StackPushV2Op>(user) || isa<TF::StackPopV2Op>(user)) {
        // Stacks will be handled by a separate pass.
        do_not_touch = true;
        break;
      }
      user->emitOpError("found unsupported operations on resource.");
      return failure();
    }
    if (!do_not_touch) (*result)[arg.getArgNumber()] = info;
  }
  return success();
}

// Merges two sets of resource arg use infos. An argument is considered used in
// the merged result as long as either set marks it as used. This is used to
// merge results from functions that have aliasing inputs, e.g., a while loop's
// body and condition. The sets of keys of the two maps must be the same.
llvm::SmallDenseMap<int64_t, ResourceArgUseInfo> MergeArgResourceUseInfo(
    const llvm::SmallDenseMap<int64_t, ResourceArgUseInfo>& infos0,
    const llvm::SmallDenseMap<int64_t, ResourceArgUseInfo>& infos1) {
  llvm::SmallDenseMap<int64_t, ResourceArgUseInfo> result;
  for (const auto& entry : infos0) {
    auto info1_it = infos1.find(entry.getFirst());
    // If the entry is missing in any input, we should not touch this entry.
    if (info1_it == infos1.end()) continue;
    auto& info = result[entry.getFirst()];
    info = entry.getSecond();
    if (info.updated) continue;
    if (info1_it->getSecond().used) {
      info.used = true;
      info.updated = info1_it->getSecond().updated;
      info.data_type = info1_it->getSecond().data_type;
    }
  }
  return result;
}

// Removes the unused resource arguments, and the return values that forward the
// removed arguments. If old_to_new_arg_indices is provided, it will store the
// new argument index that corresponds to each original index (-1 means it is
// removed). If remaining_resource_data_types is provided, it will store the
// data types of the remaining resource arguments, where the indices are after
// removing unused ones.
void RemoveUnusedResourceArgumentsAndForwardedRetvals(
    const llvm::SmallDenseMap<int64_t, ResourceArgUseInfo>& infos,
    FuncOp func_op,
    llvm::SmallVector<int64_t, 4>* old_to_new_arg_indices = nullptr,
    llvm::SmallDenseMap<int64_t, Type>* remaining_resource_data_types =
        nullptr) {
  // Remove return values forwarded from unused arguments.
  auto return_op = func_op.front().getTerminator();
  auto old_return_vals = llvm::to_vector<8>(return_op->getOperands());
  int64_t skipped_retvals = 0;
  for (auto entry : llvm::enumerate(old_return_vals)) {
    auto return_val = entry.value();
    if (auto arg = return_val.dyn_cast<BlockArgument>()) {
      auto it = infos.find(arg.getArgNumber());
      if (it != infos.end() && !it->getSecond().used) {
        return_op->eraseOperand(entry.index() - skipped_retvals++);
      }
    }
  }
  llvm::SmallVector<unsigned int, 4> indices_to_erase;
  llvm::SmallVector<Type, 4> new_types;
  int64_t skipped_args = 0;
  for (auto arg : func_op.getArguments()) {
    auto it = infos.find(arg.getArgNumber());
    if (it != infos.end() && !it->getSecond().used) {
      indices_to_erase.push_back(arg.getArgNumber());
      skipped_args++;
      if (old_to_new_arg_indices != nullptr) {
        old_to_new_arg_indices->push_back(-1);
      }
    } else {
      new_types.push_back(arg.getType());
      if (old_to_new_arg_indices != nullptr) {
        old_to_new_arg_indices->push_back(arg.getArgNumber() - skipped_args);
      }
      if (it != infos.end() && remaining_resource_data_types != nullptr) {
        (*remaining_resource_data_types)[arg.getArgNumber() - skipped_args] =
            it->second.data_type;
      }
    }
  }
  func_op.eraseArguments(indices_to_erase);
  func_op.setType(FunctionType::get(
      new_types, llvm::to_vector<4>(return_op->getOperandTypes()),
      func_op.getContext()));
}

// Lifts reads/writes of resource arguments from func_op and changes its
// signature. resource_data_types is the (index, data type) pair for each
// resource argument. handle_updated_arg_value is a caller-provided function
// that handles the updated value for an resource argument.
void LiftArgRetResourcesForFunction(
    FuncOp func_op,
    const llvm::SmallDenseMap<int64_t, Type>& resource_data_types,
    llvm::function_ref<void(int64_t, Value)> handle_updated_arg_value) {
  ForwardStoreToLoad(&func_op.front());
  // Maps a resource argument to the first read.
  llvm::SmallDenseMap<Value, TF::ReadVariableOp, 4> resource_arg_read;
  // Maps a resource argument to the last write.
  llvm::SmallDenseMap<Value, TF::AssignVariableOp, 4> resource_arg_write;
  // Use HoistResourceLoads to CSE loads and the `move_load` function only
  // records the remaining load to resource_arg_read.
  HoistResourceLoads(
      &func_op.front(),
      /*skip_load=*/
      [&](TF::ReadVariableOp read) {
        return !read.resource().isa<BlockArgument>();
      },
      /*move_load=*/
      [&](TF::ReadVariableOp read) {
        resource_arg_read[read.resource()] = read;
      });
  // Record the stores in resource_arg_read.
  for (auto& op : llvm::make_early_inc_range(func_op.front())) {
    auto write = llvm::dyn_cast<TF::AssignVariableOp>(&op);
    if (!write) continue;
    auto arg = write.resource().dyn_cast<BlockArgument>();
    if (!arg) continue;
    // After ForwardStoreToLoad(), there should be just one store for each
    // resource.
    resource_arg_write[arg] = write;
  }
  // Now change the input types to non-resource and remove the internal loads.
  auto new_types = llvm::to_vector<8>(func_op.getType().getInputs());
  for (auto& entry : resource_data_types) {
    auto arg = func_op.getArgument(entry.getFirst());
    auto read_it = resource_arg_read.find(arg);
    auto write_it = resource_arg_write.find(arg);
    arg.setType(entry.getSecond());
    new_types[arg.getArgNumber()] = entry.getSecond();
    if (read_it != resource_arg_read.end()) {
      read_it->getSecond().replaceAllUsesWith(arg);
      read_it->getSecond().erase();
    }
    if (write_it != resource_arg_write.end()) {
      handle_updated_arg_value(arg.getArgNumber(),
                               write_it->getSecond().value());
      write_it->getSecond().erase();
    }
  }
  func_op.setType(FunctionType::get(
      new_types,
      llvm::to_vector<4>(func_op.front().getTerminator()->getOperandTypes()),
      func_op.getContext()));
}

// Returns a vector filtered from range where the unused elements (specified by
// resource_arg_uses) are removed.
template <typename T, typename Range>
llvm::SmallVector<T, 4> FilterRange(
    Range range,
    const llvm::SmallDenseMap<int64_t, ResourceArgUseInfo>& resource_arg_uses) {
  llvm::SmallVector<T, 4> filtered;
  for (auto entry : llvm::enumerate(range)) {
    auto it = resource_arg_uses.find(entry.index());
    if (it == resource_arg_uses.end() || it->getSecond().used)
      filtered.push_back(entry.value());
  }
  return filtered;
}

// Changes the types of the control flow op (e.g., while, if) and adds loads and
// stores around it. arg_data_type_and_updated_output_index maps an operand (to
// be changed) index to its data type and the updated value index in the output
// (-1 means not updated.)
void AddLoadsStoresOutsideControlFlowOp(
    Operation* caller,
    const llvm::SmallDenseMap<int64_t, std::pair<Type, int64_t>>&
        arg_data_type_and_updated_output_index) {
  OpBuilder builder(caller);
  auto new_operands = llvm::to_vector<8>(caller->getOperands());
  llvm::SmallVector<int64_t, 8> changed_indices;
  // Find the operands to change, and create the loads.
  for (auto& entry : arg_data_type_and_updated_output_index) {
    int64_t index = entry.getFirst();
    Type new_type = entry.getSecond().first;
    int64_t updated_index = entry.getSecond().second;
    auto operand = caller->getOperand(index);
    builder.setInsertionPoint(caller);
    new_operands[index] = builder.create<TF::ReadVariableOp>(
        caller->getLoc(), ArrayRef<Type>{new_type}, ArrayRef<Value>{operand},
        ArrayRef<NamedAttribute>{});
    caller->setOperand(index, new_operands[index]);
    if (updated_index < 0) continue;
    builder.setInsertionPointAfter(caller);
    builder.create<TF::AssignVariableOp>(
        caller->getLoc(), ArrayRef<Type>{},
        ArrayRef<Value>{operand, caller->getResult(updated_index)},
        ArrayRef<NamedAttribute>{});
  }
}

// Lifts loads/stores from while loop's body and cond functions.
LogicalResult HandleWhileLoop(TF::WhileOp while_op, FuncOp body, FuncOp cond) {
  // Remove identity nodes to avoid aliasing.
  RemoveIdentity(&body.front());
  RemoveIdentity(&cond.front());
  auto return_op = body.front().getTerminator();
  // Sanity check: body resource input/output should alias each other.
  for (auto arg : body.getArguments()) {
    if (!getElementTypeOrSelf(arg.getType()).isa<TF::ResourceType>()) continue;
    if (return_op->getOperand(arg.getArgNumber()) != arg) {
      return return_op->emitOpError(
                 "resource used in while loop is only supported when the ")
             << "resource input and output alias each other in the loop body.";
    }
  }
  // FindResourceArgUseInfo will check supported resource ops (read and assign),
  // but loop condition has additional requirement that it cannot write
  // resources.
  if (cond.walk([&](TF::AssignVariableOp assign) {
            assign.emitOpError("found resource write in loop condition.");
            return WalkResult::interrupt();
          })
          .wasInterrupted()) {
    return failure();
  }
  llvm::SmallDenseMap<int64_t, ResourceArgUseInfo> body_use_info;
  llvm::SmallDenseMap<int64_t, ResourceArgUseInfo> cond_use_info;
  if (failed(FindResourceArgUseInfo(body, &body_use_info)) ||
      failed(FindResourceArgUseInfo(cond, &cond_use_info))) {
    return failure();
  }
  // A resource is considered used as long as it is used in either body or cond.
  auto resource_arg_uses =
      MergeArgResourceUseInfo(body_use_info, cond_use_info);
  if (resource_arg_uses.empty()) return success();
  for (const auto& entry : resource_arg_uses) {
    // Replace output resource uses with the input, so that we can later freely
    // change the output type.
    while_op.getResult(entry.getFirst())
        .replaceAllUsesWith(while_op.getOperand(entry.getFirst()));
  }
  // Remove unused resources in functions.
  llvm::SmallVector<int64_t, 4> old_to_new_indices;
  llvm::SmallDenseMap<int64_t, Type> remaining_resource_data_types;
  RemoveUnusedResourceArgumentsAndForwardedRetvals(
      resource_arg_uses, body, &old_to_new_indices,
      &remaining_resource_data_types);
  RemoveUnusedResourceArgumentsAndForwardedRetvals(resource_arg_uses, cond);
  LiftArgRetResourcesForFunction(
      body, remaining_resource_data_types,
      [&](int64_t index, Value value) { return_op->setOperand(index, value); });
  LiftArgRetResourcesForFunction(cond, remaining_resource_data_types,
                                 [&](int64_t index, Value value) {
                                   // We already checked that cond should not
                                   // have variable writes.
                                   assert(false && "Should not happen");
                                 });
  // Recreate the while op.
  OpBuilder builder(while_op);
  auto new_output_shapes = FilterRange<Attribute, ArrayRef<Attribute>>(
      while_op.output_shapes().getValue(), resource_arg_uses);
  // Now use the filtered original operands, which will be replaced by
  // AddLoadsStoresOutsideControlFlowOp().
  auto new_while = builder.create<TF::WhileOp>(
      while_op.getLoc(), body.getType().getResults(),
      FilterRange<Value, OperandRange>(while_op.getOperands(),
                                       resource_arg_uses),
      while_op.getAttrs());
  // Prepare for AddLoadsStoresOutsideControlFlowOp() and update
  // new_output_shapes.
  llvm::SmallDenseMap<int64_t, std::pair<Type, int64_t>>
      arg_data_type_and_updated_output_index;
  for (const auto& entry : remaining_resource_data_types) {
    int64_t update_index = return_op->getOperand(entry.getFirst()) ==
                                   body.getArgument(entry.getFirst())
                               ? -1
                               : entry.getFirst();
    arg_data_type_and_updated_output_index[entry.getFirst()] = {
        entry.getSecond(), update_index};
    if (!new_output_shapes.empty()) {
      new_output_shapes[entry.getFirst()] =
          tensorflow::ConvertTypeToTensorShapeAttr(entry.getSecond());
    }
  }
  AddLoadsStoresOutsideControlFlowOp(new_while,
                                     arg_data_type_and_updated_output_index);
  new_while.setAttr("output_shapes", builder.getArrayAttr(new_output_shapes));
  // Replace uses.
  for (int64_t i = 0; i < old_to_new_indices.size(); ++i) {
    if (old_to_new_indices[i] >= 0) {
      while_op.getResult(i).replaceAllUsesWith(
          new_while.getResult(old_to_new_indices[i]));
    }
  }
  while_op.erase();
  return success();
}

// Lifts loads/stores from an IfOp's branches.
LogicalResult HandleIfOP(TF::IfOp if_op, FuncOp then_branch,
                         FuncOp else_branch) {
  // Remove identity nodes to avoid aliasing.
  RemoveIdentity(&then_branch.front());
  RemoveIdentity(&else_branch.front());
  // Sanity check: branch return of resources should be aliases of inputs. If
  // so, replace the output uses with the input so that we can remove these
  // outputs.
  for (auto entry : llvm::enumerate(
           llvm::zip(then_branch.front().getTerminator()->getOperands(),
                     else_branch.front().getTerminator()->getOperands()))) {
    auto then_retval = std::get<0>(entry.value());
    auto else_retval = std::get<1>(entry.value());
    assert(then_retval.getType() == else_retval.getType());
    if (!getElementTypeOrSelf(then_retval.getType()).isa<TF::ResourceType>()) {
      continue;
    }
    auto then_aliasing_arg = then_retval.dyn_cast<BlockArgument>();
    auto else_aliasing_arg = else_retval.dyn_cast<BlockArgument>();
    if (!then_aliasing_arg || !else_aliasing_arg ||
        then_aliasing_arg.getArgNumber() != else_aliasing_arg.getArgNumber()) {
      return if_op.emitOpError("unsupported tf.IfOp output: ")
             << "resource does not alias a single input.";
    }
    if_op.getResult(entry.index())
        .replaceAllUsesWith(
            if_op.getOperand(then_aliasing_arg.getArgNumber() + 1));
  }
  // Erase the resource outputs from the branches.
  int64_t non_resource_results = 0;
  llvm::SmallVector<int64_t, 4> old_to_new_output_indices;
  llvm::SmallVector<Attribute, 4> new_output_shapes;
  bool output_removed = false;
  for (auto result : if_op.getResults()) {
    if (!getElementTypeOrSelf(result.getType()).isa<TF::ResourceType>()) {
      old_to_new_output_indices.push_back(non_resource_results++);
      if (!if_op.output_shapes().getValue().empty()) {
        new_output_shapes.push_back(
            if_op.output_shapes().getValue()[result.getResultNumber()]);
      }
      continue;
    }
    old_to_new_output_indices.push_back(-1);
    then_branch.front().getTerminator()->eraseOperand(non_resource_results);
    else_branch.front().getTerminator()->eraseOperand(non_resource_results);
    output_removed = true;
  }

  llvm::SmallDenseMap<int64_t, ResourceArgUseInfo> then_use_info;
  llvm::SmallDenseMap<int64_t, ResourceArgUseInfo> else_use_info;
  if (failed(FindResourceArgUseInfo(then_branch, &then_use_info)) ||
      failed(FindResourceArgUseInfo(else_branch, &else_use_info))) {
    return failure();
  }
  // A resource is considered used as long as it is used in either branch.
  auto resource_arg_uses =
      MergeArgResourceUseInfo(then_use_info, else_use_info);
  if (resource_arg_uses.empty() && !output_removed) return success();
  // Remove unused resources in functions.
  llvm::SmallDenseMap<int64_t, Type> remaining_resource_data_types;
  RemoveUnusedResourceArgumentsAndForwardedRetvals(
      resource_arg_uses, then_branch, /*old_to_new_arg_indices=*/nullptr,
      &remaining_resource_data_types);
  RemoveUnusedResourceArgumentsAndForwardedRetvals(resource_arg_uses,
                                                   else_branch);
  // Forward resource inputs updated in any branch to the outputs of both
  // branches. First prepare the mapping from arg to new update output.
  llvm::SmallDenseMap<int64_t, int64_t> resource_arg_to_new_output;
  {
    int64_t removed_args = 0;
    for (const auto& entry : resource_arg_uses) {
      if (!entry.getSecond().used) {
        removed_args++;
        continue;
      }
      if (!entry.getSecond().updated) continue;
      int64_t new_output_index =
          non_resource_results + resource_arg_to_new_output.size();
      resource_arg_to_new_output[entry.getFirst() - removed_args] =
          new_output_index;
    }
  }
  // Append resource updates to the return ops: now they are just forwarded
  // input resources, but will be replaced by the data value in
  // LiftArgRetResourcesForFunction().
  for (auto branch : {then_branch, else_branch}) {
    auto new_retvals =
        llvm::to_vector<4>(branch.front().getTerminator()->getOperands());
    for (const auto& entry : resource_arg_to_new_output) {
      new_retvals.push_back(branch.getArgument(entry.getFirst()));
    }
    auto old_return = branch.front().getTerminator();
    OpBuilder builder(old_return);
    auto new_return =
        builder.create<ReturnOp>(old_return->getLoc(), new_retvals);
    old_return->erase();
    LiftArgRetResourcesForFunction(
        branch, remaining_resource_data_types, [&](int64_t index, Value value) {
          new_return.setOperand(resource_arg_to_new_output[index], value);
        });
  }

  // Recreate the if op.
  OpBuilder builder(if_op);
  // Now use the filtered original operands, which will be replaced by
  // AddLoadsStoresOutsideControlFlowOp().
  auto new_operands =
      FilterRange<Value, OperandRange>(if_op.input(), resource_arg_uses);
  new_operands.insert(new_operands.begin(), if_op.cond());
  auto new_if = builder.create<TF::IfOp>(if_op.getLoc(),
                                         then_branch.getType().getResults(),
                                         new_operands, if_op.getAttrs());
  // Prepare for AddLoadsStoresOutsideControlFlowOp() and update
  // new_output_shapes.
  llvm::SmallDenseMap<int64_t, std::pair<Type, int64_t>>
      arg_data_type_and_updated_output_index;
  for (const auto& entry : remaining_resource_data_types) {
    auto new_output_it = resource_arg_to_new_output.find(entry.getFirst());
    int64_t update_index = new_output_it == resource_arg_to_new_output.end()
                               ? -1
                               : new_output_it->getSecond();
    arg_data_type_and_updated_output_index[entry.getFirst() + 1] = {
        entry.getSecond(), update_index};
    if (!if_op.output_shapes().getValue().empty() && update_index >= 0) {
      new_output_shapes.push_back(
          tensorflow::ConvertTypeToTensorShapeAttr(entry.getSecond()));
    }
  }
  AddLoadsStoresOutsideControlFlowOp(new_if,
                                     arg_data_type_and_updated_output_index);
  new_if.setAttr("output_shapes", builder.getArrayAttr(new_output_shapes));
  // Replace uses.
  for (int64_t i = 0; i < old_to_new_output_indices.size(); ++i) {
    if (old_to_new_output_indices[i] >= 0) {
      if_op.getResult(i).replaceAllUsesWith(
          new_if.getResult(old_to_new_output_indices[i]));
    }
  }
  if_op.erase();
  return success();
}

// A resource-lifted function for (potentially multiple) PartitionedCallOps and
// information about the lifting changes.
struct PartitionedCallLiftingInfo {
  // Function with resources lifted. Can be nullptr if nothing needs to change.
  FuncOp lifted_callee;
  // Mapping from old resource outputs to their aliasing output inputs.
  llvm::SmallDenseMap<int64_t, int64_t> old_outputs_aliasing_old_inputs;
  // Mapping from old to new output indices in case any output is removed.
  llvm::SmallVector<int64_t, 4> old_to_new_output_indices;
  // ResourceArgUseInfo for each old resource argument.
  llvm::SmallDenseMap<int64_t, ResourceArgUseInfo> use_info;
  // Input for AddLoadsStoresOutsideControlFlowOp(), see its comment.
  llvm::SmallDenseMap<int64_t, std::pair<Type, int64_t>>
      arg_data_type_and_updated_output_index;
};

// Lifts loads/stores from a PartitionedCallOp's callee function. If anything
// needs to be changed, the original function will be preserved, and the lifting
// happens on a clone, which will be stored in `result`.
LogicalResult HandlePartitionedCallOpCallee(
    FuncOp callee, PartitionedCallLiftingInfo* result) {
  // Remove identity nodes to avoid aliasing.
  RemoveIdentity(&callee.front());
  // Sanity check: return of resources should be aliases of inputs. Such outputs
  // will be removed later.
  int64_t non_resource_results = 0;
  for (auto entry :
       llvm::enumerate(callee.front().getTerminator()->getOperands())) {
    auto retval = entry.value();
    if (!getElementTypeOrSelf(retval.getType()).isa<TF::ResourceType>()) {
      result->old_to_new_output_indices.push_back(non_resource_results++);
      continue;
    }
    auto aliasing_arg = retval.dyn_cast<BlockArgument>();
    if (!aliasing_arg) {
      return callee.emitOpError("unsupported function call: ")
             << "resource return value does not alias an input.";
    }
    result->old_outputs_aliasing_old_inputs[entry.index()] =
        aliasing_arg.getArgNumber();
    result->old_to_new_output_indices.push_back(-1);
  }

  if (failed(FindResourceArgUseInfo(callee, &result->use_info))) {
    return failure();
  }
  if (result->use_info.empty()) {
    result->lifted_callee = nullptr;
    return success();
  }

  // Clone the callee before making changes.
  SmallString<64> name_base = callee.getName();
  auto module = callee.getParentOfType<ModuleOp>();
  name_base += "_resource_lifted";
  auto name = name_base;
  callee = callee.clone();
  callee.setVisibility(SymbolTable::Visibility::Private);
  callee.setName(name);
  SymbolTable(module).insert(callee);
  result->lifted_callee = callee;

  // Remove unused resources in functions.
  llvm::SmallDenseMap<int64_t, Type> remaining_resource_data_types;
  RemoveUnusedResourceArgumentsAndForwardedRetvals(
      result->use_info, callee, /*old_to_new_arg_indices=*/nullptr,
      &remaining_resource_data_types);
  for (const auto& entry : remaining_resource_data_types) {
    result->arg_data_type_and_updated_output_index[entry.getFirst()] = {
        entry.getSecond(), -1};
  }
  llvm::SmallVector<Value, 4> new_retvals;
  for (auto val : callee.front().getTerminator()->getOperands()) {
    // Remove resource type outputs.
    if (getElementTypeOrSelf(val.getType()).isa<TF::ResourceType>()) continue;
    new_retvals.push_back(val);
  }
  // Lift resources.
  LiftArgRetResourcesForFunction(
      callee, remaining_resource_data_types, [&](int64_t index, Value value) {
        result->arg_data_type_and_updated_output_index[index].second =
            new_retvals.size();
        new_retvals.push_back(value);
      });
  auto old_return = callee.front().getTerminator();
  // Replace old return with the new ones with update values.
  OpBuilder builder(old_return);
  auto new_return = builder.create<ReturnOp>(old_return->getLoc(), new_retvals);
  old_return->erase();
  callee.setType(FunctionType::get(
      callee.getType().getInputs(),
      llvm::to_vector<4>(new_return.getOperandTypes()), callee.getContext()));
  return success();
}

// Updates a PartitionedCallOp/StatefulPartitionedCallOp according to the
// resource-lifted new callee function in lifting_info.
template <typename CallOpType>
void UpdatePartitionedCallOpWithNewCallee(
    CallOpType call_op, const PartitionedCallLiftingInfo& lifting_info) {
  if (lifting_info.lifted_callee == nullptr) return;
  // Replace output resource uses with the aliasing input, so that we can remove
  // this output.
  for (const auto& entry : lifting_info.old_outputs_aliasing_old_inputs) {
    call_op.getResult(entry.getFirst())
        .replaceAllUsesWith(call_op.getOperand(entry.getSecond()));
  }
  // Recreate the call op.
  OpBuilder builder(call_op);
  // Now use the filtered original operands, which will be replaced by
  // AddLoadsStoresOutsideControlFlowOp().
  auto new_operands =
      FilterRange<Value, OperandRange>(call_op.args(), lifting_info.use_info);
  auto new_call = builder.create<CallOpType>(
      call_op.getLoc(),
      const_cast<FuncOp&>(lifting_info.lifted_callee).getType().getResults(),
      new_operands, call_op.getAttrs());
  new_call.setAttr(
      "f", builder.getSymbolRefAttr(
               const_cast<FuncOp&>(lifting_info.lifted_callee).getName()));
  AddLoadsStoresOutsideControlFlowOp(
      new_call, lifting_info.arg_data_type_and_updated_output_index);
  // Replace uses.
  for (int64_t i = 0; i < lifting_info.old_to_new_output_indices.size(); ++i) {
    if (lifting_info.old_to_new_output_indices[i] >= 0) {
      call_op.getResult(i).replaceAllUsesWith(
          new_call.getResult(lifting_info.old_to_new_output_indices[i]));
    }
  }
  call_op.erase();
}

LogicalResult HoistForFunctionalControlFlow(
    Block*, ModuleOp, llvm::SmallDenseMap<FuncOp, PartitionedCallLiftingInfo>*);

// A templated routine for handling both PartitionedCallOp and
// StatefulPartitionedCallOp. If the callee is already lifted, it just updates
// the caller op itself; otherwise, it first recursively handles nested control
// flow, then performs lifting on the callee.
template <typename CallOpType>
LogicalResult HandlePartitionedCallOp(
    CallOpType call_op, FuncOp callee, ModuleOp module,
    llvm::SmallDenseMap<FuncOp, PartitionedCallLiftingInfo>* lifted_callees) {
  auto emplace_res =
      lifted_callees->try_emplace(callee, PartitionedCallLiftingInfo());
  if (emplace_res.second) {
    // Unseen callee. Perform resource lifting on it.
    HoistForFunctionalControlFlow(&callee.front(), module, lifted_callees);
    if (failed(HandlePartitionedCallOpCallee(
            callee, &emplace_res.first->getSecond()))) {
      return failure();
    }
  }
  UpdatePartitionedCallOpWithNewCallee(call_op, emplace_res.first->getSecond());
  return success();
}

// Hoists resource loads/stores from control flow ops in `block` outside the
// body/cond/branch/callee functions.
LogicalResult HoistForFunctionalControlFlow(
    Block* block, ModuleOp module,
    llvm::SmallDenseMap<FuncOp, PartitionedCallLiftingInfo>*
        lifted_partitioned_call_callees) {
  // Remove identity nodes to avoid aliasing.
  RemoveIdentity(block);
  for (Operation& op : llvm::make_early_inc_range(*block)) {
    if (auto while_op = llvm::dyn_cast<TF::WhileOp>(&op)) {
      auto body = llvm::cast<FuncOp>(module.lookupSymbol(while_op.body()));
      auto cond = llvm::cast<FuncOp>(module.lookupSymbol(while_op.cond()));
      // Recursively handle the nested control flow.
      HoistForFunctionalControlFlow(&body.front(), module,
                                    lifted_partitioned_call_callees);
      HoistForFunctionalControlFlow(&cond.front(), module,
                                    lifted_partitioned_call_callees);
      if (failed(HandleWhileLoop(while_op, body, cond))) return failure();
    } else if (auto if_op = llvm::dyn_cast<TF::IfOp>(&op)) {
      auto then_branch =
          llvm::cast<FuncOp>(module.lookupSymbol(if_op.then_branch()));
      auto else_branch =
          llvm::cast<FuncOp>(module.lookupSymbol(if_op.else_branch()));
      // Recursively handle the nested control flow.
      HoistForFunctionalControlFlow(&then_branch.front(), module,
                                    lifted_partitioned_call_callees);
      HoistForFunctionalControlFlow(&else_branch.front(), module,
                                    lifted_partitioned_call_callees);
      if (failed(HandleIfOP(if_op, then_branch, else_branch))) return failure();
    } else if (auto call_op = llvm::dyn_cast<TF::PartitionedCallOp>(&op)) {
      if (!call_op.f().isa<FlatSymbolRefAttr>()) {
        return call_op.emitOpError(
            "resource lifting does not support call with nested references.");
      }
      auto callee = llvm::cast<FuncOp>(
          module.lookupSymbol(call_op.f().getRootReference()));
      if (failed(HandlePartitionedCallOp(call_op, callee, module,
                                         lifted_partitioned_call_callees))) {
        // Nested control flow handling is done in HandlePartitionedCallOp().
        return failure();
      }
    } else if (auto call_op =
                   llvm::dyn_cast<TF::StatefulPartitionedCallOp>(&op)) {
      auto callee = llvm::cast<FuncOp>(module.lookupSymbol(call_op.f()));
      if (failed(HandlePartitionedCallOp(call_op, callee, module,
                                         lifted_partitioned_call_callees))) {
        return failure();
      }
    }
  }

  // Remove unused local variables.
  ForwardStoreToLoad(block);
  llvm::SmallVector<TF::MlirLocalVarOp, 8> local_vars;
  for (Operation& op : *block) {
    if (auto local_var = llvm::dyn_cast<TF::MlirLocalVarOp>(&op)) {
      local_vars.push_back(local_var);
    }
  }
  for (auto local_var : local_vars) {
    if (llvm::all_of(local_var.resource().getUsers(),
                     [](const Operation* user) {
                       return isa<TF::AssignVariableOp>(user);
                     })) {
      for (auto user : local_var.resource().getUsers()) user->erase();
      local_var.erase();
    }
  }
  return success();
}

// Lifts resource operation from tf_device.cluster ops nested in `op` outside.
// Returns failure if there are remaining resource-type values that can not be
// lifted.
void ResourceOpLiftingPass::runOnOperation() {
  llvm::SmallDenseMap<FuncOp, PartitionedCallLiftingInfo>
      lifted_partitioned_call_callees;
  ModuleOp module = getOperation();
  auto result = module.walk([&](FuncOp func_op) {
    return func_op.walk([&](tf_device::ClusterOp cluster) {
      if (failed(HoistForFunctionalControlFlow(
              &cluster.GetBody(), module, &lifted_partitioned_call_callees)) ||
          failed(HoistResourceOpsFromCluster(cluster, module))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  });
  if (result.wasInterrupted()) {
    signalPassFailure();
  }
}

struct ResourceOpLiftingForMainFunctionPass
    : public PassWrapper<ResourceOpLiftingForMainFunctionPass,
                         OperationPass<ModuleOp>> {
  void runOnOperation() override;
};

void ResourceOpLiftingForMainFunctionPass::runOnOperation() {
  ModuleOp module = getOperation();
  FuncOp main_func = module.lookupSymbol<FuncOp>("main");
  if (!main_func) {
    return;
  }

  if (failed(TF::ResourceLiftingForFunctionalControlFlow(main_func))) {
    return signalPassFailure();
  }
}

static PassRegistration<ResourceOpLiftingForMainFunctionPass>
    lift_main_func_pass(
        "tf-resource-op-lifting-for-main-function",
        "Lifting resource operations out of control flow statements for the "
        "main function");

static PassRegistration<ResourceOpLiftingPass> pass(
    "tf-resource-op-lifting",
    "Lifting resource operations out of device computation");

}  // namespace

namespace TFDevice {
std::unique_ptr<OperationPass<ModuleOp>> CreateResourceOpLiftingPass() {
  return std::make_unique<ResourceOpLiftingPass>();
}
}  // namespace TFDevice

namespace TF {
LogicalResult ResourceLiftingForFunctionalControlFlow(FuncOp function) {
  // This routine should only be called when control flow operations are still
  // represented with TF IfOp and WhileOp operations. In this case, there should
  // be only one basic blocks in the MLIR representation.
  if (!hasSingleElement(function.getBlocks())) {
    return function.emitError()
           << "expect the function to have 1 block while it has "
           << function.getBlocks().size();
  }

  llvm::SmallDenseMap<FuncOp, PartitionedCallLiftingInfo>
      lifted_partitioned_call_callees;
  return HoistForFunctionalControlFlow(&function.front(),
                                       cast<ModuleOp>(function.getParentOp()),
                                       &lifted_partitioned_call_callees);
}
}  // namespace TF

}  // namespace mlir

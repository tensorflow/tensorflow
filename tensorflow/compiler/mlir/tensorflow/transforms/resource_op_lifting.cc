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
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Block.h"  // TF:llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Diagnostics.h"  // TF:llvm-project
#include "mlir/IR/Function.h"  // TF:llvm-project
#include "mlir/IR/Module.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/IR/TypeUtilities.h"  // TF:llvm-project
#include "mlir/IR/Types.h"  // TF:llvm-project
#include "mlir/IR/Value.h"  // TF:llvm-project
#include "mlir/IR/Visitors.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Support/LLVM.h"  // TF:llvm-project
#include "mlir/Support/LogicalResult.h"  // TF:llvm-project
#include "mlir/Transforms/RegionUtils.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"

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
//
// If the launch body contains functional control flow, the pass first lifts the
// loads/stores in the body/cond/branch functions to the launch body, then
// performs the above lifting. E.g.,
//
// func @launch_with_loop() -> () {
//   %0 = "tf.VarHandleOp"() ...
//   "tf_device.launch"() ( {
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
// func @launch_with_loop() {
//   %0 = "tf.VarHandleOp"() ...
//   %1 = "tf.ReadVariableOp"(%0)
//   %2 = "tf_device.launch"() ( {
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
struct ResourceOpLiftingPass : public ModulePass<ResourceOpLiftingPass> {
  void runOnModule() override;
};

// Removes identity nodes in the block. The device computation does not need
// such nodes to carry information.
void RemoveIdentity(Block* block) {
  for (auto& op : llvm::make_early_inc_range(*block)) {
    if (llvm::isa<TF::IdentityOp>(&op) || llvm::isa<TF::IdentityNOp>(&op)) {
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
// assumes load-store forwarding has been performed on this launch_op such that
// all loads of same resource are on its initial values. A `skip_load` functions
// is used to indicate whether a load should be skipped. If there are multiple
// loads on the same resource, only the first one will be moved, and the later
// ones will be removed and replaced with the first one.
void HoistResourceLoads(
    Block* block, llvm::function_ref<bool(TF::ReadVariableOp)> skip_load,
    llvm::function_ref<void(TF::ReadVariableOp)> move_load) {
  llvm::SmallDenseMap<Value, TF::ReadVariableOp> resource_to_read_ops;

  // Only iterate through ops directly in launch_op's body as we can't handle
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
    if (resource.getParentRegion() == &launch_op.body()) continue;

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
tf_device::LaunchOp SinkResourceStores(tf_device::LaunchOp launch_op,
                                       OpBuilder* builder) {
  // Update ReturnOp inside launch_op's body to output final values of updated
  // external resources.
  bool has_resource_store = AppendResourceStoreValueToReturn(launch_op);
  if (!has_resource_store) return launch_op;

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
    std::get<0>(p).replaceAllUsesWith(std::get<1>(p));
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
  return new_launch_op;
}

// Hoists resource variable loads and sinks stores from launch_op.
LogicalResult HoistResourceOpsFromLaunchOp(tf_device::LaunchOp launch_op) {
  ModuleOp m = launch_op.getParentOfType<ModuleOp>();
  OpBuilder builder(m);

  // Remove identity nodes to avoid aliasing.
  RemoveIdentity(&launch_op.GetBody());

  // Perform store-load forwarding. So that each resource is only loaded with
  // its initial value and is only stored with its final value.
  ForwardStoreToLoad(&launch_op.GetBody());

  // Move loads of external resources, if any, to before launch_op.
  // (Skipping resources created inside of launch_op.)
  HoistResourceLoads(
      &launch_op.GetBody(),
      /*skip_load=*/
      [&](TF::ReadVariableOp read) {
        return read.resource().getParentRegion() == &launch_op.body();
      },
      /*move_load=*/
      [&](TF::ReadVariableOp read) {
        read.getOperation()->moveBefore(launch_op);
      });

  // Move stores of external resources, if any, to after launch_op.
  auto new_launch_op = SinkResourceStores(launch_op, &builder);

  llvm::SetVector<Value> captured_values;
  getUsedValuesDefinedAbove(new_launch_op.body(), new_launch_op.body(),
                            captured_values);

  for (Value v : captured_values) {
    auto tensor_type = v.getType().dyn_cast<TensorType>();
    if (!tensor_type) continue;
    if (!tensor_type.getElementType().isa<TF::ResourceType>()) continue;

    return new_launch_op.emitOpError()
           << "has remaining resource inputs that can not be lifted";
  }

  return success();
}

// Holds information about a function's use of a resource argument.
struct ResourceArgUseInfo {
  bool used;
  Type data_type;
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
    auto& info = (*result)[arg.getArgNumber()];
    info.used = false;
    for (auto user : arg.getUsers()) {
      if (user == return_op) continue;
      if (auto read = llvm::dyn_cast<TF::ReadVariableOp>(user)) {
        info.used = true;
        info.data_type = read.getType();
        continue;
      }
      if (auto assign = llvm::dyn_cast<TF::AssignVariableOp>(user)) {
        info.used = true;
        info.data_type = assign.value().getType();
        continue;
      }
      user->emitError("Found unsupported operations on resource.");
      return failure();
    }
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
  auto result = infos0;
  for (auto& entry : result) {
    if (entry.getSecond().used) continue;
    auto& info1_entry = *infos1.find(entry.getFirst());
    if (info1_entry.getSecond().used) {
      entry.getSecond().used = true;
      entry.getSecond().data_type = info1_entry.getSecond().data_type;
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
    if (auto write = llvm::dyn_cast<TF::AssignVariableOp>(&op)) {
      auto arg = write.resource().dyn_cast<BlockArgument>();
      if (!arg) continue;
      // After ForwardStoreToLoad(), there should be just one store for each
      // resource.
      resource_arg_write[arg] = write;
    }
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
    if (it != resource_arg_uses.end() && !it->getSecond().used) continue;
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
LogicalResult HanldeWhileLoop(TF::WhileOp while_op, FuncOp body, FuncOp cond) {
  // Remove identity nodes to avoid aliasing.
  RemoveIdentity(&body.front());
  RemoveIdentity(&cond.front());
  auto return_op = body.front().getTerminator();
  // Sanity check: body resource input/output should alias each other.
  for (auto arg : body.getArguments()) {
    if (!getElementTypeOrSelf(arg.getType()).isa<TF::ResourceType>()) continue;
    if (return_op->getOperand(arg.getArgNumber()) != arg) {
      return_op->emitError(
          "Resource used in while loop is only supported when the resource "
          "input and output alias each other in the loop body.");
      return failure();
    }
  }
  // FindResourceArgUseInfo will check supported resource ops (read and assign),
  // but loop condition has additional requirement that it cannot write
  // resources.
  if (cond.walk([&](TF::AssignVariableOp assign) {
            assign.emitError("Found resource write in loop condition.");
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
      tensorflow::TensorShapeProto shape_proto;
      tensorflow::ConvertTypeToTensorShape(entry.getSecond())
          .AsProto(&shape_proto);
      new_output_shapes[entry.getFirst()] = builder.getStringAttr(
          tensorflow::mangling_util::MangleShape(shape_proto));
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

// Hoists resource loads/stores from control flow ops in `block` outside the
// body/cond/branch functions.
LogicalResult HoistForFunctionalControlFlow(Block* block, ModuleOp module) {
  for (Operation& op : llvm::make_early_inc_range(*block)) {
    if (auto while_op = llvm::dyn_cast<TF::WhileOp>(&op)) {
      auto body = llvm::cast<FuncOp>(module.lookupSymbol(while_op.body()));
      auto cond = llvm::cast<FuncOp>(module.lookupSymbol(while_op.cond()));
      // Recursively handle the nested control flow.
      HoistForFunctionalControlFlow(&body.front(), module);
      HoistForFunctionalControlFlow(&cond.front(), module);
      if (failed(HanldeWhileLoop(while_op, body, cond))) return failure();
    } else if (auto if_op = llvm::dyn_cast<TF::IfOp>(&op)) {
      // TODO(yuanzx): Add support for IfOp.
    }
  }
  return success();
}

}  // namespace

// Lifts resource operation from tf_device.launch_func ops nested in `op`
// outside. Returns failure if there are remaining resource-type values that can
// not be lifted.
void ResourceOpLiftingPass::runOnModule() {
  auto result = getModule().walk([&](FuncOp func_op) {
    return func_op.walk([&](tf_device::LaunchOp launch_op) {
      if (failed(HoistForFunctionalControlFlow(&launch_op.GetBody(),
                                               getModule())) ||
          failed(HoistResourceOpsFromLaunchOp(launch_op))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  });
  if (result.wasInterrupted()) {
    signalPassFailure();
  }
}

std::unique_ptr<OpPassBase<ModuleOp>> CreateResourceOpLiftingPass() {
  return std::make_unique<ResourceOpLiftingPass>();
}

static PassRegistration<ResourceOpLiftingPass> pass(
    "tf-resource-op-lifting",
    "Lifting resource operations out of device computation");

}  // namespace TFDevice
}  // namespace mlir

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

#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"

namespace mlir {
namespace TFTPU {

namespace {

constexpr char kDeviceAttr[] = "device";
constexpr char kXlaOutsideCompilationAttr[] = "_xla_outside_compilation";

// Mapping for `_xla_outside_compilation` attribute to ops of a cluster.
using OutsideClusterMap =
    llvm::SmallDenseMap<llvm::StringRef, llvm::SmallVector<Operation*, 8>, 8>;

// This pass extracts a CPU computation cluster with `_xla_outside_compilation`
// annotation from a TPU cluster. Each outside compilation cluster is moved to
// a parallel_execute region. The TPU cluster is also moved to a
// parallel_execute region. Communication ops between device and host are
// added to pass inputs/outputs to/from the outside compiled region.
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
//   "tf_device.parallel_execute"() ( {
//     "tf_device.launch"() ( {
//       "tf.B()
//       tf_device.return
//     })
//     tf_device.return
//   }, {
//     "tf_device.cluster"( {
//       "tf.A"()
//       "tf.C"()
//       tf_device.return
//     })
//    tf_device.return
//  })

struct TPUExtractOutsideCompilation
    : public PassWrapper<TPUExtractOutsideCompilation,
                         OperationPass<ModuleOp>> {
  void runOnOperation() override;
};

// Creates a tf._XlaSendFromHost or tf._XlaSendFromHostV2 op. If device ordinal
// is present, a tf._XlaSendFromHostV2 op is created instead.
Operation* CreateSendFromHostOp(OpBuilder& builder, Location loc,
                                ValueRange inputs, Value compilation_key,
                                Value device_ordinal,
                                llvm::StringRef communication_key) {
  if (device_ordinal)
    return builder.create<TF::_XlaSendFromHostV2Op>(
        loc, inputs,
        /*dynamic_key=*/compilation_key, device_ordinal,
        builder.getStringAttr(communication_key));

  return builder.create<TF::_XlaSendFromHostOp>(
      loc, inputs,
      /*dynamic_key=*/compilation_key, builder.getStringAttr(communication_key),
      /*device_ordinal=*/builder.getI64IntegerAttr(0));
}

// Creates a tf._XlaRecvAtHost or tf._XlaRecvAtHostV2 op. If device ordinal is
// present, a tf._XlaRecvAtHostV2 op is created instead.
Operation* CreateRecvAtHostOp(OpBuilder& builder, Location loc,
                              TypeRange output_types, Value compilation_key,
                              Value device_ordinal,
                              llvm::StringRef communication_key) {
  if (device_ordinal)
    return builder.create<TF::_XlaRecvAtHostV2Op>(
        loc, output_types, /*dynamic_key=*/compilation_key, device_ordinal,
        builder.getStringAttr(communication_key));

  return builder.create<TF::_XlaRecvAtHostOp>(
      loc, output_types, /*dynamic_key=*/compilation_key,
      builder.getStringAttr(communication_key),
      /*device_ordinal=*/builder.getI64IntegerAttr(0));
}

// Holds information about control flow operations that wrap outside compiled
// op. Currently only tf.IfRegion and tf.WhileRegion ops are supported.
class ControlFlowStackInfo {
 public:
  enum ControlFlowBranchType { kIfThen, kIfElse, kWhileCond, kWhileBody };

  explicit ControlFlowStackInfo(Operation* wrapping_op, Operation* nested_op)
      : callsite_op_(wrapping_op) {
    if (auto control_flow_op = llvm::dyn_cast<TF::IfRegionOp>(callsite_op_)) {
      auto parent_region = nested_op->getParentRegion();
      if (&control_flow_op.then_branch() == parent_region) {
        type_ = ControlFlowBranchType::kIfThen;
      } else {
        type_ = ControlFlowBranchType::kIfElse;
      }
    } else if (auto control_flow_op =
                   llvm::dyn_cast<TF::WhileRegionOp>(callsite_op_)) {
      auto parent_region = nested_op->getParentRegion();
      if (&control_flow_op.cond() == parent_region) {
        type_ = ControlFlowBranchType::kWhileCond;
      } else {
        type_ = ControlFlowBranchType::kWhileBody;
      }
    } else {
      assert(false);
    }
  }

  Value GetIfPredicateValue() {
    auto if_op = llvm::cast<TF::IfRegionOp>(callsite_op_);
    return if_op.cond();
  }

  ControlFlowBranchType GetBranchType() const { return type_; }

  Operation* GetCallSiteOp() const { return callsite_op_; }

  bool operator==(const ControlFlowStackInfo& other) const {
    return type_ == other.type_ && callsite_op_ == other.callsite_op_;
  }

 private:
  ControlFlowBranchType type_;

  // `this` does not hold ownership of `callsite_op_`.
  Operation* callsite_op_;
};

// Returns a list of ControlFlowStackInfo that represents a stack of control
// flow operations that wraps `op`.
llvm::SmallVector<ControlFlowStackInfo, 4> GetControlFlowStackForOp(
    tf_device::ClusterOp tpu_cluster, Operation* op) {
  assert(tpu_cluster.getOperation()->isProperAncestor(op));

  llvm::SmallVector<ControlFlowStackInfo, 4> controlflow_stack;
  Operation* op_in_stack = op;
  while (op_in_stack != tpu_cluster.getOperation()) {
    auto parent_op = op_in_stack->getParentOp();
    if (llvm::isa<TF::IfRegionOp, TF::WhileRegionOp>(parent_op)) {
      controlflow_stack.insert(controlflow_stack.begin(),
                               ControlFlowStackInfo(parent_op, op_in_stack));
    }
    op_in_stack = parent_op;
  }

  return controlflow_stack;
}

// Creates a IfRegionOp with `predicate` and then/else region with yield op and
// an empty block.
TF::IfRegionOp CloneEmptyIfWithPredicate(Value predicate, bool is_stateless,
                                         Location loc, OpBuilder& builder) {
  auto host_side_if = builder.create<TF::IfRegionOp>(
      loc, llvm::SmallVector<Type, 4>{}, predicate, is_stateless);

  // Create empty then branch region.
  auto& then_branch = host_side_if.then_branch();
  then_branch.push_back(new Block);
  builder.setInsertionPointToEnd(&then_branch.front());
  builder.create<TF::YieldOp>(loc, /*operands=*/ArrayRef<Value>{});

  // Create empty else branch region.
  auto& else_branch = host_side_if.else_branch();
  else_branch.push_back(new Block);
  builder.setInsertionPointToEnd(&else_branch.front());
  builder.create<TF::YieldOp>(loc, /*operands=*/ArrayRef<Value>{});
  return host_side_if;
}

// Replicates tf.IfRegion op to host side computation.
Operation* ReplicateIf(const ControlFlowStackInfo& controlflow_info,
                       llvm::StringRef outside_cluster_name,
                       Value compilation_key, Value device_ordinal,
                       OpBuilder& builder, int* send_recv_counter) {
  // Create XlaSendToHostOp to send predicate value from device to host.
  OpBuilder::InsertPoint insert_point = builder.saveInsertionPoint();
  auto if_callsite_op =
      llvm::cast<TF::IfRegionOp>(controlflow_info.GetCallSiteOp());
  builder.setInsertionPoint(if_callsite_op);

  const auto predicate_send_recv_key =
      llvm::formatv("if_predicate_channel_{0}_{1}", outside_cluster_name,
                    *send_recv_counter)
          .str();
  *send_recv_counter += 1;

  auto predicate = if_callsite_op.cond();
  auto predicate_shape = predicate.getType();
  builder.create<TF::XlaSendToHostOp>(if_callsite_op.getLoc(), predicate,
                                      predicate_send_recv_key);

  // Create XlaRecvAtHostOp to receive predicate value from host.
  builder.restoreInsertionPoint(insert_point);
  Operation* recv_predicate_at_host = CreateRecvAtHostOp(
      builder, if_callsite_op.getLoc(), TypeRange{predicate_shape},
      compilation_key, device_ordinal, predicate_send_recv_key);

  // Create host side if op.
  return CloneEmptyIfWithPredicate(recv_predicate_at_host->getResult(0),
                                   if_callsite_op.is_stateless(),
                                   if_callsite_op.getLoc(), builder);
}

// Creates a WhileRegionOp cond and body regions with yield op and
// an empty body.
TF::WhileRegionOp CloneEmptyWhile(bool is_stateless,
                                  uint64_t parallel_iterations, Location loc,
                                  OpBuilder& builder) {
  auto host_side_while = builder.create<TF::WhileRegionOp>(
      loc, /*output=*/ArrayRef<Type>{}, /*input=*/ArrayRef<Value>{},
      parallel_iterations, is_stateless, /*shape_invariant=*/false);

  // Create empty else branch region.
  auto& body = host_side_while.body();
  body.push_back(new Block);
  builder.setInsertionPointToEnd(&body.front());
  builder.create<TF::YieldOp>(loc, /*operands=*/ArrayRef<Value>{});
  return host_side_while;
}

// Replicates tf.WhileRegion op to host side computation.
Operation* ReplicateWhile(const ControlFlowStackInfo& controlflow_info,
                          llvm::StringRef outside_cluster_name,
                          Value compilation_key, Value device_ordinal,
                          OpBuilder& builder, int* send_recv_counter) {
  // Create XlaSendToHostOp to send cond region output from device to host.
  OpBuilder::InsertPoint insert_point = builder.saveInsertionPoint();
  auto while_callsite_op =
      llvm::cast<TF::WhileRegionOp>(controlflow_info.GetCallSiteOp());
  builder.setInsertionPoint(while_callsite_op.cond().front().getTerminator());

  const auto condition_send_recv_key =
      llvm::formatv("while_condition_channel_{0}_{1}", outside_cluster_name,
                    *send_recv_counter)
          .str();
  *send_recv_counter += 1;
  auto condition =
      while_callsite_op.cond().front().getTerminator()->getOperand(0);
  builder.create<TF::XlaSendToHostOp>(while_callsite_op.getLoc(), condition,
                                      condition_send_recv_key);
  builder.restoreInsertionPoint(insert_point);

  auto host_side_while = CloneEmptyWhile(
      while_callsite_op.is_stateless(), while_callsite_op.parallel_iterations(),
      while_callsite_op.getLoc(), builder);

  // Create cond region and yield the condition from the device.
  auto& cond = host_side_while.cond();
  cond.push_back(new Block);
  builder.setInsertionPointToEnd(&cond.front());
  auto recv_condition_at_host = CreateRecvAtHostOp(
      builder, while_callsite_op.getLoc(), TypeRange{condition.getType()},
      compilation_key, device_ordinal, condition_send_recv_key);
  builder.create<TF::YieldOp>(while_callsite_op.getLoc(),
                              recv_condition_at_host->getResults());

  return host_side_while;
}

// TODO(b/157054714): Use a better abstraction instead of
// _TPUCompileMlirOp and _XlaRecvAtHostOp and _XlaSendFromHostOp.
// Creates a compilation key as placeholder. A placeholder compilation cache key
// is created because it is a required input to _XlaRecvAtHost and
// _XlaSendFromHost but the _TPUCompileMlir has not yet been created for the TPU
// cluster that contains the outside compiled ops. This placeholder should be
// replaced by the TPU cluster _TPUCompileMlir in a subsequent pass.
Value CreateCompilationKeyPlaceholder(Location loc, OpBuilder& builder) {
  auto result_type =
      RankedTensorType::get({2}, builder.getType<TF::StringType>());
  return builder.create<TF::_TPUCompileMlirPlaceholderProgramKeyOp>(
      loc, /*program=*/result_type, llvm::ArrayRef<Value>{});
}

// Retrieves terminator of branch specified by `control_flow_info` of replicated
// control flow op.
Operation* GetControlFlowBranchRegionTerminator(
    const ControlFlowStackInfo& controlflow_info, Operation* controlflow_op) {
  if (auto inner_most_if = llvm::dyn_cast<TF::IfRegionOp>(controlflow_op)) {
    if (controlflow_info.GetBranchType() == ControlFlowStackInfo::kIfThen) {
      return inner_most_if.then_branch().front().getTerminator();
    } else {
      return inner_most_if.else_branch().front().getTerminator();
    }
  } else if (auto inner_most_while =
                 llvm::dyn_cast<TF::WhileRegionOp>(controlflow_op)) {
    if (controlflow_info.GetBranchType() == ControlFlowStackInfo::kWhileCond) {
      return &inner_most_while.cond().front().front();
    } else {
      return inner_most_while.body().front().getTerminator();
    }
  }
  assert(false);
  return nullptr;
}

// Replicates the control flow operations that wraps outside compiled ops to
// `destination_block`.
Operation* GetOrReplicateControlFlowStack(
    llvm::StringRef outside_cluster_name,
    const llvm::SmallVectorImpl<ControlFlowStackInfo>& stack_info,
    tf_device::ClusterOp tpu_cluster, ModuleOp module, Value compilation_key,
    Value device_ordinal, Block* destination_block, int* send_recv_counter,
    llvm::SmallDenseMap<Operation*, Operation*>* replicated_controlflow_map) {
  assert(!stack_info.empty());
  const auto& controlflow_info = stack_info.back();
  auto it = replicated_controlflow_map->find(controlflow_info.GetCallSiteOp());
  if (it != replicated_controlflow_map->end())
    return GetControlFlowBranchRegionTerminator(controlflow_info, it->second);

  OpBuilder builder = OpBuilder::atBlockTerminator(destination_block);
  Operation* previous_replicated_controlflow_op = nullptr;
  for (const auto& controlflow_stack_info : stack_info) {
    // If controlflow operation has already been created, reuse the cached
    // controlflow operation.
    auto it = replicated_controlflow_map->find(
        controlflow_stack_info.GetCallSiteOp());
    if (it != replicated_controlflow_map->end()) {
      previous_replicated_controlflow_op = it->second;
      builder.setInsertionPoint(GetControlFlowBranchRegionTerminator(
          controlflow_stack_info, previous_replicated_controlflow_op));
      continue;
    }

    // Create control flow op given provided insertion point and
    // ControlFlowStackInfo.
    if (auto control_flow_op = llvm::dyn_cast<TF::IfRegionOp>(
            controlflow_stack_info.GetCallSiteOp())) {
      previous_replicated_controlflow_op = ReplicateIf(
          controlflow_stack_info, outside_cluster_name, compilation_key,
          device_ordinal, builder, send_recv_counter);
      auto if_op =
          llvm::cast<TF::IfRegionOp>(previous_replicated_controlflow_op);
      auto type = controlflow_stack_info.GetBranchType();

      // Update the insertion point to proper region inside the newly created
      // control flow op.
      if (type == ControlFlowStackInfo::kIfThen) {
        builder.setInsertionPoint(&if_op.then_branch().front().front());
      } else {
        builder.setInsertionPoint(&if_op.else_branch().front().front());
      }
    } else if (auto control_flow_op = llvm::dyn_cast<TF::WhileRegionOp>(
                   controlflow_stack_info.GetCallSiteOp())) {
      previous_replicated_controlflow_op = ReplicateWhile(
          controlflow_stack_info, outside_cluster_name, compilation_key,
          device_ordinal, builder, send_recv_counter);
      auto while_op =
          llvm::cast<TF::WhileRegionOp>(previous_replicated_controlflow_op);
      auto type = controlflow_stack_info.GetBranchType();
      if (type == ControlFlowStackInfo::kWhileCond) {
        builder.setInsertionPoint(&while_op.cond().front().front());
      } else {
        builder.setInsertionPoint(&while_op.body().front().front());
      }
    }
  }

  replicated_controlflow_map->try_emplace(stack_info.back().GetCallSiteOp(),
                                          previous_replicated_controlflow_op);

  // Return operation which should be used to as the insertion point to create
  // send/recv ops.
  return GetControlFlowBranchRegionTerminator(
      stack_info.back(), previous_replicated_controlflow_op);
}

// Collects and clusters ops in `block` with the same `_xla_outside_compilation`
// attribute into `clusters` This returns an error if a
// `_xla_outside_compilation` attribute of an op is empty.
// TODO(b/163141763): Make sure ops inside control flow regions are not outside
// compiled if the entire control flow op is marked as outside compiled.
LogicalResult CollectAndGroupOutsideClusterOps(Block* block,
                                               OutsideClusterMap* clusters) {
  auto walk_result = block->walk([&](Operation* op) {
    if (auto attr = op->getAttrOfType<StringAttr>(kXlaOutsideCompilationAttr)) {
      if (attr.getValue().empty()) {
        op->emitError() << "attribute '" << kXlaOutsideCompilationAttr
                        << "' is empty";
        return WalkResult::interrupt();
      }

      auto it = clusters->try_emplace(attr.getValue());
      it.first->getSecond().push_back(op);
    }
    return WalkResult::advance();
  });

  return failure(walk_result.wasInterrupted());
}

// Moves `cluster_ops` before `op`.
void MoveOutsideClusterOpsBeforeOp(Operation* op,
                                   llvm::ArrayRef<Operation*> cluster_ops,
                                   MLIRContext* context) {
  for (Operation* cluster_op : cluster_ops) {
    // Remove `_xla_outside_compilation` and `device` attribute from ops in the
    // cluster as that information will be present in the `launch_op`.
    cluster_op->removeAttr(
        Identifier::get(kXlaOutsideCompilationAttr, context));
    cluster_op->removeAttr(Identifier::get(kDeviceAttr, context));
    cluster_op->moveBefore(op);
  }
}

// Creates a `tf_device.launch` to wrap cluster ops.
tf_device::LaunchOp CreateLaunchOpForOutsideCluster(
    OpBuilder& builder, Operation* last_cluster_op,
    llvm::StringRef host_device) {
  // An empty string placeholder is used for the device as that will be later
  // populated with the device of the associated TPUReplicateMetadata op.
  auto launch_op = builder.create<tf_device::LaunchOp>(
      last_cluster_op->getLoc(), builder.getStringAttr(host_device),
      /*result_types=*/ArrayRef<Type>{});

  launch_op.body().push_back(new Block);
  builder.setInsertionPointToEnd(&launch_op.GetBody());
  builder.create<tf_device::ReturnOp>(last_cluster_op->getLoc(),
                                      llvm::ArrayRef<Value>{});

  return launch_op;
}

// Extracts all externally provided operands of `host_cluster_ops`.
llvm::SmallSetVector<Value, 4> GetExternalOperands(
    tf_device::ClusterOp tpu_cluster,
    llvm::ArrayRef<Operation*> host_cluster_ops) {
  llvm::SmallSetVector<Value, 4> external_values;

  for (Operation* host_cluster_op : host_cluster_ops) {
    auto cluster_op_parent_region = host_cluster_op->getParentRegion();
    host_cluster_op->walk([&](Operation* op) {
      auto region = op->getParentRegion();

      if (region == cluster_op_parent_region) {
        // For op operands, add operand defining ops, if they are not included
        // in `host_cluster_ops`.
        for (Value v : op->getOperands()) {
          Operation* defining_op = v.getDefiningOp();
          bool is_external = false;
          if (defining_op) {
            if (!tpu_cluster.getOperation()->isAncestor(defining_op)) continue;

            is_external =
                llvm::none_of(host_cluster_ops, [&](Operation* cluster_op) {
                  return defining_op == cluster_op;
                });
          } else {
            if (auto block_arg = v.dyn_cast<BlockArgument>()) {
              if (block_arg.getParentRegion() == cluster_op_parent_region)
                is_external = true;
            }
          }
          if (is_external) external_values.insert(v);
        }
      } else {
        llvm::SetVector<Value> external_captured_inputs;
        visitUsedValuesDefinedAbove(*region, *region, [&](OpOperand* operand) {
          const bool captured_value_from_host =
              llvm::find(host_cluster_ops, operand->get().getDefiningOp()) !=
              host_cluster_ops.end();
          if (captured_value_from_host) return;

          Region* operand_defined_region = operand->get().getParentRegion();
          if (!tpu_cluster.body().isAncestor(operand_defined_region)) return;
          // If the host_cluster_op is regional control flow (if, while),
          // then check if the operand_defined_region is an ancestor of the
          // control flow regions.
          if (auto if_op = llvm::dyn_cast<TF::IfRegionOp>(host_cluster_op)) {
            if (if_op.then_branch().isAncestor(operand_defined_region) ||
                if_op.else_branch().isAncestor(operand_defined_region))
              return;
          }
          if (auto while_op =
                  llvm::dyn_cast<TF::WhileRegionOp>(host_cluster_op)) {
            if (while_op.cond().isAncestor(operand_defined_region) ||
                while_op.body().isAncestor(operand_defined_region))
              return;
          }
          external_captured_inputs.insert(operand->get());
        });
        external_values.insert(external_captured_inputs.begin(),
                               external_captured_inputs.end());
      }
    });
  }

  return external_values;
}

// Extracts all externally used outputs of `cluster_ops`.
llvm::SmallSetVector<Value, 4> GetExternalOutputs(
    llvm::ArrayRef<Operation*> cluster_ops) {
  llvm::SmallSetVector<Value, 4> external_outputs;
  llvm::SmallPtrSet<Operation*, 4> host_cluster_ops_set;
  for (auto op : cluster_ops) {
    op->walk([&](Operation* host_cluster_op) {
      host_cluster_ops_set.insert(host_cluster_op);
    });
  }

  for (Operation* op : cluster_ops) {
    for (Operation* user : op->getUsers()) {
      bool is_external = llvm::none_of(
          host_cluster_ops_set,
          [&](Operation* cluster_op) { return user == cluster_op; });
      if (!is_external) continue;
      for (Value v : user->getOperands()) {
        if (v.getDefiningOp() == op) external_outputs.insert(v);
      }
    }
  }

  return external_outputs;
}

// Sets the insertion point on `builder` for HostCompute op.  Sets insertion
// point to the first op in `cluster_ops` that has one of `external_inputs`
// as an operand.  If there are no external_inputs, set insertion point to first
// cluster_op.
void SetHostComputeInsertion(
    OpBuilder& builder, llvm::ArrayRef<Operation*> cluster_ops,
    const llvm::SmallSetVector<Value, 4>& external_inputs) {
  if (external_inputs.empty()) builder.setInsertionPoint(cluster_ops.front());
  for (const auto& cluster_op : cluster_ops) {
    for (Value v : cluster_op->getOperands()) {
      if (external_inputs.count(v)) {
        builder.setInsertionPoint(cluster_op);
        return;
      }
    }
  }

  // If no operand usage can be found, this means that external input is
  // implicitly captured inputs for ops inside internal regions of one of the
  // `cluster_ops`. In that case, set the insertion point to the last op of the
  // `cluster_ops` in the IR.
  builder.setInsertionPoint(cluster_ops.back());
}

// Creates the HostCompute with `inputs` and `outputs`
// using `communication_key`.
TF::_XlaHostComputeMlirOp CreateHostCompute(
    OpBuilder& builder, tf_device::ClusterOp tpu_cluster,
    llvm::ArrayRef<Operation*> cluster_ops,
    const llvm::SmallSetVector<Value, 4>& inputs, llvm::ArrayRef<Value> outputs,
    llvm::StringRef args_communication_key,
    llvm::StringRef retvals_communication_key) {
  llvm::SmallVector<Type, 4> device_output_types;
  for (const auto& output : outputs)
    device_output_types.push_back(output.getType());
  SetHostComputeInsertion(builder, cluster_ops, inputs);
  auto host_compute = builder.create<TF::_XlaHostComputeMlirOp>(
      tpu_cluster.getLoc(), device_output_types, inputs.getArrayRef(),
      builder.getStringAttr(args_communication_key),
      builder.getStringAttr(retvals_communication_key),
      /*tpu_core=*/builder.getI64IntegerAttr(0));
  return host_compute;
}

// Represents a set of ops inside host computation that is wrapped inside the
// same control flow.
struct HostCluster {
  // List of control flow that wraps host computation operations. May be empty.
  llvm::SmallVector<ControlFlowStackInfo, 4> controlflow_stack;

  // Set of operations that will run on host wrapped around same stack of
  // control flow.
  llvm::SmallVector<Operation*, 4> section_ops;
};

HostCluster* FindHostCluster(
    llvm::SmallVectorImpl<HostCluster>& host_cluster_sections,
    const llvm::SmallVector<ControlFlowStackInfo, 4>& control_flows) {
  for (auto& section : host_cluster_sections)
    if (control_flows == section.controlflow_stack) return &section;
  return nullptr;
}

void MoveOutsideCompiledOpsInsideControlFlow(
    ModuleOp module, tf_device::ClusterOp tpu_cluster,
    llvm::StringRef host_cluster_section_name,
    tf_device::LaunchOp host_launch_op, Value compilation_key,
    Value device_ordinal, llvm::ArrayRef<Operation*> cluster_section_ops,
    const llvm::SmallVectorImpl<ControlFlowStackInfo>& controlflow_stack,
    llvm::ArrayRef<Operation*> cluster_ops,
    const llvm::SmallSetVector<Value, 4>& section_external_inputs,
    llvm::ArrayRef<Value> section_external_outputs,
    llvm::SmallDenseMap<Operation*, Operation*>* replicated_controlflow_map) {
  Operation* insertion_op = nullptr;
  if (controlflow_stack.empty()) {
    insertion_op = host_launch_op.GetBody().getTerminator();
  } else {
    int send_recv_counter = 0;
    insertion_op = GetOrReplicateControlFlowStack(
        host_cluster_section_name, controlflow_stack, tpu_cluster, module,
        compilation_key, device_ordinal, &host_launch_op.GetBody(),
        &send_recv_counter, replicated_controlflow_map);
  }

  MLIRContext* context = host_launch_op.getContext();
  if (section_external_inputs.empty() && section_external_outputs.empty()) {
    MoveOutsideClusterOpsBeforeOp(insertion_op, cluster_section_ops, context);
    return;
  }

  OpBuilder builder(insertion_op);
  llvm::SmallVector<Type, 4> host_output_types;
  for (const auto& external_input : section_external_inputs)
    host_output_types.push_back(external_input.getType());

  std::string args_communication_key =
      llvm::formatv("host_compute_channel_{0}_args", host_cluster_section_name)
          .str();
  std::string retvals_communication_key =
      llvm::formatv("host_compute_channel_{0}_retvals",
                    host_cluster_section_name)
          .str();

  auto recv_at_host = CreateRecvAtHostOp(
      builder, tpu_cluster.getLoc(), host_output_types, compilation_key,
      device_ordinal, args_communication_key);

  auto host_compute =
      CreateHostCompute(builder, tpu_cluster, cluster_section_ops,
                        section_external_inputs, section_external_outputs,
                        args_communication_key, retvals_communication_key);
  MoveOutsideClusterOpsBeforeOp(insertion_op, cluster_section_ops, context);

  builder.setInsertionPoint(insertion_op);
  if (!section_external_outputs.empty())
    CreateSendFromHostOp(builder, tpu_cluster.getLoc(),
                         section_external_outputs, compilation_key,
                         device_ordinal, retvals_communication_key);

  for (auto result :
       llvm::zip(section_external_inputs, recv_at_host->getResults())) {
    mlir::replaceAllUsesInRegionWith(std::get<0>(result), std::get<1>(result),
                                     *insertion_op->getParentRegion());
  }

  auto operand_inside_device_cluster = [&](OpOperand& operand) {
    return tpu_cluster.body().isAncestor(
               operand.getOwner()->getParentRegion()) &&
           llvm::none_of(cluster_ops, [&](Operation* cluster_op) {
             return operand.getOwner() == cluster_op;
           });
  };

  for (auto result :
       llvm::zip(section_external_outputs, host_compute.getResults())) {
    Value external_output = std::get<0>(result);
    external_output.replaceUsesWithIf(std::get<1>(result),
                                      operand_inside_device_cluster);
  }
}

void MoveOutsideCompiledOps(
    ModuleOp module, tf_device::ClusterOp tpu_cluster,
    llvm::StringRef outside_cluster_name, tf_device::LaunchOp host_launch_op,
    llvm::ArrayRef<Operation*> cluster_ops,
    const llvm::SmallSetVector<Value, 4>& external_inputs,
    const llvm::SmallSetVector<Value, 4>& external_outputs) {
  // Identify and groups ops in `cluster_ops` by ops wrapped inside the same
  // control flows.
  llvm::SmallVector<HostCluster, 4> host_cluster_sections;
  for (Operation* host_cluster_op : cluster_ops) {
    auto controlflow_stack =
        GetControlFlowStackForOp(tpu_cluster, host_cluster_op);
    auto host_cluster_section =
        FindHostCluster(host_cluster_sections, controlflow_stack);
    if (!host_cluster_section) {
      host_cluster_sections.emplace_back(
          HostCluster{controlflow_stack, {host_cluster_op}});
    } else {
      host_cluster_section->section_ops.emplace_back(host_cluster_op);
    }
  }

  const bool has_control_flow =
      llvm::any_of(host_cluster_sections, [](const auto host_cluster_section) {
        return !host_cluster_section.controlflow_stack.empty();
      });

  Value compilation_key = nullptr;
  Value device_ordinal = nullptr;
  if (has_control_flow || !external_inputs.empty() ||
      !external_outputs.empty()) {
    OpBuilder builder(&host_launch_op.GetBody().front());
    compilation_key =
        CreateCompilationKeyPlaceholder(tpu_cluster.getLoc(), builder);

    // If there is no replication/data parallelism, it is assumed the device
    // ordinal is always 0 (e.g. /device:TPU:0). In that case, a constant 0
    // attribute can be used instead for _XlaSendFromHost/_XlaRecvAtHost ops.
    if (tpu_cluster.getParentOfType<tf_device::ReplicateOp>()) {
      auto device_ordinal_op =
          builder.create<TF::_TPUDeviceOrdinalPlaceholderOp>(
              host_launch_op.getLoc(),
              RankedTensorType::get({}, builder.getI64Type()));
      device_ordinal = device_ordinal_op.device_ordinal();
    }
  }

  // Maintains a map of control flow callsite operation in TPU device side
  // and an replicated control flow operation on host cluster.
  llvm::SmallDenseMap<Operation*, Operation*> replicated_controlflows;

  // Move `cluster_op` to host cluster, replicating control flow if ops are
  // wrapped inside a control flow.
  for (const auto& host_cluster_section_and_index :
       llvm::enumerate(host_cluster_sections)) {
    const auto& host_cluster_section = host_cluster_section_and_index.value();
    const int index = host_cluster_section_and_index.index();

    const auto& controlflow_stack = host_cluster_section.controlflow_stack;
    const auto& cluster_section_ops = host_cluster_section.section_ops;
    auto section_external_inputs =
        GetExternalOperands(tpu_cluster, cluster_section_ops);
    for (auto input : section_external_inputs) {
      if (!external_inputs.contains(input))
        section_external_inputs.remove(input);
    }
    auto section_external_outputs = GetExternalOutputs(cluster_section_ops);
    for (auto output : section_external_outputs) {
      if (!external_outputs.contains(output))
        section_external_outputs.remove(output);
    }

    const std::string host_cluster_section_name =
        llvm::formatv("{0}_{1}", outside_cluster_name, index).str();

    MoveOutsideCompiledOpsInsideControlFlow(
        module, tpu_cluster, host_cluster_section_name, host_launch_op,
        compilation_key, device_ordinal, cluster_section_ops, controlflow_stack,
        cluster_ops, section_external_inputs,
        section_external_outputs.takeVector(), &replicated_controlflows);
  }
}

// Creates a `parallel_execute` op in place of launch with 'clusters` and
// 'launch` as regions.
void CreateParallelExecuteFromOutsideClusters(ModuleOp module,
                                              tf_device::ClusterOp tpu_cluster,
                                              const OutsideClusterMap& clusters,
                                              llvm::StringRef host_device) {
  OpBuilder builder(tpu_cluster);
  // Create parallel_execute regions.  The original TPU cluster computation
  // is the extra region.
  const int num_regions = 1 + clusters.size();
  auto parallel_execute_op = builder.create<tf_device::ParallelExecuteOp>(
      tpu_cluster.getLoc(), num_regions, tpu_cluster.results().getTypes());

  // Move outside compilation clusters to parallel_execute regions.
  for (const auto& cluster : llvm::enumerate(clusters)) {
    const auto& cluster_ops = cluster.value().getSecond();

    Block& outside_block =
        parallel_execute_op.GetRegionBlockWithIndex(cluster.index());

    builder.setInsertionPointToEnd(&outside_block);
    tf_device::LaunchOp host_launch_op = CreateLaunchOpForOutsideCluster(
        builder, cluster_ops.back(), host_device);

    // Determine if there are any inputs that are provided out of cluster.
    auto external_inputs = GetExternalOperands(tpu_cluster, cluster_ops);
    auto external_outputs = GetExternalOutputs(cluster_ops);

    MoveOutsideCompiledOps(module, tpu_cluster, cluster.value().getFirst(),
                           host_launch_op, cluster_ops, external_inputs,
                           external_outputs);
    builder.setInsertionPointToEnd(&outside_block);
    builder.create<tf_device::ReturnOp>(tpu_cluster.getLoc(),
                                        ArrayRef<Value>{});
  }

  // Move the launch body to last parallel_execute block.
  Block& parallel_execute_tpu_block =
      parallel_execute_op.GetRegionBlockWithIndex(num_regions - 1);
  builder.setInsertionPointToEnd(&parallel_execute_tpu_block);
  builder.create<tf_device::ReturnOp>(tpu_cluster.getLoc(),
                                      tpu_cluster.getResults());
  tpu_cluster.getOperation()->moveBefore(
      parallel_execute_tpu_block.getTerminator());

  // Remap cluster results with parallel_execute results if user is outside of
  // parallel_execute.
  for (auto result :
       llvm::zip(tpu_cluster.getResults(), parallel_execute_op.getResults())) {
    Value tpu_cluster_result = std::get<0>(result);
    Value parallel_execute_result = std::get<1>(result);
    for (auto& use : llvm::make_early_inc_range(tpu_cluster_result.getUses()))
      if (!parallel_execute_op.getOperation()->isProperAncestor(use.getOwner()))
        use.set(parallel_execute_result);
  }
}

void TPUExtractOutsideCompilation::runOnOperation() {
  // Get runtime devices information from the closest parent module.
  auto module = getOperation();
  mlir::TF::RuntimeDevices devices;
  if (failed(tensorflow::GetDevicesFromOp(module, &devices)))
    return signalPassFailure();

  auto extract_result =
      module.walk([&](tf_device::ClusterOp tpu_cluster) {
        OutsideClusterMap clusters;
        if (failed(CollectAndGroupOutsideClusterOps(&tpu_cluster.GetBody(),
                                                    &clusters)))
          return WalkResult::interrupt();

        if (clusters.empty()) return WalkResult::advance();

        std::string host_device;
        tensorflow::GetHostDeviceOutsideComputation(devices, tpu_cluster,
                                                    &host_device);

        CreateParallelExecuteFromOutsideClusters(module, tpu_cluster, clusters,
                                                 host_device);

        return WalkResult::advance();
      });

  if (extract_result.wasInterrupted()) return signalPassFailure();
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateTPUExtractOutsideCompilationPass() {
  return std::make_unique<TPUExtractOutsideCompilation>();
}

static PassRegistration<TPUExtractOutsideCompilation> pass(
    "tf-tpu-extract-outside-compilation",
    "Extracts TPU outside compilation to separate parallel_execute.");

}  // namespace TFTPU
}  // namespace mlir

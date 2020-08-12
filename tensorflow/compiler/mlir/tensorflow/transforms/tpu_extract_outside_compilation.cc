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
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
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

// Holds information about control flow operations that wrap outside compiled
// op. Currently only tf.If op is supported.
class ControlFlowStackInfo {
 public:
  enum ControlFlowBranchType { kIfThen, kIfElse };

  explicit ControlFlowStackInfo(Operation* wrapping_op, Operation* nested_op)
      : callsite_op_(wrapping_op) {
    // Only tf.IfRegion op is supported for now.
    auto control_flow_op = llvm::cast<TF::IfRegionOp>(callsite_op_);
    assert(control_flow_op);

    auto parent_region = nested_op->getParentRegion();
    if (&control_flow_op.then_branch() == parent_region) {
      type_ = ControlFlowBranchType::kIfThen;
    } else {
      type_ = ControlFlowBranchType::kIfElse;
    }
  }

  Value GetIfPredicateValue() {
    auto if_op = llvm::cast<TF::IfRegionOp>(callsite_op_);
    return if_op.cond();
  }

  ControlFlowBranchType GetBranchType() const { return type_; }

  Operation* GetCallSiteOp() const { return callsite_op_; }

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
    if (llvm::isa<TF::IfRegionOp>(parent_op)) {
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
                                         Location loc, OpBuilder* builder) {
  auto host_side_if = builder->create<TF::IfRegionOp>(
      loc, llvm::SmallVector<Type, 4>{}, predicate, is_stateless);

  // Create empty then branch region.
  auto& then_branch = host_side_if.then_branch();
  builder->setInsertionPoint(&then_branch.front(), then_branch.front().begin());
  builder->createBlock(&then_branch);
  builder->create<TF::YieldOp>(loc, /*operands=*/ArrayRef<Value>{});

  // Create empty else branch region.
  auto& else_branch = host_side_if.else_branch();
  builder->setInsertionPoint(&else_branch.front(), else_branch.front().begin());
  builder->createBlock(&else_branch);
  builder->create<TF::YieldOp>(loc, /*operands=*/ArrayRef<Value>{});
  return host_side_if;
}

// Replicates tf.IfRegion op to host side computation.
Operation* ReplicateIf(const ControlFlowStackInfo& controlflow_info,
                       llvm::StringRef outside_cluster_name, ModuleOp module,
                       Value compilation_key, OpBuilder* builder,
                       int* send_recv_counter) {
  // Create XlaSendToHostOp to send predicate value from device to host.
  OpBuilder::InsertPoint insert_point = builder->saveInsertionPoint();
  auto if_callsite_op =
      llvm::cast<TF::IfRegionOp>(controlflow_info.GetCallSiteOp());
  builder->setInsertionPoint(if_callsite_op);

  const auto predicate_send_recv_key =
      llvm::formatv("if_predicate_channel_{0}_{1}", outside_cluster_name,
                    *send_recv_counter)
          .str();
  *send_recv_counter += 1;

  auto predicate = if_callsite_op.cond();
  auto predicate_shape = predicate.getType();
  builder->create<TF::XlaSendToHostOp>(if_callsite_op.getLoc(), predicate,
                                       predicate_send_recv_key);

  // Create XlaRecvAtHostOp to receive predicate value from host.
  builder->restoreInsertionPoint(insert_point);
  auto recv_predicate_at_host = builder->create<TF::_XlaRecvAtHostOp>(
      if_callsite_op.getLoc(), llvm::ArrayRef<Type>{predicate_shape},
      /*dynamic_key=*/compilation_key,
      builder->getStringAttr(predicate_send_recv_key),
      /*device_ordinal=*/builder->getI64IntegerAttr(0));

  // Create host side if op.
  return CloneEmptyIfWithPredicate(recv_predicate_at_host.getResult(0),
                                   if_callsite_op.is_stateless(),
                                   if_callsite_op.getLoc(), builder);
}

// TODO(b/157054714): Use a better abstraction instead of
// _TPUCompileMlirOp and _XlaRecvAtHostOp and _XlaSendFromHostOp.
// Creates a compilation key as placeholder. A placeholder compilation cache key
// is created because it is a required input to _XlaRecvAtHost and
// _XlaSendFromHost but the _TPUCompileMlir has not yet been created for the TPU
// cluster that contains the outside compiled ops. This placeholder should be
// replaced by the TPU cluster _TPUCompileMlir in a subsequent pass.
Value CreateCompilationKeyPlaceholder(Location loc, OpBuilder* builder) {
  auto result_type =
      RankedTensorType::get({2}, builder->getType<TF::StringType>());
  return builder->create<TF::_TPUCompileMlirPlaceholderProgramKeyOp>(
      loc, /*program=*/result_type, llvm::ArrayRef<Value>{});
}

// Replicates the control flow operations that wraps outside compiled ops to
// `destination_block`.
Block* ReplicateControlFlowStack(
    llvm::StringRef outside_cluster_name,
    const llvm::SmallVectorImpl<ControlFlowStackInfo>& stack_info,
    tf_device::ClusterOp tpu_cluster, ModuleOp module, Value compilation_key,
    Block* destination_block, int* send_recv_counter) {
  assert(stack_info.size());
  OpBuilder builder = OpBuilder::atBlockTerminator(destination_block);
  Operation* previous_replicated_controlflow_op = nullptr;
  for (const auto& controlflow_stack_info : stack_info) {
    // Create control flow op given provided insertion point and
    // ControlFlowStackInfo.
    previous_replicated_controlflow_op =
        ReplicateIf(controlflow_stack_info, outside_cluster_name, module,
                    compilation_key, &builder, send_recv_counter);
    auto if_op = llvm::cast<TF::IfRegionOp>(previous_replicated_controlflow_op);
    auto type = controlflow_stack_info.GetBranchType();

    // Update the insertion point to proper region inside the newly created
    // control flow op.
    if (type == ControlFlowStackInfo::kIfThen) {
      builder.setInsertionPoint(&if_op.then_branch().front().front());
    } else {
      builder.setInsertionPoint(&if_op.else_branch().front().front());
    }
  }

  // Return the inner most branch at which outside compiled op is located.
  // This block will later be used as insertion point to create send/recv ops.
  auto inner_most_controlflow_stack = stack_info.back();
  auto inner_most_if =
      llvm::cast<TF::IfRegionOp>(previous_replicated_controlflow_op);
  if (inner_most_controlflow_stack.GetBranchType() ==
      ControlFlowStackInfo::kIfThen) {
    return &inner_most_if.then_branch().front();
  } else {
    return &inner_most_if.else_branch().front();
  }
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

// Moves `cluster_ops` to associated `block`.
void MoveOutsideClusterOpsToBlock(Block& block,
                                  llvm::ArrayRef<Operation*> cluster_ops,
                                  MLIRContext* context) {
  Operation* terminator = block.getTerminator();
  for (Operation* cluster_op : cluster_ops) {
    // Remove `_xla_outside_compilation` and `device` attribute from ops in the
    // cluster as that information will be present in the `launch_op`.
    cluster_op->removeAttr(
        Identifier::get(kXlaOutsideCompilationAttr, context));
    cluster_op->removeAttr(Identifier::get(kDeviceAttr, context));
    cluster_op->moveBefore(terminator);
  }
}

// Creates a `tf_device.launch` to wrap cluster ops.
tf_device::LaunchOp CreateLaunchOpForOutsideCluster(
    OpBuilder* builder, Operation* last_cluster_op,
    llvm::StringRef host_device) {
  // An empty string placeholder is used for the device as that will be later
  // populated with the device of the associated TPUReplicateMetadata op.
  auto launch_op = builder->create<tf_device::LaunchOp>(
      last_cluster_op->getLoc(), builder->getStringAttr(host_device),
      /*result_types=*/ArrayRef<Type>{});

  launch_op.body().push_back(new Block);

  // Add terminator.
  builder->setInsertionPointToEnd(&launch_op.GetBody());
  builder->create<tf_device::ReturnOp>(last_cluster_op->getLoc(),
                                       llvm::ArrayRef<Value>{});

  return launch_op;
}

// Extracts all externally provided operands of `cluster_ops`.
llvm::SmallSetVector<Value, 4> GetExternalOperands(
    llvm::ArrayRef<Operation*> cluster_ops) {
  llvm::SmallSetVector<Value, 4> external_values;

  for (Operation* op : cluster_ops) {
    for (Value v : op->getOperands()) {
      Operation* defining_op = v.getDefiningOp();
      if (!defining_op) continue;
      bool is_external = llvm::none_of(cluster_ops, [&](Operation* cluster_op) {
        return defining_op == cluster_op;
      });

      if (is_external) external_values.insert(v);
    }
  }

  return external_values;
}

// Extracts all externally used outputs of `cluster_ops`.
llvm::SmallVector<Value, 4> GetExternalOutputs(
    llvm::ArrayRef<Operation*> cluster_ops) {
  llvm::SmallSetVector<Value, 4> external_outputs;

  for (Operation* op : cluster_ops) {
    for (Operation* user : op->getUsers()) {
      bool is_external = llvm::none_of(cluster_ops, [&](Operation* cluster_op) {
        return user == cluster_op;
      });
      if (!is_external) continue;
      for (Value v : user->getOperands()) {
        if (v.getDefiningOp() == op) external_outputs.insert(v);
      }
    }
  }

  return external_outputs.takeVector();
}

// Sets the insertion point on `builder` for HostCompute op.  Sets insertion
// point to the first op in `cluster_ops` that has one of `external_inputs`
// as an operand.  If there are no external_inputs, set insertion point to first
// cluster_op.
void SetHostComputeInsertion(
    OpBuilder* builder, llvm::ArrayRef<Operation*> cluster_ops,
    const llvm::SmallSetVector<Value, 4>& external_inputs) {
  if (external_inputs.empty()) builder->setInsertionPoint(cluster_ops.front());
  for (const auto& cluster_op : cluster_ops) {
    for (Value v : cluster_op->getOperands()) {
      if (external_inputs.count(v)) {
        builder->setInsertionPoint(cluster_op);
        return;
      }
    }
  }
}

// Creates the HostCompute with `inputs` and `outputs`
// using `communication_key`.
TF::_XlaHostComputeMlirOp CreateHostCompute(
    OpBuilder* builder, tf_device::ClusterOp tpu_cluster,
    llvm::ArrayRef<Operation*> cluster_ops,
    const llvm::SmallSetVector<Value, 4>& inputs, llvm::ArrayRef<Value> outputs,
    llvm::StringRef args_communication_key,
    llvm::StringRef retvals_communication_key) {
  llvm::SmallVector<Type, 4> device_output_types;
  for (const auto& output : outputs)
    device_output_types.push_back(output.getType());
  SetHostComputeInsertion(builder, cluster_ops, inputs);
  auto host_compute = builder->create<TF::_XlaHostComputeMlirOp>(
      tpu_cluster.getLoc(), device_output_types, inputs.getArrayRef(),
      builder->getStringAttr(args_communication_key),
      builder->getStringAttr(retvals_communication_key),
      /*tpu_core=*/builder->getI64IntegerAttr(0));
  return host_compute;
}

void MoveOutsideCompiledOps(
    ModuleOp module, tf_device::ClusterOp tpu_cluster,
    llvm::StringRef outside_cluster_name, tf_device::LaunchOp host_launch_op,
    llvm::ArrayRef<Operation*> cluster_ops,
    const llvm::SmallSetVector<Value, 4>& external_inputs,
    llvm::ArrayRef<Value> external_outputs) {
  // Since ops in `cluster_ops` do not cross function/control flow boundary, it
  // is sufficient to identify the control flow that wraps `cluster_ops` by
  // looking at any arbitary op inside `cluster_ops`.
  auto controlflow_stack =
      GetControlFlowStackForOp(tpu_cluster, cluster_ops.front());

  Value compilation_key;
  if (!controlflow_stack.empty() || !external_inputs.empty() ||
      !external_outputs.empty()) {
    OpBuilder builder(&host_launch_op.GetBody().front());
    compilation_key =
        CreateCompilationKeyPlaceholder(tpu_cluster.getLoc(), &builder);
  }

  Block* block_to_move_host_cluster = nullptr;
  if (controlflow_stack.empty()) {
    block_to_move_host_cluster = &host_launch_op.GetBody();
  } else {
    int send_recv_counter = 0;
    block_to_move_host_cluster = ReplicateControlFlowStack(
        outside_cluster_name, controlflow_stack, tpu_cluster, module,
        compilation_key, &host_launch_op.GetBody(), &send_recv_counter);
  }

  MLIRContext* context = host_launch_op.getContext();
  if (external_inputs.empty() && external_outputs.empty()) {
    MoveOutsideClusterOpsToBlock(*block_to_move_host_cluster, cluster_ops,
                                 context);
    return;
  }

  OpBuilder builder(block_to_move_host_cluster->getTerminator());
  llvm::SmallVector<Type, 4> host_output_types;
  for (const auto& external_input : external_inputs)
    host_output_types.push_back(external_input.getType());

  std::string args_communication_key =
      llvm::formatv("host_compute_channel_{0}_args", outside_cluster_name)
          .str();
  std::string retvals_communication_key =
      llvm::formatv("host_compute_channel_{0}_retvals", outside_cluster_name)
          .str();

  auto recv_at_host = builder.create<TF::_XlaRecvAtHostOp>(
      tpu_cluster.getLoc(), host_output_types,
      /*dynamic_key=*/compilation_key,
      builder.getStringAttr(args_communication_key),
      /*device_ordinal=*/builder.getI64IntegerAttr(0));

  auto host_compute = CreateHostCompute(
      &builder, tpu_cluster, cluster_ops, external_inputs, external_outputs,
      args_communication_key, retvals_communication_key);
  MoveOutsideClusterOpsToBlock(*block_to_move_host_cluster, cluster_ops,
                               context);

  builder.setInsertionPoint(block_to_move_host_cluster->getTerminator());
  builder.create<TF::_XlaSendFromHostOp>(
      tpu_cluster.getLoc(), external_outputs,
      /*dynamic_key=*/compilation_key,
      builder.getStringAttr(retvals_communication_key),
      /*device_ordinal=*/builder.getI64IntegerAttr(0));

  for (auto result : llvm::zip(external_inputs, recv_at_host.getResults()))
    mlir::replaceAllUsesInRegionWith(std::get<0>(result), std::get<1>(result),
                                     host_launch_op.body());

  for (auto result : llvm::zip(external_outputs, host_compute.getResults()))
    mlir::replaceAllUsesInRegionWith(std::get<0>(result), std::get<1>(result),
                                     tpu_cluster.body());
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
        &builder, cluster_ops.back(), host_device);

    // Determine if there are any inputs that are provided out of cluster.
    auto external_inputs = GetExternalOperands(cluster_ops);
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

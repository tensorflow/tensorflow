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

#include <algorithm>
#include <cstdint>
#include <deque>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/shape_inference.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"

namespace mlir {
namespace TFDevice {

namespace {

constexpr char kDeviceAttr[] = "device";
constexpr char kHostFunctionAttr[] = "host_func";
constexpr char kXlaOutsideCompilationAttr[] = "_xla_outside_compilation";
constexpr char kNoReplicationCluster[] = "__no_replication_cluster";

#define GEN_PASS_DEF_EXTRACTOUTSIDECOMPILATIONPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

struct ExtractOutsideCompilation
    : public impl::ExtractOutsideCompilationPassBase<
          ExtractOutsideCompilation> {
  void runOnOperation() override;
};

// Build a function containing `ops` with `inputs` and `outputs` using
// `builder`.  The `ops` are cloned and modified to use the function arguments
// as inputs.
func::FuncOp BuildFunction(llvm::ArrayRef<Operation*> ops,
                           llvm::ArrayRef<Value> inputs,
                           llvm::ArrayRef<Value> outputs, OpBuilder* builder) {
  llvm::SmallVector<Type, 4> operand_types;
  operand_types.reserve(inputs.size());
  for (Value v : inputs) operand_types.emplace_back(v.getType());
  llvm::SmallVector<Type, 4> output_types;
  output_types.reserve(outputs.size());
  for (Value v : outputs) output_types.emplace_back(v.getType());

  auto func_type = builder->getFunctionType(operand_types, output_types);

  func::FuncOp outlined_func =
      func::FuncOp::create(ops.front()->getLoc(), kHostFunctionAttr, func_type);

  // Create function body.
  Block* outlined_func_block = outlined_func.addEntryBlock();

  // Clone the operations and remap the inputs to use the function arguments.
  IRMapping mapping;
  mapping.map(inputs, outlined_func.getArguments());
  builder->setInsertionPoint(outlined_func_block, outlined_func_block->begin());
  for (Operation* op : ops) {
    builder->clone(*op, mapping);
  }

  // Set the returned values to use cloned ops results using mapping.
  llvm::SmallVector<Value, 4> results_after_mapping;
  for (Value result : outputs) {
    results_after_mapping.push_back(mapping.lookupOrDefault(result));
  }

  builder->create<func::ReturnOp>(ops.front()->getLoc(), results_after_mapping);
  return outlined_func;
}

// Encapsulates `func` in a module and serializes that module.
// `serialized_func_module` is set to the serialized module.
void EncapsulateFuncAndSerialize(func::FuncOp func,
                                 std::string* serialized_func_module) {
  // Create a new module to hold func and all referenced functions.
  OwningOpRef<mlir::ModuleOp> module_for_func =
      ModuleOp::create(mlir::UnknownLoc::get(func.getContext()));
  SymbolTable symbol_table(module_for_func.get());

  symbol_table.insert(func);
  *serialized_func_module =
      tensorflow::SerializeMlirModule(module_for_func.get());
}

// Returns whether `op` or ops nested in `op` are outside compiled.
bool HasOutsideCompilationNested(Operation* op) {
  return op
      ->walk([&](Operation* walked_op) {
        if (op == walked_op) return WalkResult::advance();
        if (walked_op->hasAttrOfType<StringAttr>(kXlaOutsideCompilationAttr)) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      })
      .wasInterrupted();
}

// Returns whether `op` or any ancestors of `op` are outside compiled.
bool HasOutsideCompilationAncestor(Operation* op) {
  while (op) {
    if (op->hasAttrOfType<StringAttr>(kXlaOutsideCompilationAttr)) {
      return true;
    }
    op = op->getParentOp();
  }
  return false;
}

// Returns whether any ancestors of `op` are outside compiled.
bool HasOutsideCompilationAncestorExclusive(Operation* op) {
  Operation* parent_op = op->getParentOp();
  if (!parent_op) return false;
  return HasOutsideCompilationAncestor(parent_op);
}

Operation* ApplyXlaHostTransferAttr(Operation* op, OpBuilder& builder) {
  op->setAttr("_xla_has_host_transfer", builder.getBoolAttr(true));
  return op;
}

// Creates a tf._XlaSendFromHost or tf._XlaSendFromHostV2 op. If device ordinal
// is present, a tf._XlaSendFromHostV2 op is created instead.
Operation* CreateSendFromHostOp(OpBuilder& builder, Location loc,
                                ValueRange inputs, Value compilation_key,
                                Value device_ordinal,
                                int default_device_ordinal,
                                StringAttr device_type_attr,
                                llvm::StringRef communication_key) {
  if (device_ordinal)
    return ApplyXlaHostTransferAttr(
        builder.create<TF::_XlaSendFromHostV2Op>(
            loc, inputs,
            /*dynamic_key=*/compilation_key, device_ordinal,
            builder.getStringAttr(communication_key), device_type_attr),
        builder);

  return ApplyXlaHostTransferAttr(
      builder.create<TF::_XlaSendFromHostOp>(
          loc, inputs,
          /*dynamic_key=*/compilation_key,
          builder.getStringAttr(communication_key),
          /*device_ordinal=*/builder.getI64IntegerAttr(default_device_ordinal),
          device_type_attr),
      builder);
}

// Creates a tf._XlaRecvAtHost or tf._XlaRecvAtHostV2 op. If device ordinal is
// present, a tf._XlaRecvAtHostV2 op is created instead.
Operation* CreateRecvAtHostOp(OpBuilder& builder, Location loc,
                              TypeRange output_types, Value compilation_key,
                              Value device_ordinal, int default_device_ordinal,
                              StringAttr device_type_attr,
                              llvm::StringRef communication_key) {
  if (device_ordinal)
    return ApplyXlaHostTransferAttr(
        builder.create<TF::_XlaRecvAtHostV2Op>(
            loc, output_types, /*dynamic_key=*/compilation_key, device_ordinal,
            builder.getStringAttr(communication_key), device_type_attr),
        builder);

  return ApplyXlaHostTransferAttr(
      builder.create<TF::_XlaRecvAtHostOp>(
          loc, output_types, /*dynamic_key=*/compilation_key,
          builder.getStringAttr(communication_key),
          /*device_ordinal=*/builder.getI64IntegerAttr(default_device_ordinal),
          device_type_attr),
      builder);
}

// Clones an IfRegionOp 'if_region' and attributes and creates then/else regions
// with yield op and an empty block.
TF::IfRegionOp CloneEmptyIfWithPredicate(TF::IfRegionOp if_region,
                                         OpBuilder& builder) {
  // Mark op as stateful due to side-effecting communication ops added later.
  auto host_side_if = builder.create<TF::IfRegionOp>(
      if_region.getLoc(), llvm::SmallVector<Type, 4>{}, if_region.getCond(),
      /*is_stateless=*/false, if_region.get_thenFuncNameAttr(),
      if_region.get_elseFuncNameAttr());

  // Create empty then branch region.
  auto& then_branch = host_side_if.getThenBranch();
  then_branch.push_back(new Block);
  builder.setInsertionPointToEnd(&then_branch.front());
  builder.create<TF::YieldOp>(if_region.getLoc(),
                              /*operands=*/ArrayRef<Value>{});

  // Create empty else branch region.
  auto& else_branch = host_side_if.getElseBranch();
  else_branch.push_back(new Block);
  builder.setInsertionPointToEnd(&else_branch.front());
  builder.create<TF::YieldOp>(if_region.getLoc(),
                              /*operands=*/ArrayRef<Value>{});
  return host_side_if;
}
// Creates a WhileRegionOp cond and body regions with yield op and
// an empty body.
TF::WhileRegionOp CloneEmptyWhile(uint64_t parallel_iterations, Location loc,
                                  OpBuilder& builder) {
  // Mark op as stateful due to side-effecting communication ops added later.
  auto host_side_while = builder.create<TF::WhileRegionOp>(
      loc, /*output=*/ArrayRef<Type>{}, /*input=*/ArrayRef<Value>{},
      parallel_iterations, /*is_stateless=*/false, /*shape_invariant=*/false);

  // Create empty else branch region.
  auto& body = host_side_while.getBody();
  body.push_back(new Block);
  builder.setInsertionPointToEnd(&body.front());
  builder.create<TF::YieldOp>(loc, /*operands=*/ArrayRef<Value>{});
  return host_side_while;
}

// TODO(b/157054714): Use a better abstraction instead of
// _TPUCompileMlirOp and _XlaRecvAtHostOp and _XlaSendFromHostOp.
// Creates a compilation key as placeholder. A placeholder compilation cache key
// is created because it is a required input to _XlaRecvAtHost and
// _XlaSendFromHost but the _TPUCompileMlir has not yet been created for device
// cluster that contains the outside compiled ops. This placeholder should be
// replaced by the TPU cluster _TPUCompileMlir in a subsequent pass.
TF::_TPUCompileMlirPlaceholderProgramKeyOp CreateCompilationKeyPlaceholder(
    Location loc, OpBuilder& builder) {
  auto result_type =
      RankedTensorType::get({3}, builder.getType<TF::StringType>());
  return builder.create<TF::_TPUCompileMlirPlaceholderProgramKeyOp>(
      loc, /*program=*/result_type, llvm::ArrayRef<Value>{});
}

TF::ConstOp CreateCpuGpuComilationKeyPlaceholder(Location loc,
                                                 OpBuilder& builder) {
  auto shape_type =
      RankedTensorType::get({3}, builder.getType<TF::StringType>());

  return builder.create<TF::ConstOp>(
      loc, DenseStringElementsAttr::get(shape_type,
                                        llvm::ArrayRef<StringRef>{"", "", ""}));
}

// Creates a `tf_device.launch` to wrap cluster ops.
tf_device::LaunchOp CreateLaunchOpForOutsideCluster(
    OpBuilder& builder, Operation* loc_op, llvm::StringRef host_device,
    llvm::SmallVector<Value, 4>& return_value_from_host) {
  llvm::SmallVector<Type, 4> host_result_types;
  host_result_types.reserve(return_value_from_host.size());
  for (Value old_result : return_value_from_host)
    host_result_types.push_back(old_result.getType());

  // An empty string placeholder is used for the device as that will be later
  // populated with the device of the associated Device op.
  // For TPU case, it is TPUReplicateMetadata op.
  auto launch_op = builder.create<tf_device::LaunchOp>(
      loc_op->getLoc(), builder.getStringAttr(host_device),
      /*result_types=*/host_result_types);

  launch_op.getBody().push_back(new Block);
  builder.setInsertionPointToEnd(&launch_op.GetBody());
  builder.create<tf_device::ReturnOp>(loc_op->getLoc(), return_value_from_host);
  return launch_op;
}

// Returns true if `op` has non-static shaped outputs.
bool HasDynamicOutputs(Operation* op) {
  for (Value v : op->getResults()) {
    if (TF::CanBeRefined(v.getType())) return true;
  }
  return false;
}

// Returns true if any op in `cluster_ops` has outputs consumed by ops not
// `cluster_ops` with a non-static shape.
bool HasDynamicOutputs(const llvm::SmallSetVector<Operation*, 4>& cluster_ops) {
  for (Operation* op : cluster_ops) {
    for (const OpOperand& use : op->getUses()) {
      if (cluster_ops.count(use.getOwner())) {
        continue;
      }
      if (TF::CanBeRefined(use.get().getType())) return true;
    }
  }
  return false;
}

bool HasDynamicExternalValues(Operation* op) {
  return op
      ->walk([](Operation* walked_op) {
        for (Value v : walked_op->getOperands()) {
          if (TF::CanBeRefined(v.getType())) {
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      })
      .wasInterrupted();
}

// Returns operands of `cluster_ops` that need to be
// communicated from device->host. This is for the case when all operands have a
// static shape.
llvm::SmallSetVector<Value, 4> GetStaticExternalOperands(
    tf_device::ClusterOp device_cluster,
    const llvm::SmallSetVector<Operation*, 4>& cluster_ops) {
  llvm::SmallSetVector<Value, 4> external_values;
  for (Operation* op : cluster_ops) {
    op->walk([&](Operation* walked_op) {
      if (llvm::isa<TF::_XlaRecvAtHostOp, TF::_XlaRecvAtHostV2Op,
                    TF::_XlaSendFromHostOp, TF::_XlaSendFromHostV2Op>(
              walked_op))
        return WalkResult::advance();
      for (Value v : walked_op->getOperands()) {
        if (!tensorflow::TypeValidForXLA(v.getType())) continue;
        if (auto* defining_op = v.getDefiningOp()) {
          if (!op->isAncestor(defining_op) &&
              device_cluster->isAncestor(defining_op) &&
              !HasOutsideCompilationAncestor(defining_op) &&
              // Ignore operands that have already been received by a previously
              // created cluster.
              !llvm::isa<TF::_XlaRecvAtHostOp, TF::_XlaRecvAtHostV2Op>(
                  defining_op)) {
            external_values.insert(v);
          }
          continue;
        }
        auto block_arg = v.cast<BlockArgument>();
        if (block_arg.getParentRegion() == op->getParentRegion())
          external_values.insert(v);
      }
      return WalkResult::advance();
    });
  }
  return external_values;
}

// Returns every operand of `cluster_ops` that does not come from an op in
// `cluster_ops`.
llvm::SmallSetVector<Value, 4> GetAllExternalOperands(
    const llvm::SmallSetVector<Operation*, 4>& cluster_ops) {
  llvm::SmallSetVector<Value, 4> external_values;
  for (Operation* op : cluster_ops) {
    op->walk([&](Operation* walked_op) {
      for (Value v : walked_op->getOperands()) {
        if (!tensorflow::TypeValidForXLA(v.getType())) continue;
        Operation* defining_op = v.getDefiningOp();
        if (!defining_op || !cluster_ops.count(defining_op)) {
          external_values.insert(v);
        }
      }
    });
  }
  return external_values;
}

// Returns a SmallSetVector containing all of the operands that need to be
// communicated from device->host.
llvm::SmallSetVector<Value, 4> GetExternalOperands(
    tf_device::ClusterOp device_cluster,
    const llvm::SmallSetVector<Operation*, 4>& cluster_ops) {
  // If there are any dynamic outputs, get all of the operands which are defined
  // external to `cluster_ops`.
  bool has_dynamic_outputs = HasDynamicOutputs(cluster_ops);
  if (has_dynamic_outputs) {
    return GetAllExternalOperands(cluster_ops);
  } else {
    return GetStaticExternalOperands(device_cluster, cluster_ops);
  }
}

// Gets all outputs that need to be communicated from host->device.
void GetExternalOutputs(const llvm::SmallSetVector<Operation*, 4>& cluster_ops,
                        llvm::SmallSetVector<Value, 4>& external_outputs,
                        llvm::SmallVector<Value, 4>& host_outputs) {
  bool has_dynamic_outputs = HasDynamicOutputs(cluster_ops);
  llvm::SmallVector<Value, 4> tmp_host_outputs;
  for (Operation* op : cluster_ops) {
    llvm::SmallDenseSet<Operation*, 4> user_set;
    for (Operation* user : op->getUsers()) {
      // We skip any operations that are in the same outside compilation
      // cluster that will be moved to the host at the same time since both
      // defining op and user op will be moved to host.
      if (cluster_ops.count(user)) {
        continue;
      }
      // This is pessimistic and in some cases will add extra communication.
      if (!HasOutsideCompilationAncestor(user) || has_dynamic_outputs ||
          HasDynamicOutputs(user)) {
        if (!user_set.insert(user).second) continue;
        for (Value v : user->getOperands()) {
          if (tensorflow::TypeValidForXLA(v.getType()) &&
              v.getDefiningOp() == op && !isa<tf_device::ReturnOp>(user))
            external_outputs.insert(v);
          if (v.getDefiningOp() == op && isa<tf_device::ReturnOp>(user))
            tmp_host_outputs.push_back(v);
        }
      }
    }
  }

  // Value in `tmp_host_outputs` may contain user in non return op, which has
  // been in `external_outputs`. We need exclude those Value.
  for (auto val : tmp_host_outputs) {
    if (!external_outputs.contains(val)) host_outputs.push_back(val);
  }
}

// Creates the HostCompute with `inputs` and `outputs`
// using `communication_key`.
TF::_XlaHostComputeMlirOp CreateHostCompute(
    OpBuilder& builder, Location loc,
    const llvm::SmallSetVector<Value, 4>& inputs, llvm::ArrayRef<Value> outputs,
    llvm::StringRef args_communication_key,
    llvm::StringRef retvals_communication_key,
    llvm::StringRef serialized_func_module) {
  llvm::SmallVector<Type, 4> device_output_types;
  for (const auto& output : outputs)
    device_output_types.push_back(output.getType());
  auto host_compute = builder.create<TF::_XlaHostComputeMlirOp>(
      loc, device_output_types, inputs.getArrayRef(),
      builder.getStringAttr(args_communication_key),
      builder.getStringAttr(retvals_communication_key),
      /*host_mlir_module=*/builder.getStringAttr(serialized_func_module));
  return host_compute;
}

void MarkOutsideCompiled(Operation* op) {
  op->setAttr(kXlaOutsideCompilationAttr,
              StringAttr::get(op->getContext(), "temp"));
}

// Returns whether an outside compilation cluster should be closed.  True when:
// 1. There is no non-XLA output.
// 2. There is a dynamically shaped output consumed by a non-outside compiled
// op.
// 3. There is no dynamically shaped output.
bool ShouldCloseCluster(llvm::ArrayRef<Value> outputs) {
  bool has_dynamic_output = false;
  bool has_nonxla_output = false;
  for (Value v : outputs) {
    if (TF::CanBeRefined(v.getType())) {
      has_dynamic_output = true;
      for (Operation* user : v.getUsers()) {
        if (!HasOutsideCompilationAncestor(user) &&
            !isa<tf_device::ReturnOp>(user))
          return true;
      }
    }
    if (!tensorflow::TypeValidForXLA(v.getType()))
      for (const Operation* user : v.getUsers())
        if (!isa<tf_device::ReturnOp>(user)) has_nonxla_output = true;
  }

  return !has_nonxla_output && !has_dynamic_output;
}

// Replaces `external_operands` with the results from `recv_at_host`.
// For non-static shapes, only replace operand usage if op is in the same
// region as insertion.
// For static-shapes, Replace operand usages if op is in the same region as
// insertion or if the op is outside compiled and will be moved to host later.
void ReplaceExternalOperandUsage(
    const llvm::SmallSetVector<Value, 4>& external_operands,
    Operation* recv_at_host, Operation* insertion_point,
    Block* original_op_block) {
  auto replace_operand_usage = [&](OpOperand& operand) {
    if (TF::CanBeRefined(operand.get().getType()) ||
        HasDynamicOutputs(operand.getOwner())) {
      return insertion_point->getParentRegion()->isAncestor(
          operand.getOwner()->getParentRegion());
    }
    return insertion_point->getParentRegion()->isAncestor(
               operand.getOwner()->getParentRegion()) ||
           (HasOutsideCompilationAncestor(operand.getOwner()) &&
            original_op_block == operand.getOwner()->getBlock());
  };
  for (auto result : llvm::zip(external_operands, recv_at_host->getResults())) {
    Value external_operand = std::get<0>(result);
    external_operand.replaceUsesWithIf(std::get<1>(result),
                                       replace_operand_usage);
  }
}

bool HasDynamicOutputs(llvm::ArrayRef<Value> outputs) {
  for (Value v : outputs) {
    if (TF::CanBeRefined(v.getType())) {
      return true;
    }
  }
  return false;
}

// Replaces usages of `external_outputs` which are values returned by outside
// compilation with the corresponding outputs from `host_compute`.
void ReplaceExternalOutputUsage(
    const llvm::SmallSetVector<Value, 4>& external_outputs,
    TF::_XlaHostComputeMlirOp host_compute) {
  bool has_dynamic_outputs = HasDynamicOutputs(external_outputs.getArrayRef());

  auto replace_output_usage = [&](OpOperand& operand) {
    // Don't replace output usages if in host computation (defining op and user
    // in same region).
    bool in_same_region =
        operand.get().getDefiningOp()->getParentRegion()->isAncestor(
            operand.getOwner()->getParentRegion());
    if (has_dynamic_outputs || HasDynamicOutputs(operand.getOwner())) {
      return !in_same_region;
    } else {
      // Don't replace output usages in host computation or for outside
      // compiled ops.
      return !in_same_region &&
             !HasOutsideCompilationAncestor(operand.getOwner());
    }
  };
  for (auto result : llvm::zip(external_outputs, host_compute.getResults())) {
    Value external_output = std::get<0>(result);
    external_output.replaceUsesWithIf(std::get<1>(result),
                                      replace_output_usage);
  }
}

// Move `clustered_ops` to run on host and adds communication ops to transfer
// `external_operands` and `external_outputs` to/from device/host.  Inserts
// ops at `insertion_point` and uses `compilation_key` and `device_ordinal` when
// creating comm ops.
void MoveOpsToHost(const llvm::SmallSetVector<Operation*, 4>& clustered_ops,
                   const llvm::SmallSetVector<Value, 4>& external_operands,
                   const llvm::SmallSetVector<Value, 4>& external_outputs,
                   Operation* insertion_point, Value compilation_key,
                   Value device_ordinal, int default_device_ordinal,
                   StringAttr device_type_attr, int& communication_key_index) {
  OpBuilder builder(insertion_point);
  Operation& op = *clustered_ops.back();
  std::string args_communication_key =
      llvm::formatv("host_compute_channel_{0}_args", (communication_key_index))
          .str();
  std::string retvals_communication_key =
      llvm::formatv("host_compute_channel_{0}_retvals",
                    (communication_key_index))
          .str();

  // Use a unique name when sending just the IfRegion predicate.  This is
  // for readable and to match the key in the TF2XLA bridge.
  if (clustered_ops.size() == 1 && llvm::isa<TF::IfRegionOp>(op) &&
      external_operands.size() == 1) {
    args_communication_key =
        llvm::formatv("if_predicate_channel_{0}", (communication_key_index))
            .str();
  }

  std::string serialized_func_module;
  if (HasDynamicOutputs(external_outputs.getArrayRef())) {
    func::FuncOp shape_op = BuildFunction(
        clustered_ops.getArrayRef(), external_operands.getArrayRef(),
        external_outputs.getArrayRef(), &builder);
    EncapsulateFuncAndSerialize(shape_op, &serialized_func_module);
  }

  builder.setInsertionPoint(&op);
  auto host_compute =
      CreateHostCompute(builder, op.getLoc(), external_operands,
                        external_outputs.getArrayRef(), args_communication_key,
                        retvals_communication_key, serialized_func_module);
  // Insert ops on the host side computation to receive data from device.
  builder.setInsertionPoint(insertion_point);
  llvm::SmallVector<Type, 4> host_operand_types;
  for (const auto& operand : external_operands)
    host_operand_types.push_back(operand.getType());

  Operation* recv_at_host = CreateRecvAtHostOp(
      builder, op.getLoc(), host_operand_types, compilation_key, device_ordinal,
      default_device_ordinal, device_type_attr, args_communication_key);
  Block* original_op_block = op.getBlock();
  Operation* after_op = recv_at_host;
  for (Operation* cluster_op : clustered_ops) {
    cluster_op->moveAfter(after_op);
    cluster_op->removeAttr(StringAttr::get(op.getContext(), kDeviceAttr));
    after_op = cluster_op;
  }

  if (!external_outputs.empty()) {
    CreateSendFromHostOp(builder, op.getLoc(), external_outputs.getArrayRef(),
                         compilation_key, device_ordinal,
                         default_device_ordinal, device_type_attr,
                         retvals_communication_key);
  }

  if (external_operands.empty()) {
    recv_at_host->erase();
  } else {
    ReplaceExternalOperandUsage(external_operands,
                                /*recv_at_host=*/recv_at_host,
                                /*insertion_point=*/insertion_point,
                                /*original_op_block=*/original_op_block);
  }

  ReplaceExternalOutputUsage(external_outputs, host_compute);

  if (external_operands.empty() && external_outputs.empty()) {
    host_compute.erase();
  } else {
    ++communication_key_index;
  }
}

// Move outside compiled ops in `src` to `insertion_point` in host
// computation (may be temporarily with `device_cluster` but moved in subsequent
// call to this method).  Communication ops are added in both `src` and at
// `insertion_point` using `compilation_key`, `device_ordinal` and
// `communication_key_index` which is incremented when used. Communication ops
// are added only when needed and at the location need.  There are checks to
// ensure that duplicate communication between device and host is not added.
// When `return_value_from_host` is not nullptr, MoveOpsToHost will also update
// its value.
LogicalResult MoveOpsToHost(
    tf_device::ClusterOp device_cluster, Block* src, Operation* insertion_point,
    Value compilation_key, Value device_ordinal, int default_device_ordinal,
    int& communication_key_index,
    llvm::SmallVector<Value, 4>* return_value_from_host = nullptr) {
  // Contains all of the outside compiled operations that should be moved to the
  // host using a single `_XlaHostComputeMlir` op.  This should only contain a
  // single op except in the case where some of the input/output shapes are
  // non-static.
  llvm::SmallSetVector<Operation*, 4> clustered_ops;
  auto device_type_attr =
      device_cluster->getAttrOfType<StringAttr>(TF::kCompileDeviceTypeAttr);

  for (Operation& op : llvm::make_early_inc_range(*src)) {
    if (HasOutsideCompilationAncestorExclusive(&op) ||
        !op.hasAttrOfType<StringAttr>(kXlaOutsideCompilationAttr))
      continue;

    llvm::SmallSetVector<Value, 4> external_outputs;
    llvm::SmallVector<Value, 4> host_outputs;
    // We want to move the clustered_ops if the op to be added has all
    // statically shaped operands since we can't ensure that the static shapes
    // has been sent back to host in all cases.  See
    // @static_shapes_sandwiched_outside_compilation MLIR test for an example.
    if (!HasDynamicExternalValues(&op) && !clustered_ops.empty()) {
      llvm::SmallSetVector<Value, 4> external_operands =
          GetExternalOperands(device_cluster, clustered_ops);
      GetExternalOutputs(clustered_ops, external_outputs, host_outputs);
      if (return_value_from_host) {
        for (auto& output : host_outputs) {
          return_value_from_host->push_back(output);
        }
      }
      MoveOpsToHost(clustered_ops, external_operands, external_outputs,
                    insertion_point, compilation_key, device_ordinal,
                    default_device_ordinal, device_type_attr,
                    communication_key_index);
      clustered_ops.clear();
    }

    clustered_ops.insert(&op);

    // Get the outputs that need to be communicated from host -> device.
    external_outputs.clear();
    host_outputs.clear();
    GetExternalOutputs(clustered_ops, external_outputs, host_outputs);

    if (ShouldCloseCluster(external_outputs.getArrayRef())) {
      // Get the operands that need to be communicated from device -> host.
      llvm::SmallSetVector<Value, 4> external_operands =
          GetExternalOperands(device_cluster, clustered_ops);
      if (return_value_from_host) {
        for (auto& output : host_outputs) {
          return_value_from_host->push_back(output);
        }
      }

      MoveOpsToHost(clustered_ops, external_operands, external_outputs,
                    insertion_point, compilation_key, device_ordinal,
                    default_device_ordinal, device_type_attr,
                    communication_key_index);
      clustered_ops.clear();
    }
  }
  return success();
}

void GetReturnValueFromDevice(
    tf_device::ClusterOp device_cluster,
    const llvm::SmallVector<Value, 4>& return_value_from_host,
    llvm::SmallVector<Value, 4>& return_value_from_device) {
  if (auto return_op = llvm::dyn_cast_or_null<tf_device::ReturnOp>(
          device_cluster.GetBody().getTerminator())) {
    for (auto v : return_op.getOperands()) {
      if (absl::c_count(return_value_from_host, v) == 0) {
        return_value_from_device.push_back(v);
      }
    }
  }
}

// Decompose control flow in `device_cluster` into device computation and host
// (outside compiled) computation into two separate control flow ops with
// communication between the device/host for data dependencies.  Both device and
// host control flow initially remain within `device_cluster` and a subsequency
// call to MoveOpsToHost moves the host side control flow to the host launch in
// tf_device.parallel_execute.  Uses `compilation_key, `device_ordinal` and
// `communication_key_index` when creating communication ops.
LogicalResult DecomposeControlFlow(tf_device::ClusterOp device_cluster,
                                   Value compilation_key, Value device_ordinal,
                                   int default_device_ordinal,
                                   int& communication_key_index) {
  auto result = device_cluster.GetBody().walk([&](Operation* op) {
    if (auto if_op = llvm::dyn_cast<TF::IfRegionOp>(op)) {
      if (!HasOutsideCompilationNested(op)) return WalkResult::advance();
      OpBuilder builder(if_op);
      auto host_if = CloneEmptyIfWithPredicate(if_op, builder);
      if (failed(MoveOpsToHost(
              device_cluster, &if_op.getThenBranch().front(),
              host_if.getThenBranch().front().getTerminator(), compilation_key,
              device_ordinal, default_device_ordinal, communication_key_index)))
        return WalkResult::interrupt();
      if (failed(MoveOpsToHost(
              device_cluster, &if_op.getElseBranch().front(),
              host_if.getElseBranch().front().getTerminator(), compilation_key,
              device_ordinal, default_device_ordinal, communication_key_index)))
        return WalkResult::interrupt();
      // Mark op as stateful due to side-effecting communication ops.
      if_op->setAttr("is_stateless", builder.getBoolAttr(false));
      MarkOutsideCompiled(host_if.getOperation());
    }
    if (auto while_op = llvm::dyn_cast<TF::WhileRegionOp>(op)) {
      if (!HasOutsideCompilationNested(op)) return WalkResult::advance();
      OpBuilder builder(while_op);
      auto host_while = CloneEmptyWhile(while_op.getParallelIterations(),
                                        while_op.getLoc(), builder);
      const auto condition_send_recv_key =
          llvm::formatv("while_condition_channel_{0}",
                        communication_key_index++)
              .str();
      auto& cond = host_while.getCond();
      cond.push_back(new Block);
      auto condition =
          while_op.getCond().front().getTerminator()->getOperand(0);
      builder.setInsertionPoint(while_op.getCond().front().getTerminator());
      builder.create<TF::XlaSendToHostOp>(while_op.getLoc(), condition,
                                          condition_send_recv_key);
      builder.setInsertionPointToEnd(&cond.front());
      auto recv_condition_at_host = CreateRecvAtHostOp(
          builder, while_op.getLoc(), TypeRange{condition.getType()},
          compilation_key, device_ordinal, default_device_ordinal,
          device_cluster->getAttrOfType<StringAttr>(TF::kCompileDeviceTypeAttr),
          condition_send_recv_key);
      builder.create<TF::YieldOp>(while_op.getLoc(),
                                  recv_condition_at_host->getResults());

      if (failed(MoveOpsToHost(device_cluster, &while_op.getCond().front(),
                               recv_condition_at_host, compilation_key,
                               device_ordinal, default_device_ordinal,
                               communication_key_index)))
        return WalkResult::interrupt();
      if (failed(MoveOpsToHost(
              device_cluster, &while_op.getBody().front(),
              host_while.getBody().front().getTerminator(), compilation_key,
              device_ordinal, default_device_ordinal, communication_key_index)))
        return WalkResult::interrupt();
      // Mark op as stateful due to side-effecting communication ops.
      while_op->setAttr("is_stateless", builder.getBoolAttr(false));
      MarkOutsideCompiled(host_while.getOperation());
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) return failure();
  return success();
}

// Removes outside compilation from all ops inside `host_launch_op`.  Should
// only be run after all outside compiled ops have been moved to
// `host_launch_op`.
void RemoveOutsideCompilation(tf_device::LaunchOp host_launch_op) {
  host_launch_op.GetBody().walk([&](Operation* op) {
    if (op->hasAttrOfType<StringAttr>(kXlaOutsideCompilationAttr)) {
      op->removeAttr(
          StringAttr::get(op->getContext(), kXlaOutsideCompilationAttr));
    }
  });
}

// This method extracts default ordinal or default device core associated with a
// host.
// If the cluster has replication attribute and it is not empty, then it means
// it is replicated case and then NO ordinal info is extracted but
// if it is non replicated cluster and there is a device attr with some
// non-empty device, then that device's ordinal (0 out of TPU:0 and
// 1 out of TPU:1) is extracted and the default ordinal is set to this value.
LogicalResult GetDefaultDeviceOrdinal(tf_device::ClusterOp device_cluster,
                                      int& default_ordinal) {
  bool has_replication = device_cluster->hasAttr(TF::kReplicationInfoAttr);

  std::string replication_info;
  if (has_replication) {
    replication_info =
        device_cluster->getAttrOfType<StringAttr>(TF::kReplicationInfoAttr)
            .str();
  }
  if (replication_info == kNoReplicationCluster || replication_info.empty()) {
    has_replication = false;
  }
  if (!has_replication &&
      device_cluster->hasAttrOfType<StringAttr>(kDeviceAttr) &&
      !device_cluster->getAttrOfType<StringAttr>(kDeviceAttr).str().empty()) {
    int64_t ordinal = 0;
    mlir::LogicalResult result = tensorflow::GetDeviceOrdinalFromDeviceString(
        mlir::UnknownLoc::get(device_cluster.getContext()),
        device_cluster->getAttrOfType<StringAttr>(kDeviceAttr).str(), &ordinal);
    if (succeeded(result)) {
      default_ordinal = ordinal;
    } else {
      return device_cluster.emitError()
             << " could not find ordinal for the given device";
    }
  }
  return success();
}

// The results of parallel executes is the combination of return values from
// both host and device.
llvm::SmallVector<Type, 4> GetParallelExecuteResultsTypes(
    const llvm::SmallVector<Value, 4>& return_value_from_host,
    const llvm::SmallVector<Value, 4>& return_value_from_device) {
  llvm::SmallVector<Type, 4> parallel_execute_result_types;
  const int num_of_outputs =
      return_value_from_host.size() + return_value_from_device.size();
  parallel_execute_result_types.reserve(num_of_outputs);
  for (Value result : return_value_from_host)
    parallel_execute_result_types.push_back(result.getType());
  for (Value result : return_value_from_device)
    parallel_execute_result_types.push_back(result.getType());
  return parallel_execute_result_types;
}

// Remap the device cluster results with parallel execute op results
llvm::SmallVector<Value, 4> GetRemappedTpuClusterResults(
    tf_device::ClusterOp device_cluster,
    const llvm::SmallVector<Value, 4>& return_value_from_host,
    const llvm::SmallVector<Value, 4>& return_value_from_device) {
  llvm::SmallVector<Value, 4> remapped_device_cluster_results;

  std::vector<int> order_from_new_output_to_previous_output;
  order_from_new_output_to_previous_output.reserve(
      return_value_from_host.size() + return_value_from_device.size());

  llvm::SmallDenseMap<Value, std::deque<int>> return_operand_map;
  auto return_op = llvm::dyn_cast<tf_device::ReturnOp>(
      device_cluster.GetBody().getTerminator());

  for (OpOperand& operand : return_op->getOpOperands()) {
    auto operand_idx = operand.getOperandNumber();
    return_operand_map[return_op.getOperand(operand_idx)].push_back(
        operand_idx);
  }
  // `return_value_from_host` and `return_value_from_device` together contain
  // all elements in operands in `return_op`, including duplicated ones. They
  // are one to one mapping. Therefore, when we iterate `return_value_from_host`
  // and `return_value_from_device`, it is safe to access return_operand_map and
  // call `front()` method.
  for (const auto& return_val : return_value_from_host) {
    order_from_new_output_to_previous_output.push_back(
        return_operand_map[return_val].front());
    return_operand_map[return_val].pop_front();
  }

  for (const Value& return_val : return_value_from_device) {
    order_from_new_output_to_previous_output.push_back(
        return_operand_map[return_val].front());
    return_operand_map[return_val].pop_front();
  }

  for (auto idx : order_from_new_output_to_previous_output)
    remapped_device_cluster_results.push_back(device_cluster.getResult(idx));

  return remapped_device_cluster_results;
}

// Remap cluster results with parallel_execute results if user is outside of
// parallel_execute.
void RemapDeviceClusterResultsWithParallelExecuteResults(
    tf_device::ClusterOp device_cluster,
    tf_device::ParallelExecuteOp parallel_execute_op,
    const llvm::SmallVector<Value, 4>& return_value_from_host,
    const llvm::SmallVector<Value, 4>& return_value_from_device) {
  llvm::SmallVector<Value, 4> remapped_device_cluster_results =
      GetRemappedTpuClusterResults(device_cluster, return_value_from_host,
                                   return_value_from_device);

  for (auto result : llvm::zip(remapped_device_cluster_results,
                               parallel_execute_op.getResults())) {
    Value device_cluster_result = std::get<0>(result);
    Value parallel_execute_result = std::get<1>(result);
    for (auto& use :
         llvm::make_early_inc_range(device_cluster_result.getUses()))
      if (!parallel_execute_op.getOperation()->isProperAncestor(use.getOwner()))
        use.set(parallel_execute_result);
  }
}

// Get the vector of results for new device cluster
llvm::SmallVector<Value, 4> GetNewDeviceResults(
    const llvm::SmallVector<Value, 4>& return_value_from_device) {
  llvm::SmallVector<Value, 4> device_results;
  device_results.reserve(return_value_from_device.size());
  for (Value old_result : return_value_from_device)
    device_results.push_back(old_result);
  return device_results;
}

// Get the vector of types of results for new device cluster
llvm::SmallVector<Type, 4> GetNewDeviceTypes(
    const llvm::SmallVector<Value, 4>& return_value_from_device) {
  llvm::SmallVector<Type, 4> device_result_types;
  device_result_types.reserve(return_value_from_device.size());
  for (Value old_result : return_value_from_device)
    device_result_types.push_back(old_result.getType());
  return device_result_types;
}

// Move ops in old device cluster to new device cluster
void MoveOldTpuClusterToNewTpuCluster(tf_device::ClusterOp device_cluster,
                                      Operation* after_op_r) {
  for (Operation& op : llvm::make_early_inc_range(device_cluster.GetBody())) {
    if (&op != device_cluster.GetBody().getTerminator()) {
      op.moveBefore(after_op_r);
    }
  }
}

// Move ops in the tmp host launch op to new host launch op
void MoveTmpLaunchOpToNewLaunchOp(tf_device::LaunchOp tmp_host_launch_op,
                                  Operation* after_op_host_cluster) {
  for (Operation& op :
       llvm::make_early_inc_range(tmp_host_launch_op.GetBody())) {
    if (&op != tmp_host_launch_op.GetBody().getTerminator()) {
      op.moveBefore(after_op_host_cluster);
    }
  }
}

// Since we have the outputs from host and device computation after moving
// outside compiled ops, we can create the actual parallel_execute regions.
// Still, one region is for the host computation for outside compilation and
// the other one is for the original Device cluster computation.
tf_device::ParallelExecuteOp CreateFinalParallelExecuteOp(
    OpBuilder& builder, int num_regions, llvm::StringRef host_device,
    tf_device::ClusterOp device_cluster, tf_device::LaunchOp tmp_host_launch_op,
    const llvm::SmallVector<Value, 4>& return_value_from_host,
    const llvm::SmallVector<Value, 4>& return_value_from_device) {
  llvm::SmallVector<Type, 4> parallel_execute_result_types =
      GetParallelExecuteResultsTypes(return_value_from_host,
                                     return_value_from_device);

  builder.setInsertionPoint(device_cluster);
  auto parallel_execute_op = builder.create<tf_device::ParallelExecuteOp>(
      device_cluster.getLoc(), num_regions, parallel_execute_result_types);
  Block& host_computation_block =
      parallel_execute_op.GetRegionBlockWithIndex(0);
  builder.setInsertionPointToEnd(&host_computation_block);

  // Create a single launch op for all outside compiled ops.
  llvm::SmallVector<Value, 4> host_results;
  host_results.insert(host_results.end(), return_value_from_host.begin(),
                      return_value_from_host.end());
  tf_device::LaunchOp host_launch_op = CreateLaunchOpForOutsideCluster(
      builder, device_cluster, host_device, host_results);

  // Create a return op for host computation block
  builder.setInsertionPointToEnd(&host_computation_block);
  builder.create<tf_device::ReturnOp>(device_cluster.getLoc(),
                                      host_launch_op->getResults());

  // Move the launch body to last parallel_execute block.
  Block& parallel_execute_device_block =
      parallel_execute_op.GetRegionBlockWithIndex(1);
  builder.setInsertionPointToEnd(&parallel_execute_device_block);

  // Get the vector of results and types of results for new device cluster
  llvm::SmallVector<Value, 4> device_results =
      GetNewDeviceResults(return_value_from_device);
  llvm::SmallVector<Type, 4> device_result_types =
      GetNewDeviceTypes(return_value_from_device);

  // Create a empty device cluster op with same attribute but different return
  // type
  auto new_device_cluster = builder.create<tf_device::ClusterOp>(
      device_cluster.getLoc(), device_result_types,
      /*operands=*/llvm::ArrayRef<Value>{}, device_cluster->getAttrs());

  new_device_cluster.getBody().push_back(new Block);
  builder.setInsertionPointToEnd(&new_device_cluster.GetBody());

  // Create return op for device computation region in the paralle_execute op
  Operation* after_op_r = builder.create<tf_device::ReturnOp>(
      new_device_cluster.getLoc(), device_results);

  builder.setInsertionPointToEnd(&parallel_execute_device_block);

  // Create return op for the new device cluster op
  builder.create<tf_device::ReturnOp>(device_cluster.getLoc(),
                                      new_device_cluster.getResults());

  MoveOldTpuClusterToNewTpuCluster(device_cluster, after_op_r);

  Operation* after_op_host_cluster = host_launch_op.GetBody().getTerminator();
  MoveTmpLaunchOpToNewLaunchOp(tmp_host_launch_op, after_op_host_cluster);

  return parallel_execute_op;
}

// Creates a `parallel_execute` op with a region for host computation and
// a region for `device_cluster` computation by extracting outside compiled ops
// to host computation.
LogicalResult CreateParallelExecuteForOutsideCompilation(
    ModuleOp module, tf_device::ClusterOp device_cluster,
    llvm::StringRef host_device,
    llvm::SmallVector<tf_device::ParallelExecuteOp, 4>& ops,
    bool has_tpu_device) {
  OpBuilder builder(device_cluster);
  llvm::SmallVector<Value, 4> returns_from_host;

  // Create a temporary parallel_execute. This is temporary because the result
  // type is not determined until after it is filled. There are two regions in
  // `tmp_parallel_execute_op`. The first one is for the host computation for
  // outside compilation and the second one is for the original Device cluster
  // computation.
  const int num_regions = 2;
  auto tmp_parallel_execute_op = builder.create<tf_device::ParallelExecuteOp>(
      device_cluster.getLoc(), num_regions, llvm::ArrayRef<Type>{});
  Block& tmp_host_computation_block =
      tmp_parallel_execute_op.GetRegionBlockWithIndex(0);
  builder.setInsertionPointToEnd(&tmp_host_computation_block);

  // Create a single tmp launch op for all outside compiled ops.
  llvm::SmallVector<Value, 4> tmp_host_results;
  tf_device::LaunchOp tmp_host_launch_op = CreateLaunchOpForOutsideCluster(
      builder, device_cluster, host_device, tmp_host_results);

  // Create a tmp return op for tmp host computation block
  builder.setInsertionPointToEnd(&tmp_host_computation_block);
  builder.create<tf_device::ReturnOp>(device_cluster.getLoc(),
                                      llvm::ArrayRef<Value>{});

  builder.setInsertionPoint(tmp_host_launch_op.GetBody().getTerminator());

  Operation* compilation_key_op = nullptr;
  Value compilation_key = nullptr;
  Operation* device_ordinal_op = nullptr;

  if (has_tpu_device) {
    compilation_key_op =
        CreateCompilationKeyPlaceholder(device_cluster.getLoc(), builder);
    compilation_key =
        llvm::dyn_cast<TF::_TPUCompileMlirPlaceholderProgramKeyOp>(
            compilation_key_op)
            .getProgram();
    device_ordinal_op = builder.create<TF::_TPUDeviceOrdinalPlaceholderOp>(
        device_cluster.getLoc(),
        RankedTensorType::get({}, builder.getI64Type()));
  } else {
    compilation_key_op =
        CreateCpuGpuComilationKeyPlaceholder(device_cluster.getLoc(), builder);
    compilation_key =
        llvm::dyn_cast<Value>(compilation_key_op->getResults()[0]);
    device_ordinal_op = builder.create<TF::ConstOp>(
        device_cluster.getLoc(),
        DenseIntElementsAttr::get(
            RankedTensorType::get({}, builder.getI64Type()),
            static_cast<int64_t>(0)));
  }

  Value device_ordinal = nullptr;

  if (device_cluster->getParentOfType<tf_device::ReplicateOp>()) {
    device_ordinal = device_ordinal_op->getResults()[0];
  }
  int default_device_ordinal = 0;
  if (failed(GetDefaultDeviceOrdinal(device_cluster, default_device_ordinal))) {
    return failure();
  }
  int communication_key_index = 0;
  // Decompose control flow into device and host control flow when outside
  // compilation is included.
  if (failed(DecomposeControlFlow(device_cluster, compilation_key,
                                  device_ordinal, default_device_ordinal,
                                  communication_key_index)))
    return failure();

  // Move all outside compiled ops including control flow to tmp host launch.
  // Also set the values returned from the host when ops are moved.
  if (failed(MoveOpsToHost(device_cluster, &device_cluster.GetBody(),
                           tmp_host_launch_op.GetBody().getTerminator(),
                           compilation_key, device_ordinal,
                           default_device_ordinal, communication_key_index,
                           &returns_from_host)))
    return failure();

  llvm::SmallVector<Value, 4> returns_from_device;
  GetReturnValueFromDevice(device_cluster, returns_from_host,
                           returns_from_device);

  if (communication_key_index == 0) compilation_key_op->erase();
  if (communication_key_index == 0 || device_ordinal == nullptr)
    device_ordinal_op->erase();

  RemoveOutsideCompilation(tmp_host_launch_op);

  tf_device::ParallelExecuteOp parallel_execute_op =
      CreateFinalParallelExecuteOp(builder, num_regions, host_device,
                                   device_cluster, tmp_host_launch_op,
                                   returns_from_host, returns_from_device);

  ops.push_back(tmp_parallel_execute_op);
  // Remap cluster results with parallel_execute results if user is outside of
  // parallel_execute.
  RemapDeviceClusterResultsWithParallelExecuteResults(
      device_cluster, parallel_execute_op, returns_from_host,
      returns_from_device);

  device_cluster.erase();

  return success();
}

// Check that cluster results are valid. An result is invalid when it does not
// have a valid XLA type.
LogicalResult CheckClusterResults(tf_device::ClusterOp cluster) {
  for (OpResult result : cluster.getResults()) {
    if (!tensorflow::TypeValidForXLA(result.getType())) {
      cluster.emitError()
          << "The ExtractHeadTailOutsideCompilation pass produced a Device "
             "cluster with a result with a non-XLA type: "
          << result.getType();
      return failure();
    }
  }
  return success();
}

// Check that op marked for outside compilation has an ancestor also marked for
// outside compilation.
LogicalResult CheckAncestorNotOutsideComp(Operation* op) {
  if (!op->getAttrOfType<StringAttr>(kXlaOutsideCompilationAttr))
    return success();
  Operation* iter_op = op;
  while (auto* parent_op = iter_op->getParentOp()) {
    if (parent_op->getAttrOfType<StringAttr>(kXlaOutsideCompilationAttr)) {
      op->emitOpError()
          << "An op marked for outside compilation (having attribute "
          << kXlaOutsideCompilationAttr
          << ") has an ancestor marked for outside compilation.";
      return failure();
    }
    iter_op = parent_op;
  }
  return success();
}

// Check the validity of the module, pre-pass.
LogicalResult CheckPreconditions(ModuleOp module) {
  auto walk_result = module.walk([&](Operation* op) {
    if (failed(CheckAncestorNotOutsideComp(op))) return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (walk_result.wasInterrupted()) return failure();
  return success();
}

// Check the validity of the module, post-pass.
LogicalResult CheckPostconditions(ModuleOp module) {
  auto walk_result = module.walk([&](tf_device::ClusterOp cluster) {
    if (failed(CheckClusterResults(cluster))) return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (walk_result.wasInterrupted()) return failure();
  return success();
}

void ExtractOutsideCompilation::runOnOperation() {
  // Get runtime devices information from the closest parent module.
  auto module = getOperation();
  if (failed(CheckPreconditions(module))) signalPassFailure();

  mlir::TF::RuntimeDevices devices;
  if (failed(tensorflow::GetDevicesFromOp(module, &devices)))
    return signalPassFailure();

  llvm::SmallVector<tf_device::ParallelExecuteOp, 4> tmp_parallel_execute_ops;

  module.walk([&](tf_device::ClusterOp device_cluster) {
    if (HasOutsideCompilationNested(device_cluster.getOperation())) {
      std::string host_device;
      if (failed(tensorflow::GetHostDeviceOutsideComputation(
              devices, device_cluster, &host_device)))
        return signalPassFailure();
      if (failed(CreateParallelExecuteForOutsideCompilation(
              module, device_cluster, host_device, tmp_parallel_execute_ops,
              tensorflow::HasTPUDevice(devices))))
        return signalPassFailure();
    }
  });

  for (auto parallel_execute_op : tmp_parallel_execute_ops) {
    parallel_execute_op.erase();
  }
  // Remove `_xla_outside_compilation` attribute from all ops.  These ops will
  // be outside of the device cluster. The `_xla_outside_compilation` attribute
  // on ops outside of tf_device.cluster don't have any meaning and can lead to
  // errors later on.  These ops were likely lifted out of the
  // tf_device.cluster in an earlier pass.
  module.walk(
      [](Operation* op) { op->removeAttr("_xla_outside_compilation"); });

  if (failed(CheckPostconditions(module))) return signalPassFailure();
}
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateExtractOutsideCompilationPass() {
  return std::make_unique<ExtractOutsideCompilation>();
}

}  // namespace TFDevice
}  // namespace mlir

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

#include <cstdint>
#include <string>
#include <type_traits>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/xla_sharding_util.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace mlir {
namespace TFTPU {

constexpr char kStepMarkerLocationAttr[] = "step_marker_location";
constexpr char kDevicesAttr[] = "devices";
constexpr char kVersionsAttr[] = "tf.versions";
constexpr char kUseXlaSpmdAttr[] = "use_spmd_for_xla_partitioning";

constexpr char kBadStringArrayElementMsg[] =
    "bad '{0}' attribute at index {1}, not a string";
constexpr char kBadArrayElementMsg[] =
    "bad '{0}' attribute at index {1} with value '{2}': failed to parse to {3}";
constexpr char kBadArrayAttrLengthMsg[] =
    "bad '{0}' attribute, expected array attribute of size {1}, got size {2}";

namespace {
struct TPURewritePass : public TF::TPURewritePassBase<TPURewritePass> {
  void runOnOperation() override;
};

// Creates a missing attribute error message.
std::string CreateMissingAttributeMsg(llvm::StringRef attribute) {
  return llvm::formatv("requires attribute '{0}'", attribute).str();
}

LogicalResult EncapsulateFuncAndSerialize(func::FuncOp entry_func,
                                          std::string* serialized_func_module) {
  ModuleOp module = entry_func->getParentOfType<ModuleOp>();
  SymbolTable entry_module_table(module);
  llvm::SmallVector<func::FuncOp, 4> referenced({entry_func});

  // Create a new module to hold func and all referenced functions.
  OwningOpRef<mlir::ModuleOp> module_for_func =
      ModuleOp::create(mlir::UnknownLoc::get(entry_func.getContext()));
  auto parent_module = entry_func->getParentOfType<ModuleOp>();
  auto versions_attr = parent_module->getAttr(kVersionsAttr);
  if (!versions_attr)
    return parent_module.emitError(CreateMissingAttributeMsg(kVersionsAttr));

  module_for_func.get().getOperation()->setAttr(kVersionsAttr, versions_attr);
  SymbolTable symbol_table(module_for_func.get());

  while (!referenced.empty()) {
    auto func = referenced.pop_back_val();

    // Skip functions that have already been cloned into new module.
    if (symbol_table.lookup<func::FuncOp>(func.getName())) continue;

    // Find any SymbolRefAttr in func that maps to a FuncOp. We need to clone
    // all found FuncOps to new_module to make sure new_module is
    // self-contained.
    Optional<SymbolTable::UseRange> uses = SymbolTable::getSymbolUses(func);
    assert(uses && "expected to be able to collect symbol uses");
    for (SymbolTable::SymbolUse use : *uses) {
      func::FuncOp referenced_func = entry_module_table.lookup<func::FuncOp>(
          use.getSymbolRef().cast<FlatSymbolRefAttr>().getValue());

      // Skip Symbols that do not map to a function.
      if (!referenced_func) continue;

      referenced.emplace_back(referenced_func);
    }

    auto clone = func.clone();
    if (clone.getName() == entry_func.getName()) {
      // We can simply change name of TPU program's main function because there
      // should be no other reference to it.
      clone.setName(StringAttr::get(clone.getContext(), "main"));
      clone.setPublic();
    } else {
      clone.setPrivate();
    }
    symbol_table.insert(clone);
  }

  *serialized_func_module =
      tensorflow::SerializeMlirModule(module_for_func.get());
  return success();
}

// Populates a TPUCompileMetadataProto with StepMarkerLocation from a
// `tf_device::ClusterFuncOp`.
LogicalResult SetMetadataProtoStepMarkerLocation(
    tf_device::ClusterFuncOp op,
    tensorflow::tpu::TPUCompileMetadataProto* metadata) {
  auto step_marker_location =
      op->getAttrOfType<StringAttr>(kStepMarkerLocationAttr);
  if (!step_marker_location)
    return op.emitOpError(CreateMissingAttributeMsg(kStepMarkerLocationAttr));

  // Default to `STEP_MARK_AT_ENTRY` for step marker location if attribute is
  // empty.
  xla::DebugOptions::StepMarkerLocation location =
      xla::DebugOptions::STEP_MARK_AT_ENTRY;
  if (!step_marker_location.getValue().empty() &&
      !xla::DebugOptions::StepMarkerLocation_Parse(
          std::string(step_marker_location.getValue()), &location))
    return op.emitOpError(llvm::formatv("bad '{0}' attribute with value '{1}'",
                                        kStepMarkerLocationAttr,
                                        step_marker_location.getValue()));

  metadata->set_step_marker_location(location);

  return success();
}

// Parses a xla::OpSharding from a string attribute.
LogicalResult SetOpSharding(Operation* op, Attribute attr, llvm::StringRef name,
                            int index, xla::OpSharding* sharding) {
  auto sharding_str = attr.dyn_cast<StringAttr>();
  if (!sharding_str)
    return op->emitOpError(
        llvm::formatv(kBadStringArrayElementMsg, name, index));

  if (!sharding->ParseFromString(sharding_str.getValue().str()))
    return op->emitOpError(llvm::formatv(kBadArrayElementMsg, name, index,
                                         sharding_str.getValue(),
                                         "xla::OpSharding"));

  return success();
}

// Populates a TPUCompileMetadataProto with argument types and sharding from a
// `tf_device::ClusterFuncOp`.
LogicalResult SetMetadataProtoArgs(
    tf_device::ClusterFuncOp op,
    tensorflow::tpu::TPUCompileMetadataProto* metadata) {
  auto input_shardings =
      op->getAttrOfType<ArrayAttr>(tensorflow::kInputShardingAttr);
  if (!input_shardings)
    return op.emitOpError(
        CreateMissingAttributeMsg(tensorflow::kInputShardingAttr));

  if (input_shardings.size() != op.getNumOperands())
    return op.emitOpError(
        llvm::formatv(kBadArrayAttrLengthMsg, tensorflow::kInputShardingAttr,
                      op.getNumOperands(), input_shardings.size()));

  // Set args metadata in proto.
  mlir::StringAttr replication_attr_name = mlir::StringAttr::get(
      op.getContext(), "mhlo.is_same_data_across_replicas");
  for (auto operand_type_and_idx : llvm::enumerate(op.getOperandTypes())) {
    Type operand_type = operand_type_and_idx.value();
    int index = operand_type_and_idx.index();
    tensorflow::tpu::TPUCompileMetadataProto::Arg* arg = metadata->add_args();
    tensorflow::DataType dtype;
    tensorflow::Status status =
        tensorflow::ConvertToDataType(operand_type, &dtype);
    if (!status.ok())
      return op.emitOpError(
          llvm::formatv("failed to determine operand type at index {0}: {1}",
                        index, status.error_message()));

    arg->set_dtype(dtype);
    // TODO(lyandy): Support other arg kinds.
    if (dtype == tensorflow::DT_RESOURCE)
      arg->set_kind(tensorflow::tpu::TPUCompileMetadataProto::Arg::VARIABLE);
    else
      arg->set_kind(tensorflow::tpu::TPUCompileMetadataProto::Arg::PARAMETER);

    // Populate argument shapes.
    *arg->mutable_shape() = tensorflow::TensorShapeProto();
    if (auto ranked_tensor_type = operand_type.dyn_cast<RankedTensorType>()) {
      tensorflow::TensorShapeProto shape_proto;
      ConvertToTensorShapeProto(ranked_tensor_type.getShape(), &shape_proto);
      *arg->mutable_shape() = std::move(shape_proto);
    } else {
      arg->mutable_shape()->set_unknown_rank(true);
    }

    if (failed(SetOpSharding(op, input_shardings.getValue()[index],
                             tensorflow::kInputShardingAttr, index,
                             arg->mutable_sharding())))
      return failure();

    // Populate set_is_same_data_across_replicas
    // Note: this information is duplicated and can be removed from the proto
    // and here once MLIR bridge phase 2 doesn't fallback to the old bridge.
    mlir::UnitAttr attr = op.getFunc().getArgAttrOfType<mlir::UnitAttr>(
        index, replication_attr_name);
    arg->set_is_same_data_across_replicas(attr != nullptr);
  }

  return success();
}

// Populates a TPUCompileMetadataProto with result sharding from a
// `tf_device::ClusterFuncOp`.
LogicalResult SetMetadataProtoRetvals(
    tf_device::ClusterFuncOp op,
    tensorflow::tpu::TPUCompileMetadataProto* metadata) {
  auto output_shardings =
      op->getAttrOfType<ArrayAttr>(tensorflow::kOutputShardingAttr);
  if (!output_shardings)
    return op.emitOpError(
        CreateMissingAttributeMsg(tensorflow::kOutputShardingAttr));

  if (output_shardings.size() != op.getNumResults())
    return op.emitOpError(
        llvm::formatv(kBadArrayAttrLengthMsg, tensorflow::kOutputShardingAttr,
                      op.getNumResults(), output_shardings.size()));

  // Set retvals metadata in proto.
  for (auto output_sharding_and_idx : llvm::enumerate(output_shardings))
    if (failed(SetOpSharding(op, output_sharding_and_idx.value(),
                             tensorflow::kOutputShardingAttr,
                             output_sharding_and_idx.index(),
                             metadata->add_retvals()->mutable_sharding())))
      return failure();

  return success();
}

// Populates a TPUCompileMetadataProto from attributes of a
// `tf_device::ClusterFuncOp`. If any necessary attributes are missing from the
// op, a failure will be returned.
// TODO(lyandy): Support session handle and guaranteed consts.
LogicalResult SetMetadataProtoFromClusterFuncOp(
    tf_device::ClusterFuncOp op, int num_replicas, int num_cores_per_replica,
    llvm::Optional<xla::DeviceAssignmentProto>&& xla_device_assignment,
    tensorflow::tpu::TPUCompileMetadataProto* metadata) {
  if (auto options_attr =
          op->getAttrOfType<StringAttr>("tpu_compile_options_proto")) {
    if (!metadata->mutable_compile_options()->ParseFromArray(
            options_attr.data(), options_attr.size())) {
      return failure();
    }
  }
  metadata->set_num_replicas(num_replicas);
  metadata->set_num_cores_per_replica(num_cores_per_replica);

  if (failed(SetMetadataProtoStepMarkerLocation(op, metadata)))
    return failure();

  if (xla_device_assignment.hasValue())
    *metadata->mutable_device_assignment() =
        std::move(xla_device_assignment.getValue());
  auto use_spmd_attr = op->getAttrOfType<BoolAttr>(kUseXlaSpmdAttr);
  if (!use_spmd_attr)
    return op.emitOpError(CreateMissingAttributeMsg(kUseXlaSpmdAttr));
  metadata->set_use_spmd_for_xla_partitioning(use_spmd_attr.getValue());

  if (failed(SetMetadataProtoArgs(op, metadata))) return failure();

  return SetMetadataProtoRetvals(op, metadata);
}

// Wraps single op in `tf_device.launch` for explicit device assignment.
tf_device::LaunchOp WrapOpInLaunch(OpBuilder* builder, Location loc,
                                   Operation* op, llvm::StringRef device) {
  OpBuilder::InsertPoint insert_point = builder->saveInsertionPoint();

  auto launch = builder->create<tf_device::LaunchOp>(
      loc, builder->getStringAttr(device), op->getResultTypes());
  launch.body().push_back(new Block);

  builder->setInsertionPointToEnd(&launch.GetBody());
  builder->create<tf_device::ReturnOp>(loc, op->getResults());

  // Move op inside cluster.
  op->moveBefore(launch.GetBody().getTerminator());

  builder->restoreInsertionPoint(insert_point);

  return launch;
}

// Create a `tf._TPUCompileMlir` that contains a MLIR module that is
// functionally equivalent to the function referenced by cluster_func.
Operation* BuildCompileOp(
    tf_device::ClusterFuncOp cluster_func, int num_replicas,
    int num_cores_per_replica, llvm::StringRef compilation_device,
    llvm::Optional<xla::DeviceAssignmentProto>&& xla_device_assignment,
    OpBuilder* builder, bool tpu_compile_metadata_debug) {
  // Set metadata from attributes.
  tensorflow::tpu::TPUCompileMetadataProto metadata;
  if (failed(SetMetadataProtoFromClusterFuncOp(
          cluster_func, num_replicas, num_cores_per_replica,
          std::move(xla_device_assignment), &metadata)))
    return nullptr;

  // Build a shape op for each input to cluster_func.
  // TODO(b/139377366): When shape inference is ready, we can use compile time
  // shape inference to get inputs that have static shapes and only use shape
  // ops for the rest.
  llvm::SmallVector<Value, 4> compile_op_operands;
  compile_op_operands.reserve(cluster_func.getNumOperands());

  for (auto operand_and_idx : llvm::enumerate(cluster_func.getOperands())) {
    // Skip adding shape op for operands that have static shapes.
    tensorflow::PartialTensorShape shape(
        metadata.args(operand_and_idx.index()).shape());
    if (shape.IsFullyDefined()) continue;

    auto shape_op = builder->create<TF::ShapeOp>(
        cluster_func.getLoc(),
        RankedTensorType::get({-1}, builder->getIntegerType(64)),
        operand_and_idx.value());
    compile_op_operands.emplace_back(shape_op.getResult());
  }

  FlatSymbolRefAttr func_attr = cluster_func.funcAttr();
  func::FuncOp func =
      cluster_func->getParentOfType<ModuleOp>().lookupSymbol<func::FuncOp>(
          func_attr.getValue());

  std::string txt_module;
  if (failed(EncapsulateFuncAndSerialize(func, &txt_module))) return nullptr;

  auto compilation_status_type =
      RankedTensorType::get({}, builder->getType<TF::StringType>());
  auto program_type =
      RankedTensorType::get({3}, builder->getType<TF::StringType>());

  // Add MLIR module's fingerprint to compile metadata.
  uint64_t mlir_fingerprint = tensorflow::Fingerprint64(txt_module);
  metadata.set_mlir_fingerprint(mlir_fingerprint);

  std::string txt_metadata;
  if (tpu_compile_metadata_debug) {
    ::tensorflow::protobuf::TextFormat::Printer printer;
    printer.SetExpandAny(true);
    printer.PrintToString(metadata, &txt_metadata);
  } else {
    metadata.SerializeToString(&txt_metadata);
  }

  auto compile_op = builder->create<TF::_TPUCompileMlirOp>(
      cluster_func.getLoc(),
      /*compilation_status=*/compilation_status_type, /*program=*/
      llvm::SmallVector<Type, 8>(num_cores_per_replica, program_type),
      compile_op_operands, txt_module, txt_metadata);

  return WrapOpInLaunch(builder, compile_op.getLoc(), compile_op,
                        compilation_device);
}

// Assigns explicit devices to replicate op. An aliased device is created per
// core, and all replica devices per core are grouped together.
void AssignDevicesToReplicate(
    tf_device::ReplicateOp replicate,
    llvm::ArrayRef<llvm::SmallVector<tensorflow::TPUDeviceAndHost, 8>>
        tpu_devices,
    OpBuilder* builder) {
  if (!replicate) return;

  const int num_replicas = tpu_devices.size();
  const int num_cores_per_replica = tpu_devices.front().size();

  llvm::SmallVector<NamedAttribute, 8> device_attrs;
  for (int core = 0; core < num_cores_per_replica; ++core) {
    llvm::SmallVector<StringRef, 8> devices_by_core;
    devices_by_core.reserve(num_replicas);
    for (int replica = 0; replica < num_replicas; ++replica)
      devices_by_core.push_back(tpu_devices[replica][core].device);

    device_attrs.push_back(
        builder->getNamedAttr(tensorflow::GetDeviceAliasForLogicalCore(core),
                              builder->getStrArrayAttr(devices_by_core)));
  }

  // For data parallelism, also add replicated host devices, as these are
  // necessary for outside compilation.
  llvm::SmallVector<StringRef, 8> hosts;
  hosts.reserve(num_replicas);
  for (int replica = 0; replica < num_replicas; ++replica)
    hosts.push_back(tpu_devices[replica][0].host);

  device_attrs.push_back(builder->getNamedAttr(
      tensorflow::kTPUReplicatedHost, builder->getStrArrayAttr(hosts)));

  replicate->setAttr(kDevicesAttr, builder->getDictionaryAttr(device_attrs));
}

// Creates a `tf.TPUExecute` op that executes TPU program.
LogicalResult BuildExecuteOp(
    const int core_id, llvm::ArrayRef<xla::OpSharding> output_sharding_config,
    llvm::ArrayRef<Value> inputs, tf_device::ClusterFuncOp cluster_func,
    OpBuilder* builder, TF::TPUExecuteOp* execute_op) {
  // TODO(b/139377366): Need to snapshot all resource variable inputs in
  // follow-up CLs.
  llvm::SmallVector<Type, 4> output_types;
  auto result = tensorflow::GetOutputTypesForLogicalDeviceComputation(
      core_id, output_sharding_config, cluster_func, &output_types);
  if (failed(result)) return failure();

  // TPUExecute has same output types as cluster_func.
  *execute_op = builder->create<TF::TPUExecuteOp>(cluster_func.getLoc(),
                                                  output_types, inputs);
  auto producer_name_attr = cluster_func->getAttr("_producer_name");
  if (producer_name_attr)
    (*execute_op)->setAttr("_producer_name", producer_name_attr);
  return success();
}

// Move child processes of the ParallelExecute that do not change. These are all
// children except for the child with the ClusterFunc.
//
// Returns the index of the child with the ClusterFunc.
int MovePreservedParallelExecuteChildren(
    const int num_cores_per_replica,
    llvm::SmallVector<Type, 8>& concatenated_output_types, OpBuilder* builder,
    tf_device::ClusterFuncOp cluster_func,
    tf_device::ParallelExecuteOp old_parallel_execute,
    tf_device::ParallelExecuteOp* new_parallel_execute) {
  // `num_moved_children` is the number of children that will be preserved.
  const int num_moved_children = old_parallel_execute.regions().size() - 1;
  *new_parallel_execute = builder->create<tf_device::ParallelExecuteOp>(
      old_parallel_execute->getLoc(),
      num_moved_children + num_cores_per_replica, concatenated_output_types);

  // `cluster_idx` is the index of the child with the `ClusterFuncOp`, which
  // will be replaced.
  int cluster_idx = -1;
  for (int child_idx = 0; child_idx < old_parallel_execute.regions().size();
       ++child_idx) {
    auto& block = old_parallel_execute.GetRegionBlockWithIndex(child_idx);
    if (cluster_func->getBlock() == &block) {
      assert(cluster_idx == -1);
      cluster_idx = child_idx;
    }
  }
  assert(cluster_idx != -1);

  // Move children that are preserved.
  for (int child_idx = 0; child_idx < num_moved_children; ++child_idx) {
    int old_idx = child_idx >= cluster_idx ? child_idx + 1 : child_idx;
    int new_idx = child_idx >= cluster_idx ? child_idx + num_cores_per_replica
                                           : child_idx;
    new_parallel_execute->getRegions()[new_idx]->takeBody(
        *old_parallel_execute.getRegions()[old_idx]);
  }

  return cluster_idx;
}

// Given a `ParallelExecute`, replace it with a new `ParallelExecute`. The
// new `ParallelExecute` will replace the child that contains the
// `ClusterFunc` with `num_cores_per_replica` children. It keep other children
// the same. Return values from the child with the `ClusterFunc` will be
// duplicated `num_cores_per_replica` times.
LogicalResult AddToParallelExecuteOp(
    llvm::ArrayRef<llvm::SmallVector<tensorflow::TPUDeviceAndHost, 8>>
        tpu_devices,
    llvm::ArrayRef<xla::OpSharding> output_sharding_config,
    Operation* compile_op, tf_device::ClusterFuncOp cluster_func,
    OpBuilder* builder, tf_device::ParallelExecuteOp old_parallel_execute,
    tf_device::ParallelExecuteOp* new_parallel_execute, int* cluster_idx) {
  const int num_cores_per_replica = tpu_devices.front().size();
  // parallel_execute op returns concatenated list of return values of
  // all its regions.
  //
  // TODO(b/149102702): Correctly map inputs to parallel_execute op via
  // identifying xla_sharding op in the cluster_func function.
  const auto cluster_result_types = cluster_func.getResultTypes();
  llvm::SmallVector<Type, 8> concatenated_output_types;
  concatenated_output_types.reserve(cluster_result_types.size() *
                                    num_cores_per_replica);

  for (int core = 0; core < num_cores_per_replica; ++core) {
    llvm::SmallVector<Type, 4> output_types;
    auto result = tensorflow::GetOutputTypesForLogicalDeviceComputation(
        core, output_sharding_config, cluster_func, &output_types);
    if (failed(result)) return failure();

    for (Type t : output_types) concatenated_output_types.emplace_back(t);
  }

  *cluster_idx = MovePreservedParallelExecuteChildren(
      num_cores_per_replica, concatenated_output_types, builder, cluster_func,
      old_parallel_execute, new_parallel_execute);

  // Extract inputs for each block of the parallel_execute op. The i-th
  // element in the list represents the input lists to TPU computation for
  // i-th logical core.
  llvm::SmallVector<llvm::SmallVector<mlir::Value, 4>, 4> input_list;
  builder->setInsertionPoint(*new_parallel_execute);
  auto result = tensorflow::ExtractInputsForLogicalDevices(
      num_cores_per_replica, cluster_func, builder, &input_list);
  if (failed(result)) return failure();

  const bool replicated = tpu_devices.size() != 1;
  // For each logical core, create a region with TPUExecute op.
  assert(input_list.size() == num_cores_per_replica);
  for (int core = 0; core < num_cores_per_replica; ++core) {
    auto& block =
        new_parallel_execute->GetRegionBlockWithIndex((*cluster_idx) + core);
    builder->setInsertionPointToEnd(&block);

    // Create Execute op.
    //
    // TODO(b/148913294): Identify inputs/return values specific to each
    // logical core TPU execution by parsing xla_sharding op in
    // cluster_func.
    auto execute_inputs = input_list[core];
    execute_inputs.emplace_back(compile_op->getResult(core + 1));

    TF::TPUExecuteOp execute;
    result = BuildExecuteOp(core, output_sharding_config, execute_inputs,
                            cluster_func, builder, &execute);
    if (failed(result)) return failure();

    // If computation is replicated, use aliased device. Otherwise there is only
    // one execution device per core and the device is assigned to the execute
    // op.
    std::string device = replicated
                             ? tensorflow::GetDeviceAliasForLogicalCore(core)
                             : tpu_devices.front()[core].device;

    auto block_launch_op =
        WrapOpInLaunch(builder, block.getParent()->getLoc(), execute, device);

    builder->create<tf_device::ReturnOp>(block.getParent()->getLoc(),
                                         block_launch_op.getResults());
  }

  return success();
}

tf_device::LaunchOp AssignDevicesToReplicatedExecute(
    llvm::ArrayRef<llvm::SmallVector<tensorflow::TPUDeviceAndHost, 8>>
        tpu_devices,
    Operation* execute_op, OpBuilder* builder) {
  const bool replicated = tpu_devices.size() != 1;
  // If computation is replicated, use aliased device. Otherwise there is only
  // one execution device and the device is assigned to the execute op.
  std::string device = replicated ? tensorflow::GetDeviceAliasForLogicalCore(0)
                                  : tpu_devices.front().front().device;

  return WrapOpInLaunch(builder, execute_op->getLoc(), execute_op, device);
}

// Creates a `tf.TPUCompileSucceededAssert` operation that parses compilation
// status of `compile_op` to check whether compilation is successful.
void BuildTPUCompileSucceededAssertOp(Operation* compile_op,
                                      Operation* result_id,
                                      llvm::StringRef compilation_device,
                                      OpBuilder* builder) {
  auto assert_op = builder->create<TF::TPUCompileSucceededAssertOp>(
      compile_op->getLoc(), result_id->getResult(0));
  WrapOpInLaunch(builder, compile_op->getLoc(), assert_op, compilation_device);
}

// Wrap the `ClusterFunc` in a `ParallelExecute` with only one child. This
// creates a canonical form, regardless of whether a `ParallelExecute` was
// created in a previous pass.
void BuildSingletonParallelExecuteOp(
    tf_device::ClusterFuncOp cluster_func, OpBuilder* builder,
    tf_device::ParallelExecuteOp* parallel_execute) {
  if (!*parallel_execute) {
    const auto output_types = cluster_func.getResultTypes();
    builder->setInsertionPoint(cluster_func);
    *parallel_execute = builder->create<tf_device::ParallelExecuteOp>(
        cluster_func.getLoc(), 1, output_types);
    cluster_func->remove();
    auto& block = parallel_execute->GetRegionBlockWithIndex(0);
    builder->setInsertionPointToEnd(&block);
    builder->insert(cluster_func);
    cluster_func.replaceAllUsesWith(*parallel_execute);
    builder->create<tf_device::ReturnOp>(block.getParent()->getLoc(),
                                         cluster_func.getResults());
  }
}

// Unwrap the `ParallelExecute`'s contents if it only has one child.
LogicalResult RemoveSingletonParallelExecuteOp(
    tf_device::ParallelExecuteOp parallel_execute, OpBuilder* builder) {
  if (parallel_execute.regions().size() == 1) {
    builder->setInsertionPoint(parallel_execute);
    auto& block = parallel_execute.GetRegionBlockWithIndex(0);
    llvm::SmallVector<Operation*, 2> ops_move;
    for (Operation& op : block) {
      ops_move.push_back(&op);
    }
    if (ops_move.size() != 2) {
      parallel_execute.emitError() << "Expected 2 ops in parallel_execute.";
      return failure();
    }
    auto launch = llvm::dyn_cast<tf_device::LaunchOp>(ops_move[0]);
    if (!launch) {
      parallel_execute.emitError()
          << "Expected the op in parallel_execute to be a tf_device.launch";
      return failure();
    }
    launch->remove();
    builder->insert(launch);
    parallel_execute.replaceAllUsesWith(launch);
    parallel_execute.erase();
  }
  return success();
}

LogicalResult CheckTPUPartitionedInputAndOutputAreValid(
    tf_device::ClusterFuncOp cluster,
    tf_device::ParallelExecuteOp parallel_execute) {
  for (auto cluster_result : parallel_execute.execute_outputs()) {
    for (Operation* user :
         llvm::make_early_inc_range(cluster_result.getUsers())) {
      // Check that user has no outputs that are TPUPartitionedOutput
      for (auto result : user->getResults()) {
        for (Operation* user : llvm::make_early_inc_range(result.getUsers())) {
          if (llvm::isa<TF::TPUPartitionedOutputOp>(user)) {
            user->emitError() << "Input of TPUPartitionedOutput must "
                              << "be in tpu computation.";
            return failure();
          }
        }
      }
    }
  }
  for (auto cluster_operand : cluster.operands()) {
    Operation* def = cluster_operand.getDefiningOp();
    // This pass assumes that a TPUPartitionedInput is preceeded by
    // ReadVariable ops, and not vice versa. An earlier pass,
    // TPUResourceReadsWritesPartitioning, should have ensured this
    // precondition.
    if (!def) continue;
    for (auto operand : def->getOperands()) {
      Operation* def_of_read = operand.getDefiningOp();
      if (llvm::isa_and_nonnull<TF::TPUPartitionedInputOp>(def_of_read)) {
        def_of_read->emitError() << "Output of TPUPartitionedInput must "
                                 << "be in tpu computation.";
        return failure();
      }
    }
  }
  return success();
}

LogicalResult Rewrite(
    tf_device::ClusterFuncOp cluster_func,
    llvm::ArrayRef<tensorflow::DeviceNameUtils::ParsedName> devices,
    ArrayRef<TF::TPUCompilationResultOp> compilation_result, OpBuilder* builder,
    bool tpu_compile_metadata_debug) {
  // Fetch the ParallelExecute parent of `cluster_func`, or create it if it does
  // not exist.
  tf_device::ParallelExecuteOp old_parallel_execute =
      cluster_func->getParentOfType<tf_device::ParallelExecuteOp>();
  if (old_parallel_execute &&
      cluster_func->getParentOp() != old_parallel_execute) {
    cluster_func->emitError() << "The ParallelExecute ancestor of a "
                                 "ClusterFunc must be its direct parent.";
    return failure();
  }
  BuildSingletonParallelExecuteOp(cluster_func, builder, &old_parallel_execute);

  // check TPUPartitionedInput and TPUPartitionedOutput are in valid pattern
  if (failed(CheckTPUPartitionedInputAndOutputAreValid(cluster_func,
                                                       old_parallel_execute)))
    return failure();

  // Collect `num_replicas` and `num_cores_per_replica` attributes.
  int num_replicas = 1;
  tf_device::ReplicateOp replicate =
      cluster_func->getParentOfType<tf_device::ReplicateOp>();
  if (replicate) num_replicas = replicate.n();

  auto num_cores_per_replica_attr = cluster_func->getAttrOfType<IntegerAttr>(
      tensorflow::kNumCoresPerReplicaAttr);
  if (!num_cores_per_replica_attr)
    return cluster_func.emitOpError(
        CreateMissingAttributeMsg(tensorflow::kNumCoresPerReplicaAttr));

  int num_cores_per_replica = num_cores_per_replica_attr.getInt();

  auto topology_attr =
      cluster_func->getAttrOfType<StringAttr>(tensorflow::kTopologyAttr);
  if (!topology_attr)
    return cluster_func.emitOpError(
        CreateMissingAttributeMsg(tensorflow::kTopologyAttr));

  auto device_assignment_attr = cluster_func->getAttrOfType<mlir::ArrayAttr>(
      tensorflow::kDeviceAssignmentAttr);
  if (!device_assignment_attr)
    return cluster_func.emitOpError(
        llvm::formatv("requires attribute '{0}'",
                      tensorflow::kDeviceAssignmentAttr)
            .str());

  auto status_or_device_coodinates =
      tensorflow::GetDeviceCoordinates(device_assignment_attr);
  if (!status_or_device_coodinates.ok())
    return cluster_func.emitError()
           << "error in fetching tpu device coordinates: "
           << status_or_device_coodinates.status().error_message();

  // Determine compilation and execution devices.
  auto status_or_tpu_device_assignment =
      tensorflow::GetTPUCompilationAndExecutionDevices(
          devices, num_replicas, num_cores_per_replica,
          topology_attr.getValue(), status_or_device_coodinates.value());
  if (!status_or_tpu_device_assignment.ok())
    return cluster_func.emitError()
           << "error in fetching TPU compilation/execution devices: "
           << status_or_tpu_device_assignment.status().error_message();

  // Create compile op.
  auto& tpu_device_assignment = status_or_tpu_device_assignment.ValueOrDie();
  builder->setInsertionPoint(cluster_func);

  // Create the TPUCompileMlir and TPUCompileSucceededAssert outside of
  // the parallel_execute.
  builder->setInsertionPoint(old_parallel_execute);
  Operation* compile_op =
      BuildCompileOp(cluster_func, num_replicas, num_cores_per_replica,
                     tpu_device_assignment.compilation_device,
                     std::move(tpu_device_assignment.xla_device_assignment),
                     builder, tpu_compile_metadata_debug);
  if (!compile_op) return failure();

  // This replaces _TPUCompileMlir placeholder ops that are required
  // by XlaRecvAtHost and XlaSendFromHost ops add in earlier pass.
  // TODO(b/157054714): When a better abstraction instead of _TPUCompileMlirOp
  // and _XlaRecvAtHostOp and _XlaSendFromHostOp are used, update to a more
  // structured lowering.
  old_parallel_execute.walk(
      [&](TF::_TPUCompileMlirPlaceholderProgramKeyOp key_op) {
        key_op.replaceAllUsesWith(compile_op->getResult(1));
        key_op.erase();
      });

  // After rewrite, if there is a TPUCompilationResultOp from the same cluster,
  // replace it with the result of the compile op. The TPUCompilationResultOp is
  // used as a placeholder to hook during graph creation the other ops that are
  // intended to consume the compile result.
  Operation* result_id = compile_op;
  // TODO(jpienaar): Remove this later.
  auto compile_device_op = compile_op->getAttr("device");
  for (auto res : compilation_result) {
    // Build identity op with the same location/name as the original compilation
    // result op.
    result_id = builder->create<TF::IdentityOp>(
        res.getLoc(), compile_op->getResult(0).getType(),
        result_id->getResult(0));
    // Assign to same device as result is currently set, unless unset and then
    // assign to the device on which compilation will happen.
    // TODO(jpienaar): Remove this later.
    if (auto device = res->getAttrOfType<StringAttr>("device")) {
      if (!device.getValue().empty())
        result_id->setAttr("device", device);
      else
        result_id->setAttr("device", compile_device_op);
    } else if (compile_device_op) {
      result_id->setAttr("device", compile_device_op);
    }
    res.output().replaceAllUsesWith(compile_op->getResult(0));
  }

  BuildTPUCompileSucceededAssertOp(
      compile_op, result_id, tpu_device_assignment.compilation_device, builder);

  AssignDevicesToReplicate(replicate, tpu_device_assignment.tpu_devices,
                           builder);

  llvm::SmallVector<xla::OpSharding, 4> output_shardings;
  auto result = tensorflow::ParseAndValidateOutputSharding(
      num_cores_per_replica, cluster_func, &output_shardings);
  if (failed(result)) return failure();

  // For model parallelism, tf_device.parallel_execute is used to express
  // concurrent device execution across multiple logical devices.
  tf_device::ParallelExecuteOp new_parallel_execute;
  int cluster_idx;
  result = AddToParallelExecuteOp(tpu_device_assignment.tpu_devices,
                                  output_shardings, compile_op, cluster_func,
                                  builder, old_parallel_execute,
                                  &new_parallel_execute, &cluster_idx);
  if (failed(result)) return failure();

  // As tf_device.parallel_execute wraps # logical cores number of TPUExecute
  // ops, the number of return values of parallel_execute op exceeds that of
  // cluster_func op. As such, each return value of parallel_execute op must
  // be mapped with corresponding return value usages of cluster_func.
  result = tensorflow::RemapOutputsFromLogicalDevices(
      cluster_func.getLoc(), output_shardings, old_parallel_execute,
      cluster_idx, new_parallel_execute, builder);
  if (failed(result)) return failure();

  return RemoveSingletonParallelExecuteOp(new_parallel_execute, builder);
}

// Erase rewritten ClusterFuncOp(s). If TPUPartitionedInputOp /
// TPUPartitionedOutputOp are present, they must be removed along with the
// ClusterFuncOp(s).
void EraseClusterFuncs(
    llvm::MutableArrayRef<tf_device::ClusterFuncOp> to_be_erased) {
  for (auto cluster : to_be_erased) {
    auto old_parallel_execute =
        cluster->getParentOfType<tf_device::ParallelExecuteOp>();
    assert(old_parallel_execute);

    for (auto result : old_parallel_execute.execute_outputs()) {
      for (Operation* user : llvm::make_early_inc_range(result.getUsers())) {
        if (llvm::isa<TF::TPUPartitionedOutputOp>(user)) {
          assert(user->use_empty());
          user->erase();
        }
      }
    }

    for (auto operand : cluster.operands()) {
      Operation* def = operand.getDefiningOp();
      if (operand.hasOneUse() &&
          llvm::isa_and_nonnull<TF::TPUPartitionedInputOp>(def)) {
        operand.dropAllUses();
        def->erase();
      }
    }
    assert(old_parallel_execute->use_empty());
    old_parallel_execute->erase();
  }
}

void TPURewritePass::runOnOperation() {
  mlir::TF::RuntimeDevices devices;
  if (failed(tensorflow::GetDevicesFromOp(getOperation(), &devices)))
    return signalPassFailure();

  // Collect compilation results.
  llvm::DenseMap<Attribute, SmallVector<TF::TPUCompilationResultOp, 1>>
      compilation_results;
  auto result_init = getOperation().walk([&](TF::TPUCompilationResultOp op) {
    auto cluster_id = op->getAttrOfType<StringAttr>("_tpu_compilation_status");
    if (!cluster_id) {
      op->emitOpError("missing '_tpu_compilation_status'");
      return WalkResult::interrupt();
    }
    compilation_results[cluster_id].push_back(op);
    return WalkResult::advance();
  });
  if (result_init.wasInterrupted()) return signalPassFailure();
  llvm::SmallVector<tf_device::ClusterFuncOp> to_be_erased;
  OpBuilder builder(&getContext());
  auto result = getOperation().walk([&](tf_device::ClusterFuncOp op) {
    if (failed(TF::HasValidCompilationAndReplicationAttributes(*op)))
      return WalkResult::interrupt();
    // Skip non-tpu device cluster_func.
    auto cluster_id = op->getAttrOfType<StringAttr>(TF::kReplicationInfoAttr);
    if (!cluster_id) return WalkResult::advance();

    if (failed(Rewrite(op, devices.device_names(),
                       compilation_results[cluster_id], &builder,
                       tpu_compile_metadata_debug_)))
      return WalkResult::interrupt();

    to_be_erased.push_back(op);
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) return signalPassFailure();

  EraseClusterFuncs(to_be_erased);

  // Eliminate TPUCompilationResultOp now that the rewrite is complete.
  for (auto& it : compilation_results) {
    for (auto op : it.second) {
      if (!op.use_empty()) {
        mlir::InFlightDiagnostic err = op.emitError("uses remain post rewrite");
        for (auto user : op->getUsers())
          err.attachNote(user->getLoc()) << "remaining user";
        return signalPassFailure();
      }
      op.erase();
    }
  }

  // TODO(b/139377366): Remove functions that are no longer needed.
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateTPURewritePass() {
  return std::make_unique<TPURewritePass>();
}

}  // namespace TFTPU
}  // namespace mlir

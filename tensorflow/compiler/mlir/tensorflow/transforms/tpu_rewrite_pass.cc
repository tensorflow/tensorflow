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
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/xla_sharding_util.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/protobuf/tpu/dynamic_padding.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace mlir {
namespace TFTPU {

// NOLINTNEXTLINE
static llvm::cl::opt<bool> tpu_compile_metadata_debug(
    "tpu_compile_metadata_debug",
    llvm::cl::desc("Serialize TPUCompileMetadataProto metadata in "
                   "'tf._TPUCompileMlir' op as a proto debug string"));

constexpr char kNumReplicasAttr[] = "num_replicas";
constexpr char kNumCoresPerReplicaAttr[] = "num_cores_per_replica";
constexpr char kStepMarkerLocationAttr[] = "step_marker_location";
constexpr char kPaddingMapAttr[] = "padding_map";
constexpr char kTopologyAttr[] = "topology";
constexpr char kDeviceAssignmentAttr[] = "device_assignment";
constexpr char kDeviceAttr[] = "device";
constexpr char kDevicesAttr[] = "devices";
constexpr char kVersionsAttr[] = "tf.versions";

constexpr char kBadStringArrayElementMsg[] =
    "bad '{0}' attribute at index {1}, not a string";
constexpr char kBadIntArrayElementMsg[] =
    "bad '{0}' attribute at index {1}, not an int";
constexpr char kBadArrayElementMsg[] =
    "bad '{0}' attribute at index {1} with value '{2}': failed to parse to {3}";
constexpr char kBadArrayAttrLengthMsg[] =
    "bad '{0}' attribute, expected array attribute of size {1}, got size {2}";

// Rewrites `tf_device.cluster_func` operations assigned to TPU into actual TPU
// jit-compile runtime ops.
//
// For example:
//   %1 = "tf_device.cluster_func"(%0) {_tpu_replicate = "cluster", func =
//         @tpu_func}
//   %2 = "tf.SomeOp"(%1)
//
// Would become following ops (unimportant attributes, types are omitted):
//    %1 = "tf.Shape"(%0)
//    %2:2 = "tf._TPUCompileMlir"(%1) {module = "<Serialized @tpu_func>"}
//    "tf.TPUCompileSucceededAssert"(%2#0)
//    %3 = "tf.TPUExecute"(%0, %2#1)
//    %4 = "tf.SomeOp"(%3)

namespace {
struct TPURewritePass
    : public PassWrapper<TPURewritePass, OperationPass<ModuleOp>> {
  void runOnOperation() override;
};

// Creates a missing attribute error message.
std::string CreateMissingAttributeMsg(llvm::StringRef attribute) {
  return llvm::formatv("requires attribute '{0}'", attribute).str();
}

LogicalResult EncapsulateFuncAndSerialize(FuncOp entry_func,
                                          std::string* serialized_func_module) {
  ModuleOp module = entry_func.getParentOfType<ModuleOp>();
  SymbolTable entry_module_table(module);
  llvm::SmallVector<FuncOp, 4> referenced({entry_func});

  // Create a new module to hold func and all referenced functions.
  OwningModuleRef module_for_func =
      ModuleOp::create(mlir::UnknownLoc::get(entry_func.getContext()));
  auto parent_module = entry_func.getParentOfType<ModuleOp>();
  auto versions_attr = parent_module.getAttr(kVersionsAttr);
  if (!versions_attr)
    return parent_module.emitError(CreateMissingAttributeMsg(kVersionsAttr));

  module_for_func.get().getOperation()->setAttr(kVersionsAttr, versions_attr);
  SymbolTable symbol_table(module_for_func.get());

  while (!referenced.empty()) {
    auto func = referenced.pop_back_val();

    // Skip functions that have already been cloned into new module.
    if (symbol_table.lookup<FuncOp>(func.getName())) continue;

    // Find any SymbolRefAttr in func that maps to a FuncOp. We need to clone
    // all found FuncOps to new_module to make sure new_module is
    // self-contained.
    Optional<SymbolTable::UseRange> uses = SymbolTable::getSymbolUses(func);
    assert(uses && "expected to be able to collect symbol uses");
    for (SymbolTable::SymbolUse use : *uses) {
      FuncOp referenced_func = entry_module_table.lookup<FuncOp>(
          use.getSymbolRef().cast<FlatSymbolRefAttr>().getValue());

      // Skip Symbols that do not map to a function.
      if (!referenced_func) continue;

      referenced.emplace_back(referenced_func);
    }

    auto clone = func.clone();
    if (clone.getName() == entry_func.getName()) {
      // We can simply change name of TPU program's main function because there
      // should be no other reference to it.
      clone.setName("main");
    }
    symbol_table.insert(clone);
  }

  // Serialize module and return.
  {
    llvm::raw_string_ostream os(*serialized_func_module);
    module_for_func.get().print(os);
  }
  return success();
}

// Extracts device coordinates from a device assignment attribute on an op.
LogicalResult GetDeviceCoordinates(
    tf_device::ClusterFuncOp op,
    llvm::SmallVectorImpl<int64_t>* device_assignment) {
  auto device_assignment_attr =
      op.getAttrOfType<ArrayAttr>(kDeviceAssignmentAttr);
  if (!device_assignment_attr)
    return op.emitOpError(CreateMissingAttributeMsg(kDeviceAssignmentAttr));

  device_assignment->reserve(device_assignment_attr.size());

  for (auto device_coordinate_and_idx :
       llvm::enumerate(device_assignment_attr)) {
    auto device_coordinate =
        device_coordinate_and_idx.value().dyn_cast<IntegerAttr>();
    if (!device_coordinate)
      return op.emitOpError(llvm::formatv(kBadIntArrayElementMsg,
                                          kDeviceAssignmentAttr,
                                          device_coordinate_and_idx.index()));

    device_assignment->push_back(device_coordinate.getInt());
  }

  return success();
}

// Populates a TPUCompileMetadataProto with StepMarkerLocation from a
// `tf_device::ClusterFuncOp`.
LogicalResult SetMetadataProtoStepMarkerLocation(
    tf_device::ClusterFuncOp op,
    tensorflow::tpu::TPUCompileMetadataProto* metadata) {
  auto step_marker_location =
      op.getAttrOfType<StringAttr>(kStepMarkerLocationAttr);
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

// Populates a TPUCompileMetadataProto with PaddingMap from a
// `tf_device::ClusterFuncOp`.
LogicalResult SetMetadataProtoPaddingMap(
    tf_device::ClusterFuncOp op,
    tensorflow::tpu::TPUCompileMetadataProto* metadata) {
  auto padding_map = op.getAttrOfType<ArrayAttr>(kPaddingMapAttr);
  if (!padding_map)
    return op.emitOpError(CreateMissingAttributeMsg(kPaddingMapAttr));

  for (const auto& padding_and_idx : llvm::enumerate(padding_map)) {
    auto& padding_attr = padding_and_idx.value();
    auto padding_attr_str = padding_attr.dyn_cast<StringAttr>();
    if (!padding_attr_str)
      return op.emitOpError(llvm::formatv(
          kBadStringArrayElementMsg, kPaddingMapAttr, padding_and_idx.index()));

    tensorflow::tpu::PaddingMap* padding =
        metadata->mutable_padding_maps()->Add();
    if (!padding->ParseFromString(std::string(padding_attr_str.getValue())))
      return op.emitOpError(llvm::formatv(
          kBadArrayElementMsg, kPaddingMapAttr, padding_and_idx.index(),
          padding_attr_str.getValue(), "tpu::PaddingMap"));
  }

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
      op.getAttrOfType<ArrayAttr>(tensorflow::kInputShardingAttr);
  if (!input_shardings)
    return op.emitOpError(
        CreateMissingAttributeMsg(tensorflow::kInputShardingAttr));

  if (input_shardings.size() != op.getNumOperands())
    return op.emitOpError(
        llvm::formatv(kBadArrayAttrLengthMsg, tensorflow::kInputShardingAttr,
                      op.getNumOperands(), input_shardings.size()));

  // Set args metadata in proto.
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
  }

  return success();
}

// Populates a TPUCompileMetadataProto with result sharding from a
// `tf_device::ClusterFuncOp`.
LogicalResult SetMetadataProtoRetvals(
    tf_device::ClusterFuncOp op,
    tensorflow::tpu::TPUCompileMetadataProto* metadata) {
  auto output_shardings =
      op.getAttrOfType<ArrayAttr>(tensorflow::kOutputShardingAttr);
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
  metadata->set_num_replicas(num_replicas);
  metadata->set_num_cores_per_replica(num_cores_per_replica);

  if (failed(SetMetadataProtoStepMarkerLocation(op, metadata)))
    return failure();

  if (failed(SetMetadataProtoPaddingMap(op, metadata))) return failure();

  if (xla_device_assignment.hasValue())
    *metadata->mutable_device_assignment() =
        std::move(xla_device_assignment.getValue());

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
    OpBuilder* builder) {
  // Set metadata from attributes.
  tensorflow::tpu::TPUCompileMetadataProto metadata;
  if (failed(SetMetadataProtoFromClusterFuncOp(
          cluster_func, num_replicas, num_cores_per_replica,
          std::move(xla_device_assignment), &metadata)))
    return nullptr;

  std::string txt_metadata;
  if (tpu_compile_metadata_debug)
    txt_metadata = metadata.DebugString();
  else
    metadata.SerializeToString(&txt_metadata);

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
  FuncOp func = cluster_func.getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(
      func_attr.getValue());

  std::string txt_module;
  if (failed(EncapsulateFuncAndSerialize(func, &txt_module))) return nullptr;

  auto result_type =
      RankedTensorType::get({}, builder->getType<TF::StringType>());

  auto compile_op = builder->create<TF::_TPUCompileMlirOp>(
      cluster_func.getLoc(), /*compilation_status=*/result_type, /*program=*/
      llvm::SmallVector<Type, 8>(num_cores_per_replica, result_type),
      compile_op_operands, txt_module, txt_metadata);

  return WrapOpInLaunch(builder, compile_op.getLoc(), compile_op,
                        compilation_device);
}

// Assigns explicit devices to replicate op. An aliased device is created per
// core, and all replica devices per core are grouped together.
void AssignDevicesToReplicate(
    tf_device::ReplicateOp replicate,
    llvm::ArrayRef<llvm::SmallVector<std::string, 8>> execution_devices,
    OpBuilder* builder) {
  if (!replicate) return;

  const int num_replicas = execution_devices.size();
  const int num_cores_per_replica = execution_devices.front().size();

  llvm::SmallVector<NamedAttribute, 8> device_attrs;
  for (int core = 0; core < num_cores_per_replica; ++core) {
    llvm::SmallVector<StringRef, 8> devices_by_core;
    devices_by_core.reserve(num_replicas);
    for (int replica = 0; replica < num_replicas; ++replica)
      devices_by_core.push_back(execution_devices[replica][core]);

    device_attrs.push_back(
        builder->getNamedAttr(tensorflow::GetDeviceAliasForLogicalCore(core),
                              builder->getStrArrayAttr(devices_by_core)));
  }

  replicate.setAttr(kDevicesAttr, builder->getDictionaryAttr(device_attrs));
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
  *execute_op = builder->create<TF::TPUExecuteOp>(
      cluster_func.getLoc(), output_types, inputs,
      llvm::ArrayRef<NamedAttribute>{});
  return success();
}

// Creates a tf_device.parallel_execute op that wraps TPUExecute op to
// represent execution of TPU program in multiple logical cores.
LogicalResult BuildParallelExecuteOp(
    llvm::ArrayRef<llvm::SmallVector<std::string, 8>> execution_devices,
    llvm::ArrayRef<xla::OpSharding> output_sharding_config,
    Operation* compile_op, tf_device::ClusterFuncOp cluster_func,
    OpBuilder* builder, tf_device::ParallelExecuteOp* parallel_execute_op) {
  const int num_cores_per_replica = execution_devices.front().size();
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

  *parallel_execute_op = builder->create<tf_device::ParallelExecuteOp>(
      cluster_func.getLoc(), num_cores_per_replica, concatenated_output_types);

  // Extract inputs for each region of the parallel_execute op. The i-th
  // element in the list represents the input lists to TPU computation for
  // i-th logical core.
  llvm::SmallVector<llvm::SmallVector<mlir::Value, 4>, 4> input_list;
  builder->setInsertionPoint(*parallel_execute_op);
  auto result = tensorflow::ExtractInputsForLogicalDevices(
      num_cores_per_replica, cluster_func, builder, &input_list);
  if (failed(result)) return failure();

  const bool replicated = execution_devices.size() != 1;
  // For each logical core, create a region with TPUExecute op.
  assert(input_list.size() == num_cores_per_replica);
  for (int core = 0; core < num_cores_per_replica; ++core) {
    auto& region = parallel_execute_op->GetRegionBlockWithIndex(core);
    builder->setInsertionPointToEnd(&region);

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
                             : execution_devices.front()[core];

    auto region_launch_op =
        WrapOpInLaunch(builder, region.getParent()->getLoc(), execute, device);

    builder->create<tf_device::ReturnOp>(region.getParent()->getLoc(),
                                         region_launch_op.getResults());
  }

  return success();
}

tf_device::LaunchOp AssignDevicesToReplicatedExecute(
    llvm::ArrayRef<llvm::SmallVector<std::string, 8>> execution_devices,
    Operation* execute_op, OpBuilder* builder) {
  const bool replicated = execution_devices.size() != 1;
  // If computation is replicated, use aliased device. Otherwise there is only
  // one execution device and the device is assigned to the execute op.
  std::string device = replicated ? tensorflow::GetDeviceAliasForLogicalCore(0)
                                  : execution_devices.front().front();

  return WrapOpInLaunch(builder, execute_op->getLoc(), execute_op, device);
}

// Creates a `tf.TPUCompileSucceededAssert` operation that parses compilation
// status of `compile_op` to check whether compilation is successful.
void BuildTPUCompileSucceededAssertOp(Operation* compile_op,
                                      llvm::StringRef compilation_device,
                                      OpBuilder* builder) {
  auto assert_op = builder->create<TF::TPUCompileSucceededAssertOp>(
      compile_op->getLoc(), compile_op->getResult(0));
  WrapOpInLaunch(builder, compile_op->getLoc(), assert_op, compilation_device);
}

// Rewrites a `tf_device.cluster_func` operation into a set of TPU Runtime
// Operations that jit-compiles and executes function in
// `tf_device.cluster_func` on TPU. Device assignment is determined from
// available devices in `devices`. If it is not possible to rewrite the
// operation or device assignment fails, a failure will be returned.
//
// For example, a non replicated `tf_device.cluster_func`:
//
// func @main(%arg0: tensor<i1>) {
//   %0 = "tf_device.cluster_func"(%arg0)
//          {_tpu_replicate = "cluster0", device = "", func = @_func} :
//          (tensor<i1>) -> tensor<i1>
//   return
// }
//
// will be rewritten as:
//
// func @main(%arg0: tensor<i1>) {
//   %0 = "tf.Shape"(%arg0) : (tensor<i1>) -> tensor<?xi32>
//   %1:2 = "tf._TPUCompileMlir"(%0) {device = "/CPU:0"} :
//            (tensor<?xi32>) -> (tensor<!tf.string>, tensor<!tf.string>)
//   %2 = "tf.TPUExecute"(%arg0, %1#0) {device = "/TPU:0"} :
//            (tensor<i1>, tensor<!tf.string>) -> tensor<i1>
//   return
// }
//
// and a replicated `tf_device.cluster_func`:
//
// func @main(%arg0: tensor<i1>, %arg1: tensor<i1>) {
//   %0:2 = tf_device.replicate([%arg0, %arg1] as %ri: tensor<i1>)
//                              {n = 2 : i32} {
//     %1 = "tf_device.cluster_func"(%ri)
//            {_tpu_replicate = "cluster0", device = "", func = @_func} :
//            (tensor<i1>) -> tensor<i1>
//     tf_device.return %1 : tensor<i1>
//   }
//   return
// }
//
// will be rewritten as:
//
// func @main(%arg0: tensor<i1>, %arg1: tensor<i1>) {
//   %0:2 = tf_device.replicate([%arg0, %arg1] as %ri: tensor<i1>)
//                              {n = 2 : i32, devices = ["/TPU:0", "/TPU:1"]} {
//     %1 = "tf.Shape"(%ri) : (tensor<i1>) -> tensor<?xi32>
//     %2:2 = "tf._TPUCompileMlir"(%1) {device = "/CPU:0"} :
//              (tensor<?xi32>) -> (tensor<!tf.string>, tensor<!tf.string>)
//     %3 = "tf.TPUExecute"(%ri, %2#0) :
//            (tensor<i1>, tensor<!tf.string>) -> tensor<i1>
//     tf_device.return %3 : tensor<i1>
//   }
//   return
// }
LogicalResult Rewrite(
    tf_device::ClusterFuncOp cluster_func,
    llvm::ArrayRef<tensorflow::DeviceNameUtils::ParsedName> devices,
    OpBuilder* builder) {
  // Skip non-tpu device cluster_func.
  auto replicate_attr =
      cluster_func.getAttrOfType<StringAttr>("_tpu_replicate");
  if (!replicate_attr) return success();

  // Collect `num_replicas` and `num_cores_per_replica` attributes.
  int num_replicas = 1;
  tf_device::ReplicateOp replicate =
      cluster_func.getParentOp()
          ? llvm::dyn_cast_or_null<tf_device::ReplicateOp>(
                cluster_func.getParentOp())
          : nullptr;
  if (replicate) num_replicas = replicate.n().getLimitedValue();

  auto num_cores_per_replica_attr =
      cluster_func.getAttrOfType<IntegerAttr>(kNumCoresPerReplicaAttr);
  if (!num_cores_per_replica_attr)
    return cluster_func.emitOpError(
        CreateMissingAttributeMsg(kNumCoresPerReplicaAttr));

  int num_cores_per_replica = num_cores_per_replica_attr.getInt();

  auto topology_attr = cluster_func.getAttrOfType<StringAttr>(kTopologyAttr);
  if (!topology_attr)
    return cluster_func.emitOpError(CreateMissingAttributeMsg(kTopologyAttr));

  llvm::SmallVector<int64_t, 6> device_assignment;
  if (failed(GetDeviceCoordinates(cluster_func, &device_assignment)))
    return failure();

  // Determine compilation and execution devices.
  auto status_or_tpu_device_assignment =
      tensorflow::GetTPUCompilationAndExecutionDevices(
          devices, num_replicas, num_cores_per_replica,
          topology_attr.getValue(), device_assignment);
  if (!status_or_tpu_device_assignment.ok())
    return cluster_func.emitError()
           << "error in fetching TPU compilation/execution devices: "
           << status_or_tpu_device_assignment.status().error_message();

  // Create compile op.
  auto& tpu_device_assignment = status_or_tpu_device_assignment.ValueOrDie();
  builder->setInsertionPoint(cluster_func);

  // Create the TPUCompileMlir and TPUCompileSucceededAssert outside of
  // parallel_execute region if it exists.
  if (llvm::isa<tf_device::ParallelExecuteOp>(cluster_func.getParentOp())) {
    // Currently, outside compilation and model parallelism are not supported
    // together.
    assert(num_cores_per_replica == 1);
    builder->setInsertionPoint(cluster_func.getParentOp());
  }

  Operation* compile_op = BuildCompileOp(
      cluster_func, num_replicas, num_cores_per_replica,
      tpu_device_assignment.compilation_device,
      std::move(tpu_device_assignment.xla_device_assignment), builder);
  if (!compile_op) return failure();

  // After rewrite, find if there is a TPUCompilationResultOp in the block with
  // the same _tpu_replicate attribute and replace it with the result of the
  // compile op. This op is used as a placeholder to hook during graph creation
  // the other ops that are intended to consume the compile result.
  Block* block = cluster_func.getOperation()->getBlock();
  for (auto compile_result_op : block->getOps<TF::TPUCompilationResultOp>())
    compile_result_op.output().replaceAllUsesWith(compile_op->getResult(0));

  BuildTPUCompileSucceededAssertOp(
      compile_op, tpu_device_assignment.compilation_device, builder);

  AssignDevicesToReplicate(replicate, tpu_device_assignment.execution_devices,
                           builder);

  llvm::SmallVector<xla::OpSharding, 4> output_shardings;
  auto result = tensorflow::ParseAndValidateOutputSharding(
      num_cores_per_replica, cluster_func, &output_shardings);
  if (failed(result)) return failure();

  builder->setInsertionPoint(cluster_func);
  if (num_cores_per_replica > 1) {
    // For model parallelism, tf_device.parallel_execute is used to express
    // concurrent device execution across multiple logical devices.

    tf_device::ParallelExecuteOp execute_op;
    result = BuildParallelExecuteOp(tpu_device_assignment.execution_devices,
                                    output_shardings, compile_op, cluster_func,
                                    builder, &execute_op);
    if (failed(result)) return failure();

    // As tf_device.parallel_execute wraps # logical cores number of TPUExecute
    // ops, the number of return values of parallel_execute op exceeds that of
    // cluster_func op. As so, each return value of parallel_execute op must be
    // mapped with corresponding return value usages of cluster_func.
    tensorflow::RemapOutputsFromLogicalDevices(cluster_func.getLoc(),
                                               output_shardings, cluster_func,
                                               execute_op, builder);
  } else {
    llvm::SmallVector<Value, 4> execute_inputs(cluster_func.getOperands());
    execute_inputs.emplace_back(compile_op->getResult(1));

    TF::TPUExecuteOp execute_op;
    result = BuildExecuteOp(
        /*core_id=*/0, output_shardings, execute_inputs, cluster_func, builder,
        &execute_op);
    if (failed(result)) return failure();

    tf_device::LaunchOp launch_op = AssignDevicesToReplicatedExecute(
        tpu_device_assignment.execution_devices, execute_op, builder);
    cluster_func.replaceAllUsesWith(launch_op);
  }

  cluster_func.erase();

  return success();
}

void TPURewritePass::runOnOperation() {
  mlir::TF::RuntimeDevices devices;
  if (failed(tensorflow::GetDevicesFromOp(getOperation(), &devices)))
    return signalPassFailure();

  OpBuilder builder(&getContext());
  auto result = getOperation().walk([&](tf_device::ClusterFuncOp op) {
    if (failed(Rewrite(op, devices.device_names(), &builder)))
      return WalkResult::interrupt();

    return WalkResult::advance();
  });

  if (result.wasInterrupted()) return signalPassFailure();

  // Eliminate TPUCompilationResultOp now that the rewrite is complete.
  getOperation().walk([&](TF::TPUCompilationResultOp op) { op.erase(); });

  // TODO(b/139377366): Remove functions that are no longer needed.
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateTPURewritePass() {
  return std::make_unique<TPURewritePass>();
}

static PassRegistration<TPURewritePass> pass(
    "tf-tpu-rewrite",
    "Rewriting `tf_device.cluster_func` on TPUs into TPU runtime ops");

}  // namespace TFTPU
}  // namespace mlir

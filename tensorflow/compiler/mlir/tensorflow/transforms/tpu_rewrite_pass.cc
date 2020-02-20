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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Module.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/IR/Types.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Pass/PassRegistry.h"  // TF:llvm-project
#include "mlir/Support/LogicalResult.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"
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
constexpr char kDeviceAttr[] = "device";
constexpr char kDevicesAttr[] = "devices";
constexpr char kVersionsAttr[] = "tf.versions";

// Rewrites `tf_device.launch_func` operations assigned to TPU into actual TPU
// jit-compile runtime ops.
//
// For example:
//   %1 = "tf_device.launch_func"(%0) {_tpu_replicate = "cluster", func =
//         @tpu_func}
//   %2 = "tf.SomeOp"(%1)
//
// Would become following ops (unimportant attributes, types are omitted):
//    %1 = "tf.Shape"(%0)
//    %2:2 = "tf.MLIRCompileToTPU"(%1) {module = "<Serialized @tpu_func>"}
//    "tf.TPUCompileSucceededAssert"(%2#0)
//    %3 = "tf.TPUExecute"(%0, %2#1)
//    %4 = "tf.SomeOp"(%3)

namespace {
struct TPURewritePass : public ModulePass<TPURewritePass> {
  void runOnModule() override;
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

// Populates a TPUCompileMetadataProto from attributes of a
// `tf_device::LaunchFuncOp`. If any necessary attributes are missing from the
// op, a failure will be returned.
// TODO(lyandy): Support session handle and guaranteed consts.
LogicalResult SetMetadataProtoFromLaunchFuncOp(
    tf_device::LaunchFuncOp op, int num_replicas, int num_cores_per_replica,
    tensorflow::tpu::TPUCompileMetadataProto* metadata) {
  metadata->set_num_replicas(num_replicas);
  metadata->set_num_cores_per_replica(num_cores_per_replica);

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

  auto padding_map = op.getAttrOfType<ArrayAttr>(kPaddingMapAttr);
  if (!padding_map)
    return op.emitOpError(CreateMissingAttributeMsg(kPaddingMapAttr));

  for (const auto padding_and_idx : llvm::enumerate(padding_map)) {
    auto& padding_attr = padding_and_idx.value();
    auto padding_attr_str = padding_attr.dyn_cast<StringAttr>();
    if (!padding_attr_str)
      return op.emitOpError(
          llvm::formatv("bad '{0}' attribute at index {1}, not a string",
                        kPaddingMapAttr, padding_and_idx.index()));

    tensorflow::tpu::PaddingMap* padding =
        metadata->mutable_padding_maps()->Add();
    if (!padding->ParseFromString(std::string(padding_attr_str.getValue())))
      return op.emitOpError(llvm::formatv(
          "bad '{0}' attribute at index {1} with value '{2}'", kPaddingMapAttr,
          padding_and_idx.index(), padding_attr_str.getValue()));
  }

  // Set args metadata in proto.
  for (auto operand_type_and_idx : llvm::enumerate(op.getOperandTypes())) {
    Type operand_type = operand_type_and_idx.value();
    tensorflow::tpu::TPUCompileMetadataProto::Arg* arg = metadata->add_args();
    tensorflow::DataType dtype;
    tensorflow::Status status =
        tensorflow::ConvertToDataType(operand_type, &dtype);
    if (!status.ok())
      return op.emitOpError(
          llvm::formatv("failed to determine operand type at index {0}: {1}",
                        operand_type_and_idx.index(), status.error_message()));

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

    // TODO(lyandy): Determine proper sharding of args once topology and devices
    // are propagated to the pass.
    xla::OpSharding sharding;
    sharding.set_type(xla::OpSharding::MAXIMAL);
    sharding.add_tile_assignment_dimensions(1);
    sharding.add_tile_assignment_devices(0);
    *arg->mutable_sharding() = std::move(sharding);
  }

  // Set retvals metadata in proto.
  // TODO(lyandy): Determine proper sharding of retvals once topology and
  // devices is propagated to the pass.
  for (int i = 0; i < op.getNumResults(); ++i) {
    xla::OpSharding sharding;
    sharding.set_type(xla::OpSharding::MAXIMAL);
    sharding.add_tile_assignment_dimensions(1);
    sharding.add_tile_assignment_devices(0);
    *metadata->add_retvals()->mutable_sharding() = std::move(sharding);
  }

  return success();
}

// Create a `tf._TPUCompileMlir` that contains a MLIR module that is
// functionally equivalent to the function referenced by launch_func.
Operation* BuildCompileOp(tf_device::LaunchFuncOp launch_func, int num_replicas,
                          int num_cores_per_replica,
                          llvm::StringRef compilation_device,
                          OpBuilder* builder) {
  // TODO(b/139377366): Use tf_tpu.compile build method when it is defined.
  OperationState compile_op_state(launch_func.getLoc(), "tf._TPUCompileMlir");

  // Set metadata from attributes.
  tensorflow::tpu::TPUCompileMetadataProto metadata;
  if (failed(SetMetadataProtoFromLaunchFuncOp(
          launch_func, num_replicas, num_cores_per_replica, &metadata)))
    return nullptr;

  std::string txt_metadata;
  if (tpu_compile_metadata_debug)
    txt_metadata = metadata.DebugString();
  else
    metadata.SerializeToString(&txt_metadata);

  compile_op_state.addAttribute("metadata",
                                builder->getStringAttr(txt_metadata));

  // Build a shape op for each input to launch_func.
  // TODO(b/139377366): When shape inference is ready, we can use compile time
  // shape inference to get inputs that have static shapes and only use shape
  // ops for the rest.
  llvm::SmallVector<Value, 4> compile_op_operands;
  compile_op_operands.reserve(launch_func.getNumOperands());

  for (auto operand_and_idx : llvm::enumerate(launch_func.getOperands())) {
    // Skip adding shape op for operands that have static shapes.
    tensorflow::PartialTensorShape shape(
        metadata.args(operand_and_idx.index()).shape());
    if (shape.IsFullyDefined()) continue;

    auto shape_op = builder->create<TF::ShapeOp>(
        launch_func.getLoc(),
        RankedTensorType::get({-1}, builder->getIntegerType(64)),
        operand_and_idx.value());
    compile_op_operands.emplace_back(shape_op.getResult());
  }
  compile_op_state.addOperands(compile_op_operands);
  compile_op_state.addAttribute(
      "NumDynamicShapes",
      builder->getI64IntegerAttr(compile_op_operands.size()));

  FlatSymbolRefAttr func_attr =
      launch_func.getAttrOfType<FlatSymbolRefAttr>("func");
  if (!func_attr) {
    launch_func.emitOpError("does not have `func` attribute");
    return nullptr;
  }
  FuncOp func = launch_func.getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(
      func_attr.getValue());

  std::string txt_module;
  if (failed(EncapsulateFuncAndSerialize(func, &txt_module))) return nullptr;
  compile_op_state.addAttribute("mlir_module",
                                builder->getStringAttr(txt_module));

  compile_op_state.addAttribute(kDeviceAttr,
                                builder->getStringAttr(compilation_device));

  // Result #0 is a string indicating whether compilation is successful or not.
  compile_op_state.addTypes(
      RankedTensorType::get({}, builder->getType<TF::StringType>()));

  // Result #1 is key to look up executable binary in compilation cache.
  compile_op_state.addTypes(
      RankedTensorType::get({}, builder->getType<TF::StringType>()));

  return builder->createOperation(compile_op_state);
}

// Creates a `tf.TPUExecute` op that executes TPU program generated by
// `compile_op`.
Operation* BuildExecuteOp(Operation* compile_op,
                          tf_device::LaunchFuncOp launch_func,
                          OpBuilder* builder) {
  // TPUExecute inherits all launch_func inputs, and takes an additional input
  // for compilation cache key.
  llvm::SmallVector<Value, 4> tensor_inputs(launch_func.getOperands());
  tensor_inputs.push_back(compile_op->getResult(1));

  // TODO(b/139377366): Need to snapshot all resource variable inputs in
  // follow-up CLs.

  // TPUExecute has same output types as launch_func.
  return builder->create<TF::TPUExecuteOp>(
      launch_func.getLoc(), launch_func.getResultTypes(), tensor_inputs,
      llvm::ArrayRef<NamedAttribute>{});
}

// Creates a tf_device.parallel_execute op that wraps TPUExecute op to
// represent execution of TPU program in multiple logical cores.
Operation* BuildParallelExecuteOp(int num_logical_cores, Operation* compile_op,
                                  tf_device::LaunchFuncOp launch_func,
                                  OpBuilder* builder) {
  // parallel_execute op returns concatenated list of return values of
  // all its regions.
  //
  // TODO(b/149102702): Correctly map inputs to parallel_execute op via
  // identifying xla_sharding op in the launch_func function.
  const auto& launch_result_types = launch_func.getResultTypes();
  llvm::SmallVector<Type, 8> concatenated_output_types;
  concatenated_output_types.reserve(launch_result_types.size() *
                                    num_logical_cores);

  for (int core_id = 0; core_id < num_logical_cores; ++core_id)
    for (Type t : launch_result_types)
      concatenated_output_types.emplace_back(t);

  auto parallel_execute_op = builder->create<tf_device::ParallelExecuteOp>(
      launch_func.getLoc(), num_logical_cores, concatenated_output_types);

  // For each logical core, create a region with TPUExecute op.
  for (int core_id = 0; core_id < num_logical_cores; ++core_id) {
    auto& region = parallel_execute_op.GetRegionBlockWithIndex(core_id);

    // Create Execute op.
    //
    // TODO(b/148913294): Identify inputs/return values specific to each
    // logical core TPU execution by parsing xla_sharding op in
    // launch_func.
    auto execute = BuildExecuteOp(compile_op, launch_func, builder);

    // Create a launch op for each region of parallel_execute.
    //
    // TODO(b/149102679): Add device attribute to launch op once device
    // topology for multiple logical cores can be correctly parsed.
    builder->setInsertionPointToStart(&region);
    auto region_launch_op = builder->create<tf_device::LaunchOp>(
        region.getParent()->getLoc(), builder->getStringAttr(""),
        launch_func.results().getTypes());
    region_launch_op.body().push_back(new Block);

    builder->setInsertionPointToEnd(&region_launch_op.GetBody());
    builder->create<tf_device::ReturnOp>(region_launch_op.getLoc(),
                                         execute->getResults());

    // Move execute inside the launch op.
    execute->moveBefore(&region_launch_op.GetBody().back());

    builder->setInsertionPointToEnd(&region);
    builder->create<tf_device::ReturnOp>(region.getParent()->getLoc(),
                                         region_launch_op.getResults());
  }

  return parallel_execute_op;
}

// As tf_device.parallel_execute wraps # logical cores number of TPUExecute
// ops, the number of return values of parallel_execute op exceeds that of
// launch_func op. As so, each return value of parallel_execute op must be
// mapped with corresponding return value usages of launch_func.
//
// TODO(b/148913294): Once argument and return value sharding of tpu computation
// is determined, correctly map outputs of parallel_execute op.
void RemapOutputsOfParallelExecute(tf_device::LaunchFuncOp launch_func,
                                   Operation* op) {
  for (auto outputs : llvm::zip(launch_func.getResults(), op->getResults()))
    std::get<0>(outputs).replaceAllUsesWith(std::get<1>(outputs));
}

void AssignDevicesToReplicatedExecute(
    const llvm::SmallVector<std::string, 8>& execution_devices,
    tf_device::ReplicateOp replicate, Operation* execute_op,
    OpBuilder* builder) {
  // If computation is replicated, execution devices are assigned to the
  // replicate. Otherwise there is only one execution device and the device is
  // assigned to the execute op.
  if (replicate) {
    // Model parallelism is not support for now. Therefore, assign all ops
    // in replicate op with virtual device alias specifying that ops will be
    // executed on the zeroth core.
    auto device_attr = builder->getNamedAttr(
        tensorflow::GetDeviceAliasForLogicalCore(0),
        builder->getStrArrayAttr(llvm::SmallVector<llvm::StringRef, 4>{
            execution_devices.begin(), execution_devices.end()}));
    replicate.setAttr(kDevicesAttr, builder->getDictionaryAttr(device_attr));
  } else {
    execute_op->setAttr(kDeviceAttr,
                        builder->getStringAttr(execution_devices.front()));
  }
}

// Creates a `tf.TPUCompileSucceededAssert` operation that parses compilation
// status of `compile_op` to check whether compilation is successful.
void BuildTPUCompileSucceededAssertOp(Operation* compile_op,
                                      OpBuilder* builder) {
  OperationState assert_op_state(compile_op->getLoc(),
                                 "tf.TPUCompileSucceededAssert");
  assert_op_state.addOperands(compile_op->getResult(0));
  builder->createOperation(assert_op_state);
}

// Rewrites a `tf_device.launch_func` operation into a set of TPU Runtime
// Operations that jit-compiles and executes function in `tf_device.launch_func`
// on TPU. Device assignment is determined from available devices in `devices`.
// If it is not possible to rewrite the operation or device assignment fails, a
// failure will be returned.
//
// For example, a non replicated `tf_device.launch_func`:
//
// func @main(%arg0: tensor<i1>) {
//   %0 = "tf_device.launch_func"(%arg0)
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
// and a replicated `tf_device.launch_func`:
//
// func @main(%arg0: tensor<i1>, %arg1: tensor<i1>) {
//   %0:2 = tf_device.replicate([%arg0, %arg1] as %ri: tensor<i1>)
//                              {n = 2 : i32} {
//     %1 = "tf_device.launch_func"(%ri)
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
    tf_device::LaunchFuncOp launch_func,
    llvm::ArrayRef<tensorflow::DeviceNameUtils::ParsedName> devices,
    OpBuilder* builder) {
  // Skip non-tpu device launch_func.
  auto replicate_attr = launch_func.getAttrOfType<StringAttr>("_tpu_replicate");
  if (!replicate_attr) return success();

  // Collect `num_replicas` and `num_cores_per_replica` attributes.
  int num_replicas = 1;
  tf_device::ReplicateOp replicate =
      launch_func.getParentOp()
          ? llvm::dyn_cast_or_null<tf_device::ReplicateOp>(
                launch_func.getParentOp())
          : nullptr;
  if (replicate) num_replicas = replicate.n().getLimitedValue();

  auto num_cores_per_replica_attr =
      launch_func.getAttrOfType<IntegerAttr>(kNumCoresPerReplicaAttr);
  if (!num_cores_per_replica_attr)
    return launch_func.emitOpError(
        CreateMissingAttributeMsg(kNumCoresPerReplicaAttr));

  int num_cores_per_replica = num_cores_per_replica_attr.getInt();

  // Determine compilation and execution devices.
  std::string compilation_device;
  llvm::SmallVector<std::string, 8> execution_devices;
  auto status = tensorflow::GetTPUCompilationAndExecutionDevices(
      devices, num_replicas, num_cores_per_replica, &compilation_device,
      &execution_devices);
  if (!status.ok())
    return launch_func.emitError()
           << "error in fetching TPU compilation/execution devices: "
           << status.error_message();

  // Create compile op;
  builder->setInsertionPoint(launch_func);
  Operation* compile_op =
      BuildCompileOp(launch_func, num_replicas, num_cores_per_replica,
                     compilation_device, builder);
  if (!compile_op) return failure();

  // After rewrite, find if there is a TPUCompilationResultOp in the block with
  // the same _tpu_replicate attribute and replace it with the result of the
  // compile op. This op is used as a placeholder to hook during graph creation
  // the other ops that are intended to consume the compile result.
  Block* block = launch_func.getOperation()->getBlock();
  for (auto compile_result_op : block->getOps<TF::TPUCompilationResultOp>())
    compile_result_op.output().replaceAllUsesWith(compile_op->getResult(0));

  BuildTPUCompileSucceededAssertOp(compile_op, builder);

  Operation* execute_op;
  if (num_cores_per_replica > 1) {
    // For model parallelism, tf_device.parallel_execute is used to express
    // concurrent device execution across multiple logical devices.
    execute_op = BuildParallelExecuteOp(num_cores_per_replica, compile_op,
                                        launch_func, builder);

    RemapOutputsOfParallelExecute(launch_func, execute_op);

    // TODO(hongjunchoi): Correctly parse TPU topology and assign logical device
    // attributes to launch_op's within parallel_execute op.
  } else {
    execute_op = BuildExecuteOp(compile_op, launch_func, builder);
    AssignDevicesToReplicatedExecute(execution_devices, replicate, execute_op,
                                     builder);
    launch_func.replaceAllUsesWith(execute_op);
  }

  launch_func.erase();

  return success();
}

void TPURewritePass::runOnModule() {
  llvm::SmallVector<tensorflow::DeviceNameUtils::ParsedName, 8> devices;
  if (failed(tensorflow::GetDevicesFromOp(getModule(), &devices)))
    return signalPassFailure();

  OpBuilder builder(&getContext());
  auto result = getModule().walk([&](tf_device::LaunchFuncOp op) {
    if (failed(Rewrite(op, devices, &builder))) return WalkResult::interrupt();

    return WalkResult::advance();
  });

  if (result.wasInterrupted()) return signalPassFailure();

  // Eliminate TPUCompilationResultOp now that the rewrite is complete.
  getModule().walk([&](TF::TPUCompilationResultOp op) { op.erase(); });

  // TODO(b/139377366): Remove functions that are no longer needed.
}

}  // namespace

std::unique_ptr<OpPassBase<ModuleOp>> CreateTPURewritePass() {
  return std::make_unique<TPURewritePass>();
}

static PassRegistration<TPURewritePass> pass(
    "tf-tpu-rewrite",
    "Rewriting `tf_device.launch_func` on TPUs into TPU runtime ops");

}  // namespace TFTPU
}  // namespace mlir

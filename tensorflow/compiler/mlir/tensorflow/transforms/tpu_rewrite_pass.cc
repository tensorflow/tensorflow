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

#include <string>
#include <type_traits>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassRegistry.h"  // TF:local_config_mlir
#include "mlir/Support/LogicalResult.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/protobuf/tpu/dynamic_padding.pb.h"

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

// Recursively visits all attributes of `op` to find any Attribute of type
// `SymbolRefAttr`.
llvm::SmallVector<SymbolRefAttr, 8> GetAllSymbolRefAttrs(Operation* op) {
  llvm::SmallVector<SymbolRefAttr, 8> symbol_ref_attrs;

  llvm::SmallVector<Attribute, 8> worklist;
  for (auto named_attr : op->getAttrs()) {
    worklist.push_back(named_attr.second);
  }

  while (!worklist.empty()) {
    Attribute attr = worklist.pop_back_val();

    if (SymbolRefAttr symbol_ref_attr = attr.dyn_cast<SymbolRefAttr>()) {
      // Found a SymbolRefAttr, add it to result list.
      symbol_ref_attrs.push_back(symbol_ref_attr);
    } else if (ArrayAttr array_attr = attr.dyn_cast<ArrayAttr>()) {
      // Found an ArrayAttr, add its nested Attributes to worklist for further
      // inspection.
      worklist.append(array_attr.begin(), array_attr.end());
    } else if (DictionaryAttr dict_attr = attr.dyn_cast<DictionaryAttr>()) {
      // Found a DictionaryAttr, add its nested value Attributes to worklist for
      // further inspection.
      for (NamedAttribute named_attr : dict_attr.getValue()) {
        worklist.push_back(named_attr.second);
      }
    }
  }

  return symbol_ref_attrs;
}

// Creates a new self-contained module that contains `entry_func` and all
// referenced functions in `entry_func`. entry_func is renamed to "main".
// Return value is serialized text formate of newly-created module.
std::string EncapsulateFuncAndSerialize(FuncOp entry_func) {
  ModuleOp module = entry_func.getParentOfType<ModuleOp>();
  llvm::SmallVector<FuncOp, 4> referenced({entry_func});

  // Create a new module to hold func and all referenced functions.
  OwningModuleRef module_for_func =
      ModuleOp::create(mlir::UnknownLoc::get(entry_func.getContext()));
  ModuleManager module_manager(module_for_func.get());

  while (!referenced.empty()) {
    auto func = referenced.pop_back_val();

    // Skip functions that have already been cloned into new module.
    if (module_manager.lookupSymbol<FuncOp>(func.getName())) continue;

    // Find any SymbolRefAttr in func that maps to a FuncOp. We need to clone
    // all found FuncOps to new_module to make sure new_module is
    // self-contained.
    func.walk([&](Operation* op) {
      for (auto symbol_ref_attr : GetAllSymbolRefAttrs(op)) {
        FuncOp referenced_func =
            module.lookupSymbol<FuncOp>(symbol_ref_attr.getValue());

        // Skip Symbols that do not map to a function.
        if (!referenced_func) continue;

        referenced.emplace_back(referenced_func);
      }
    });

    auto clone = func.clone();
    if (clone.getName() == entry_func.getName()) {
      // We can simply change name of TPU program's main function because there
      // should be no other reference to it.
      clone.setName("main");
    }
    module_manager.insert(clone);
  }

  // Serialize module and return.
  std::string txt_module;
  {
    llvm::raw_string_ostream os(txt_module);
    module_for_func.get().print(os);
  }
  return txt_module;
}

// Creates a missing attribute error message.
std::string CreateMissingAttributeMsg(llvm::StringRef attribute) {
  return llvm::formatv("requires attribute '{0}'", attribute).str();
}

// Populates a TPUCompileMetadataProto from attributes of a
// `tf_device::LaunchFuncOp`. If any necessary attributes are missing from the
// op, a failure will be returned.
// TODO(lyandy): Propagate and support device assignment.
// TODO(lyandy): Support session handle and guaranteed consts.
LogicalResult SetMetadataProtoFromLaunchFuncOp(
    tf_device::LaunchFuncOp op,
    tensorflow::tpu::TPUCompileMetadataProto* metadata) {
  auto num_replicas = op.getAttrOfType<IntegerAttr>(kNumReplicasAttr);
  if (!num_replicas)
    return op.emitOpError(CreateMissingAttributeMsg(kNumReplicasAttr));

  metadata->set_num_replicas(num_replicas.getInt());

  auto num_cores_per_replica =
      op.getAttrOfType<IntegerAttr>(kNumCoresPerReplicaAttr);
  if (!num_cores_per_replica)
    return op.emitOpError(CreateMissingAttributeMsg(kNumCoresPerReplicaAttr));

  metadata->set_num_cores_per_replica(num_cores_per_replica.getInt());

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
          step_marker_location.getValue(), &location))
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
    if (!padding->ParseFromString(padding_attr_str.getValue()))
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
Operation* BuildCompileOp(tf_device::LaunchFuncOp launch_func,
                          OpBuilder* builder) {
  // TODO(b/139377366): Use tf_tpu.compile build method when it is defined.
  OperationState compile_op_state(launch_func.getLoc(), "tf._TPUCompileMlir");

  // Set metadata from attributes.
  tensorflow::tpu::TPUCompileMetadataProto metadata;
  if (failed(SetMetadataProtoFromLaunchFuncOp(launch_func, &metadata)))
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
  llvm::SmallVector<Value*, 4> compile_op_operands;
  compile_op_operands.reserve(launch_func.getNumOperands());

  for (auto operand_and_idx : llvm::enumerate(launch_func.getOperands())) {
    // Skip adding shape op for operands that have static shapes.
    tensorflow::PartialTensorShape shape(
        metadata.args(operand_and_idx.index()).shape());
    if (shape.IsFullyDefined()) continue;

    auto shape_op = builder->create<TF::ShapeOp>(
        launch_func.getLoc(),
        builder->getTensorType({-1}, builder->getIntegerType(64)),
        operand_and_idx.value());
    compile_op_operands.emplace_back(shape_op.getResult());
  }
  compile_op_state.addOperands(compile_op_operands);
  compile_op_state.addAttribute(
      "NumDynamicShapes",
      builder->getI64IntegerAttr(compile_op_operands.size()));

  SymbolRefAttr func_attr = launch_func.getAttrOfType<SymbolRefAttr>("func");
  if (!func_attr) {
    launch_func.emitOpError("does not have `func` attribute");
    return nullptr;
  }
  FuncOp func = launch_func.getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(
      func_attr.getValue());

  std::string txt_module = EncapsulateFuncAndSerialize(func);
  compile_op_state.addAttribute("mlir_module",
                                builder->getStringAttr(txt_module));

  // Result #0 is a string indicating whether compilation is successful or not.
  compile_op_state.addTypes(
      builder->getTensorType({}, builder->getType<TF::StringType>()));

  // Result #1 is key to look up executable binary in compilation cache.
  compile_op_state.addTypes(
      builder->getTensorType({}, builder->getType<TF::StringType>()));

  return builder->createOperation(compile_op_state);
}

// Creates a `tf.TPUExecute` op that executes TPU program generated by
// `compile_op`.
Operation* BuildExecuteOp(Operation* compile_op,
                          tf_device::LaunchFuncOp launch_func,
                          OpBuilder* builder) {
  // TODO(b/139377366): Use tf.TPUExecute build method when it is defined.
  OperationState execute_op_state(launch_func.getLoc(), "tf.TPUExecute");

  // TPUExecute inherits all launch_func inputs.
  llvm::SmallVector<Value*, 4> tensor_inputs(launch_func.getOperands());
  execute_op_state.addOperands(tensor_inputs);

  // TODO(b/139377366): Need to snapshot all resource variable inputs in
  // follow-up CLs.

  // Set Targs of TPUExecute according to launch_func input types.
  llvm::SmallVector<Attribute, 4> tensor_input_types_attrs;
  tensor_input_types_attrs.reserve(tensor_inputs.size());
  for (Value* v : tensor_inputs) {
    tensor_input_types_attrs.emplace_back(builder->getTypeAttr(v->getType()));
  }
  execute_op_state.addAttribute(
      "Targs", builder->getArrayAttr(tensor_input_types_attrs));

  // TPUExecute takes an additional input for compilation cache key.
  execute_op_state.addOperands(compile_op->getResult(1));

  // Set Tresults of TPUExecute according to launch_func results types.
  llvm::SmallVector<Attribute, 4> output_types_attrs;
  output_types_attrs.reserve(launch_func.getNumResults());
  for (Value* v : launch_func.getResults()) {
    output_types_attrs.emplace_back(builder->getTypeAttr(v->getType()));
  }
  execute_op_state.addAttribute("Tresults",
                                builder->getArrayAttr(output_types_attrs));

  // TPUExecute has same output types as launch_func.
  llvm::SmallVector<Type, 4> output_types(launch_func.getResultTypes());
  execute_op_state.addTypes(output_types);

  return builder->createOperation(execute_op_state);
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
// on TPU. If it is not possible to rewrite the operation, a failure will be
// returned.
LogicalResult Rewrite(tf_device::LaunchFuncOp launch_func, OpBuilder* builder) {
  // Skip non-tpu device launch_func.
  auto replicate_attr = launch_func.getAttrOfType<StringAttr>("_tpu_replicate");
  if (!replicate_attr) return success();

  builder->setInsertionPoint(launch_func);
  Operation* compile_op = BuildCompileOp(launch_func, builder);
  if (!compile_op) return failure();

  // After rewrite, find if there is a TPUCompilationResultOp in the block with
  // the same _tpu_replicate attribute and replace it with the result of the
  // compile op. This op is used as a placeholder to hook during graph creation
  // the other ops that are intended to consume the compile result.
  Block* block = launch_func.getOperation()->getBlock();
  for (auto compile_result_op : block->getOps<TF::TPUCompilationResultOp>())
    compile_result_op.output()->replaceAllUsesWith(compile_op->getResult(0));

  BuildTPUCompileSucceededAssertOp(compile_op, builder);
  // TODO(ycao): Right now we only support single-core case. The right thing to
  // do is to read from launch_func attributes to determine how many execute
  // ops to build.
  Operation* execute_op = BuildExecuteOp(compile_op, launch_func, builder);
  launch_func.replaceAllUsesWith(execute_op);
  launch_func.erase();

  return success();
}

void TPURewritePass::runOnModule() {
  OpBuilder builder(&getContext());
  auto result = getModule().walk([&](tf_device::LaunchFuncOp op) {
    if (failed(Rewrite(op, &builder))) return WalkResult::interrupt();

    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    signalPassFailure();
    return;
  }

  // Eliminate TPUReplicatedInput and TPUReplicatedOutput now that the rewrite
  // is complete.
  getModule().walk([&](Operation* op) {
    auto op_name = op->getName().getStringRef();
    if (op_name != "tf.TPUReplicatedInput" &&
        op_name != "tf.TPUReplicatedOutput")
      return;
    op->getResult(0)->replaceAllUsesWith(op->getOperand(0));
    op->erase();
  });

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

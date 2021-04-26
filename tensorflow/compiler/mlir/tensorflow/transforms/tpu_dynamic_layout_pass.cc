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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/xla_sharding_util.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace mlir {
namespace TFTPU {

namespace {

constexpr char kDeviceAttr[] = "device";
constexpr char kDeviceCPU[] = "CPU";
constexpr char kFuncDeviceAttr[] = "tf.device";

// A pass that allows TPU input layout to be determined after JIT compilation.
// This is done by adding run-time ops that interpret compilation result and
// copy the input to device with that layout.
//
// Example: original program:
//
//   %input = "tf.IteratorGetNext"(...) {device = "/CPU:0"}
//   %compile:2 = "tf._TPUCompileMlir"(...)
//   %execute = "tf.TPUExecute"(%input, ..., %compile#1) {device = "/TPU:0"}
//
// Without this pass, later TF graph partitioning passes will insert send/recv
// between %input and %execute and data will be copied to device in a fixed
// layout. With this pass, the program will be transformed into:
//
//   %input = "tf.IteratorGetNext"(...) {device = "/CPU:0"}
//   %compile:2 = "tf._TPUCompileMlir"(...)
//   %get_layout = "tf.TPUGetLayoutOp"(%compile#1) {...}
//   %copy_to_device = "tf.TPUCopyWithLayout"(%input, %get_layout)
//       {device = "/TPU:0"}
//   %execute = "tf.TPUExecute"(%copy_to_device, ..., %compile#1)
//       {device = "/TPU:0"}
//
// This way, %compile will determine the layout, which will be respected by
// %copy_to_device. There will not be send/recv ops added by later passes,
// because tf.TPUCopyWithLayout accepts a host input and produces a device
// output.
struct TPUDynamicLayoutPass
    : public TF::PerFunctionAggregateAnalysisConsumerPass<
          TPUDynamicLayoutPass, TF::ResourceAliasAnalysis> {
  void runOnFunction(
      FuncOp func,
      const TF::ResourceAliasAnalysis::Info& resource_alias_analysis);
};

// Checks if the input producer op is supported in this transform. Right now, we
// only check if it is a tf.IteratorGetNext where resource input is coming from
// a VarHandle on CPU or a function argument assigned to CPU.
bool IsSupportedInputOp(
    Operation* op,
    const TF::ResourceAliasAnalysis::Info& resource_alias_analysis) {
  TF::IteratorGetNextOp iterator_op = llvm::dyn_cast<TF::IteratorGetNextOp>(op);
  if (!iterator_op) return false;

  Value resource_iterator = iterator_op.iterator();

  if (resource_alias_analysis.IsUnknownResource(resource_iterator))
    return false;
  llvm::SmallSetVector<Value, 8> aliases =
      resource_alias_analysis.GetResourceAliases(resource_iterator);

  auto is_generator = [](Value val) {
    if (val.isa<BlockArgument>()) return true;
    Operation* definition = val.getDefiningOp();
    return definition->getNumOperands() == 0 &&
           definition->getNumResults() == 1;
  };

  // Check all generator aliases (ops or function argument) are on CPU.
  FuncOp func = iterator_op->getParentOfType<FuncOp>();
  return llvm::all_of(aliases, [&](Value alias) {
    // Ignore non-generator aliases.
    if (!is_generator(alias)) return true;

    StringAttr device;
    if (auto arg = alias.dyn_cast<BlockArgument>()) {
      device = func.getArgAttrOfType<mlir::StringAttr>(arg.getArgNumber(),
                                                       kFuncDeviceAttr);
    } else {
      device = alias.getDefiningOp()->getAttrOfType<StringAttr>(kDeviceAttr);
    }

    if (!device) return false;
    tensorflow::DeviceNameUtils::ParsedName parsed_device;
    if (!tensorflow::DeviceNameUtils::ParseFullName(device.getValue().str(),
                                                    &parsed_device)) {
      return false;
    }
    return parsed_device.has_type && parsed_device.type == kDeviceCPU;
  });
}

OpBuilder CreateBuilderAfterOp(Operation* op) {
  return OpBuilder(op->getBlock(), ++Block::iterator(op));
}

// Builds a TPUGetLayoutOp with the given compile op and input index.
TF::TPUGetLayoutOp BuildGetLayout(const int64_t execute_arg_index,
                                  Value compilation_key,
                                  tf_device::LaunchOp compile_launch,
                                  OpBuilder* builder) {
  return builder->create<TF::TPUGetLayoutOp>(
      compile_launch.getLoc(),
      llvm::ArrayRef<Type>{RankedTensorType::get({ShapedType::kDynamicSize},
                                                 builder->getIntegerType(64))},
      llvm::ArrayRef<Value>{compilation_key},
      llvm::ArrayRef<NamedAttribute>{
          builder->getNamedAttr("index",
                                builder->getI64IntegerAttr(execute_arg_index)),
          builder->getNamedAttr("is_output", builder->getBoolAttr(false))});
}

// Builds a TPUCopyWithLayoutOp with the given get_layout op and input.
TF::TPUCopyWithLayoutOp BuildCopyWithLayout(tf_device::LaunchOp execute_launch,
                                            tf_device::LaunchOp compile_launch,
                                            TF::TPUGetLayoutOp get_layout,
                                            Value input, OpBuilder* builder) {
  return builder->create<TF::TPUCopyWithLayoutOp>(
      execute_launch.getLoc(), llvm::ArrayRef<Type>{input.getType()},
      llvm::ArrayRef<Value>{input, get_layout.layout()});
}

// Performs transformation for a non-replicated input.
void HandleInput(Value input, const int64_t execute_arg_index,
                 TF::TPUExecuteOp execute, tf_device::LaunchOp execute_launch,
                 tf_device::LaunchOp compile_launch) {
  OpBuilder builder = CreateBuilderAfterOp(compile_launch);
  auto get_layout = BuildGetLayout(execute_arg_index, execute.key(),
                                   compile_launch, &builder);
  builder.setInsertionPoint(execute_launch);
  auto copy_with_layout = BuildCopyWithLayout(execute_launch, compile_launch,
                                              get_layout, input, &builder);
  copy_with_layout->setAttr(kDeviceAttr, execute_launch.deviceAttr());
  execute.setOperand(execute_arg_index, copy_with_layout);
}

// Performs transformation for replicated inputs. Returns true if this is a
// supported case (thus transform happened).
bool HandleReplicatedInputs(
    const int64_t execute_arg_index, Value compilation_key,
    tf_device::LaunchOp execute_launch, tf_device::LaunchOp compile_launch,
    mlir::BlockArgument replicate_arg, tf_device::ReplicateOp replicate,
    const TF::ResourceAliasAnalysis::Info& resource_alias_analysis) {
  // We need to know the devices to copy to.
  if (!replicate.devices()) return false;

  MutableArrayRef<OpOperand> inputs =
      replicate.GetOperandsForBlockArgument(replicate_arg);
  for (auto entry : llvm::enumerate(inputs)) {
    auto input_op = entry.value().get().getDefiningOp();
    if (!input_op || !IsSupportedInputOp(input_op, resource_alias_analysis))
      return false;
  }
  OpBuilder builder = CreateBuilderAfterOp(compile_launch);
  auto get_layout = BuildGetLayout(execute_arg_index, compilation_key,
                                   compile_launch, &builder);
  builder.setInsertionPoint(replicate);
  for (auto entry : llvm::enumerate(inputs)) {
    auto copy_with_layout =
        BuildCopyWithLayout(execute_launch, compile_launch, get_layout,
                            entry.value().get(), &builder);

    auto device_list = replicate.devices()
                           .getValue()
                           .get(execute_launch.getDevice())
                           .cast<ArrayAttr>();
    copy_with_layout->setAttr(kDeviceAttr,
                              device_list.getValue()[entry.index()]);

    entry.value().set(copy_with_layout);
  }
  return true;
}

// Performs transformation on a compile and associated execute(s) ops. The
// compile should not have other uses.
void HandleCompileAndExecutes(
    tf_device::LaunchOp compile_launch,
    llvm::MutableArrayRef<tf_device::LaunchOp> execute_launches,
    const TF::ResourceAliasAnalysis::Info& resource_alias_analysis) {
  auto compile =
      llvm::cast<TF::_TPUCompileMlirOp>(compile_launch.GetBody().front());
  tensorflow::tpu::TPUCompileMetadataProto metadata;
  metadata.ParseFromString(compile.metadata().str());
  llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> input_mappings =
      tensorflow::GetMetadataArgumentMapping(metadata);

  bool metadata_updated = false;
  auto maybe_replicate =
      execute_launches.front()->getParentOfType<tf_device::ReplicateOp>();

  for (auto execute_and_input_mapping :
       llvm::zip(execute_launches, input_mappings)) {
    auto& execute_launch = std::get<0>(execute_and_input_mapping);
    auto execute =
        llvm::cast<TF::TPUExecuteOp>(execute_launch.GetBody().front());
    const auto& input_mapping = std::get<1>(execute_and_input_mapping);

    for (auto& input_and_idx : llvm::enumerate(execute.args())) {
      Value input = input_and_idx.value();
      const int64_t execute_arg_index = input_and_idx.index();
      if (auto block_arg = input.dyn_cast<BlockArgument>()) {
        // For a block argument, consider transforms only when it is a
        // replicated input (defining ops will be outside the replicate node).
        if (maybe_replicate != block_arg.getParentRegion()->getParentOp() ||
            !HandleReplicatedInputs(execute_arg_index, execute.key(),
                                    execute_launch, compile_launch, block_arg,
                                    maybe_replicate, resource_alias_analysis)) {
          continue;
        }
      } else {
        // For an op output, consider transforms only when 1) there is no
        // replication or 2) it is outside the replicate node that encloses the
        // execute node. (Because if the op is inside replicate, it is probably
        // not on the host.)
        auto* input_op = input.getDefiningOp();
        if (maybe_replicate &&
            maybe_replicate.body().isAncestor(input_op->getParentRegion())) {
          continue;
        }
        if (!IsSupportedInputOp(input_op, resource_alias_analysis)) continue;
        HandleInput(input, execute_arg_index, execute, execute_launch,
                    compile_launch);
      }

      metadata.mutable_args(input_mapping[execute_arg_index])
          ->set_unrestricted_layout(true);
      metadata_updated = true;
    }
  }

  if (metadata_updated)
    compile->setAttr("metadata", StringAttr::get(compile.getContext(),
                                                 metadata.SerializeAsString()));
}

void TPUDynamicLayoutPass::runOnFunction(
    FuncOp func,
    const TF::ResourceAliasAnalysis::Info& resource_alias_analysis) {
  func.walk([&](TF::_TPUCompileMlirOp compile) {
    // Detect tf._TPUCompileMlir -> tf.TPUExecute(s).
    auto compile_launch =
        llvm::dyn_cast<tf_device::LaunchOp>(compile->getParentOp());
    if (!compile_launch || !compile_launch.WrapsSingleOp()) return;

    llvm::SmallVector<tf_device::LaunchOp, 4> execute_launches;
    execute_launches.reserve(compile_launch.getNumResults() - 1);
    for (Value program_result : llvm::drop_begin(compile_launch.results(), 1)) {
      if (!program_result.hasOneUse()) return;
      Operation* user = *program_result.user_begin();
      auto execute = llvm::dyn_cast<TF::TPUExecuteOp>(user);
      if (!execute) return;
      auto execute_launch =
          llvm::dyn_cast<tf_device::LaunchOp>(execute->getParentOp());
      if (!execute_launch || !execute_launch.WrapsSingleOp()) return;
      execute_launches.push_back(execute_launch);
    }

    HandleCompileAndExecutes(compile_launch, execute_launches,
                             resource_alias_analysis);
  });
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateTPUDynamicLayoutPass() {
  return std::make_unique<TPUDynamicLayoutPass>();
}

static PassRegistration<TPUDynamicLayoutPass> pass(
    "tf-tpu-dynamic-layout-pass",
    "Adds ops that allow TPU program inputs to have layouts determined at JIT "
    "compile time.");

}  // namespace TFTPU
}  // namespace mlir

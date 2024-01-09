/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

// This transformation pass converts tf_device.cluster_func op into
// tf._XlaCompile and tf._XlaRun ops.

#include <memory>
#include <string>

#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/parallel_execute_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/xla_rewrite_util.h"

#define DEBUG_TYPE "tf-xla-rewrite-v2"

namespace mlir {
namespace {

#define GEN_PASS_DEF_XLAREWRITEV2PASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_device_passes.h.inc"

constexpr absl::string_view kDeviceAttr = "device";

struct XlaRewriteV2Pass : public impl::XlaRewriteV2PassBase<XlaRewriteV2Pass> {
  void runOnOperation() override;
};

// Get the device from `tf_device.cluster_func` op
mlir::LogicalResult GetClusterFuncDevice(tf_device::ClusterFuncOp cluster_func,
                                         std::string& compilation_device) {
  auto device_attr = cluster_func->getAttrOfType<StringAttr>(kDeviceAttr);
  if (device_attr) {
    compilation_device = device_attr.str();
  } else {
    return cluster_func.emitOpError("No device assigned for cluster_func ");
  }
  return success();
}

// Rearrange the input order by putting resource args after non resource args
// Returns true when the inputs is in order, otherwise return false
bool RearrangeInputOrder(llvm::SmallVector<mlir::Value, 4> inputs,
                         llvm::SmallVector<Value>& non_resource_args,
                         llvm::SmallVector<Value>& resource_args) {
  bool has_resources = false;
  bool in_order = true;
  for (const Value& arg : inputs) {
    if (!getElementTypeOrSelf(arg.getType()).template isa<TF::ResourceType>()) {
      non_resource_args.push_back(arg);
      if (has_resources) in_order = false;
    } else {
      resource_args.push_back(arg);
      has_resources = true;
    }
  }
  return in_order;
}

// Move the resource args to the end of the function operand list.
void MoveResourceArgsToEnd(func::FuncOp callee) {
  llvm::DenseMap<BlockArgument, BlockArgument> mapping;
  unsigned num_params = callee.getNumArguments();
  llvm::BitVector removed_params(num_params);
  // Copy the resource-type parameters to the end.
  for (unsigned i = 0; i < num_params; ++i) {
    BlockArgument param = callee.getArgument(i);
    if (getElementTypeOrSelf(param.getType())
            .template isa<TF::ResourceType>()) {
      removed_params.set(i);
      callee.getBody().addArgument(param.getType(), param.getLoc());
      param.replaceAllUsesWith(callee.getArguments().back());
      removed_params.push_back(false);
    }
  }
  // Remove old resource-type parameters.
  callee.getBody().front().eraseArguments(removed_params);
  // Update function type.
  callee.setFunctionType(FunctionType::get(callee.getContext(),
                                           callee.getBody().getArgumentTypes(),
                                           callee.getResultTypes()));
}

mlir::LogicalResult GetOutputTypesForClusterFunc(
    mlir::tf_device::ClusterFuncOp cluster_func,
    llvm::SmallVectorImpl<mlir::Type>* output_types) {
  output_types->reserve(cluster_func.getNumResults());
  for (const auto& result_and_index :
       llvm::enumerate(cluster_func.getResults())) {
    const auto cluster_func_output_type =
        result_and_index.value().getType().cast<mlir::TensorType>();
    output_types->emplace_back(cluster_func_output_type);
  }
  return mlir::success();
}

mlir::LogicalResult ExtractInputsForLogicalDevices(
    const int num_cores_per_replica,
    mlir::tf_device::ClusterFuncOp cluster_func, mlir::OpBuilder* builder,
    llvm::SmallVectorImpl<llvm::SmallVector<mlir::Value, 4>>* input_list) {
  // Initialize the input list for each logical devices.
  input_list->reserve(num_cores_per_replica);
  for (int i = 0; i < num_cores_per_replica; ++i)
    input_list->emplace_back(llvm::SmallVector<mlir::Value, 4>());

  llvm::SmallVector<mlir::Value, 4> cluster_func_inputs(
      cluster_func.getOperands());

  // If sharding attribute does not exist, then all inputs are placed on 0th
  // logical core by default.
  (*input_list)[0] = cluster_func_inputs;
  return mlir::success();
}

// Creates a `tf._XlaRun` op that executes XLA program.
LogicalResult BuildExecuteOp(llvm::SmallVector<mlir::Value, 4> input,
                             tf_device::ClusterFuncOp cluster_func,
                             Operation* compile_op, int core,
                             OpBuilder* builder, TF::_XlaRunOp* execute_op) {
  llvm::SmallVector<Type, 4> output_types;
  llvm::SmallVector<int, 4> cluster_to_core_index;
  auto result = GetOutputTypesForClusterFunc(cluster_func, &output_types);
  if (failed(result)) return failure();

  llvm::SmallVector<Value> non_resource_args, resource_args;
  bool in_order = RearrangeInputOrder(input, non_resource_args, resource_args);

  llvm::SmallVector<mlir::Value, 4> execute_inputs;
  if (!in_order) {
    for (auto non_resource_arg : non_resource_args) {
      execute_inputs.emplace_back(non_resource_arg);
    }
    for (auto resource_arg : resource_args) {
      execute_inputs.emplace_back(resource_arg);
    }
  } else {
    execute_inputs = input;
  }
  execute_inputs.emplace_back(compile_op->getResult(core));

  // _XlaRun op has same output types as cluster_func.
  *execute_op = builder->create<TF::_XlaRunOp>(cluster_func.getLoc(),
                                               output_types, execute_inputs);
  return success();
}

// parallel_execute op returns concatenated list of return values of all its
// regions.
mlir::LogicalResult GetConcatenatedOutputTypes(
    const int num_cores_per_replica, tf_device::ClusterFuncOp cluster_func,
    tf_device::ParallelExecuteOp old_parallel_execute,
    const ValueTypeRange<ResultRange>& cluster_result_types,
    llvm::SmallVector<Type, 8>& concatenated_output_types) {
  // parallel_execute op returns concatenated list of return values of
  // all its regions.
  concatenated_output_types.reserve(cluster_result_types.size() *
                                    num_cores_per_replica);
  for (mlir::Region& region : old_parallel_execute.getRegions()) {
    if (!isa<tf_device::ClusterFuncOp>(region.front().front())) {
      for (Type t : region.front().front().getResultTypes())
        concatenated_output_types.emplace_back(t);
    }
  }

  for (int core = 0; core < num_cores_per_replica; ++core) {
    llvm::SmallVector<Type, 4> output_types;
    auto result = GetOutputTypesForClusterFunc(cluster_func, &output_types);
    if (failed(result)) return failure();
    for (Type t : output_types) {
      concatenated_output_types.emplace_back(t);
    }
  }
  return success();
}

// Given a `ParallelExecute`, replace it with a new `ParallelExecute`. The
// new `ParallelExecute` will replace the child that contains the
// `ClusterFunc` with `num_cores_per_replica` children. It keep other children
// the same. Return values from the child with the `ClusterFunc` will be
// duplicated `num_cores_per_replica` times.
LogicalResult AddToParallelExecuteOp(
    llvm::SmallVectorImpl<llvm::SmallVector<int, 4>>* cluster_to_core_index,
    Operation* compile_op, tf_device::ClusterFuncOp cluster_func,
    OpBuilder* builder, tf_device::ParallelExecuteOp old_parallel_execute,
    tf_device::ParallelExecuteOp* new_parallel_execute, int* cluster_idx) {
  const int num_cores_per_replica = 1;
  const auto cluster_result_types = cluster_func.getResultTypes();
  llvm::SmallVector<Type, 8> concatenated_output_types;

  if (failed(GetConcatenatedOutputTypes(
          num_cores_per_replica, cluster_func, old_parallel_execute,
          cluster_result_types, concatenated_output_types)))
    return failure();

  *cluster_idx = tensorflow::MovePreservedParallelExecuteChildren(
      num_cores_per_replica, concatenated_output_types, builder, cluster_func,
      old_parallel_execute, new_parallel_execute);

  // Extract inputs for each block of the parallel_execute op. The i-th
  // element in the list represents the input lists to XLA computation for
  // i-th logical core.
  llvm::SmallVector<llvm::SmallVector<mlir::Value, 4>, 4> input_list;
  builder->setInsertionPoint(*new_parallel_execute);
  auto result = ExtractInputsForLogicalDevices(
      num_cores_per_replica, cluster_func, builder, &input_list);
  if (failed(result)) return failure();

  // For each logical core, create a region with tf._XlaRun op.
  for (int core = 0; core < num_cores_per_replica; ++core) {
    auto& block =
        new_parallel_execute->GetRegionBlockWithIndex((*cluster_idx) + core);
    builder->setInsertionPointToEnd(&block);

    // Create Execute op _XlaRun.
    TF::_XlaRunOp execute;
    if (failed(BuildExecuteOp(input_list[core], cluster_func, compile_op, core,
                              builder, &execute)))
      return failure();

    std::string execute_device;
    if (failed(GetClusterFuncDevice(cluster_func, execute_device)))
      return failure();

    auto block_launch_op = tensorflow::WrapOpInLaunch(
        builder, block.getParent()->getLoc(), execute, execute_device);

    builder->create<tf_device::ReturnOp>(block.getParent()->getLoc(),
                                         block_launch_op.getResults());
  }

  return success();
}

// Replace the uses of old parallel execute outputs with new outputs
mlir::LogicalResult RemapOutputsFromLogicalDevices(
    mlir::tf_device::ParallelExecuteOp old_parallel_execute, int cluster_idx,
    mlir::tf_device::ParallelExecuteOp new_parallel_execute,
    mlir::OpBuilder* builder) {
  for (auto [output_index, old_parallel_execute_output] :
       llvm::enumerate(old_parallel_execute.getResults())) {
    const auto output_from_logical_device =
        new_parallel_execute.GetRegionOutputs(cluster_idx)[output_index];
    old_parallel_execute_output.replaceAllUsesWith(output_from_logical_device);
  }
  return mlir::success();
}

// Create a `tf._XlaCompile` op
Operation* BuildCompileOp(tf_device::ClusterFuncOp cluster_func,
                          llvm::StringRef compilation_device,
                          SymbolTable& symtab, OpBuilder* builder) {
  llvm::SmallVector<Value> non_resource_args, resource_args;
  bool in_order = RearrangeInputOrder(cluster_func.getOperands(),
                                      non_resource_args, resource_args);
  if (!in_order) {
    // Functions do not get reused in practice, so skip the check for if the
    // callee has been updated.
    StringAttr callee_sym = cluster_func.getFuncAttr().getAttr();
    MoveResourceArgsToEnd(symtab.lookup<func::FuncOp>(callee_sym));
  }

  auto program_type =
      RankedTensorType::get({3}, builder->getType<TF::StringType>());
  auto compilation_status_type =
      RankedTensorType::get({}, builder->getType<TF::BoolRefType>());
  auto compile_op = builder->create<TF::_XlaCompileOp>(
      cluster_func.getLoc(), program_type, compilation_status_type,
      /*constants=*/ValueRange({}), ValueRange(non_resource_args),
      ValueRange(resource_args), builder->getBoolAttr(true),
      cluster_func.getFuncAttr());
  return tensorflow::WrapOpInLaunch(builder, compile_op.getLoc(), compile_op,
                                    compilation_device);
}

mlir::LogicalResult GetCompilationDeviceFromParallelExecuteOp(
    tf_device::ParallelExecuteOp& old_parallel_execute,
    std::string& compilation_device) {
  auto& first_block = old_parallel_execute.GetRegionBlockWithIndex(0);
  if (isa<tf_device::LaunchOp>(first_block.front())) {
    auto device_attr =
        first_block.front().getAttrOfType<StringAttr>(kDeviceAttr);
    if (device_attr) {
      compilation_device = device_attr.str();
    } else {
      return failure();
    }
  }
  return success();
}

mlir::LogicalResult Rewrite(tf_device::ClusterFuncOp cluster_func,
                            SymbolTable& symtab, OpBuilder& builder) {
  // Fetch the ParallelExecute parent of `cluster_func`, or create it if
  // it does not exist.
  tf_device::ParallelExecuteOp old_parallel_execute =
      cluster_func->getParentOfType<tf_device::ParallelExecuteOp>();
  if (old_parallel_execute &&
      cluster_func->getParentOp() != old_parallel_execute) {
    cluster_func->emitError() << "The ParallelExecute ancestor of a "
                                 "ClusterFunc must be its direct parent.";
  }

  // Fetch compilation device
  std::string compilation_device;
  if (failed(GetClusterFuncDevice(cluster_func, compilation_device)))
    return failure();

  if (!old_parallel_execute) {
    old_parallel_execute =
        mlir::TF::BuildParallelExecuteOp(cluster_func, &builder);
  }

  // Build compile op _XlaCompile
  builder.setInsertionPoint(old_parallel_execute);
  Operation* compile_op =
      BuildCompileOp(cluster_func, compilation_device, symtab, &builder);
  if (!compile_op) {
    return failure();
  }

  old_parallel_execute.walk(
      [&](TF::_XlaCompileMlirPlaceholderProgramKeyOp key_op) {
        key_op.replaceAllUsesWith(compile_op->getResult(0));
        key_op.erase();
      });

  // Build new parallel execute op
  tf_device::ParallelExecuteOp new_parallel_execute;
  int num_cores_per_replica = 1;
  int cluster_idx;
  llvm::SmallVector<llvm::SmallVector<int, 4>, 4> cluster_to_core_index;
  cluster_to_core_index.reserve(num_cores_per_replica);

  if (failed(AddToParallelExecuteOp(
          &cluster_to_core_index, compile_op, cluster_func, &builder,
          old_parallel_execute, &new_parallel_execute, &cluster_idx)))
    return failure();

  // As tf_device.parallel_execute wraps # logical cores number of tf._XlaRun
  // ops, the number of return values of parallel_execute op may exceed that of
  // cluster_func op. As such, each return value of parallel_execute op must
  // be mapped with corresponding return value usages of cluster_func.
  if (failed(RemapOutputsFromLogicalDevices(old_parallel_execute, cluster_idx,
                                            new_parallel_execute, &builder)))
    return failure();

  if (failed(mlir::TF::RemoveSingletonParallelExecuteOp(new_parallel_execute,
                                                        &builder)))
    return failure();

  return success();
}

void XlaRewriteV2Pass::runOnOperation() {
  ModuleOp module = getOperation();
  SymbolTable symtab(module);
  OpBuilder builder(&getContext());
  llvm::SmallVector<tf_device::ClusterFuncOp, 4> cluster_func_ops;
  module.walk([&](tf_device::ClusterFuncOp cluster_func) {
    cluster_func_ops.push_back(cluster_func);
  });

  for (tf_device::ClusterFuncOp cluster_func : cluster_func_ops) {
    if (failed(Rewrite(cluster_func, symtab, builder)))
      return signalPassFailure();
  }

  // Erase all the tf_device.cluster_func ops
  if (failed(tensorflow::EraseClusterFuncs(cluster_func_ops))) {
    return signalPassFailure();
  }
}

}  // namespace

namespace TFDevice {
std::unique_ptr<OperationPass<ModuleOp>> CreateXlaRewriteV2Pass() {
  return std::make_unique<XlaRewriteV2Pass>();
}

}  // namespace TFDevice
}  // namespace mlir

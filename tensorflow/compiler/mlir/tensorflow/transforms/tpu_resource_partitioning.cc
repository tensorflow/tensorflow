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
#include <tuple>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"

namespace mlir {
namespace TFTPU {
namespace {

constexpr char kReplicateSharding[] = "";

struct TPUResourceReadsWritesPartitioningPass
    : public TF::TPUResourceReadsWritesPartitioningPassBase<
          TPUResourceReadsWritesPartitioningPass> {
  void runOnFunction() override;
};

bool AllResourceTypesHaveSubtypes(TypeRange resources) {
  for (Type resource : resources)
    if (!llvm::hasSingleElement(resource.cast<TensorType>()
                                    .getElementType()
                                    .cast<TF::ResourceType>()
                                    .getSubtypes()))
      return false;

  return true;
}

Type GetResourceSubtype(Type type) {
  return type.cast<TensorType>()
      .getElementType()
      .cast<TF::ResourceType>()
      .getSubtypes()
      .front();
}

Type GetResourceSubtype(Value resource) {
  return GetResourceSubtype(resource.getType());
}

// Rewrites unpartitioned resource reads and writes to partitioned resource
// reads and writes. The TPU computation from the frontend is generated in such
// a way that resource operations operate on the unpartitioned resource handle
// (from a `tf.TPUReplicatedInput`). This results in resource reads and writes
// on the unpartitioned resource handle post resource op decomposition/lifting.
// Here the unpartitioned resource read and write is expanded to individual
// resource reads and writes per associated partitioned resource handle.
void PartitionResourceReadsWrites(tf_device::ClusterFuncOp cluster_func) {
  bool use_spmd = false;
  if (auto use_spmd_attr = cluster_func->getAttrOfType<BoolAttr>(
          "use_spmd_for_xla_partitioning"))
    use_spmd = use_spmd_attr.getValue();

  if (!use_spmd) return;

  OpBuilder builder(cluster_func);
  // Rewrite results before rewriting operands as `tf.TPUPartitionedInput`
  // resource handle results is an indicator for a partitioned resource
  // variable. These `tf.TPUPartitionedInput` will be removed when rewriting
  // the operands.
  for (Value result : cluster_func.results()) {
    if (!result.hasOneUse()) continue;
    auto assign_var =
        llvm::dyn_cast<TF::AssignVariableOp>(*result.getUsers().begin());
    if (!assign_var || assign_var.value() != result) continue;
    auto partitioned_input = llvm::dyn_cast_or_null<TF::TPUPartitionedInputOp>(
        assign_var.resource().getDefiningOp());
    if (!partitioned_input ||
        !AllResourceTypesHaveSubtypes(partitioned_input.inputs().getTypes()))
      continue;

    builder.setInsertionPoint(assign_var);
    llvm::SmallVector<Type, 4> partitioned_output_types;
    partitioned_output_types.reserve(partitioned_input.N());
    for (Type input_type : partitioned_input.inputs().getTypes())
      partitioned_output_types.push_back(GetResourceSubtype(input_type));
    auto partitioned_output = builder.create<TF::TPUPartitionedOutputOp>(
        cluster_func->getLoc(), partitioned_output_types, result,
        partitioned_input.partition_dimAttr(),
        partitioned_input._XlaShardingAttr());
    for (auto resource_write :
         llvm::zip(partitioned_input.inputs(), partitioned_output.output()))
      builder.create<TF::AssignVariableOp>(
          assign_var->getLoc(), /*resource=*/std::get<0>(resource_write),
          /*value=*/std::get<1>(resource_write));
    assign_var.erase();
  }

  for (OpOperand& operand : cluster_func->getOpOperands()) {
    auto read_var = llvm::dyn_cast_or_null<TF::ReadVariableOp>(
        operand.get().getDefiningOp());
    if (!read_var || !read_var.value().hasOneUse()) continue;
    auto partitioned_input = llvm::dyn_cast_or_null<TF::TPUPartitionedInputOp>(
        read_var.resource().getDefiningOp());
    if (!partitioned_input ||
        !AllResourceTypesHaveSubtypes(partitioned_input.inputs().getTypes()))
      continue;

    builder.setInsertionPoint(partitioned_input);
    llvm::SmallVector<Value, 4> partitioned_reads;
    for (Value input : partitioned_input.inputs()) {
      auto partitioned_read = builder.create<TF::ReadVariableOp>(
          read_var->getLoc(), GetResourceSubtype(input), input);
      partitioned_reads.push_back(partitioned_read.value());
    }
    auto partitioned_read = builder.create<TF::TPUPartitionedInputOp>(
        partitioned_input->getLoc(), read_var.value().getType(),
        partitioned_reads, partitioned_input.partition_dimAttr(),
        partitioned_input._XlaShardingAttr());
    operand.set(partitioned_read);
    read_var->erase();
    if (partitioned_input->use_empty()) partitioned_input->erase();
  }
}

void TPUResourceReadsWritesPartitioningPass::runOnFunction() {
  llvm::SmallVector<tf_device::ClusterFuncOp, 4> cluster_funcs;
  getFunction()->walk([&cluster_funcs](tf_device::ClusterFuncOp cluster_func) {
    cluster_funcs.push_back(cluster_func);
  });
  for (tf_device::ClusterFuncOp cluster_func : cluster_funcs)
    PartitionResourceReadsWrites(cluster_func);
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>>
CreateTPUResourceReadsWritesPartitioningPass() {
  return std::make_unique<TPUResourceReadsWritesPartitioningPass>();
}

}  // namespace TFTPU
}  // namespace mlir

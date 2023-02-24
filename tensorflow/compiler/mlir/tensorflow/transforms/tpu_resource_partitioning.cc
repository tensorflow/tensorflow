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

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/parallel_execute_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace mlir {
namespace TFTPU {
namespace {

#define GEN_PASS_DEF_TPURESOURCEREADSWRITESPARTITIONINGPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

constexpr char kUseSpmdAttr[] = "use_spmd_for_xla_partitioning";
constexpr char kNumCoresPerReplicaAttr[] = "num_cores_per_replica";

struct TPUResourceReadsWritesPartitioningPass
    : public impl::TPUResourceReadsWritesPartitioningPassBase<
          TPUResourceReadsWritesPartitioningPass> {
  void runOnOperation() override;
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

// Updates uses of `old_read` to `new_partitioned_input` and `new_reads`.
// `old_partitioned_input` is the predecessor of `old_read`. `new_reads`
// contains the predecessors of `new_partitioned_input`.
LogicalResult UpdateReadUses(TF::ReadVariableOp old_read,
                             TF::TPUPartitionedInputV2Op old_partitioned_input,
                             TF::TPUPartitionedInputV2Op new_partitioned_input,
                             llvm::SmallVector<Value, 4> new_reads) {
  xla::OpSharding sharding;
  sharding.ParseFromString(
      old_partitioned_input.get_XlaShardingAttr().getValue().str());
  for (OpOperand& read_use :
       llvm::make_early_inc_range(old_read.getValue().getUses())) {
    if (dyn_cast_or_null<tf_device::ClusterFuncOp>(read_use.getOwner())) {
      // ClusterFunc's use of the Read is replaced with use of the
      // TPUPartitionedInputV2.
      read_use.set(new_partitioned_input);
    } else {
      // Outside compiled code's use of the Read after TPUPartitionedInputV2 is
      // replaced with use of the first Read before the TPUPartitionedInputV2.
      if (sharding.type() != xla::OpSharding::REPLICATED) {
        // TODO(b/243077297): Generalize to any sharding.
        old_partitioned_input.emitOpError(
            "TPUPartitionedInputV2 variable used in outside compiled code is "
            "only supported with REPLICATED sharding");
        return failure();
      }
      read_use.set(new_reads[0]);
    }
  }
  return success();
}

// Rewrites unpartitioned resource reads and writes to partitioned resource
// reads and writes. The TPU computation from the frontend is generated in such
// a way that resource operations operate on the unpartitioned resource handle
// (from a `tf.TPUReplicatedInput`). This results in resource reads and writes
// on the unpartitioned resource handle post resource op decomposition/lifting.
// Here the unpartitioned resource read and write is expanded to individual
// resource reads and writes per associated partitioned resource handle.
LogicalResult PartitionResourceReadsWrites(
    tf_device::ClusterFuncOp cluster_func) {
  bool use_spmd = false;
  if (auto use_spmd_attr = cluster_func->getAttrOfType<BoolAttr>(kUseSpmdAttr))
    use_spmd = use_spmd_attr.getValue();

  if (!use_spmd) return success();

  auto num_cores_per_replica_attr =
      cluster_func->getAttrOfType<IntegerAttr>(kNumCoresPerReplicaAttr);

  // Wrap the ClusterFunc with a ParallelExecute if it does not already exist.
  OpBuilder builder(cluster_func);
  tf_device::ParallelExecuteOp parallel_execute =
      cluster_func->getParentOfType<tf_device::ParallelExecuteOp>();
  if (!parallel_execute)
    parallel_execute = BuildParallelExecuteOp(cluster_func, &builder);

  // Rewrite results before rewriting operands as `tf.TPUPartitionedInputV2`
  // resource handle results is an indicator for a partitioned resource
  // variable. These `tf.TPUPartitionedInputV2` will be removed when rewriting
  // the operands.
  for (Value result : parallel_execute.getExecuteOutputs()) {
    if (!result.hasOneUse()) continue;
    auto assign_var =
        llvm::dyn_cast<TF::AssignVariableOp>(*result.getUsers().begin());
    if (!assign_var || assign_var.getValue() != result) continue;
    auto partitioned_input =
        llvm::dyn_cast_or_null<TF::TPUPartitionedInputV2Op>(
            assign_var.getResource().getDefiningOp());
    if (!partitioned_input ||
        !AllResourceTypesHaveSubtypes(partitioned_input.getInputs().getTypes()))
      continue;

    const auto inputs = partitioned_input.getInputs();
    const bool packed_input = partitioned_input.getIsPacked();
    int num_cores_per_replica = partitioned_input.getN();
    if (num_cores_per_replica_attr) {
      num_cores_per_replica = num_cores_per_replica_attr.getInt();
    } else if (packed_input) {
      return partitioned_input->emitOpError()
             << "num cores per replica unavailable";
    }

    const int num_operands_expected = packed_input ? 1 : num_cores_per_replica;
    if (num_cores_per_replica_attr && num_operands_expected != inputs.size()) {
      return partitioned_input->emitOpError()
             << "expects " << num_operands_expected << " operands but found "
             << partitioned_input.getNumOperands();
    }

    builder.setInsertionPoint(assign_var);
    llvm::SmallVector<Type, 4> partitioned_output_types;
    partitioned_output_types.reserve(num_cores_per_replica);
    for (int i = 0; i < num_cores_per_replica; ++i) {
      const auto& input = packed_input ? inputs[0] : inputs[i];
      partitioned_output_types.push_back(GetResourceSubtype(input.getType()));
    }

    auto partitioned_output = builder.create<TF::TPUPartitionedOutputV2Op>(
        cluster_func->getLoc(), partitioned_output_types, result,
        partitioned_input.getPartitionDimsAttr(),
        partitioned_input.get_XlaShardingAttr());
    for (auto [i, value] : llvm::enumerate(partitioned_output.getOutput())) {
      const auto& resource = packed_input ? inputs[0] : inputs[i];
      builder.create<TF::AssignVariableOp>(
          assign_var->getLoc(), /*resource=*/resource, /*value=*/value);
    }
    assign_var.erase();
  }

  for (OpOperand& operand : cluster_func->getOpOperands()) {
    auto read_var = llvm::dyn_cast_or_null<TF::ReadVariableOp>(
        operand.get().getDefiningOp());
    if (!read_var) continue;
    auto partitioned_input =
        llvm::dyn_cast_or_null<TF::TPUPartitionedInputV2Op>(
            read_var.getResource().getDefiningOp());
    if (!partitioned_input || !AllResourceTypesHaveSubtypes(
                                  partitioned_input.getInputs().getTypes())) {
      continue;
    }

    // we only want to create one read variable op per unique input
    // otherwise tpu rewriting will fail to clean up the duplicates
    llvm::SmallMapVector<Value, Value, 4> read_variable_ops;
    llvm::SmallVector<Value, 4> partitioned_reads;
    builder.setInsertionPoint(partitioned_input);

    for (Value input : partitioned_input.getInputs()) {
      auto search = read_variable_ops.find(input);
      // if a read variable op already doesn't exist for this input, create it
      if (search == read_variable_ops.end()) {
        auto partitioned_read = builder.create<TF::ReadVariableOp>(
            read_var->getLoc(), GetResourceSubtype(input), input);
        search = read_variable_ops.insert({input, partitioned_read.getValue()})
                     .first;
      }
      partitioned_reads.push_back(search->second);
    }

    auto partitioned_read = builder.create<TF::TPUPartitionedInputV2Op>(
        partitioned_input->getLoc(), read_var.getValue().getType(),
        partitioned_reads, partitioned_input.getPartitionDimsAttr(),
        partitioned_input.getIsPackedAttr(),
        partitioned_input.get_XlaShardingAttr());
    if (failed(UpdateReadUses(read_var, partitioned_input, partitioned_read,
                              partitioned_reads)))
      return failure();
    read_var->erase();
    if (partitioned_input->use_empty()) partitioned_input->erase();
  }
  return RemoveSingletonParallelExecuteOp(parallel_execute, &builder);
}

void TPUResourceReadsWritesPartitioningPass::runOnOperation() {
  llvm::SmallVector<tf_device::ClusterFuncOp, 4> cluster_funcs;
  getOperation()->walk([&cluster_funcs](tf_device::ClusterFuncOp cluster_func) {
    cluster_funcs.push_back(cluster_func);
  });
  for (tf_device::ClusterFuncOp cluster_func : cluster_funcs)
    if (failed(PartitionResourceReadsWrites(cluster_func)))
      return signalPassFailure();
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateTPUResourceReadsWritesPartitioningPass() {
  return std::make_unique<TPUResourceReadsWritesPartitioningPass>();
}

}  // namespace TFTPU
}  // namespace mlir

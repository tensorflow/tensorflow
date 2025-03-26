/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <iterator>
#include <memory>
#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"

namespace tensorflow {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSORDEVICEMESHCLUSTERCOARSENING
#include "tensorflow/dtensor/mlir/dtensor_passes.h.inc"

constexpr char kMissingMeshAttributeErrorMessage[] =
    "failed to merge mesh cluster as cluster does not have mesh attribute. "
    "This is likely due to problem in mesh propagation.";

// Determines whether two adjoining clusters should be merged.
mlir::LogicalResult ShouldMergeClusters(mlir::tf_device::ClusterOp cluster_a,
                                        mlir::tf_device::ClusterOp cluster_b,
                                        bool* should_merge) {
  if (cluster_a->getParentRegion() != cluster_b->getParentRegion()) {
    *should_merge = false;
    return mlir::success();
  }

  auto mesh_a_or_status = ExtractDeviceMeshFromOp(cluster_a.getOperation());
  if (!mesh_a_or_status.ok())
    return cluster_a.emitOpError(
        absl::StatusMessageAsCStr(mesh_a_or_status.status()));

  auto mesh_b_or_status = ExtractDeviceMeshFromOp(cluster_b.getOperation());
  if (!mesh_b_or_status.ok())
    return cluster_b.emitOpError(
        absl::StatusMessageAsCStr(mesh_b_or_status.status()));

  auto mesh_a = mesh_a_or_status.value();
  auto mesh_b = mesh_b_or_status.value();
  if (!mesh_a || !mesh_b) {
    return !mesh_a ? cluster_a.emitOpError(kMissingMeshAttributeErrorMessage)
                   : cluster_b.emitOpError(kMissingMeshAttributeErrorMessage);
  }

  *should_merge = mesh_a == mesh_b;
  return mlir::success();
}

// Moves all ops (except tf_device.return op) inside `src_cluster` to
// block inside `target_cluster`. Ops are moved before the `exit_op`
// inside the `target_cluster`.
void MoveOpsInsideCluster(mlir::tf_device::ClusterOp src_cluster,
                          mlir::tf_device::ClusterOp target_cluster,
                          mlir::Operation* exit_op) {
  auto& cluster_body = src_cluster.GetBody().getOperations();
  target_cluster.GetBody().getOperations().splice(
      exit_op->getIterator(), cluster_body, cluster_body.begin(),
      std::prev(cluster_body.end()));
}

// Returns a list of pair of mlir Values that represent <return values of ops
// inside the merged_cluster, output values of merged cluster>.
//
// If outputs of `current_cluster` is used as operands to ops in
// `merging_cluster`, then make sure to replace operands such that
// results values from the inner ops of `current_cluster` is used instead.
//
// For example,
//    %0 = "tf_device.cluster"() ({
//      %1 = "tf.A"() : () -> tensor<i32>
//      "tf_device.return"(%1) : (tensor<i32>) -> ()
//    }) { mesh = "mesh_config: cpu[1, 1]"} : () -> (tensor<i32>)
//
//    %2 = "tf_device.cluster"() ({
//      %3 = "tf.B"(%0) : (tenosr<i32>) -> tensor<f32>
//      "tf_device.return"(%3) : (tensor<f32>) -> ()
//    }) { mesh = "mesh_config: cpu[1, 1]"} : () -> (tensor<f32>)
//
// will be:
//    %0 = "tf_device.cluster"() ({
//      %1 = "tf.A"() : () -> tensor<i32>
//
//      # NOTE: `tf.B` op now takes operand directly from
//      # `tf.A` instead of `tf_dtensor.cluster op.
//      %2 = "tf.B"(%1) : (tenosr<i32>) -> tensor<f32>
//      "tf_device.return"(%1, %2) : (tensor<i32>, tensor<f32>)) -> ()
//    }) {mesh = "mesh_config: cpu[1, 1]"} : () -> (tensor<i32>, tensor<f32>)
llvm::SmallVector<std::pair<mlir::Value, mlir::Value>, 8>
GetMergedMeshClusterResults(mlir::tf_device::ClusterOp current_cluster,
                            mlir::tf_device::ClusterOp merging_cluster) {
  llvm::SmallVector<std::pair<mlir::Value, mlir::Value>, 8>
      merged_cluster_results;
  merged_cluster_results.reserve(current_cluster.getNumResults() +
                                 merging_cluster.getNumResults());

  auto current_cluster_return_op = current_cluster.GetBody().getTerminator();
  for (auto result : llvm::zip(current_cluster_return_op->getOpOperands(),
                               current_cluster.getResults())) {
    mlir::Value inner_op_result = std::get<0>(result).get();
    mlir::Value outer_op_result = std::get<1>(result);

    // If the output value of `current_cluster` is only used by ops
    // inside the `merged_cluster`, do not add the value as a return
    // value for newly created tf_device.cluster op.
    bool result_only_used_by_merging_cluster = true;
    for (auto& use : llvm::make_early_inc_range(outer_op_result.getUses())) {
      if (merging_cluster.GetBody().findAncestorOpInBlock(*use.getOwner())) {
        use.set(inner_op_result);
      } else {
        result_only_used_by_merging_cluster = false;
      }
    }

    if (!result_only_used_by_merging_cluster) {
      merged_cluster_results.emplace_back(inner_op_result, outer_op_result);
    }
  }

  auto merging_cluster_return_op = merging_cluster.GetBody().getTerminator();
  for (auto result : llvm::zip(merging_cluster_return_op->getOpOperands(),
                               merging_cluster.getResults())) {
    mlir::Value inner_op_result = std::get<0>(result).get();
    mlir::Value outer_op_result = std::get<1>(result);

    if (!outer_op_result.getUses().empty())
      merged_cluster_results.emplace_back(inner_op_result, outer_op_result);
  }

  return merged_cluster_results;
}

// Updates the users of `merging_cluster` so that they use values
// from `merged_cluster` instead.
void ReplaceOperandUsagesWithMergedClusterOutputs(
    mlir::ValueRange values_to_replace,
    mlir::tf_device::ClusterOp merged_cluster) {
  for (auto result :
       llvm::zip(values_to_replace, merged_cluster.getResults())) {
    std::get<0>(result).replaceAllUsesWith(std::get<1>(result));
  }
}

// Creates a new tf_device.cluster op that merges
// `current_cluster` and `merging_cluster`.
mlir::LogicalResult CreateMergedMeshCluster(
    mlir::OpBuilder* builder, mlir::tf_device::ClusterOp current_cluster,
    mlir::tf_device::ClusterOp merging_cluster,
    mlir::tf_device::ClusterOp* merged_cluster) {
  auto return_values =
      GetMergedMeshClusterResults(current_cluster, merging_cluster);

  llvm::SmallVector<mlir::Type, 8> merged_cluster_output_types;
  llvm::SmallVector<mlir::Value, 8> merged_cluster_output_values;
  llvm::SmallVector<mlir::Value, 8> output_values_to_replace;
  merged_cluster_output_types.reserve(return_values.size());
  merged_cluster_output_values.reserve(return_values.size());
  output_values_to_replace.reserve(return_values.size());
  for (auto cluster_return_value : return_values) {
    auto inner_op_return_value = std::get<0>(cluster_return_value);
    merged_cluster_output_types.emplace_back(inner_op_return_value.getType());
    merged_cluster_output_values.emplace_back(inner_op_return_value);
    output_values_to_replace.emplace_back(std::get<1>(cluster_return_value));
  }

  *merged_cluster = builder->create<mlir::tf_device::ClusterOp>(
      current_cluster.getLoc(), merged_cluster_output_types);
  auto mesh_attr = current_cluster->getAttrOfType<mlir::StringAttr>(kMeshAttr);
  if (!mesh_attr)
    return current_cluster.emitOpError(kMissingMeshAttributeErrorMessage);

  (*merged_cluster)->setAttr(kMeshAttr, mesh_attr);

  // Create a terminator op that returns all return values from
  // `current_cluster` and `merging_cluster`.
  merged_cluster->getBody().push_back(new mlir::Block);
  builder->setInsertionPointToEnd(&merged_cluster->GetBody());
  builder->create<mlir::tf_device::ReturnOp>(merged_cluster->getLoc(),
                                             merged_cluster_output_values);

  // Make sure to replace usages of tf_device.cluster ops to be merged-away with
  // newly created tf_device.cluster op.
  ReplaceOperandUsagesWithMergedClusterOutputs(output_values_to_replace,
                                               *merged_cluster);

  return mlir::success();
}

// Merges `current_cluster` and `merging_cluster` and returns a new merged
// tf_device.cluster.
mlir::LogicalResult MergeClusters(mlir::OpBuilder* builder,
                                  mlir::tf_device::ClusterOp current_cluster,
                                  mlir::tf_device::ClusterOp merging_cluster,
                                  mlir::tf_device::ClusterOp* merged_cluster) {
  builder->setInsertionPoint(current_cluster);

  // Create new tf_device.cluster op that outputs results of both
  // `current_cluster` and `merging_cluster`.
  if (mlir::failed(CreateMergedMeshCluster(builder, current_cluster,
                                           merging_cluster, merged_cluster)))
    return mlir::failure();

  // Move all ops to newly created merged cluster.
  auto exit_op = merged_cluster->GetBody().getTerminator();
  MoveOpsInsideCluster(current_cluster, *merged_cluster, exit_op);
  MoveOpsInsideCluster(merging_cluster, *merged_cluster, exit_op);

  // Remove mesh clusters as they are now merged to a new cluster.
  current_cluster.erase();
  merging_cluster.erase();
  return mlir::success();
}

// Loops through tf_device.Cluster ops and merge clusters with same execution
// device set.
mlir::LogicalResult ClusterDeviceClusterOpsInBlock(mlir::OpBuilder* builder,
                                                   mlir::Block* block) {
  llvm::SmallVector<mlir::tf_device::ClusterOp, 4> block_ops;
  block->walk([&](mlir::Operation* op) {
    if (auto cluster = llvm::dyn_cast<mlir::tf_device::ClusterOp>(op))
      block_ops.emplace_back(cluster);
  });

  std::optional<mlir::tf_device::ClusterOp> current_cluster;
  for (mlir::tf_device::ClusterOp cluster :
       llvm::make_early_inc_range(block_ops)) {
    if (!current_cluster.has_value()) {
      current_cluster = cluster;
      continue;
    }
    bool should_merge;
    if (failed(ShouldMergeClusters(*current_cluster, cluster, &should_merge)))
      return mlir::failure();

    if (should_merge) {
      mlir::tf_device::ClusterOp new_cluster;
      if (mlir::failed(
              MergeClusters(builder, *current_cluster, cluster, &new_cluster)))
        return mlir::failure();

      current_cluster.emplace(new_cluster);
    } else {
      current_cluster.emplace(cluster);
    }
  }
  return mlir::success();
}

}  // namespace

// MLIR pass that merges cluster ops with the same mesh attribute.
struct DTensorDeviceMeshClusterCoarsening
    : public impl::DTensorDeviceMeshClusterCoarseningBase<
          DTensorDeviceMeshClusterCoarsening> {
  void runOnOperation() override {
    mlir::MLIRContext& context = getContext();
    mlir::OpBuilder builder(&context);
    for (mlir::Block& block : getOperation())
      if (mlir::failed(ClusterDeviceClusterOpsInBlock(&builder, &block)))
        return signalPassFailure();
  }
};

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorDeviceMeshClusterCoarsening() {
  return std::make_unique<DTensorDeviceMeshClusterCoarsening>();
}

}  // namespace dtensor
}  // namespace tensorflow

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
#include <string>
#include <utility>

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/core/platform/logging.h"

namespace mlir {
namespace TFTPU {

namespace {

constexpr char kXlaOutsideCompilationAttr[] = "_xla_outside_compilation";
constexpr char kDeviceAttr[] = "device";

// Mapping for `_xla_outside_compilation` attribute to ops of a cluster.
using ClusterMap =
    llvm::SmallDenseMap<llvm::StringRef, llvm::SmallVector<Operation*, 8>, 8>;

// This pass extracts a CPU computation cluster with `_xla_outside_compilation`
// annotation from a TPU cluster.  Each outside compilation cluster is moved to
// a parallel_execute region.  The TPU cluster is also moved to a
// parallel_execute region.
// TODO(b/154363171): Add example tranformations.

struct TPUExtractOutsideCompilation
    : public PassWrapper<TPUExtractOutsideCompilation, FunctionPass> {
  void runOnFunction() override;
};

// Collects and clusters ops in `block` with the same `_xla_outside_compilation`
// attribute into `clusters` This returns an error if a
// `_xla_outside_compilation` attribute of an op is empty.
LogicalResult CollectAndGroupClusterOps(Block* block, ClusterMap* clusters) {
  for (Operation& op : *block) {
    if (auto attr = op.getAttrOfType<StringAttr>(kXlaOutsideCompilationAttr)) {
      if (attr.getValue().empty())
        return op.emitError()
               << "attribute '" << kXlaOutsideCompilationAttr << "' is empty";

      auto it = clusters->try_emplace(attr.getValue());
      it.first->getSecond().push_back(&op);
    }
  }

  return success();
}

// Moves `cluster_ops` to associated `launch_op` body.
void MoveClusterOpsToLaunchOp(
    tf_device::LaunchOp launch_op,
    const llvm::SmallVector<Operation*, 8>& cluster_ops) {
  MLIRContext* context = launch_op.getContext();
  Operation* terminator = launch_op.GetBody().getTerminator();

  for (Operation* cluster_op : cluster_ops) {
    // Remove `_xla_outside_compilation` and `device` attribute from ops in the
    // cluster as that information will be present in the `launch_op`.
    cluster_op->removeAttr(
        Identifier::get(kXlaOutsideCompilationAttr, context));
    cluster_op->removeAttr(Identifier::get(kDeviceAttr, context));
    cluster_op->moveBefore(terminator);
  }
}

// Creates a `tf_device::LaunchOp` to wrap cluster ops.
tf_device::LaunchOp CreateLaunchOpForCluster(OpBuilder* builder,
                                             Operation* last_cluster_op) {
  // TODO(b/154363171): Set the CPU device.
  // An empty string placeholder is used for the device as that will be later
  // populated with the device of the associated TPUReplicateMetadata op.
  llvm::SmallVector<Type, 8> result_types;
  auto launch_op = builder->create<tf_device::LaunchOp>(
      last_cluster_op->getLoc(), builder->getStringAttr(""), result_types);

  launch_op.body().push_back(new Block);

  // Add terminator.
  builder->setInsertionPointToEnd(&launch_op.GetBody());
  builder->create<tf_device::ReturnOp>(last_cluster_op->getLoc(),
                                       llvm::ArrayRef<Value>{});

  return launch_op;
}

// Creates a `parallel_execute` op in place of launch with 'clusters` and
// 'launch` as regions.
void CreateParallelExecuteFromClusters(tf_device::LaunchOp launch,
                                       const ClusterMap& clusters) {
  OpBuilder builder(launch);
  // Create parallel_execute regions.  The original TPU cluster computation
  // is the extra region.
  int num_regions = 1 + clusters.size();
  // TODO(b/154363171): Correctly determine output_types. Add tests to confirm
  // that the types for parallel_execute_op match the concatenated output
  // types of the contained regions.
  // TODO(b/154363171): Remap the results of the `launch` op to use the
  // results of the `parallel_execute` op.
  llvm::SmallVector<Type, 8> concatenated_output_types;
  auto parallel_execute_op = builder.create<tf_device::ParallelExecuteOp>(
      launch.getLoc(), num_regions, concatenated_output_types);

  // Move outside compilation clusters to parallel_execute regions.
  for (const auto& cluster : llvm::enumerate(clusters)) {
    const auto& cluster_ops = cluster.value().getSecond();

    Block& outside_block =
        parallel_execute_op.GetRegionBlockWithIndex(cluster.index());
    builder.setInsertionPointToEnd(&outside_block);
    tf_device::LaunchOp launch_op =
        CreateLaunchOpForCluster(&builder, cluster_ops.back());
    MoveClusterOpsToLaunchOp(launch_op, cluster_ops);
    builder.setInsertionPointToEnd(&outside_block);
    builder.create<tf_device::ReturnOp>(launch.getLoc(), launch.getResults());
  }

  // Move the launch body to last parallel_execute block.
  Block& inside_block =
      parallel_execute_op.GetRegionBlockWithIndex(num_regions - 1);
  builder.setInsertionPointToEnd(&inside_block);
  builder.create<tf_device::ReturnOp>(launch.getLoc(), launch.getResults());
  launch.getOperation()->moveBefore(inside_block.getTerminator());
}

void TPUExtractOutsideCompilation::runOnFunction() {
  auto extract_result = getFunction().walk([&](tf_device::LaunchOp launch) {
    ClusterMap clusters;
    if (failed(CollectAndGroupClusterOps(&launch.GetBody(), &clusters)))
      return WalkResult::interrupt();

    if (clusters.empty()) return WalkResult::advance();

    CreateParallelExecuteFromClusters(launch, clusters);

    return WalkResult::advance();
  });

  if (extract_result.wasInterrupted()) return signalPassFailure();
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>>
CreateTPUExtractOutsideCompilationPass() {
  return std::make_unique<TPUExtractOutsideCompilation>();
}

static PassRegistration<TPUExtractOutsideCompilation> pass(
    "tf-tpu-extract-outside-compilation",
    "Extracts TPU outside compilation to separate parallel_execute.");

}  // namespace TFTPU
}  // namespace mlir

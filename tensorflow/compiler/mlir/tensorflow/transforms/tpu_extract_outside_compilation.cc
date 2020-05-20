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

#include "absl/strings/str_cat.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TFTPU {

namespace {

constexpr char kAncestorsAttr[] = "ancestors";
constexpr char kDeviceAttr[] = "device";
constexpr char kKeyAttr[] = "key";
constexpr char kShapesAttr[] = "shapes";
constexpr char kXlaOutsideCompilationAttr[] = "_xla_outside_compilation";

// Mapping for `_xla_outside_compilation` attribute to ops of a cluster.
using OutsideClusterMap =
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
LogicalResult CollectAndGroupOutsideClusterOps(Block* block,
                                               OutsideClusterMap* clusters) {
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
void MoveOutsideClusterOpsToLaunchOp(
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
tf_device::LaunchOp CreateLaunchOpForOutsideCluster(
    OpBuilder* builder, Operation* last_cluster_op) {
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

// Propagates the return from `parallel_execute_op` to parent replicate
// op if it exists.
void PropagateParallelExecuteReturnToReplicate(
    tf_device::ParallelExecuteOp parallel_execute_op) {
  // Update the return for the parallel_execute op parent.
  auto replicate = llvm::dyn_cast_or_null<tf_device::ReplicateOp>(
      parallel_execute_op.getParentOp());
  if (replicate)
    replicate.GetBody().getTerminator()->setOperands(
        parallel_execute_op.execute_outputs());
}

// Extracts all externally provided operands of `cluster_ops`.
llvm::SmallSetVector<Value, 4> GetExternalOperands(
    const llvm::SmallVector<Operation*, 8>& cluster_ops) {
  llvm::SmallSetVector<Value, 4> external_values;

  for (Operation* op : cluster_ops) {
    for (Value v : op->getOperands()) {
      Operation* defining_op = v.getDefiningOp();
      if (!defining_op) continue;
      bool is_external = llvm::none_of(cluster_ops, [&](Operation* cluster_op) {
        return defining_op == cluster_op;
      });

      if (is_external) external_values.insert(v);
    }
  }

  return external_values;
}

void MoveOutsideCompiledOps(
    tf_device::ClusterOp tpu_cluster, llvm::StringRef outside_cluster_name,
    tf_device::LaunchOp host_launch_op,
    const llvm::SmallVector<Operation*, 8>& cluster_ops,
    const llvm::SmallSetVector<Value, 4>& external_inputs,
    const llvm::SmallVector<Value, 4>& external_outputs) {
  if (external_inputs.empty() && external_outputs.empty()) {
    MoveOutsideClusterOpsToLaunchOp(host_launch_op, cluster_ops);
    return;
  }

  OpBuilder builder(host_launch_op.GetBody().getTerminator());
  auto result_type =
      RankedTensorType::get({}, builder.getType<TF::StringType>());

  std::string txt_metadata;
  std::string txt_module;
  // TODO(b/157054714): Use a better abstraction instead of _TPUCompileMlirOp
  // and _XlaRecvAtHostOp and _XlaSendFromHostOp.

  // A placeholder _TpuCompileMlirOp is created because it is required input to
  // XlaRecvAtHostOp and XlaSendFromHostOp but the _TpuCompileMlirOp has not yet
  // been created for the TPU cluster that contains the outside compiled ops.
  // This placeholder should be replaced by the TPU cluster _TPUCompileMlirOp in
  // a subsequent pass.
  auto compile_op = builder.create<TF::_TPUCompileMlirOp>(
      tpu_cluster.getLoc(), /*compilation_status=*/result_type, /*program=*/
      llvm::ArrayRef<Type>{result_type}, llvm::ArrayRef<Value>{}, txt_module,
      txt_metadata);

  llvm::SmallVector<Type, 4> host_output_types;
  for (const auto& external_input : external_inputs)
    host_output_types.push_back(external_input.getType());

  std::string communication_key =
      absl::StrCat("host_compute_channel_", outside_cluster_name.str());
  // XlaRecvAtHostOp takes both the program key(dynamic_key) from the
  // _TpuCompileMlirOp and the communication_key.
  auto recv_at_host = builder.create<TF::_XlaRecvAtHostOp>(
      tpu_cluster.getLoc(), host_output_types,
      /*dynamic_key=*/compile_op.getResult(1),
      builder.getStringAttr(communication_key),
      builder.getIntegerAttr(builder.getIntegerType(64), 0));

  // TODO(b/156006200): Handle host->device outputs.
  builder.setInsertionPoint(cluster_ops.front());
  auto host_compute = builder.create<TF::_HostComputeMlirOp>(
      tpu_cluster.getLoc(), llvm::ArrayRef<Type>{},
      external_inputs.getArrayRef(), llvm::ArrayRef<NamedAttribute>{});
  host_compute.setAttr(kAncestorsAttr, builder.getArrayAttr({}));
  host_compute.setAttr(kShapesAttr, builder.getArrayAttr({}));
  host_compute.setAttr(kKeyAttr, builder.getStringAttr(communication_key));
  MoveOutsideClusterOpsToLaunchOp(host_launch_op, cluster_ops);

  for (auto result : llvm::zip(external_inputs, recv_at_host.getResults()))
    mlir::replaceAllUsesInRegionWith(std::get<0>(result), std::get<1>(result),
                                     host_launch_op.body());
}

// Creates a `parallel_execute` op in place of launch with 'clusters` and
// 'launch` as regions.
void CreateParallelExecuteFromOutsideClusters(
    tf_device::ClusterOp tpu_cluster, const OutsideClusterMap& clusters) {
  OpBuilder builder(tpu_cluster);
  // Create parallel_execute regions.  The original TPU cluster computation
  // is the extra region.
  const int num_regions = 1 + clusters.size();
  auto parallel_execute_op = builder.create<tf_device::ParallelExecuteOp>(
      tpu_cluster.getLoc(), num_regions, tpu_cluster.results().getTypes());

  // Move outside compilation clusters to parallel_execute regions.
  for (const auto& cluster : llvm::enumerate(clusters)) {
    const auto& cluster_ops = cluster.value().getSecond();

    Block& outside_block =
        parallel_execute_op.GetRegionBlockWithIndex(cluster.index());
    builder.setInsertionPointToEnd(&outside_block);
    tf_device::LaunchOp host_launch_op =
        CreateLaunchOpForOutsideCluster(&builder, cluster_ops.back());

    // Determine if there are any inputs that are provided out of cluster.
    auto external_inputs = GetExternalOperands(cluster_ops);
    llvm::SmallVector<Value, 4> external_outputs;
    // TODO(b/156006200): Compute the external outputs.

    MoveOutsideCompiledOps(tpu_cluster, cluster.value().getFirst(),
                           host_launch_op, cluster_ops, external_inputs,
                           external_outputs);

    builder.setInsertionPointToEnd(&outside_block);
    // TODO(b/154363171): Handle returns from OutsideCompiled parallel_execute
    // regions either through communication with TPU parallel_execute regions
    // or modifying parallel_execute returns.
    builder.create<tf_device::ReturnOp>(tpu_cluster.getLoc(),
                                        ArrayRef<Value>{});
  }

  // Move the launch body to last parallel_execute block.
  Block& parallel_execute_tpu_block =
      parallel_execute_op.GetRegionBlockWithIndex(num_regions - 1);
  builder.setInsertionPointToEnd(&parallel_execute_tpu_block);
  builder.create<tf_device::ReturnOp>(tpu_cluster.getLoc(),
                                      tpu_cluster.getResults());
  tpu_cluster.getOperation()->moveBefore(
      parallel_execute_tpu_block.getTerminator());

  PropagateParallelExecuteReturnToReplicate(parallel_execute_op);
  // TODO(b/154363171): Handle returns from OutsideCompiled parallel_execute
  // regions either through communication with TPU parallel_execute regions
  // or modifying parallel_execute returns.
}

void TPUExtractOutsideCompilation::runOnFunction() {
  auto extract_result =
      getFunction().walk([&](tf_device::ClusterOp tpu_cluster) {
        OutsideClusterMap clusters;
        if (failed(CollectAndGroupOutsideClusterOps(&tpu_cluster.GetBody(),
                                                    &clusters)))
          return WalkResult::interrupt();

        if (clusters.empty()) return WalkResult::advance();

        CreateParallelExecuteFromOutsideClusters(tpu_cluster, clusters);

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

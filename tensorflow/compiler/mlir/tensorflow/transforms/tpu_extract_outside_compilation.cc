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
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"

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
// annotation from a TPU cluster. Each outside compilation cluster is moved to
// a parallel_execute region. The TPU cluster is also moved to a
// parallel_execute region. Communication ops between device and host are
// added to pass inputs/outputs to/from the outside compiled region.
//
// A simple example:
//   "tf_device.cluster"() ( {
//     "tf.A"()
//     "tf.B"() {_xla_outside_compilation = "cluster1"}
//     "tf.C"()
//     tf_device.return
//   }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []}
//
// Would become the following ops (unimportant attribute, type are omitted):
//   "tf_device.parallel_execute"() ( {
//     "tf_device.launch"() ( {
//       "tf.B()
//       tf_device.return
//     })
//     tf_device.return
//   }, {
//     "tf_device.cluster"( {
//       "tf.A"()
//       "tf.C"()
//       tf_device.return
//     })
//    tf_device.return
//  })

struct TPUExtractOutsideCompilation
    : public PassWrapper<TPUExtractOutsideCompilation,
                         OperationPass<ModuleOp>> {
  void runOnOperation() override;
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
void MoveOutsideClusterOpsToLaunchOp(tf_device::LaunchOp launch_op,
                                     llvm::ArrayRef<Operation*> cluster_ops) {
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
    OpBuilder* builder, Operation* last_cluster_op,
    llvm::StringRef host_device) {
  // An empty string placeholder is used for the device as that will be later
  // populated with the device of the associated TPUReplicateMetadata op.
  llvm::SmallVector<Type, 8> result_types;
  auto launch_op = builder->create<tf_device::LaunchOp>(
      last_cluster_op->getLoc(), builder->getStringAttr(host_device),
      result_types);

  launch_op.body().push_back(new Block);

  // Add terminator.
  builder->setInsertionPointToEnd(&launch_op.GetBody());
  builder->create<tf_device::ReturnOp>(last_cluster_op->getLoc(),
                                       llvm::ArrayRef<Value>{});

  return launch_op;
}

// Extracts all externally provided operands of `cluster_ops`.
llvm::SmallSetVector<Value, 4> GetExternalOperands(
    llvm::ArrayRef<Operation*> cluster_ops) {
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

// Extracts all externally used outputs of `cluster_ops`.
llvm::SmallVector<Value, 4> GetExternalOutputs(
    llvm::ArrayRef<Operation*> cluster_ops) {
  llvm::SmallSetVector<Value, 4> external_outputs;

  for (Operation* op : cluster_ops) {
    for (Operation* user : op->getUsers()) {
      bool is_external = llvm::none_of(cluster_ops, [&](Operation* cluster_op) {
        return user == cluster_op;
      });
      if (!is_external) continue;
      for (Value v : user->getOperands()) {
        if (v.getDefiningOp() == op) external_outputs.insert(v);
      }
    }
  }

  return external_outputs.takeVector();
}

// Sets the insertion point on `builder` for HostCompute op.  Sets insertion
// point to the first op in `cluster_ops` that has one of `external_inputs`
// as an operand.  If there are no external_inputs, set insertion point to first
// cluster_op.
void SetHostComputeInsertion(
    OpBuilder* builder, llvm::ArrayRef<Operation*> cluster_ops,
    const llvm::SmallSetVector<Value, 4>& external_inputs) {
  if (external_inputs.empty()) builder->setInsertionPoint(cluster_ops.front());
  for (const auto& cluster_op : cluster_ops) {
    for (Value v : cluster_op->getOperands()) {
      if (external_inputs.count(v)) {
        builder->setInsertionPoint(cluster_op);
        return;
      }
    }
  }
}

// Creates the HostCompute with `inputs` and `outputs`
// using `communication_key`.
TF::_HostComputeMlirOp CreateHostCompute(
    OpBuilder* builder, tf_device::ClusterOp tpu_cluster,
    llvm::ArrayRef<Operation*> cluster_ops,
    const llvm::SmallSetVector<Value, 4>& inputs, llvm::ArrayRef<Value> outputs,
    llvm::StringRef communication_key) {
  llvm::SmallVector<Type, 4> device_output_types;
  for (const auto& output : outputs)
    device_output_types.push_back(output.getType());
  SetHostComputeInsertion(builder, cluster_ops, inputs);
  auto host_compute = builder->create<TF::_HostComputeMlirOp>(
      tpu_cluster.getLoc(), device_output_types, inputs.getArrayRef(),
      llvm::ArrayRef<NamedAttribute>{});
  host_compute.setAttr(kAncestorsAttr, builder->getArrayAttr({}));
  host_compute.setAttr(kShapesAttr, builder->getArrayAttr({}));
  host_compute.setAttr(kKeyAttr, builder->getStringAttr(communication_key));
  return host_compute;
}

void MoveOutsideCompiledOps(
    tf_device::ClusterOp tpu_cluster, llvm::StringRef outside_cluster_name,
    tf_device::LaunchOp host_launch_op, llvm::ArrayRef<Operation*> cluster_ops,
    const llvm::SmallSetVector<Value, 4>& external_inputs,
    llvm::ArrayRef<Value> external_outputs) {
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
      llvm::formatv("host_compute_channel_{0}", outside_cluster_name).str();
  // XlaRecvAtHostOp takes both the program key(dynamic_key) from the
  // _TpuCompileMlirOp and the communication_key.
  auto recv_at_host = builder.create<TF::_XlaRecvAtHostOp>(
      tpu_cluster.getLoc(), host_output_types,
      /*dynamic_key=*/compile_op.getResult(1),
      builder.getStringAttr(communication_key),
      builder.getIntegerAttr(builder.getIntegerType(64), 0));

  auto host_compute =
      CreateHostCompute(&builder, tpu_cluster, cluster_ops, external_inputs,
                        external_outputs, communication_key);
  MoveOutsideClusterOpsToLaunchOp(host_launch_op, cluster_ops);

  builder.setInsertionPoint(host_launch_op.GetBody().getTerminator());
  builder.create<TF::_XlaSendFromHostOp>(
      tpu_cluster.getLoc(), external_outputs,
      /*dynamic_key=*/compile_op.getResult(1),
      builder.getStringAttr(communication_key),
      /*device_ordinal=*/builder.getIntegerAttr(builder.getIntegerType(64), 0));

  for (auto result : llvm::zip(external_inputs, recv_at_host.getResults()))
    mlir::replaceAllUsesInRegionWith(std::get<0>(result), std::get<1>(result),
                                     host_launch_op.body());

  for (auto result : llvm::zip(external_outputs, host_compute.getResults()))
    mlir::replaceAllUsesInRegionWith(std::get<0>(result), std::get<1>(result),
                                     tpu_cluster.body());
}

// Creates a `parallel_execute` op in place of launch with 'clusters` and
// 'launch` as regions.
void CreateParallelExecuteFromOutsideClusters(tf_device::ClusterOp tpu_cluster,
                                              const OutsideClusterMap& clusters,
                                              llvm::StringRef host_device) {
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
    tf_device::LaunchOp host_launch_op = CreateLaunchOpForOutsideCluster(
        &builder, cluster_ops.back(), host_device);

    // Determine if there are any inputs that are provided out of cluster.
    auto external_inputs = GetExternalOperands(cluster_ops);
    auto external_outputs = GetExternalOutputs(cluster_ops);

    MoveOutsideCompiledOps(tpu_cluster, cluster.value().getFirst(),
                           host_launch_op, cluster_ops, external_inputs,
                           external_outputs);

    builder.setInsertionPointToEnd(&outside_block);
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

  // Remap cluster results with parallel_execute results if user is outside of
  // parallel_execute.
  for (auto result :
       llvm::zip(tpu_cluster.getResults(), parallel_execute_op.getResults())) {
    Value tpu_cluster_result = std::get<0>(result);
    Value parallel_execute_result = std::get<1>(result);
    for (auto& use : llvm::make_early_inc_range(tpu_cluster_result.getUses()))
      if (!parallel_execute_op.getOperation()->isProperAncestor(use.getOwner()))
        use.set(parallel_execute_result);
  }
}

void TPUExtractOutsideCompilation::runOnOperation() {
  // Get runtime devices information from the closest parent module.
  auto module = getOperation();
  mlir::TF::RuntimeDevices devices;
  if (failed(tensorflow::GetDevicesFromOp(module, &devices)))
    return signalPassFailure();

  auto extract_result =
      module.walk([&](tf_device::ClusterOp tpu_cluster) {
        OutsideClusterMap clusters;
        if (failed(CollectAndGroupOutsideClusterOps(&tpu_cluster.GetBody(),
                                                    &clusters)))
          return WalkResult::interrupt();

        if (clusters.empty()) return WalkResult::advance();

        std::string host_device;
        tensorflow::GetHostDeviceOutsideComputation(devices, tpu_cluster,
                                                    &host_device);
        CreateParallelExecuteFromOutsideClusters(tpu_cluster, clusters,
                                                 host_device);

        return WalkResult::advance();
      });

  if (extract_result.wasInterrupted()) return signalPassFailure();
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateTPUExtractOutsideCompilationPass() {
  return std::make_unique<TPUExtractOutsideCompilation>();
}

static PassRegistration<TPUExtractOutsideCompilation> pass(
    "tf-tpu-extract-outside-compilation",
    "Extracts TPU outside compilation to separate parallel_execute.");

}  // namespace TFTPU
}  // namespace mlir

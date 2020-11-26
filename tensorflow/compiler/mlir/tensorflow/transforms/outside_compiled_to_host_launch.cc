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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"

namespace mlir {
namespace TFTPU {

namespace {

constexpr char kDeviceAttr[] = "device";
constexpr char kXlaOutsideCompilationAttr[] = "_xla_outside_compilation";

// Mapping for `_xla_outside_compilation` attribute to ops of a cluster.
using OutsideClusterMap =
    llvm::SmallDenseMap<llvm::StringRef, llvm::SmallVector<Operation*, 8>, 8>;

// This pass wraps ops with the same `_xla_outside_compilation`
// attribute value in a tf_device.launch op with host device assignment.
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
//   "tf_device.cluster"() ( {
//     "tf.A"()
//     "tf_device.launch"() {
//       "tf.B"() {_xla_outside_compilation = "cluster1"}
//       tf_device.return
//     } {device = "TPU_REPLICATED_HOST"} : () -> ()
//     "tf.C"()
//     tf_device.return
//   }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []}
//

struct OutsideCompiledToHostLaunch
    : public PassWrapper<OutsideCompiledToHostLaunch, OperationPass<ModuleOp>> {
  void runOnOperation() override;
};

// Collects and clusters ops in `block` with the same `_xla_outside_compilation`
// attribute into `clusters` This returns an error if a
// `_xla_outside_compilation` attribute of an op is empty.
LogicalResult CollectAndGroupOutsideClusterOps(Block* block,
                                               OutsideClusterMap* clusters) {
  auto walk_result = block->walk([&](Operation* op) {
    if (auto attr = op->getAttrOfType<StringAttr>(kXlaOutsideCompilationAttr)) {
      if (attr.getValue().empty()) {
        op->emitOpError() << "requires non empty '"
                          << kXlaOutsideCompilationAttr << "' string attribute";
        return WalkResult::interrupt();
      }

      auto it = clusters->try_emplace(attr.getValue());
      it.first->getSecond().push_back(op);
    }
    return WalkResult::advance();
  });

  return failure(walk_result.wasInterrupted());
}

// Extracts all externally used outputs of `cluster_ops`.
llvm::SmallSetVector<Value, 4> GetExternalOutputs(
    llvm::ArrayRef<Operation*> cluster_ops) {
  llvm::SmallSetVector<Value, 4> external_outputs;
  llvm::SmallPtrSet<Operation*, 4> host_cluster_ops_set;
  for (auto op : cluster_ops) {
    op->walk([&](Operation* host_cluster_op) {
      host_cluster_ops_set.insert(host_cluster_op);
    });
  }

  for (Operation* op : cluster_ops) {
    for (Operation* user : op->getUsers()) {
      bool is_external = llvm::none_of(
          host_cluster_ops_set,
          [&](Operation* cluster_op) { return user == cluster_op; });
      if (!is_external) continue;
      for (Value v : user->getOperands()) {
        if (v.getDefiningOp() == op) external_outputs.insert(v);
      }
    }
  }

  return external_outputs;
}

void WrapClusterInLaunch(llvm::ArrayRef<Operation*> cluster_ops,
                         llvm::StringRef host_device) {
  auto* last_cluster_op = cluster_ops.back();
  OpBuilder builder(last_cluster_op);
  llvm::SmallVector<Type, 4> launch_output_types;
  auto external_outputs = GetExternalOutputs(cluster_ops);
  for (const auto& external_output : external_outputs)
    launch_output_types.push_back(external_output.getType());

  auto launch_op = builder.create<tf_device::LaunchOp>(
      last_cluster_op->getLoc(), builder.getStringAttr(host_device),
      /*result_types=*/launch_output_types);
  for (auto result : llvm::zip(external_outputs, launch_op.getResults())) {
    std::get<0>(result).replaceAllUsesWith(std::get<1>(result));
  }

  launch_op.body().push_back(new Block);
  builder.setInsertionPointToEnd(&launch_op.GetBody());
  auto* return_op =
      builder
          .create<tf_device::ReturnOp>(last_cluster_op->getLoc(),
                                       external_outputs.getArrayRef())
          .getOperation();
  MLIRContext* context = launch_op.getContext();
  for (Operation* cluster_op : cluster_ops) {
    cluster_op->removeAttr(
        Identifier::get(kXlaOutsideCompilationAttr, context));
    cluster_op->removeAttr(Identifier::get(kDeviceAttr, context));
    cluster_op->moveBefore(return_op);
  }
}

void OutsideCompiledToHostLaunch::runOnOperation() {
  // Get runtime devices information from the closest parent module.
  auto module = getOperation();
  mlir::TF::RuntimeDevices devices;
  if (failed(tensorflow::GetDevicesFromOp(module, &devices)))
    return signalPassFailure();

  auto result = module.walk([&](tf_device::ClusterOp tpu_cluster) {
    OutsideClusterMap clusters;
    if (failed(CollectAndGroupOutsideClusterOps(&tpu_cluster.GetBody(),
                                                &clusters)))
      return WalkResult::interrupt();

    if (clusters.empty()) return WalkResult::advance();

    std::string host_device;
    tensorflow::GetHostDeviceOutsideComputation(devices, tpu_cluster,
                                                &host_device);
    for (const auto& cluster : clusters) {
      WrapClusterInLaunch(cluster.getSecond(), host_device);
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) return signalPassFailure();
}

}  // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateOutsideCompiledToHostLaunchPass() {
  return std::make_unique<OutsideCompiledToHostLaunch>();
}

static PassRegistration<OutsideCompiledToHostLaunch> pass(
    "tf-outside-compiled-to-host-launch",
    "Wraps ops with ithe same _xla_outside_compiled attribute in "
    "tf_device.launch on replicated host device.");

}  // namespace TFTPU
}  // namespace mlir

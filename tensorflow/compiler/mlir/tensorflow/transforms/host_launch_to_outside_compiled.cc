/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Analysis/CallGraph.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_device_passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"

namespace mlir {
namespace TFDevice {

namespace {

constexpr char kDeviceAttr[] = "device";
constexpr char kXlaOutsideCompilationAttr[] = "_xla_outside_compilation";

struct HostLaunchToOutsideCompiledPass
    : public HostLaunchToOutsideCompiledPassBase<
          HostLaunchToOutsideCompiledPass> {
  void runOnOperation() override;
};

// Assign all ops in region with _xla_outside_compilation attribute.
void MarkOutsideCompiledInRegion(Region& region) {
  region.walk([&](Operation* op) {
    op->setAttr(kXlaOutsideCompilationAttr,
                StringAttr::get(op->getContext(), "from_launch"));
  });
}

void HoistOpsAndAnnotateWithOutsideCompilation(tf_device::LaunchOp launch) {
  // Forward launch inner op results to launch op results.
  launch.replaceAllUsesWith(launch.GetBody().getTerminator()->getOperands());

  // For all inner ops, assign the launch device as a `device` attribute.
  MarkOutsideCompiledInRegion(launch.body());

  // Move all inner ops of the launch to the block containing the launch.
  auto body = launch.GetBody().without_terminator();
  Operation* launch_op = launch.getOperation();
  launch_op->getBlock()->getOperations().splice(
      launch_op->getIterator(), launch.GetBody().getOperations(), body.begin(),
      body.end());

  launch.erase();
}

void HostLaunchToOutsideCompiledPass::runOnOperation() {
  ModuleOp module = getOperation();
  mlir::TF::RuntimeDevices devices;
  if (failed(tensorflow::GetDevicesFromOp(module, &devices)))
    return signalPassFailure();
  const CallGraph call_graph(module);
  // symbol_table caches callees in the CallGraph.
  SymbolTableCollection symbol_table;
  // List pending nodes to traverse with their root TPU cluster.
  llvm::SmallVector<std::pair<CallGraphNode*, tf_device::ClusterOp>>
      pending_call_nodes;
  // Cache the host device for each TPU cluster.
  std::unordered_map<Operation*, std::string> cluster_to_host;

  // traverse_op(op, c) is applied to each op reachable from tpu_cluster c.
  auto traverse_op = [&](Operation* op, tf_device::ClusterOp tpu_cluster) {
    // Add callee nodes to pending_call_nodes.
    if (CallOpInterface call = dyn_cast<CallOpInterface>(op)) {
      CallGraphNode* node = call_graph.resolveCallable(call, symbol_table);
      pending_call_nodes.emplace_back(node, tpu_cluster);
    }
    // Hoist launch.
    if (tf_device::LaunchOp launch = dyn_cast<tf_device::LaunchOp>(op)) {
      StringAttr device_attr = launch->getAttrOfType<StringAttr>(kDeviceAttr);
      std::string host_device = cluster_to_host[tpu_cluster.getOperation()];
      if (device_attr && device_attr.getValue().equals(host_device))
        HoistOpsAndAnnotateWithOutsideCompilation(launch);
    }
  };

  // Traverse ops in each TPU cluster.
  module.walk([&](tf_device::ClusterOp tpu_cluster) {
    std::string host_device;
    // If there is model parallelism, we return early since
    // GetHostDeviceOutsideComputation will fail and an error should have been
    // returned in an earlier pass.
    // TODO(b/186420116): Remove this check once outside compilation and model
    // parallelism work together.
    if (tensorflow::HasModelParallelism(tpu_cluster)) return;
    if (failed(tensorflow::GetHostDeviceOutsideComputation(devices, tpu_cluster,
                                                           &host_device)))
      return signalPassFailure();
    cluster_to_host[tpu_cluster.getOperation()] = host_device;
    tpu_cluster.walk([&](Operation* op) { traverse_op(op, tpu_cluster); });
  });

  // Traverse ops that are reachable from some TPU cluster.
  // node_to_cluster is used to avoid traversing the same node twice, and to
  // check that no node is reachable from multiple TPU clusters.
  std::unordered_map<CallGraphNode*, tf_device::ClusterOp> node_to_cluster;
  while (!pending_call_nodes.empty()) {
    auto pair = pending_call_nodes.back();
    pending_call_nodes.pop_back();
    CallGraphNode* node = pair.first;
    tf_device::ClusterOp tpu_cluster = pair.second;
    if (node_to_cluster.count(node)) {
      if (node_to_cluster[node].getOperation() != tpu_cluster) {
        node->getCallableRegion()->getParentOp()->emitOpError(
            "The same function is reachable from multiple TPU Clusters.");
      }
    } else {
      node_to_cluster[node] = tpu_cluster;
      node->getCallableRegion()->walk(
          [&](Operation* op) { traverse_op(op, tpu_cluster); });
    }
  }
}

}  // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateHostLaunchToOutsideCompiledPass() {
  return std::make_unique<HostLaunchToOutsideCompiledPass>();
}

}  // namespace TFDevice
}  // namespace mlir

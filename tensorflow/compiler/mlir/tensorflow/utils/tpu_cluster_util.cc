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

#include "mlir/Analysis/CallGraph.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"

namespace mlir {
namespace TFTPU {

namespace {
mlir::LogicalResult WalkReachableFromTpuCluster(
    bool pass_host_device, ModuleOp module,
    std::function<WalkResult(Operation*, tf_device::ClusterOp,
                             std::optional<std::string>)>
        callback) {
  mlir::TF::RuntimeDevices devices;
  if (failed(tensorflow::GetDevicesFromOp(module, &devices))) return failure();
  const CallGraph call_graph(module);
  // symbol_table caches callees in the CallGraph.
  SymbolTableCollection symbol_table;
  // List pending nodes to traverse with their root TPU cluster.
  llvm::SmallVector<std::pair<CallGraphNode*, tf_device::ClusterOp>>
      pending_call_nodes;
  // Cache the host device for each TPU cluster.
  std::unordered_map<Operation*, std::optional<std::string>> cluster_to_host;

  auto insert_pending_op = [&](Operation* op,
                               tf_device::ClusterOp tpu_cluster) {
    // Add callee nodes to pending_call_nodes.
    if (CallOpInterface call = dyn_cast<CallOpInterface>(op)) {
      CallGraphNode* node = call_graph.resolveCallable(call, symbol_table);
      pending_call_nodes.emplace_back(node, tpu_cluster);
    }
  };

  // Traverse ops in each TPU cluster.
  auto result = module.walk([&](tf_device::ClusterOp tpu_cluster) {
    std::optional<std::string> host_device;
    if (pass_host_device) {
      std::string host_device_value;
      if (failed(tensorflow::GetHostDeviceOutsideComputation(
              devices, tpu_cluster, &host_device_value)))
        return WalkResult::interrupt();
      host_device = host_device_value;
    }
    cluster_to_host[tpu_cluster.getOperation()] = host_device;
    return tpu_cluster.walk([&](Operation* op) {
      insert_pending_op(op, tpu_cluster);
      return callback(op, tpu_cluster,
                      cluster_to_host[tpu_cluster.getOperation()]);
    });
  });
  if (result.wasInterrupted()) return failure();

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
      auto result = node->getCallableRegion()->walk([&](Operation* op) {
        insert_pending_op(op, tpu_cluster);
        return callback(op, tpu_cluster,
                        cluster_to_host[tpu_cluster.getOperation()]);
      });
      if (result.wasInterrupted()) return failure();
    }
  }

  return success();
}
}  // namespace

mlir::LogicalResult WalkReachableFromTpuCluster(
    ModuleOp module, std::function<WalkResult(Operation*, tf_device::ClusterOp,
                                              std::optional<std::string>)>
                         callback) {
  return WalkReachableFromTpuCluster(true, module, callback);
}

mlir::LogicalResult WalkReachableFromTpuCluster(
    ModuleOp module,
    std::function<WalkResult(Operation*, tf_device::ClusterOp)> callback) {
  auto with_host_op = [&](Operation* op, tf_device::ClusterOp tpu_cluster,
                          std::optional<std::string> host_device) {
    return callback(op, tpu_cluster);
  };
  return WalkReachableFromTpuCluster(false, module, with_host_op);
}

}  // namespace TFTPU
}  // namespace mlir

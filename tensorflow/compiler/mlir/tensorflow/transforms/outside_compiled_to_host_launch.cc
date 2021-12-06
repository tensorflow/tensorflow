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

#include "llvm/ADT/SmallVector.h"
#include "mlir/Analysis/CallGraph.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"

namespace mlir {
namespace TFTPU {

namespace {

constexpr char kDeviceAttr[] = "device";
constexpr char kXlaOutsideCompilationAttr[] = "_xla_outside_compilation";

struct OutsideCompiledToHostLaunchPass
    : public TF::OutsideCompiledToHostLaunchPassBase<
          OutsideCompiledToHostLaunchPass> {
  void runOnOperation() override;
};

void WrapOpInLaunch(Operation* host_op, llvm::StringRef host_device) {
  OpBuilder builder(host_op);

  auto launch_op = builder.create<tf_device::LaunchOp>(
      host_op->getLoc(), builder.getStringAttr(host_device),
      /*result_types=*/host_op->getResultTypes());
  host_op->replaceAllUsesWith(launch_op);

  launch_op.body().push_back(new Block);
  builder.setInsertionPointToEnd(&launch_op.GetBody());
  auto* return_op =
      builder
          .create<tf_device::ReturnOp>(host_op->getLoc(), host_op->getResults())
          .getOperation();
  MLIRContext* context = launch_op.getContext();
  host_op->removeAttr(Identifier::get(kXlaOutsideCompilationAttr, context));
  host_op->removeAttr(Identifier::get(kDeviceAttr, context));
  host_op->moveBefore(return_op);
}

void OutsideCompiledToHostLaunchPass::runOnOperation() {
  // Get runtime devices information from the closest parent module.
  auto module = getOperation();
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
    // Apply WrapOpInLaunch when the op has _xla_outside_compilation.
    if (op->hasAttrOfType<StringAttr>(kXlaOutsideCompilationAttr)) {
      if (tensorflow::HasModelParallelism(tpu_cluster)) {
        tpu_cluster.emitOpError(
            "outside compilation is not supported with model parallelism.");
        return WalkResult::interrupt();
      }
      WrapOpInLaunch(op, cluster_to_host[tpu_cluster.getOperation()]);
    }
    return WalkResult::advance();
  };

  // Traverse ops in each TPU cluster.
  auto result = module.walk([&](tf_device::ClusterOp tpu_cluster) {
    std::string host_device;
    if (failed(tensorflow::GetHostDeviceOutsideComputation(devices, tpu_cluster,
                                                           &host_device)))
      return WalkResult::interrupt();
    cluster_to_host[tpu_cluster.getOperation()] = host_device;
    return tpu_cluster.walk(
        [&](Operation* op) { return traverse_op(op, tpu_cluster); });
  });
  if (result.wasInterrupted()) return signalPassFailure();

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
      auto result = node->getCallableRegion()->walk(
          [&](Operation* op) { return traverse_op(op, tpu_cluster); });
      if (result.wasInterrupted()) return signalPassFailure();
    }
  }
}

}  // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateOutsideCompiledToHostLaunchPass() {
  return std::make_unique<OutsideCompiledToHostLaunchPass>();
}

}  // namespace TFTPU
}  // namespace mlir

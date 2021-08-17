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

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/cluster_ops_by_policy.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_cpurt_clustering.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_cpurt_passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_cpurt_passes.h.inc"

using llvm::ArrayRef;
using mlir::TFDevice::Cluster;
using mlir::TFDevice::ClusteringPolicySet;
using mlir::TFDevice::CreateClusterOp;
using mlir::TFDevice::FindClustersInTheBlock;

// -------------------------------------------------------------------------- //
// Cluster operations based on the TF CPURT clustering policy.
// -------------------------------------------------------------------------- //
struct ClusteringPass : public ClusteringBase<ClusteringPass> {
  ClusteringPass() = default;
  ClusteringPass(ArrayRef<std::string> cluster_oplist, int cluster_min_size) {
    oplist = cluster_oplist;
    min_cluster_size = cluster_min_size;
  }

  void runOnFunction() override {
    ClusteringPolicySet policies;

    // Parse clustering tier and operations filter from the oplist.
    llvm::DenseSet<llvm::StringRef> opset;
    llvm::Optional<CpurtClusteringTier> tier;

    for (const auto& op : oplist) {
      if (op == "tier1") {
        tier = CpurtClusteringTier::kTier1;
      } else if (op == "all") {
        tier = CpurtClusteringTier::kAll;
      } else {
        opset.insert(op);
      }
    }

    // Run clustering only if the clustering tier or supported operations are
    // explicitly defined by the oplist.
    if (!tier.hasValue() && opset.empty()) return;

    // If the clustering tier is not defined, it means that the opset will later
    // filter supported operations, so it's ok to use `all` tier.
    populateTfCpurtClusteringPolicies(
        policies, tier.getValueOr(CpurtClusteringTier::kAll));

    // If opset is not empty restrict operations that are enabled for
    // clustering.
    auto filter = [&](mlir::Operation* op) -> bool {
      return opset.empty() || opset.contains(op->getName().getStringRef());
    };

    // Annotate all formed clusters with an attribute.
    auto policy = mlir::StringAttr::get(&getContext(), "tfrt.auto-fusion");

    getFunction().walk([&](mlir::Block* block) {
      for (Cluster& cluster : FindClustersInTheBlock(block, policies, filter)) {
        // Do not create too small clusters.
        if (cluster.operations.size() < min_cluster_size) continue;
        // Verify that JIT runtime can compile the cluster.
        if (failed(VerifyCluster(cluster))) continue;

        CreateClusterOp(cluster, policy);
      }
    });
  }
};

}  // namespace

std::unique_ptr<mlir::FunctionPass> CreateTfCpurtClusteringPass() {
  return std::make_unique<ClusteringPass>();
}

std::unique_ptr<mlir::FunctionPass> CreateTfCpurtClusteringPass(
    llvm::ArrayRef<std::string> oplist, int min_cluster_size) {
  return std::make_unique<ClusteringPass>(oplist, min_cluster_size);
}

}  // namespace tensorflow

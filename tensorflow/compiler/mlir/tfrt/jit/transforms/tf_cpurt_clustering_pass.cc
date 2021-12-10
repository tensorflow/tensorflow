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

#include <string>

#include "llvm/ADT/STLExtras.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/cluster_ops_by_policy.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_cpurt_clustering.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_cpurt_passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_cpurt_passes.h.inc"

using llvm::ArrayRef;

using mlir::TF::ConstOp;
using mlir::TF::HashTableV2Op;
using mlir::TF::ReadVariableOp;

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
      if (op == "tier0") {
        tier = CpurtClusteringTier::kTier0;
      } else if (op == "tier1") {
        tier = CpurtClusteringTier::kTier1;
      } else if (op == "tier1metadata") {
        tier = CpurtClusteringTier::kTier1Metadata;
      } else if (op == "tier1reductions") {
        tier = CpurtClusteringTier::kTier1Reductions;
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
    auto opset_filter = [&](mlir::Operation* op) -> bool {
      return opset.empty() || opset.contains(op->getName().getStringRef());
    };

    // Find operations that could be hoisted from the function body into the
    // TFRT resource initialization function. Currently it is an approximation
    // of hoisting rules in the TFRT, we just find all the operations that
    // depend only on ConstOp, ReadVariableOp or HashTableV2Op operations. We
    // don't do any side effects analysis and conservatively can mark as
    // hoistable operations that will not be hoisted by TFRT because of side
    // effect dependencies.
    //
    // TODO(ezhulenev): This should be shared with TFRT hoisting implementation.

    // Initialize a set of operations that we assume we will hoist.
    llvm::DenseSet<mlir::Operation*> hoisted_ops;
    getFunction().walk([&](mlir::Operation* op) {
      if (mlir::isa<ReadVariableOp, ConstOp, HashTableV2Op>(op))
        hoisted_ops.insert(op);
    });

    // Initialize work list with users of ReadVariableOp results.
    llvm::SmallVector<mlir::Operation*> work_list;
    for (mlir::Operation* hoisted : hoisted_ops)
      work_list.append(hoisted->user_begin(), hoisted->user_end());

    // Traverse all users until we find all operations that could be hoisted.
    while (!work_list.empty()) {
      mlir::Operation* op = work_list.pop_back_val();

      // Add operation to hoisted ops set if all operands can be hoisted.
      bool all_operands_hoisted =
          llvm::all_of(op->getOperands(), [&](mlir::Value value) {
            return hoisted_ops.contains(value.getDefiningOp());
          });
      if (!all_operands_hoisted) continue;

      hoisted_ops.insert(op);
      work_list.append(op->user_begin(), op->user_end());
    }

    auto hoist_filter = [&](mlir::Operation* op) {
      return !hoisted_ops.contains(op);
    };

    // Combine together opset and hoist filters.
    auto filter = [&](mlir::Operation* op) -> bool {
      return opset_filter(op) && hoist_filter(op);
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

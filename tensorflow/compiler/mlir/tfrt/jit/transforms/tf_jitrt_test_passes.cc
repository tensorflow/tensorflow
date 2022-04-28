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

#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_test_passes.h"

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/cluster_ops_by_policy.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.h"

namespace tensorflow {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_test_passes.h.inc"

using mlir::OperationPass;
using mlir::func::FuncOp;
using mlir::TFDevice::Cluster;
using mlir::TFDevice::ClusteringPolicySet;
using mlir::TFDevice::CreateClusterOp;
using mlir::TFDevice::FindClustersInTheBlock;
using mlir::TFDevice::ValuesConstraintSet;

// -------------------------------------------------------------------------- //
// Cluster operations based on the TF JitRt clustering policy.
// -------------------------------------------------------------------------- //
struct TestClusteringPass : public TestClusteringBase<TestClusteringPass> {
  void runOnOperation() override {
    ClusteringPolicySet policies;
    populateTfJitRtClusteringPolicies(policies);

    getOperation().walk([&](mlir::Block* block) {
      for (Cluster& cluster : FindClustersInTheBlock(block, policies)) {
        // Do not create too small clusters.
        if (cluster.operations.size() < min_cluster_size) continue;
        // Verify that JIT runtime can compile the cluster.
        if (failed(VerifyCluster(cluster))) continue;

        CreateClusterOp(cluster, {});
        EmitInputsConstraintsRemarks(getOperation(), cluster.constraints);
      }
    });
  }
};

// -------------------------------------------------------------------------- //
// Test TF JitRt clustering policy by annotating ops with constraints.
// -------------------------------------------------------------------------- //
struct TestClusteringPolicyPass
    : public TestClusteringPolicyBase<TestClusteringPolicyPass> {
  void runOnOperation() override {
    FuncOp func = getOperation();
    ValuesConstraintSet constraints;

    ClusteringPolicySet policies;
    populateTfJitRtClusteringPolicies(policies);

    // Initialize constraints based on the return type attributes.
    if (failed(InferFunctionBodyValuesConstraints(func, constraints)))
      return signalPassFailure();

    // Propagate constraints though the function body.
    auto result =
        PropagateValuesConstraints(func.getBody(), policies, constraints,
                                   /*resolve=*/false, /*emit_remarks=*/true);
    (void)result;

    // Emit remarks for all operations that use constrained values.
    EmitValueConstraintsRemarks(constraints);
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateTestTfJitRtClusteringPass() {
  return std::make_unique<TestClusteringPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateTestTfJitRtClusteringPolicyPass() {
  return std::make_unique<TestClusteringPolicyPass>();
}

}  // namespace tensorflow

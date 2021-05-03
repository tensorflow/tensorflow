/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/cluster_ops_by_policy.h"

namespace mlir {
namespace TFDevice {

namespace {
struct TestClusteringPolicyPass
    : public PassWrapper<TestClusteringPolicyPass, mlir::FunctionPass> {
  void runOnFunction() override;
};

// Clustering policy for `test.OpA` and `test.OpB` operations;
class TestOpsClusteringPolicy : public ClusteringPolicy {
  LogicalResult MatchAndUpdateConstraints(
      Operation* op, const ValuesConstraintSet& results,
      ValuesConstraintSet& operands) const final {
    // Check if operation is `test.OpA` or `test.OpB`.
    bool is_op_a = op->getName().getStringRef() == "test.OpA";
    bool is_op_b = op->getName().getStringRef() == "test.OpB";
    if (!is_op_a && !is_op_b) return failure();

    if (auto result_constraint = results.GetConstraint(op->getResult(0))) {
      // `test.OpA` converts shape constraint to rank constraint.
      if (is_op_a && *result_constraint == ValueConstraint::kShape)
        operands.Insert(op->getOperand(0), ValueConstraint::kRank);

      // `test.OpB` converts value constraint to shape constraint.
      if (*result_constraint == ValueConstraint::kValue)
        operands.Insert(op->getOperand(0), ValueConstraint::kShape);
    }

    return success();
  }
};

}  // anonymous namespace

void TestClusteringPolicyPass::runOnFunction() {
  FuncOp func = getFunction();
  ValuesConstraintSet constraints;

  ClusteringPolicySet policies;
  policies.Add<TestOpsClusteringPolicy>();

  // Initialize constraints based on the return type attributes.
  if (failed(InferFunctionBodyValuesConstraints(func, constraints)))
    return signalPassFailure();

  // Propagate constraints though the function body.
  auto result = PropagateValuesConstraints(func.body(), policies, constraints);
  (void)result;

  // Emit remarks for all operations that use constrained values.
  EmitValueConstraintsRemarks(constraints);
}

static mlir::PassRegistration<TestClusteringPolicyPass>
    tf_cpurt_test_clustering_policy(
        "tf-test-clustering-policy",
        "Annotates operations with clustering constraints inferred from set of "
        "test policies");

}  // namespace TFDevice
}  // namespace mlir

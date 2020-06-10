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

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TFTPU {

namespace {

constexpr char kXlaOutsideCompilationAttr[] = "_xla_outside_compilation";

struct TPUOutsideCompilationCluster
    : public PassWrapper<TPUOutsideCompilationCluster, FunctionPass> {
  void runOnFunction() override;
};

// Represents an outside compiled cluster. All ops that are added to the same
// cluster will be extracted together in a later pass.
class OutsideCompiledCluster {
 public:
  explicit OutsideCompiledCluster(int number)
      : cluster_name_(llvm::formatv("cluster{0}", number).str()) {}

  // Attempts to add an op to this cluster.
  // This function requires all ops to be added before their uses.
  bool AddOp(Operation* op) {
    // Check if the op is safe to add before adding it.
    bool add = IsSafeToAdd(op);
    if (add) {
      // Set the ops kXlaOutsideCompilationAttr to the cluster name.
      op->setAttr(kXlaOutsideCompilationAttr,
                  StringAttr::get(cluster_name_, op->getContext()));

      // Since we are adding the op to the cluster, the op is no longer
      // considered a user of this cluster.
      users_.erase(op);
    }

    // Add this op's users to the cluster users.
    users_.insert(op->user_begin(), op->user_end());
    return add;
  }

 private:
  // Checks if it is safe for an op to be merged into this cluster.
  bool IsSafeToAdd(Operation* op) {
    // If the op is not marked for outside compilation it doesn't belong in a
    // cluster.
    if (!op->getAttrOfType<StringAttr>(kXlaOutsideCompilationAttr))
      return false;

    // Checks to see if the op's operands are related to this
    // clusters users. If they are related, then there is an op between this
    // op and the cluster. Since ops are added before their uses, there
    // is no way for the op in-between to ever be added to this cluster
    // therefore there is no way this op can ever be added to the cluster.
    for (const Value& value : op->getOperands()) {
      Operation* op_operand = value.getDefiningOp();
      if (op_operand && users_.find(op_operand) != users_.end()) return false;
    }
    return true;
  }

  // users_ stores the direct and indirect users of the outside compiled ops in
  // this cluster. It does NOT store the outside compiled ops that are a part
  // of this cluster that will be collectively extracted and run on the cpu.
  // users_ is consulted when attempting to add a new outside compiled to the
  // cluster. If the new op's operand(s) are already in users_, it means that
  // the operand(s) were not added to the cluster so it is not safe to add the
  // new op to the cluster either.
  llvm::SmallPtrSet<Operation*, 8> users_;
  std::string cluster_name_;
};

void TPUOutsideCompilationCluster::runOnFunction() {
  llvm::SmallVector<OutsideCompiledCluster, 8> clusters;
  int cluster_counter = 0;

  getFunction().walk([&](tf_device::ClusterOp tpu_cluster) {
    for (Operation& op : tpu_cluster.GetBody()) {
      // Try to add the op to existing clusters.
      bool added = false;
      for (auto& cluster : clusters)
        if ((added = cluster.AddOp(&op))) break;

      // If the op cannot be added to existing clusters, create a new cluster.
      if (!added) {
        OutsideCompiledCluster new_cluster(cluster_counter++);
        new_cluster.AddOp(&op);
        clusters.push_back(new_cluster);
      }
    }
  });
}

}  // anonymous namespace

std::unique_ptr<OperationPass<FuncOp>>
CreateTPUOutsideCompilationClusterPass() {
  return std::make_unique<TPUOutsideCompilationCluster>();
}

static PassRegistration<TPUOutsideCompilationCluster> pass(
    "tf-tpu-outside-compilation-cluster",
    "Identifies clusters of operations assigned to outside compilation");

}  // namespace TFTPU
}  // namespace mlir

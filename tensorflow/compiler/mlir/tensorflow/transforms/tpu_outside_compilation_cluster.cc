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
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TFTPU {

namespace {

constexpr char kXlaOutsideCompilationAttr[] = "_xla_outside_compilation";

struct TPUOutsideCompilationCluster
    : public TF::PerFunctionAggregateAnalysisConsumerPass<
          TPUOutsideCompilationCluster, TF::SideEffectAnalysis> {
  void runOnFunction(FuncOp func,
                     const TF::SideEffectAnalysis::Info& side_effect_analysis);
};

bool IsVariant(Value value) {
  return getElementTypeOrSelf(value.getType()).isa<TF::VariantType>();
}

bool HasOutsideCompiledAncestor(Operation* op) {
  Operation* parent = op->getParentOp();
  while (parent) {
    if (parent->getAttrOfType<StringAttr>(kXlaOutsideCompilationAttr))
      return true;
    parent = parent->getParentOp();
  }
  return false;
}

// Represents an outside compiled cluster. All ops that are added to the same
// cluster will be extracted together in a later pass.
class OutsideCompiledCluster {
 public:
  explicit OutsideCompiledCluster(int number)
      : cluster_name_(llvm::formatv("cluster{0}", number).str()) {}

  // Attempts to add an op to this cluster. Ops can be grouped to the same
  // cluster if they have data dependency and are inside the same block.
  bool AddOp(Operation* op,
             const TF::SideEffectAnalysis::Info& side_effect_analysis) {
    // Check if the op is safe to add before adding it.
    if (IsSafeToAdd(op, side_effect_analysis)) {
      op->setAttr(kXlaOutsideCompilationAttr,
                  StringAttr::get(cluster_name_, op->getContext()));
      host_cluster_ops_.insert(op);
      return true;
    }
    return false;
  }

  // If any tf.variants are inputs/outputs to the cluster, add them to the
  // cluster unless they are already marks with outside compilation attribute.
  bool AddVariantInputsOutputs() {
    bool added_op = false;
    llvm::SmallPtrSet<Operation*, 8> expanded_cluster_ops(host_cluster_ops_);
    for (Operation* cluster_op : host_cluster_ops_) {
      // Walk the clustered operations to handle nested ops.
      cluster_op->walk([&](Operation* op) {
        // Add any operations that provide variant inputs to the cluster.
        for (auto value : op->getOperands()) {
          auto input_defining_op = value.getDefiningOp();
          if (IsVariant(value) && input_defining_op &&
              !HasOutsideCompiledAncestor(input_defining_op) &&
              !input_defining_op->getAttrOfType<StringAttr>(
                  kXlaOutsideCompilationAttr)) {
            expanded_cluster_ops.insert(input_defining_op);
            input_defining_op->setAttr(
                kXlaOutsideCompilationAttr,
                StringAttr::get(cluster_name_,
                                input_defining_op->getContext()));
            added_op = true;
          }
        }
        // Add any operations that consume variant outputs to the cluster.
        for (auto value : op->getResults()) {
          if (IsVariant(value)) {
            for (auto user : value.getUsers()) {
              if (!host_cluster_ops_.contains(user) &&
                  !HasOutsideCompiledAncestor(user) &&
                  !user->getAttrOfType<StringAttr>(
                      kXlaOutsideCompilationAttr)) {
                expanded_cluster_ops.insert(user);
                user->setAttr(
                    kXlaOutsideCompilationAttr,
                    StringAttr::get(cluster_name_, user->getContext()));
                added_op = true;
              }
            }
          }
        }
      });
    }
    host_cluster_ops_.swap(expanded_cluster_ops);

    return added_op;
  }

 private:
  // Checks if it is safe for `op` to be merged into this cluster.
  bool IsSafeToAdd(Operation* op,
                   const TF::SideEffectAnalysis::Info& side_effect_analysis) {
    // If the op is not marked for outside compilation it doesn't belong in a
    // cluster.
    if (!op->getAttrOfType<StringAttr>(kXlaOutsideCompilationAttr))
      return false;

    if (host_cluster_ops_.empty()) return true;

    // If there is an intermediate data or side effect dependency between the op
    // and ops in the cluster, it's not safe to add.
    llvm::SmallSetVector<Operation*, 4> op_stack;
    for (auto* user : op->getUsers()) {
      if (!host_cluster_ops_.contains(user)) op_stack.insert(user);
    }
    for (auto* successor : side_effect_analysis.DirectControlSuccessors(op)) {
      if (!host_cluster_ops_.contains(successor)) op_stack.insert(successor);
    }
    bool safe_to_add = true;
    while (!op_stack.empty()) {
      auto* next_op = op_stack.pop_back_val();
      for (auto* user : next_op->getUsers()) {
        if (host_cluster_ops_.contains(user)) {
          safe_to_add = false;
          break;
        } else {
          op_stack.insert(user);
        }
      }
      for (auto* successor :
           side_effect_analysis.DirectControlSuccessors(next_op)) {
        if (host_cluster_ops_.contains(successor)) {
          safe_to_add = false;
          break;
        } else {
          op_stack.insert(successor);
        }
      }
      if (!safe_to_add) break;
    }

    return safe_to_add;
  }

  // `host_cluster_op_` stores a set of ops that will be grouped and computed
  // on host as single XlaHostCompute op. An outside compiled op can be grouped
  // to a single cluster if it has data dependency to another op already in the
  // cluster.
  llvm::SmallPtrSet<Operation*, 8> host_cluster_ops_;
  std::string cluster_name_;
};

void TPUOutsideCompilationCluster::runOnFunction(
    FuncOp func, const TF::SideEffectAnalysis::Info& side_effect_analysis) {
  llvm::SmallVector<OutsideCompiledCluster, 8> clusters;
  int cluster_counter = 0;

  func.walk([&](tf_device::ClusterOp tpu_cluster) {
    llvm::SmallVector<Operation*, 4> tpu_cluster_ops;
    tpu_cluster_ops.reserve(tpu_cluster.getBody()->getOperations().size());

    tpu_cluster.walk([&](Operation* op) { tpu_cluster_ops.emplace_back(op); });

    // In order to cluster ops feeding results to the same operation, traverse
    // the ops in reverse order.
    for (Operation* op : llvm::reverse(tpu_cluster_ops)) {
      // Try to add the op to existing clusters.
      bool added = false;
      for (auto& cluster : clusters)
        if ((added = cluster.AddOp(op, side_effect_analysis))) break;

      // If the op cannot be added to existing clusters, create a new cluster.
      if (!added) {
        OutsideCompiledCluster new_cluster(cluster_counter++);
        new_cluster.AddOp(op, side_effect_analysis);
        clusters.push_back(new_cluster);
      }
    }
  });
  for (auto& cluster : clusters) {
    bool variants_to_add = true;
    while (variants_to_add) variants_to_add = cluster.AddVariantInputsOutputs();
  }
}

}  // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateTPUOutsideCompilationClusterPass() {
  return std::make_unique<TPUOutsideCompilationCluster>();
}

static PassRegistration<TPUOutsideCompilationCluster> pass(
    "tf-tpu-outside-compilation-cluster",
    "Identifies clusters of operations assigned to outside compilation");

}  // namespace TFTPU
}  // namespace mlir

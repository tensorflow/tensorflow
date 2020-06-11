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

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TFTPU {

// This pass expands outside compilation attributes to Identity/Cast ops
// at the head of TPU computation if it's only used by outside compiled ops.

namespace {

constexpr char kXlaOutsideCompilationAttr[] = "_xla_outside_compilation";

bool HasOutsideCompilationAttribute(Operation* op) {
  return op->getAttrOfType<StringAttr>(kXlaOutsideCompilationAttr) != nullptr;
}

// Finds op that created a given value. If the value is a BlockArgument, this
// returns the owner of the Block.
Operation* GetOpOfValue(Value value) {
  if (auto block_arg = value.dyn_cast<BlockArgument>())
    return block_arg.getOwner()->getParentOp();

  return value.getDefiningOp();
}

// TODO(b/158596585): Replace this with a cost model analysis.
bool IsTrivialUnaryOperation(Operation* op) {
  return llvm::isa<TF::CastOp>(op) || llvm::isa<TF::IdentityOp>(op);
}

// Adds outside compilation attributes to unary ops such as Identity/Cast ops
// at the head of TPU computation that is used only by other outside compiled
// ops. Identity ops and Cast ops is commonly added to the start of TPU
// computation. Adding/expanding outside compilation attributes to these ops
// will ensure that head outside compiled ops are correctly located and moved to
// host.
// TODO(b/158691733): Also handle ops inside function calls/control flows.
void ExpandHeadOutsideCompiledOps(tf_device::ClusterOp cluster,
                                  OpBuilder* builder) {
  Region* cluster_region = &cluster.body();
  llvm::SmallSetVector<Operation*, 4> head_outside_compiled_ops;

  // Traverse the graph in topological order to find all outside compiled ops
  // at head of TPU computation or unary ops that are only used by other outside
  // compiled ops.
  auto cluster_ops = cluster.GetBody().without_terminator();
  for (Operation& cluster_op : cluster_ops) {
    if (IsTrivialUnaryOperation(&cluster_op) ||
        HasOutsideCompilationAttribute(&cluster_op)) {
      auto walk_result = cluster_op.walk([&](Operation* op) {
        for (Value operand : op->getOperands()) {
          Operation* operand_op = GetOpOfValue(operand);
          if (head_outside_compiled_ops.count(operand_op)) continue;

          if (operand_op->getParentRegion() == cluster_region)
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });

      if (!walk_result.wasInterrupted())
        head_outside_compiled_ops.insert(&cluster_op);
    }
  }

  for (auto head_outside_compiled_op :
       llvm::reverse(head_outside_compiled_ops)) {
    if (HasOutsideCompilationAttribute(head_outside_compiled_op)) continue;

    bool should_expand_op_to_host_computation = true;
    for (auto consumer_op : head_outside_compiled_op->getUsers()) {
      if (should_expand_op_to_host_computation &&
          !HasOutsideCompilationAttribute(consumer_op)) {
        should_expand_op_to_host_computation = false;
        continue;
      }
    }

    if (should_expand_op_to_host_computation)
      head_outside_compiled_op->setAttr(kXlaOutsideCompilationAttr,
                                        builder->getStringAttr(""));
  }
}

struct TPUHostComputationExpansion
    : public PassWrapper<TPUHostComputationExpansion, FunctionPass> {
  void runOnFunction() override;
};

void TPUHostComputationExpansion::runOnFunction() {
  OpBuilder builder(&getContext());
  getFunction().walk([&](tf_device::ClusterOp cluster) {
    ExpandHeadOutsideCompiledOps(cluster, &builder);
  });
}

}  // anonymous namespace

std::unique_ptr<OperationPass<FuncOp>> CreateTPUHostComputationExpansionPass() {
  return std::make_unique<TPUHostComputationExpansion>();
}

static PassRegistration<TPUHostComputationExpansion> pass(
    "tf-tpu-host-computation-expansion",
    "Expands host computation before and after TPU computation.");

}  // namespace TFTPU
}  // namespace mlir

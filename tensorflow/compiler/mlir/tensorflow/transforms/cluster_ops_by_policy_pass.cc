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

// This pass clusters operations according to the policy specified by the pass
// options. Clustered operations are placed in 'tf_device::ClusterOp'.

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/core/platform/logging.h"

#define DEBUG_TYPE "cluster-ops-by-policy"

namespace mlir {
namespace TFDevice {

namespace {

constexpr char kDeviceAttr[] = "device";

// Pass definition.
struct ClusterOpsByPolicyPass
    : public TF::ClusterOpsByPolicyPassBase<ClusterOpsByPolicyPass> {
  void runOnFunction() override;
};

// Returns true if `op` starts a sequence of ops that match ops in `oplist`.
// The found ops are written into 'matched_ops' and added to 'is_matched' set.
// The next matched op must be the only user of the previous matched op's
// result. The matched ops do not have to be consecutive. For example,
//    %1 = "tf.Add" %a, %b
//    %2 = "tf.Neg" %a
//    %3 = "tf.Sub" %c, %1 // the only use of %1
// matches "tf.Add, tf.Sub".
bool IsOplistMatch(Operation *op, ArrayRef<std::string> oplist,
                   llvm::DenseSet<Operation *> &is_matched,
                   llvm::SmallVectorImpl<Operation *> &matched_ops) {
  MLIRContext *ctx = op->getContext();

  // Skip 'op' if it's already part of another matched sequence of ops.
  if (is_matched.contains(op)) return false;

  // Does this operation match first element in the oplist?
  StringRef op_name = *oplist.begin();
  if (op->getName().getIdentifier() != Identifier::get(op_name, ctx))
    return false;

  matched_ops.push_back(op);

  // Check for match with the rest of oplist elements.
  auto oplist_iter = oplist.begin() + 1;
  auto oplist_end = oplist.end();
  Block *block = op->getBlock();
  auto device = op->getAttr(kDeviceAttr);
  Operation *curr_op = op;

  while (oplist_iter != oplist_end) {
    // Find the next op to match.
    if (!curr_op->hasOneUse()) return false;
    curr_op = *curr_op->getUsers().begin();

    // Skip 'op' if it's already part of another matched sequence of ops.
    if (is_matched.contains(curr_op)) return false;

    // Check that the op matches the next op in the oplist.
    op_name = *oplist_iter;
    if (curr_op->getName().getIdentifier() != Identifier::get(op_name, ctx))
      return false;

    // Don't cluster operations assigned to different devices.
    if (curr_op->getAttr(kDeviceAttr) != device) return false;

    // Don't cluster ops across blocks.
    if (curr_op->getBlock() != block) return false;

    // Check that op has no side effects. This guarantees that we will not
    // reorder side-effecting ops during cluster formation.
    if (!MemoryEffectOpInterface::hasNoEffect(curr_op)) return false;

    ++oplist_iter;
    matched_ops.push_back(curr_op);
  }

  is_matched.insert(matched_ops.begin(), matched_ops.end());

  return true;
}

// Move matched operations into tf_device::ClusterOp.
void ClusterMatchedOps(ArrayRef<Operation *> matched_ops) {
  LLVM_DEBUG({
    llvm::dbgs() << "Creating a cluster for matched ops:\n";
    for (auto e : matched_ops) {
      e->print(llvm::dbgs());
      llvm::dbgs() << "\n";
    }
    llvm::dbgs() << "\n";
  });

  // Create tf_device::ClusterOp before the last matched operation.
  Operation *lastOp = matched_ops.back();
  OpBuilder builder(lastOp);
  auto loc = lastOp->getLoc();
  auto clusterOp =
      builder.create<tf_device::ClusterOp>(loc, lastOp->getResultTypes());

  // Create block in clusterOp's region and move 'matched_ops' into it.
  auto block = builder.createBlock(&clusterOp.body());
  auto block_end = block->end();
  for (auto e : matched_ops) e->moveBefore(block, block_end);

  // Replace uses of lastOp results with uses of tf_device.cluster op.h
  lastOp->replaceAllUsesWith(clusterOp);

  // Add 'tf_device::ReturnOp' at the end of the block.
  builder.setInsertionPointToEnd(block);
  builder.create<tf_device::ReturnOp>(loc, lastOp->getResults());

  // Set device attribute
  if (auto device = lastOp->getAttr(kDeviceAttr))
    clusterOp->setAttr(kDeviceAttr, device);
}

// Define type to store list of operations.
typedef llvm::SmallVector<Operation *> OpList;

// Find operations that match 'oplist' and extract them into clusters.
void ClusterOpsByPolicyPass::runOnFunction() {
  if (oplist.empty()) return;

  llvm::SmallVector<OpList> clusters;
  llvm::DenseSet<Operation *> is_matched;

  // Find matching op sequences within this function.
  getFunction().walk([&](Operation *op) {
    llvm::SmallVector<Operation *> matched_ops;

    // Skip 'op' if it's already part of another matched sequence of ops.
    if (is_matched.contains(op)) return;

    // Try to match 'op' to the sequence of ops in 'oplist'.
    if (!IsOplistMatch(op, oplist, is_matched, matched_ops)) return;

    // We found a matching sequence of ops. Record it.
    clusters.push_back(matched_ops);
  });

  // Create clusters.
  for (const OpList &c : clusters) ClusterMatchedOps(c);
}

}  // namespace

std::unique_ptr<FunctionPass> CreateClusterOpsByPolicyPass() {
  return std::make_unique<TFDevice::ClusterOpsByPolicyPass>();
}

}  // namespace TFDevice
}  // namespace mlir

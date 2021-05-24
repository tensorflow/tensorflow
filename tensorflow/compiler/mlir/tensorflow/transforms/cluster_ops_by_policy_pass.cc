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

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"

#define DEBUG_TYPE "cluster-ops-by-policy"

namespace mlir {
namespace TFDevice {

namespace {

constexpr char kDeviceAttr[] = "device";

// Pass definition.
struct ClusterOpsByPolicyPass
    : public TF::ClusterOpsByPolicyPassBase<ClusterOpsByPolicyPass> {
 public:
  ClusterOpsByPolicyPass() = default;
  ClusterOpsByPolicyPass(ArrayRef<std::string> cluster_oplist,
                         int cluster_min_size, StringRef cluster_algorithm,
                         StringRef cluster_policy) {
    oplist = cluster_oplist;
    min_cluster_size = cluster_min_size;
    algorithm = cluster_algorithm.str();
    policy_name = cluster_policy.str();
  }
  void runOnFunction() override;
};

using OpList = llvm::SmallVector<Operation *>;

}  // namespace

// Move matched operations into tf_device::ClusterOp.
static tf_device::ClusterOp ClusterMatchedOps(
    MutableArrayRef<Operation *> matched_ops, StringAttr policy) {
  LLVM_DEBUG({
    llvm::dbgs() << "Creating a cluster for matched ops:\n";
    for (auto e : matched_ops) {
      e->print(llvm::dbgs());
      llvm::dbgs() << "\n";
    }
    llvm::dbgs() << "\n";
  });

  // Find all the values that are source outside of the cluster. These values
  // will be returned from the created cluster operation.
  llvm::DenseSet<Operation *> in_cluster;
  for (Operation *op : matched_ops) in_cluster.insert(op);

  llvm::SetVector<Value> return_values;
  llvm::SmallVector<Type> return_types;

  for (Operation *op : matched_ops)
    for (OpOperand &use : op->getUses()) {
      // User is inside the cluster.
      if (in_cluster.contains(use.getOwner())) continue;
      // Do not return the same value multiple times.
      if (return_values.contains(use.get())) continue;

      return_values.insert(use.get());
      return_types.emplace_back(use.get().getType());
    }

  LLVM_DEBUG(llvm::dbgs() << "Cluster has " << return_values.size()
                          << " return values\n");

  // Sort matched operations by their position in the block.
  llvm::sort(matched_ops, [](Operation *a, Operation *b) -> bool {
    return a->isBeforeInBlock(b);
  });

  // Create tf_device::ClusterOp before the last operation in the block that is
  // a part of a match set.
  auto back = matched_ops.back();
  auto loc = back->getLoc();
  OpBuilder builder(back);

  auto cluster_op =
      builder.create<tf_device::ClusterOp>(loc, return_types, policy);

  // Create block in cluster_op's region and move 'matched_ops' into it.
  auto block = builder.createBlock(&cluster_op.body());
  auto block_end = block->end();
  for (auto op : matched_ops) op->moveBefore(block, block_end);

  // Add 'tf_device::ReturnOp' at the end of the block.
  builder.setInsertionPointToEnd(block);
  builder.create<tf_device::ReturnOp>(loc, return_values.getArrayRef());

  // Set device attribute
  if (auto device = back->getAttr(kDeviceAttr))
    cluster_op->setAttr(kDeviceAttr, device);

  // Update all users of the operations moved into the cluster region.
  for (auto tuple : llvm::zip(return_values, cluster_op.getResults())) {
    Value old_value = std::get<0>(tuple);
    Value new_value = std::get<1>(tuple);
    old_value.replaceUsesWithIf(new_value, [&](OpOperand &operand) -> bool {
      // Do not update users in the same cluster.
      return operand.getOwner()->getBlock() != block;
    });
  }

  return cluster_op;
}

// -------------------------------------------------------------------------- //
// Form clusters using use-def chains.
// -------------------------------------------------------------------------- //

// Returns true if `op` starts a sequence of ops that match ops in `oplist`.
// The found ops are written into 'matched_ops' and added to 'is_matched' set.
// The next matched op must be the only user of the previous matched op's
// result. The matched ops do not have to be consecutive. For example,
//    %1 = "tf.Add" %a, %b
//    %2 = "tf.Neg" %a
//    %3 = "tf.Sub" %c, %1 // the only use of %1
// matches "tf.Add, tf.Sub".
static bool IsOplistMatch(Operation *op, ArrayRef<Identifier> oplist,
                          llvm::DenseSet<Operation *> &is_matched,
                          llvm::SmallVectorImpl<Operation *> &matched_ops) {
  // Skip 'op' if it's already part of another matched sequence of ops.
  if (is_matched.contains(op)) return false;

  // Does this operation match first element in the oplist?
  if (op->getName().getIdentifier() != *oplist.begin()) return false;

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
    if (curr_op->getName().getIdentifier() != *oplist_iter) return false;

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

// Form clusters of operations using `use-def` algorithm and appends the to the
// `clusters` list.
static void FormUseDefClusters(mlir::FuncOp func, ArrayRef<std::string> oplist,
                               llvm::SmallVectorImpl<OpList> *clusters) {
  MLIRContext *context = func.getContext();

  // Do not place the same operation into multiple cluster.
  llvm::DenseSet<Operation *> is_matched;

  // Convert 'oplist' of strings into a list of identifiers.
  std::vector<Identifier> op_id_list;
  for (const auto &op : oplist)
    op_id_list.push_back(Identifier::get(op, context));

  // Find matching op sequences within this function.
  func.walk([&](Operation *op) {
    llvm::SmallVector<Operation *> matched_ops;

    // Skip 'op' if it's already part of another matched sequence of ops.
    if (is_matched.contains(op)) return;

    // Try to match 'op' to the sequence of ops in 'op_id_list'.
    if (!IsOplistMatch(op, op_id_list, is_matched, matched_ops)) return;

    // We found a matching sequence of ops. Record it.
    clusters->push_back(matched_ops);
  });
}

// -------------------------------------------------------------------------- //
// Form clusters using union-find algorithm.
// -------------------------------------------------------------------------- //

namespace {

// A type that abstracts over types that have uses accessible via `getUses`.
using Source = PointerUnion<Operation *, BlockArgument *>;
struct Member {
  Member(unsigned root, Source source, Operation *first_user)
      : root(root), source(source), first_user(first_user) {}

  unsigned root;
  Source source;
  // After construction:
  //   First user of the `source` results in the same block where `source` is
  //   defined. If there are no users in the same block, then it is a pointer to
  //   the block terminator.
  //
  // During the union-find cluster formation:
  //   The root member will have a pointer to the first user of any result of
  //   any operation that belongs to the cluster identified by the root member.
  Operation *first_user;
};

using Members = llvm::SmallVector<Member>;
}  // namespace

// Returns the root member of the `id`.
static unsigned FindRoot(const Members &members, unsigned id) {
  if (members[id].root == id) return id;
  return FindRoot(members, members[id].root);
}

// Puts `a` and `b` members under the same root member.
static void Union(Members &members, unsigned a, unsigned b) {
  unsigned a_root = FindRoot(members, a);
  unsigned b_root = FindRoot(members, b);

  if (a_root != b_root) {
    // b_root becomes the new root.
    members[a_root].root = b_root;
    // Update first user of the new root.
    Operation *a_user = members[a_root].first_user;
    Operation *b_user = members[b_root].first_user;
    members[b_root].first_user =
        a_user->isBeforeInBlock(b_user) ? a_user : b_user;
  }
}

// Cluster operations connected with def-use chains and present in the
// `cluster_ops` set using union-find algorithm.
//
// Example: oplist = tf.Add,tf.Sub
//
//   %0 = "tf.Sub" ...
//   %1 = "tf.Sub" ...
//   %2 = "tf.Add" %0, %1
//
// Will be clustered together into:
//
//   tf_device.cluster {
//     %0 = "tf.Sub" ...
//     %1 = "tf.Sub" ...
//     %2 = "tf.Add" %0, %1
//     tf_device.return %2
//   }
//
// Although %0, %1, %2 do not form a single use-def chain, they are still
// clustered together based on the union-find algorigthm.
static void ClusterOpsInTheBlock(Block *block,
                                 const llvm::DenseSet<Identifier> &cluster_ops,
                                 llvm::SmallVectorImpl<OpList> *clusters) {
  // Returns true if op can be clustered.
  auto can_be_clustered = [&](Operation &op) -> bool {
    // Check that op has no side effects. This guarantees that we will not
    // reorder side-effecting ops during cluster formation.
    if (!MemoryEffectOpInterface::hasNoEffect(&op)) return false;

    return cluster_ops.contains(op.getName().getIdentifier());
  };

  // Use an array based union-find algorithm to cluster operations together
  // (index in this vector is the member id).
  llvm::SmallVector<Member> members;

  // Find arguments and operations that are candidates for clustering.
  for (BlockArgument &arg : block->getArguments()) {
    // Find the first user that can't be clustered.
    Operation *first_user = block->getTerminator();
    for (Operation *user : arg.getUsers())
      if (user->getBlock() == block && user->isBeforeInBlock(first_user) &&
          !can_be_clustered(*user))
        first_user = user;

    members.emplace_back(members.size(), &arg, first_user);
  }
  for (Operation &op : block->getOperations())
    if (can_be_clustered(op)) {
      // Find the first user that can't be clustered.
      Operation *first_user = block->getTerminator();
      for (Operation *user : op.getUsers())
        if (user->getBlock() == block && user->isBeforeInBlock(first_user) &&
            !can_be_clustered(*user))
          first_user = user;

      members.emplace_back(members.size(), &op, first_user);
    }

  // Mapping from the member operation to the id.
  llvm::DenseMap<Source, unsigned> member_ids;
  for (auto kv : llvm::enumerate(members))
    member_ids.try_emplace(kv.value().source, kv.index());

  LLVM_DEBUG(llvm::dbgs() << "Found " << members.size()
                          << " clustering candidate operations in the block\n");

  // Try to cluster members with their result users.
  for (auto &tuple : llvm::enumerate(members)) {
    size_t member_id = tuple.index();
    Member &member = tuple.value();

    // Candidates for clustering with a `member` operation.
    llvm::SmallVector<Operation *> users;
    if (auto op = member.source.dyn_cast<Operation *>()) {
      auto users_rng =
          llvm::make_filter_range(op->getUsers(), [&](Operation *user) {
            bool same_block = user->getBlock() == block;
            bool same_device =
                op->getAttr(kDeviceAttr) == user->getAttr(kDeviceAttr);
            return same_block && same_device && can_be_clustered(*user);
          });
      users.assign(users_rng.begin(), users_rng.end());
    } else if (auto arg = member.source.dyn_cast<BlockArgument *>()) {
      auto users_rng =
          llvm::make_filter_range(arg->getUsers(), [&](Operation *user) {
            bool same_block = user->getBlock() == block;
            return same_block && can_be_clustered(*user);
          });
      users.assign(users_rng.begin(), users_rng.end());
    } else {
      llvm_unreachable("Unexpected type in the union.");
    }

    // We need to process users according to their order in the block to be sure
    // that we do not create clusters that break dominance property.
    llvm::sort(users, [](auto *a, auto *b) { return a->isBeforeInBlock(b); });

    for (Operation *user : users) {
      // Skip users that are past the first cluster result user in the block,
      // because otherwise after clustering we would violate dominance property
      // (the cluster operation would be defined after the first user in the
      // block).
      unsigned root = FindRoot(members, member_id);
      Operation *first_cluster_user = members[root].first_user;
      if (first_cluster_user->isBeforeInBlock(user)) continue;

      Union(members, member_id, member_ids.lookup(user));
    }
  }

  // Form clusters found by the union-find algorithm.
  llvm::DenseMap<unsigned, OpList> root_clusters;
  for (auto &tuple : llvm::enumerate(members)) {
    if (auto op = tuple.value().source.dyn_cast<Operation *>()) {
      root_clusters.FindAndConstruct(FindRoot(members, tuple.index()))
          .getSecond()
          .emplace_back(op);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Found " << root_clusters.size() << " clusters\n");

  // Return found clusters through the output parameters.
  for (auto &kv : root_clusters)
    clusters->emplace_back(std::move(kv.getSecond()));
}

static void FormUnionFindClusters(mlir::FuncOp func,
                                  ArrayRef<std::string> oplist,
                                  llvm::SmallVectorImpl<OpList> *clusters) {
  MLIRContext *context = func->getContext();

  llvm::DenseSet<Identifier> opset;
  for (const auto &op : oplist) opset.insert(Identifier::get(op, context));
  func->walk(
      [&](Block *block) { ClusterOpsInTheBlock(block, opset, clusters); });
}

// -------------------------------------------------------------------------- //

// Find operations that match 'oplist' and extract them into clusters.
void ClusterOpsByPolicyPass::runOnFunction() {
  if (oplist.empty()) return;

  llvm::SmallVector<OpList> clusters;

  if (algorithm == "use-def") {
    FormUseDefClusters(getFunction(), oplist, &clusters);
  } else if (algorithm == "union-find") {
    FormUnionFindClusters(getFunction(), oplist, &clusters);
  } else {
    emitError(getFunction()->getLoc(), "Unsupported clustering algorithm");
    signalPassFailure();
    return;
  }
  // Create clusters tagged with a policy name.
  auto policy = StringAttr::get(&getContext(), policy_name);
  for (OpList &c : clusters) {
    if (c.size() < min_cluster_size) continue;
    ClusterMatchedOps(c, policy);
  }
}

std::unique_ptr<FunctionPass> CreateClusterOpsByPolicyPass() {
  return std::make_unique<TFDevice::ClusterOpsByPolicyPass>();
}

std::unique_ptr<FunctionPass> CreateClusterOpsByPolicyPass(
    ArrayRef<std::string> oplist, int min_cluster_size, StringRef algorithm,
    StringRef policy_name) {
  return std::make_unique<TFDevice::ClusterOpsByPolicyPass>(
      oplist, min_cluster_size, algorithm, policy_name);
}

}  // namespace TFDevice
}  // namespace mlir

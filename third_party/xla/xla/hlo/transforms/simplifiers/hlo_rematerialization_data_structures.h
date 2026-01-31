/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_REMATERIALIZATION_DATA_STRUCTURES_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_REMATERIALIZATION_DATA_STRUCTURES_H_

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_schedule.h"

namespace xla {
// Type holding a unique identifier for each Buffer object.
using BufferId = int64_t;
using BufferIdList = absl::InlinedVector<BufferId, 3>;

// We wrap HloInstruction* with an HloRematItem that holds auxiliary
// per-instruction state.
struct HloRematItem {
  HloInstruction* instruction;

  // True once the instruction is marked as placed (when BeginInstruction
  // has been called for this instruction).
  bool placed = false;

  // To avoid an infinite loop rematerializing the same set of
  // instructions ad infinitum, keep a denylist of instructions
  // which should not be rematerialized.
  bool denylisted = false;

  // The buffers defined by this instruction.
  BufferIdList buffers_defined;

  // Output buffers of this instruction. This is used to track outputs by GTE
  // instructions (where the instruction doesn't define a buffer).
  BufferIdList buffers_output;

  // The buffers used by this instruction.
  BufferIdList buffers_used;

  bool is_skip_node = false;

 private:
  friend class HloRematInstructionList;

  // Items are arranged in a doubly linked list.
  HloRematItem* next = nullptr;
  HloRematItem* prev = nullptr;

  HloRematItem* prev_skip_node = nullptr;
  HloRematItem* next_skip_node = nullptr;

  // List is ordered by position, which can however be duplicated as
  // new instructions are inserted.  See InsertBeforeInstructions
  // comment for details.
  int64_t position;
};

// Data structure meant to record the user of the buffer defined from an
// HloRematItem. It records also the operand_number from where such use derives,
// so that indirect uses can be better identified (like for example a buffer
// used through a bitcast).
struct HloRematItemUse {
  HloRematItem* user;
  int64_t operand_number;
  std::optional<int64_t> index;

  HloRematItemUse(HloRematItem* user, int64_t op_num,
                  std::optional<int64_t> index)
      : user(user), operand_number(op_num), index(index) {}
  bool operator==(const HloRematItemUse& other) const {
    return user == other.user && operand_number == other.operand_number &&
           index == other.index;
  }
};

using ItemList = absl::InlinedVector<HloRematItem*, 3>;
using UsesList = absl::InlinedVector<HloRematItemUse, 3>;

// Class which maintains an ordered list of instructions with fast insertion
// before arbitrary elements.
//
// This is a skip list structure that has two lanes: express lane and slow lane.
// All nodes are presented on the slow lane but a node can be promoted into
// express lane for fast iteration.
//
// In the following case, node 2 and node + 1 are connected via an express lane.
//                    +--------------------------+----------->: Express lane
//                    |                          |
//       node1<-> node 2 <-> .. <-> node n <-> node n+1 <->...: Slow lane
//
class HloRematInstructionList {
 public:
  explicit HloRematInstructionList(const HloInstructionSequence& order) {
    CHECK_OK(UpdateFromSequence(order, /*preserve_denylist=*/false));
  }

  absl::Status UpdateFromSequence(const HloInstructionSequence& order,
                                  bool preserve_denylist = true);

  size_t size() const { return item_map_.size(); }

  // For ordered iteration over items.
  //    for (auto item = q.first(); item != nullptr; item = q.next(item)) {...}
  HloRematItem* first() const { return first_; }
  HloRematItem* next(HloRematItem* item) const { return item->next; }
  const HloRematItem* next(const HloRematItem* item) const {
    return item->next;
  }
  HloRematItem* prev(HloRematItem* item) const { return item->prev; }
  const HloRematItem* prev(const HloRematItem* item) const {
    return item->prev;
  }

  HloRematItem* first_skip_node() const { return first_skip_node_; }
  HloRematItem* next_skip_node(HloRematItem* item) const {
    return item->next_skip_node;
  }

  // Creates an HloRematItem for the given instruction, but doesn't add it to
  // the list. (Use InsertBeforeInstructions to add the HloRematItem to the
  // list.)
  HloRematItem* CreateItem(HloInstruction* inst) {
    auto item = std::make_unique<HloRematItem>();
    item->instruction = inst;
    CHECK(item_map_.insert({inst, std::move(item)}).second)
        << "inserting inst twice " << inst->name();
    return item_map_[inst].get();
  }

  // Return the HloRematItem corresponding to inst.
  HloRematItem* GetItem(const HloInstruction* inst) const {
    auto iter = item_map_.find(inst);
    CHECK(iter != item_map_.end()) << "Did not find " << inst->name();
    return iter->second.get();
  }

  // Insert instruction 'to_insert' immediately before the earliest instruction
  // in 'before_instructions'.
  //
  // Each instruction gets a non-decreasing ordinal number. We use this to let
  // InsertBeforeInstructions quickly insert an instruction before the earliest
  // instruction in a set of instructions.  If position_number_[a] <
  // position_number_[b] then 'a' comes before 'b' in the list. If the position
  // numbers are the same then nothing can be said about their order without
  // examining the list.
  //
  // On object construction this ordinal is precisely the instruction's index
  // in the list. Later, instructions inserted via InsertBefore receive
  // duplicate values. However, monotonicity is preserved.
  void InsertBeforeInstructions(
      HloRematItem* to_insert,
      absl::Span<HloRematItem* const> before_instructions);

  // Scan the list and promote nodes to express lane if
  // should_promote(HloRematItem) returns true;
  void PromoteNodesToSkip(
      absl::FunctionRef<bool(HloRematItem*)> should_promote);

  void InsertAfterInstructions(
      HloRematItem* to_insert,
      absl::Span<HloRematItem* const> after_instructions);

  void Denylist(const HloInstruction* inst) {
    GetItem(inst)->denylisted = true;
  }

 private:
  // Insert instruction 'item' immediately before 'before' in the list.
  void InsertBefore(HloRematItem* item, HloRematItem* before);

  HloRematItem* first_;

  // First skip node of this list.
  HloRematItem* first_skip_node_;

  // Last skip node of this list.
  HloRematItem* last_skip_node_;

  // HloRematItem for each instruction.
  absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloRematItem>>
      item_map_;
};

// Based on Section 2, "A Simple O(log n) Amortized Time Algorithm", available
// at https://www.cs.cmu.edu/~sleator/papers/maintaining-order.pdf.
class SleatorDietzOrderMaintenance {
 public:
  SleatorDietzOrderMaintenance();

  // Inserts `new_instruction` after `old_instruction`. If `old_instruction` is
  // nullptr, then `new_instruction` is inserted at the beginning.
  absl::Status InsertBeforeInstruction(
      const HloInstruction* absl_nullable old_instruction,
      const HloInstruction* absl_nonnull new_instruction);
  // Inserts `new_instruction` before `old_instruction`. If `old_instruction` is
  // nullptr, then `new_instruction` is inserted at the end.
  absl::Status InsertAfterInstruction(
      const HloInstruction* absl_nullable old_instruction,
      const HloInstruction* absl_nonnull new_instruction);
  // Deletes `instruction`.
  absl::Status DeleteInstruction(
      const HloInstruction* absl_nonnull instruction);

  // Whether `first_instruction` comes strictly before `second_instruction`.
  absl::StatusOr<bool> CompareOrder(
      const HloInstruction* absl_nonnull first_instruction,
      const HloInstruction* absl_nonnull second_instruction) const;

  // Whether `instruction` is in the ordering at all.
  bool ContainsInstruction(
      const HloInstruction* absl_nonnull instruction) const;

  // The first instruction, if there is one, or nullptr otherwise.
  const HloInstruction* GetFirstInstruction() const;

  // The instruction that immediately precedes/succeeds `instruction`, or
  // nullptr if there is none.
  absl::StatusOr<const HloInstruction*> GetPreviousInstruction(
      const HloInstruction* absl_nonnull instruction) const;
  absl::StatusOr<const HloInstruction*> GetNextInstruction(
      const HloInstruction* absl_nonnull instruction) const;

  std::string ToString() const;

  // Method for unit tests to check that this data structure obeys certain
  // invariants.
  absl::Status VerifyInvariantsForTesting() const;

 private:
  struct SleatorDietzRecord {
    const HloInstruction* instruction;
    // We think of labels as integers mod 2^64. Since we are running
    // Sleator-Dietz with a fixed arena size of 2^64, we can have at most
    // (2^32 - 1) records.
    uint64_t label;
    // Pointers for a circular doubly-linked-list.
    SleatorDietzRecord *prev, *next;
  };

  // Allocates a new record, sets its `instruction` field, and returns a pointer
  // to it. Caller is responsible for setting the other fields.
  absl::StatusOr<SleatorDietzOrderMaintenance::SleatorDietzRecord* absl_nonnull>
  AllocateRecord(const HloInstruction* absl_nullable instruction);

  // Finds an existing record.
  absl::StatusOr<SleatorDietzOrderMaintenance::SleatorDietzRecord* absl_nonnull>
  GetRecord(const HloInstruction* absl_nullable instruction);
  absl::StatusOr<
      const SleatorDietzOrderMaintenance::SleatorDietzRecord* absl_nonnull>
  GetRecord(const HloInstruction* absl_nullable instruction) const;

  // Returns the label of `record` relative to the base label.
  uint64_t GetRelativeLabel(
      const SleatorDietzRecord* absl_nonnull record) const;

  // Returns the label of the record after `record` relative to the base label,
  // with special handling if the next label is the base label (returns 2^64
  // instead of zero).
  absl::uint128 GetNextRelativeLabel(
      SleatorDietzRecord* absl_nonnull record) const;

  // Rearranges the labels of `starting_record` and the next few records if they
  // are too close together.
  void RearrangeLabels(SleatorDietzRecord* absl_nonnull starting_record);

  // Actual implementation of both Insert*() methods.
  absl::Status InsertHelper(const HloInstruction* absl_nullable old_instruction,
                            const HloInstruction* absl_nonnull new_instruction,
                            bool insert_before);

  absl::flat_hash_map<const HloInstruction*,
                      std::unique_ptr<SleatorDietzRecord>>
      records_;
  SleatorDietzRecord* base_record_;
};

// Holds the memory usage and instruction at a given program point (usually the
// peak memory).
struct MemoryUsageAndInstruction {
  int64_t memory_usage;
  const HloInstruction* instruction;
};

// Node for AVLLazySegmentTree.
struct AVLTreeNode {
  // If nullptr, this node is the root.
  AVLTreeNode* absl_nullable parent;
  // If nullptr, this node is a leaf.
  AVLTreeNode* absl_nullable left_child;
  // If nullptr, this node is a leaf.
  AVLTreeNode* absl_nullable right_child;
  // Distance to furthest leaf in subtree rooted at this node (starts at zero
  // for the leaves), used for AVL balancing.
  int height;
  // The peak memory usage among nodes in this subtree along with the
  // HloInstruction* for which it occurs. Ties are broken for the leftmost
  // HloInstruction*, according to `left_child` and `right_child` pointers.
  MemoryUsageAndInstruction memory_usage_and_instruction;
  // All recursive children of this node (but not this node itself) need to
  // have their memory usage adjusted by this amount.
  int64_t lazy_additive_memory_update;
  // Iterator to this node for quick deletion.
  std::list<std::unique_ptr<AVLTreeNode>>::iterator node_storage_iterator;
};

class AVLLazySegmentTree {
 public:
  // Sets the leaves of the tree to be `initial_memory_usage_and_instructions`
  // (the left-to-right order of the leaves will match).
  explicit AVLLazySegmentTree(absl::Span<const MemoryUsageAndInstruction>
                                  initial_memory_usage_and_instructions);

  // Inserts a leaf containing `new_instruction` to the left of or to the right
  // of the leaf containing `old_instruction`.
  absl::Status InsertBeforeInstruction(
      const HloInstruction* absl_nonnull old_instruction,
      const MemoryUsageAndInstruction& new_memory_usage_and_instruction);
  absl::Status InsertAfterInstruction(
      const HloInstruction* absl_nonnull old_instruction,
      const MemoryUsageAndInstruction& new_memory_usage_and_instruction);
  // Deletes the leaf containing `instruction`.
  absl::Status Delete(const HloInstruction* absl_nonnull instruction);

  // Queries all leaves between `first_instruction` and `last_instruction`,
  // inclusive, returning the maximum memory usage among them and the
  // instruction at which it occurs. Since we use lazy updates, this may result
  // in changes to the data structure and hence is not a const operation.
  absl::StatusOr<MemoryUsageAndInstruction> Query(
      const HloInstruction* absl_nonnull first_instruction,
      const HloInstruction* absl_nonnull last_instruction);

  // Queries all leaves, never changing the data structure. Runs in O(1) time.
  absl::StatusOr<MemoryUsageAndInstruction> Query() const;

  // Updates all leaves between `first_instruction` and `last_instruction`,
  // inclusive, changing their each of their `memory_usage` terms by the update
  // amount.
  absl::Status Update(const HloInstruction* absl_nonnull first_instruction,
                      const HloInstruction* absl_nonnull last_instruction,
                      int64_t additive_memory_update);

  // Method for unit tests to check that this tree obeys certain invariants.
  absl::Status VerifyInvariantsForTesting() const;

 private:
  // Allocates a new node, sets its `node_storage_iterator` field, and returns a
  // pointer to it.
  AVLTreeNode* absl_nonnull AllocateNode();
  // Appropriately sets the fields of a leaf node.
  void SetLeafNode(
      AVLTreeNode* absl_nonnull leaf_node,
      const MemoryUsageAndInstruction& memory_usage_and_instruction,
      AVLTreeNode* absl_nullable parent = nullptr);
  // Appropriately sets the field of an interior node.
  void SetInteriorNode(AVLTreeNode* absl_nonnull interior_node,
                       AVLTreeNode* absl_nonnull left_child,
                       AVLTreeNode* absl_nonnull right_child,
                       AVLTreeNode* absl_nullable parent = nullptr);

  // Balances the subtree rooted at node. Returns the new root of this subtree.
  AVLTreeNode* absl_nonnull Balance(AVLTreeNode* absl_nonnull node);
  // See en.wikipedia.org/wiki/AVL_tree for balance factor definition.
  int BalanceFactor(AVLTreeNode* absl_nonnull node) const;
  // Tree rotations, see en.wikipedia.org/wiki/Tree_rotation (left/right given
  // the node-centric way). Both return the new root of the subtree so that the
  // parent (or root_) can be made to point to it. Can only be called on
  // interior nodes.
  AVLTreeNode* RotateLeft(AVLTreeNode* absl_nonnull node);
  AVLTreeNode* RotateRight(AVLTreeNode* absl_nonnull node);
  // Pushes the accumulated lazy update at a node to its children. Safe to call
  // on leaves.
  void PushDownLaziness(AVLTreeNode* absl_nonnull node);

  // Finds the path from the leaf that holds `instruction` to the root,
  // including both leaf and root. Also pushes down all lazy updates along this
  // path.
  absl::StatusOr<std::vector<AVLTreeNode* absl_nonnull>> GetPathFromLeafToRoot(
      const HloInstruction* absl_nonnull instruction);

  // Finds a sequence of nodes that covers the interval from `first_instruction`
  // to `last_instruction`, inclusive. It is guaranteed that:
  // - this sequence is nonempty
  // - no node in this sequence is a child (or recursive child) of another node
  //   in this sequence
  // - nodes are sequenced in agreement with a in-order traversal of the tree
  //   (also in agreement with pre-order traversal of the tree due to the
  //   previous property)
  // - any parent of a node in the sequence is either on the path from the leaf
  //   containing `first_instruction` going to the root or on the path from the
  //   leaf containing `last_instruction` going to the root (as a consequence,
  //   there are at most O(log n) nodes in the sequence, where n is the total
  //   number of instructions)
  // Furthermore, this method pushes down all lazy updates on all paths from the
  // root to this set of nodes.
  absl::StatusOr<std::deque<AVLTreeNode* absl_nonnull>>
  GetNodesSpanningInterval(const HloInstruction* absl_nonnull first_instruction,
                           const HloInstruction* absl_nonnull last_instruction);

  absl::Status InsertHelper(const HloInstruction* absl_nonnull old_instruction,
                            const MemoryUsageAndInstruction& new_instruction,
                            bool insert_before);

  AVLTreeNode* absl_nullable root_;
  std::list<std::unique_ptr<AVLTreeNode>> node_storage_;
  absl::flat_hash_map<const HloInstruction* absl_nonnull,
                      AVLTreeNode* absl_nonnull>
      instruction_to_leaf_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_REMATERIALIZATION_DATA_STRUCTURES_H_

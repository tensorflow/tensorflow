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

#include "xla/hlo/transforms/simplifiers/hlo_rematerialization_data_structures.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_schedule.h"

namespace xla {
namespace {

// A pointer to either the `left_node` or `right_node` field of `node` that
// points to `child`.
absl::StatusOr<AVLTreeNode** absl_nonnull> FindChildPointer(
    AVLTreeNode* absl_nonnull node, AVLTreeNode* absl_nonnull child) {
  if (node->left_child == child) {
    return &node->left_child;
  }
  if (node->right_child == child) {
    return &node->right_child;
  }
  return absl::InternalError(absl::StrFormat(
      "Node %p should have a child pointer to %p but does not.", node, child));
}

// Combines two MemoryUsageAndInstruction, tiebreaking in favor of the first.
MemoryUsageAndInstruction Merge(
    const MemoryUsageAndInstruction& memory_usage_and_instruction1,
    const MemoryUsageAndInstruction& memory_usage_and_instruction2) {
  return (memory_usage_and_instruction1.memory_usage >=
          memory_usage_and_instruction2.memory_usage)
             ? memory_usage_and_instruction1
             : memory_usage_and_instruction2;
}

}  // namespace

absl::Status HloRematInstructionList::UpdateFromSequence(
    const HloInstructionSequence& order, bool preserve_denylist) {
  absl::flat_hash_map<const HloInstruction*, bool> denylist;
  if (preserve_denylist) {
    for (HloInstruction* inst : order.instructions()) {
      if (item_map_.contains(inst) && item_map_[inst]->denylisted) {
        denylist[inst] = true;
      }
    }
  }
  // Clear item map
  item_map_.clear();
  int64_t position = 0;
  HloRematItem* last = nullptr;
  last_skip_node_ = nullptr;
  first_skip_node_ = nullptr;
  for (HloInstruction* inst : order.instructions()) {
    // Add a new item to the linked list.
    auto item = std::make_unique<HloRematItem>();
    if (preserve_denylist && denylist.contains(inst)) {
      item->denylisted = true;
    }
    item->next = nullptr;
    item->prev = last;
    if (last == nullptr) {
      first_ = item.get();
    } else {
      last->next = item.get();
    }
    last = item.get();

    // Initially position numbers are uniquely assigned in order. Later as
    // instructions are added with InsertBefore* methods, some instructions
    // may have duplicate position numbers, but the values will be guaranteed
    // to be monotonically increasing through the list, and so is still useful
    // for quickly(-ish) determining the order of arbitrary instructions in
    // the list.
    item->instruction = inst;
    item->position = position;
    position++;

    item_map_[inst] = std::move(item);
  }
  return absl::OkStatus();
}

void HloRematInstructionList::InsertBeforeInstructions(
    HloRematItem* to_insert,
    absl::Span<HloRematItem* const> before_instructions) {
  VLOG(3) << "InsertBeforeInstructions: " << to_insert->instruction->name()
          << " before {"
          << absl::StrJoin(before_instructions, ", ",
                           [](std::string* out, HloRematItem* item) {
                             absl::StrAppend(out, item->instruction->name());
                           })
          << "}";

  // Find the minimal position number of any instruction in
  // 'before_instructions'.
  CHECK(!before_instructions.empty());
  HloRematItem* min_position_item = nullptr;
  for (HloRematItem* item : before_instructions) {
    if (min_position_item == nullptr ||
        item->position < min_position_item->position) {
      min_position_item = item;
    }
  }

  // Because more than one instruction in 'before_instructions' may have a
  // position number of 'min_position_number', find the first such instruction
  // with position number 'min_position_number'.

  // First find first instruction with the min position.
  CHECK(min_position_item != nullptr);
  while (min_position_item->prev != nullptr &&
         min_position_item->position == min_position_item->prev->position) {
    min_position_item = min_position_item->prev;
  }

  // Now scan forwards until we find one of the before_instructions.
  while (!absl::c_linear_search(before_instructions, min_position_item)) {
    min_position_item = min_position_item->next;
  }
  return InsertBefore(to_insert, min_position_item);
}

void HloRematInstructionList::PromoteNodesToSkip(
    absl::FunctionRef<bool(HloRematItem*)> should_promote) {
  int64_t count = 0;
  for (auto* item = first(); item != nullptr; item = next(item)) {
    if (should_promote(item)) {
      count += 1;
      if (first_skip_node_ == nullptr) {
        first_skip_node_ = item;
      }
      item->is_skip_node = true;
      item->prev_skip_node = last_skip_node_;
      if (last_skip_node_ != nullptr) {
        last_skip_node_->next_skip_node = item;
      }
      last_skip_node_ = item;
    }
  }
  VLOG(1) << " Rematerialization has " << count << " items in express lane";
}

void HloRematInstructionList::InsertAfterInstructions(
    HloRematItem* to_insert,
    absl::Span<HloRematItem* const> after_instructions) {
  VLOG(3) << "InsertAfterInstructions: " << to_insert->instruction->name()
          << " after {"
          << absl::StrJoin(after_instructions, ", ",
                           [](std::string* out, HloRematItem* item) {
                             absl::StrAppend(out, item->instruction->name());
                           })
          << "}";

  // Find the max position number of any instruction in
  // 'after_instructions'.
  CHECK(!after_instructions.empty());
  HloRematItem* max_position_item = nullptr;
  for (HloRematItem* item : after_instructions) {
    if (max_position_item == nullptr ||
        item->position > max_position_item->position) {
      max_position_item = item;
    }
  }
  // No rematerializable instruction should be inserted at the end of the
  // computation.
  CHECK(max_position_item != nullptr);
  CHECK(max_position_item->next != nullptr);
  InsertBeforeInstructions(to_insert, {max_position_item->next});
}

void HloRematInstructionList::InsertBefore(HloRematItem* item,
                                           HloRematItem* before) {
  VLOG(3) << "InsertBefore: " << item->instruction->name() << " before "
          << before->instruction->name();
  // Always place new nodes on express lane for the ease of implementation.
  item->is_skip_node = true;
  // Find the next express node starting from 'before'. Set up the node's
  // express pointers.
  HloRematItem* cursor = before;
  while (cursor != nullptr && !cursor->is_skip_node) {
    cursor = cursor->next;
  }
  CHECK(cursor == nullptr || cursor->is_skip_node);
  if (cursor == nullptr) {
    //
    // last_skip_node_<---+                              : express lane
    //                    |
    //           ...<->`item`<-> .. <-> `cursor`(null)   : slow lane
    //
    // Reached the end. Set the prev_express to last_skip_node, and reset
    // last_skip.
    item->prev_skip_node = last_skip_node_;
    item->next_skip_node = nullptr;
    last_skip_node_ = item;
  } else {
    //
    //     <-+------------+----------------+--------->   : express lane
    //       |            |                |
    // prev_express..<->`item`<-> .. <-> `cursor` <-> ...: slow lane
    //
    // Reached the next skip node, sets up express pointers accordingly.
    CHECK(cursor->is_skip_node);
    item->prev_skip_node = cursor->prev_skip_node;
    if (item->prev_skip_node != nullptr) {
      item->prev_skip_node->next_skip_node = item;
    }
    item->next_skip_node = cursor;
    cursor->prev_skip_node = item;
  }
  if (first_skip_node_ == cursor) {
    first_skip_node_ = item;
  }
  // Insert new item into linked list.
  item->prev = before->prev;
  item->next = before;
  before->prev = item;
  if (item->prev != nullptr) {
    item->prev->next = item;
  } else {
    first_ = item;
  }

  // Assign the same position number to the newly added instruction as
  // 'before'. This guarantees monotonicity of the position numbers, but not
  // uniqueness.
  item->position = before->position;
}

SleatorDietzOrderMaintenance::SleatorDietzOrderMaintenance() {
  // First allocation cannot fail, since `records_` is empty.
  base_record_ = AllocateRecord(nullptr).value();
  base_record_->instruction = nullptr;
  base_record_->label = 0;
  base_record_->next = base_record_->prev = base_record_;
}

absl::Status SleatorDietzOrderMaintenance::InsertBeforeInstruction(
    const HloInstruction* absl_nullable old_instruction,
    const HloInstruction* absl_nonnull new_instruction) {
  return InsertHelper(old_instruction, new_instruction, /*insert_before=*/true);
}

absl::Status SleatorDietzOrderMaintenance::InsertAfterInstruction(
    const HloInstruction* absl_nullable old_instruction,
    const HloInstruction* absl_nonnull new_instruction) {
  return InsertHelper(old_instruction, new_instruction,
                      /*insert_before=*/false);
}

absl::Status SleatorDietzOrderMaintenance::DeleteInstruction(
    const HloInstruction* absl_nonnull instruction) {
  ASSIGN_OR_RETURN(SleatorDietzRecord * record, GetRecord(instruction));
  record->prev->next = record->next;
  record->next->prev = record->prev;
  records_.erase(instruction);
  return absl::OkStatus();
}

absl::StatusOr<bool> SleatorDietzOrderMaintenance::CompareOrder(
    const HloInstruction* absl_nonnull first_instruction,
    const HloInstruction* absl_nonnull second_instruction) const {
  ASSIGN_OR_RETURN(const SleatorDietzRecord* first_record,
                   GetRecord(first_instruction));
  ASSIGN_OR_RETURN(const SleatorDietzRecord* second_record,
                   GetRecord(second_instruction));
  return GetRelativeLabel(first_record) < GetRelativeLabel(second_record);
}

bool SleatorDietzOrderMaintenance::ContainsInstruction(
    const HloInstruction* absl_nonnull instruction) const {
  return GetRecord(instruction).ok();
}

const HloInstruction* SleatorDietzOrderMaintenance::GetFirstInstruction()
    const {
  if (base_record_->next == base_record_) {
    return nullptr;
  }
  return base_record_->next->instruction;
}

absl::StatusOr<const HloInstruction*>
SleatorDietzOrderMaintenance::GetPreviousInstruction(
    const HloInstruction* absl_nonnull instruction) const {
  ASSIGN_OR_RETURN(const SleatorDietzRecord* record, GetRecord(instruction));
  if (record->prev == base_record_) {
    return nullptr;
  }
  return record->prev->instruction;
}

absl::StatusOr<const HloInstruction*>
SleatorDietzOrderMaintenance::GetNextInstruction(
    const HloInstruction* absl_nonnull instruction) const {
  ASSIGN_OR_RETURN(const SleatorDietzRecord* record, GetRecord(instruction));
  if (record->next == base_record_) {
    return nullptr;
  }
  return record->next->instruction;
}

std::string SleatorDietzOrderMaintenance::ToString() const {
  std::vector<std::string> pieces;
  pieces.push_back(
      absl::StrFormat("{Base Record, Label %d}", base_record_->label));
  for (SleatorDietzRecord* record = base_record_->next; record != base_record_;
       record = record->next) {
    pieces.push_back(absl::StrFormat("{Instruction %p, Label %d}",
                                     record->instruction, record->label));
  }
  return absl::StrJoin(pieces, " ");
}

absl::Status SleatorDietzOrderMaintenance::VerifyInvariantsForTesting() const {
  for (SleatorDietzRecord* record = base_record_; record->next != base_record_;
       record = record->next) {
    if (record->next->prev != record) {
      return absl::InternalError(absl::StrFormat(
          "Record for instruction %p's successor's predecessor is not itself.",
          record->instruction));
    }
    if (GetRelativeLabel(record) >= GetRelativeLabel(record->next)) {
      return absl::InternalError(absl::StrFormat(
          "Misordered labels: record for instruction %p has relative label %d "
          "and next record for instruction %p has relative label %d",
          record->instruction, GetRelativeLabel(record),
          record->next->instruction, GetRelativeLabel(record->next)));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<SleatorDietzOrderMaintenance::SleatorDietzRecord* absl_nonnull>
SleatorDietzOrderMaintenance::AllocateRecord(
    const HloInstruction* absl_nullable instruction) {
  if (records_.find(instruction) != records_.end()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Instruction %p already has a record.", instruction));
  }
  records_.emplace(instruction, new SleatorDietzRecord());
  records_[instruction]->instruction = instruction;
  return records_[instruction].get();
}

absl::StatusOr<SleatorDietzOrderMaintenance::SleatorDietzRecord* absl_nonnull>
SleatorDietzOrderMaintenance::GetRecord(
    const HloInstruction* absl_nullable instruction) {
  if (instruction == nullptr) {
    return base_record_;
  }
  if (auto it = records_.find(instruction); it != records_.end()) {
    return it->second.get();
  } else {
    return absl::NotFoundError(
        absl::StrFormat("Sleator-Dietz Order Maintenance data structure does "
                        "not contain instruction %p",
                        instruction));
  }
}

absl::StatusOr<
    const SleatorDietzOrderMaintenance::SleatorDietzRecord* absl_nonnull>
SleatorDietzOrderMaintenance::GetRecord(
    const HloInstruction* absl_nullable instruction) const {
  if (instruction == nullptr) {
    return base_record_;
  }
  if (auto it = records_.find(instruction); it != records_.end()) {
    return it->second.get();
  } else {
    return absl::NotFoundError(
        absl::StrFormat("Sleator-Dietz Order Maintenance data structure does "
                        "not contain instruction %p",
                        instruction));
  }
}

uint64_t SleatorDietzOrderMaintenance::GetRelativeLabel(
    const SleatorDietzRecord* absl_nonnull record) const {
  return record->label - base_record_->label;
}

absl::uint128 SleatorDietzOrderMaintenance::GetNextRelativeLabel(
    SleatorDietzRecord* absl_nonnull record) const {
  if (record->next == base_record_) {
    // Return the arena size, 2^64.
    return absl::MakeUint128(1, 0);
  } else {
    return GetRelativeLabel(record->next);
  }
}

void SleatorDietzOrderMaintenance::RearrangeLabels(
    SleatorDietzRecord* absl_nonnull starting_record) {
  SleatorDietzRecord* current_record = starting_record->next;
  // Our cap on number of elements guarantees that squaring this never
  // overflows.
  uint64_t num_hops = 1;
  while (current_record->label - starting_record->label <=
         num_hops * num_hops) {
    ++num_hops;
    current_record = current_record->next;
    if (current_record == starting_record) break;
  }
  const SleatorDietzRecord* ending_record = current_record;
  // Relabel everything strictly between `starting_record` and `ending_record`
  // to make the gaps as balanced as possible. Looping all the way is treated as
  // a gap of 2^64, otherwise we take the difference between endpoints mod 2^64.
  absl::uint128 total_gap =
      (starting_record == ending_record)
          ? absl::MakeUint128(1, 0)
          : (ending_record->label - starting_record->label);

  uint64_t hop = 1;
  for (current_record = starting_record->next; current_record != ending_record;
       current_record = current_record->next) {
    // Note that the first term is computed mod 2^128 to avoid overflow before
    // dividing, but then the first and second terms are added mod 2^64.
    current_record->label = static_cast<uint64_t>(total_gap * hop / num_hops) +
                            starting_record->label;
    ++hop;
  }
}

absl::Status SleatorDietzOrderMaintenance::InsertHelper(
    const HloInstruction* absl_nullable old_instruction,
    const HloInstruction* absl_nonnull new_instruction, bool insert_before) {
  // Due to our arena size of 2^64, we can have at most (2^32 - 1) records.
  // Conveniently, that number is UINT32_MAX.
  if (records_.size() >= UINT32_MAX) {
    return absl::ResourceExhaustedError(
        "Sleator-Dietz Order Maintenance data structure cannot support any "
        "more records.");
  }

  // The sequencing of these next few steps is important.
  // - We first look up the record for `old_instruction`, checking for the case
  //   where it does not have a record.
  // - We next allocate a new record for `new_instruction`, checking for the
  //   case where it already has a record (if it does, nothing is allocated and
  //   we just throw an error). This is the last way to throw an error, so we
  //   are now safe to fiddle with data values.
  // - We rearrange the labels while `new_record` hasn't been inserted into the
  //   doubly-linked list yet.
  // - Finally, with the labels fixed, we can insert `new_record` and compute
  //   the correct label for it off of its two neighbors.
  TF_ASSIGN_OR_RETURN(SleatorDietzRecord * old_record,
                      GetRecord(old_instruction));
  if (insert_before) {
    old_record = old_record->prev;
    // We are now inserting after `old_record`.
  }
  TF_ASSIGN_OR_RETURN(SleatorDietzRecord * new_record,
                      AllocateRecord(new_instruction));

  RearrangeLabels(old_record);

  absl::uint128 new_label = GetRelativeLabel(old_record);
  new_label += GetNextRelativeLabel(old_record);  // Do this mod 2^128 to avoid
                                                  // overflow before we divide.
  new_label /= 2;                                 // Integer division
  new_record->label = static_cast<uint64_t>(new_label);  // Round down mod 2^64.
  new_record->label += base_record_->label;  // Correct from relative label to
                                             // actual label.

  // Four pointers for us to fix.
  new_record->next = old_record->next;
  new_record->prev = old_record;
  old_record->next = new_record;
  new_record->next->prev = new_record;

  return absl::OkStatus();
}

AVLLazySegmentTree::AVLLazySegmentTree(
    absl::Span<const MemoryUsageAndInstruction>
        initial_memory_usage_and_instructions) {
  if (initial_memory_usage_and_instructions.empty()) {
    root_ = nullptr;
    return;
  }

  // Construct a leaf node for every initial instruction.
  std::deque<AVLTreeNode*> subtrees;
  for (const MemoryUsageAndInstruction& memory_usage_and_instruction :
       initial_memory_usage_and_instructions) {
    AVLTreeNode* leaf_node = AllocateNode();
    SetLeafNode(leaf_node, memory_usage_and_instruction, /*parent=*/nullptr);
    instruction_to_leaf_[memory_usage_and_instruction.instruction] = leaf_node;
    subtrees.push_back(leaf_node);
  }
  subtrees.push_back(nullptr);  // special signaling value
  // Construct the interior nodes. To maintain the AVL guarantee (the heights of
  // subtrees hanging off any node differ by at most one), we alternate
  // between sweeping left-to-right and right-to-left when merging nodes. This
  // guarantees that no node can be skipped in consecutive sweep rounds (which
  // could happen if e.g. we only swept left-to-right).
  bool left_to_right = true;
  while (subtrees.size() > 1) {
    AVLTreeNode* subtree1 = left_to_right ? subtrees.front() : subtrees.back();
    left_to_right ? subtrees.pop_front() : subtrees.pop_back();
    if (subtree1 == nullptr) {
      // We finished this sweep, and prepare to sweep in the opposite direction.
      left_to_right = !left_to_right;
      if (subtrees.size() == 1) break;
      left_to_right ? subtrees.push_back(nullptr)
                    : subtrees.push_front(nullptr);
      continue;
    }
    AVLTreeNode* subtree2 = left_to_right ? subtrees.front() : subtrees.back();
    left_to_right ? subtrees.pop_front() : subtrees.pop_back();
    if (subtree2 == nullptr) {
      // We finished this sweep, but had an extra element.
      left_to_right ? subtrees.push_back(subtree1)
                    : subtrees.push_front(subtree1);
      left_to_right = !left_to_right;
      if (subtrees.size() == 1) break;
      left_to_right ? subtrees.push_back(nullptr)
                    : subtrees.push_front(nullptr);
      continue;
    }
    AVLTreeNode* interior_node = AllocateNode();
    SetInteriorNode(interior_node, left_to_right ? subtree1 : subtree2,
                    left_to_right ? subtree2 : subtree1, /*parent=*/nullptr);
    subtree1->parent = subtree2->parent = interior_node;
    left_to_right ? subtrees.push_back(interior_node)
                  : subtrees.push_front(interior_node);
  }
  root_ = subtrees.front();
}

absl::Status AVLLazySegmentTree::InsertBeforeInstruction(
    const HloInstruction* absl_nonnull old_instruction,
    const MemoryUsageAndInstruction& new_memory_usage_and_instruction) {
  return InsertHelper(old_instruction, new_memory_usage_and_instruction,
                      /*before=*/true);
}

absl::Status AVLLazySegmentTree::InsertAfterInstruction(
    const HloInstruction* absl_nonnull old_instruction,
    const MemoryUsageAndInstruction& new_memory_usage_and_instruction) {
  return InsertHelper(old_instruction, new_memory_usage_and_instruction,
                      /*before=*/false);
}

absl::Status AVLLazySegmentTree::Delete(
    const HloInstruction* absl_nonnull instruction) {
  TF_ASSIGN_OR_RETURN(std::vector<AVLTreeNode*> leaf_to_root,
                      GetPathFromLeafToRoot(instruction));
  AVLTreeNode* leaf_node = leaf_to_root[0];
  // Handle the edge case where this is the last node in the entire tree.
  if (root_ == leaf_node) {
    root_ = nullptr;
    node_storage_.clear();
    instruction_to_leaf_.clear();
    return absl::OkStatus();
  }
  // With that edge case out of the way, we are sure that this leaf has an
  // interior node above it that we need to delete as well. If `leaf` is a left
  // child, the situation looks like:
  //
  //    parent_of_interior_node       parent_of_interior_node
  //               |                             |
  //         interior_node       -->         other_leaf
  //           /       \
  //          leaf other_leaf
  AVLTreeNode* interior_node = leaf_node->parent;
  AVLTreeNode* parent_of_interior_node = interior_node->parent;
  AVLTreeNode* other_leaf = (interior_node->left_child == leaf_node)
                                ? interior_node->right_child
                                : interior_node->left_child;

  // Directly connect `other_leaf` and `parent_of_interior_node`.
  if (parent_of_interior_node == nullptr) {
    root_ = other_leaf;
  } else {
    TF_ASSIGN_OR_RETURN(
        AVLTreeNode * *child_pointer,
        FindChildPointer(parent_of_interior_node, interior_node));
    *child_pointer = other_leaf;
  }
  other_leaf->parent = parent_of_interior_node;

  // Can now delete these two nodes since we don't need the information in them
  // anymore.
  node_storage_.erase(leaf_node->node_storage_iterator);
  node_storage_.erase(interior_node->node_storage_iterator);

  for (AVLTreeNode* node : leaf_to_root) {
    // We skip over both `leaf` and `interior_node` and start at
    // `parent_of_interior_node`.
    if (node == leaf_node || node == interior_node) {
      continue;
    }
    // Recompute the memory_usage and height
    SetInteriorNode(node, node->left_child, node->right_child, node->parent);
    if (node->parent == nullptr) {
      root_ = Balance(node);
    } else {
      TF_ASSIGN_OR_RETURN(AVLTreeNode * *child_pointer,
                          FindChildPointer(node->parent, node));
      *child_pointer = Balance(node);
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<MemoryUsageAndInstruction> AVLLazySegmentTree::Query(
    const HloInstruction* absl_nonnull first_instruction,
    const HloInstruction* absl_nonnull last_instruction) {
  TF_ASSIGN_OR_RETURN(
      std::deque<AVLTreeNode*> spanning_nodes,
      GetNodesSpanningInterval(first_instruction, last_instruction));
  MemoryUsageAndInstruction partial_merge =
      spanning_nodes.front()->memory_usage_and_instruction;
  spanning_nodes.pop_front();
  while (!spanning_nodes.empty()) {
    partial_merge = Merge(partial_merge,
                          spanning_nodes.front()->memory_usage_and_instruction);
    spanning_nodes.pop_front();
  }

  return partial_merge;
}

absl::StatusOr<MemoryUsageAndInstruction> AVLLazySegmentTree::Query() const {
  if (root_ == nullptr) {
    return absl::NotFoundError(
        "Tree does not have a root node and cannot be queried");
  }
  return root_->memory_usage_and_instruction;
}

absl::Status AVLLazySegmentTree::Update(
    const HloInstruction* absl_nonnull first_instruction,
    const HloInstruction* absl_nonnull last_instruction,
    int64_t additive_memory_update) {
  // Get the relevant nodes using a subroutine.
  TF_ASSIGN_OR_RETURN(
      std::deque<AVLTreeNode*> spanning_nodes,
      GetNodesSpanningInterval(first_instruction, last_instruction));
  for (AVLTreeNode* spanning_node : spanning_nodes) {
    spanning_node->memory_usage_and_instruction.memory_usage +=
        additive_memory_update;
    spanning_node->lazy_additive_memory_update += additive_memory_update;
  }

  // We need to recalculate all the nodes that are parents or recursive parents
  // of any node in `spanning_nodes`. Fortunately, because all of these nodes
  // lie very close to the paths `first_leaf_to_root` and `last_leaf_to_root`,
  // this only involves touching O(log n) nodes.
  TF_ASSIGN_OR_RETURN(std::vector<AVLTreeNode*> first_leaf_to_root,
                      GetPathFromLeafToRoot(first_instruction));
  for (AVLTreeNode* node : first_leaf_to_root) {
    // We skip the first node because it is a leaf and does not have a left or
    // right child.
    if (node == first_leaf_to_root[0]) {
      continue;
    }
    node->memory_usage_and_instruction =
        Merge(node->left_child->memory_usage_and_instruction,
              node->right_child->memory_usage_and_instruction);
  }
  TF_ASSIGN_OR_RETURN(std::vector<AVLTreeNode*> last_leaf_to_root,
                      GetPathFromLeafToRoot(last_instruction));
  for (AVLTreeNode* node : last_leaf_to_root) {
    // We skip the first node because it is a leaf and does not have a left or
    // right child.
    if (node == last_leaf_to_root[0]) {
      continue;
    }
    node->memory_usage_and_instruction =
        Merge(node->left_child->memory_usage_and_instruction,
              node->right_child->memory_usage_and_instruction);
  }
  return absl::OkStatus();
}

absl::Status AVLLazySegmentTree::VerifyInvariantsForTesting() const {
  for (auto&& node : node_storage_) {
    // Check that the balance factor is in {-1, 0, +1}.
    int balance_factor = BalanceFactor(node.get());
    if (balance_factor != -1 && balance_factor != 0 && balance_factor != 1) {
      return absl::InternalError(absl::StrFormat(
          "Node %p has balance factor %d", node.get(), balance_factor));
    }
    // Check that both or none of the child pointers are nullptr.
    if ((node->left_child == nullptr && node->right_child != nullptr) ||
        (node->left_child != nullptr && node->right_child == nullptr)) {
      return absl::InternalError(
          absl::StrFormat("Node %p has exactly one child", node.get()));
    }
    // Check that child pointers and parent pointer are correct.
    if (node->left_child != nullptr && node->left_child->parent != node.get()) {
      return absl::InternalError(absl::StrFormat(
          "Node %p's left child's parent is not that node", node.get()));
    }
    if (node->right_child != nullptr &&
        node->right_child->parent != node.get()) {
      return absl::InternalError(absl::StrFormat(
          "Node %p's right child's parent is not that node", node.get()));
    }
    if (node->parent == nullptr && root_ != node.get()) {
      return absl::InternalError(absl::StrFormat(
          "Node %p's parent is null but node is not root", node.get()));
    }
    if (node->parent != nullptr && node->parent->left_child != node.get() &&
        node->parent->right_child != node.get()) {
      return absl::InternalError(absl::StrFormat(
          "Node %p's parent's children are both not that node", node.get()));
    }

    const bool is_leaf = node->left_child == nullptr;
    // Check that the height is correct.
    if (is_leaf) {
      if (node->height != 0) {
        return absl::InternalError(absl::StrFormat(
            "Node %p is a leaf but has height %d", node.get(), node->height));
      }
    } else {
      int height_according_to_children =
          1 + std::max(node->left_child->height, node->right_child->height);
      if (node->height != height_according_to_children) {
        return absl::InternalError(absl::StrFormat(
            "Node %p has height %d but should have height %d", node.get(),
            node->height, height_according_to_children));
      }
    }
  }
  return absl::OkStatus();
}

AVLTreeNode* absl_nonnull AVLLazySegmentTree::AllocateNode() {
  node_storage_.emplace_back(new AVLTreeNode());
  // Set the `node_storage_iterator` field to point to the node we just added.
  node_storage_.back()->node_storage_iterator = node_storage_.end();
  --(node_storage_.back()->node_storage_iterator);
  return node_storage_.back().get();
}

void AVLLazySegmentTree::SetLeafNode(
    AVLTreeNode* leaf_node,
    const MemoryUsageAndInstruction& memory_usage_and_instruction,
    AVLTreeNode* parent) {
  leaf_node->left_child = leaf_node->right_child = nullptr;
  leaf_node->parent = parent;
  leaf_node->height = 0;
  leaf_node->memory_usage_and_instruction = memory_usage_and_instruction;
  leaf_node->lazy_additive_memory_update = 0;
}

void AVLLazySegmentTree::SetInteriorNode(AVLTreeNode* interior_node,
                                         AVLTreeNode* left_child,
                                         AVLTreeNode* right_child,
                                         AVLTreeNode* parent) {
  interior_node->left_child = left_child;
  interior_node->right_child = right_child;
  interior_node->parent = parent;
  interior_node->height = 1 + std::max(left_child->height, right_child->height);
  interior_node->memory_usage_and_instruction =
      Merge(left_child->memory_usage_and_instruction,
            right_child->memory_usage_and_instruction);
  interior_node->lazy_additive_memory_update = 0;
}

AVLTreeNode* absl_nonnull AVLLazySegmentTree::Balance(
    AVLTreeNode* absl_nonnull node) {
  const int balance_factor = BalanceFactor(node);
  if (balance_factor == -1 || balance_factor == 0 || balance_factor == 1) {
    return node;  // Already balanced enough.
  } else if (balance_factor == -2) {
    // Left subtree is larger, need to rotate right to fix.
    if (BalanceFactor(node->left_child) > 0) {
      node->left_child = RotateLeft(node->left_child);
    }
    return RotateRight(node);
  } else {  // balance_factor == 2
    if (BalanceFactor(node->right_child) < 0) {
      node->right_child = RotateRight(node->right_child);
    }
    return RotateLeft(node);
  }
  return node;
}

int AVLLazySegmentTree::BalanceFactor(AVLTreeNode* absl_nonnull node) const {
  if (node->left_child == nullptr || node->right_child == nullptr) {
    // Leaves have balance-factor zero.
    return 0;
  }
  return node->right_child->height - node->left_child->height;
}

AVLTreeNode* absl_nonnull AVLLazySegmentTree::RotateLeft(
    AVLTreeNode* absl_nonnull node) {
  AVLTreeNode* parent_of_node = node->parent;
  AVLTreeNode* new_root = node->right_child;

  // Push lazy updates down before moving nodes.
  PushDownLaziness(node);
  PushDownLaziness(new_root);

  AVLTreeNode* subtree1 = node->left_child;
  AVLTreeNode* subtree2 = new_root->left_child;
  AVLTreeNode* subtree3 = new_root->right_child;

  // Use the SetInteriorNode() function bottom-up so that merging of
  // MemoryUsageAndInstruction is done correctly.
  SetInteriorNode(node, /*left_child=*/subtree1, /*right_child=*/subtree2,
                  /*parent=*/new_root);
  SetInteriorNode(new_root, /*left_child=*/node, /*right_child=*/subtree3,
                  /*parent=*/parent_of_node);
  subtree1->parent = node;
  subtree2->parent = node;
  subtree3->parent = new_root;
  return new_root;
}

AVLTreeNode* absl_nonnull AVLLazySegmentTree::RotateRight(
    AVLTreeNode* absl_nonnull node) {
  AVLTreeNode* parent_of_node = node->parent;
  AVLTreeNode* new_root = node->left_child;

  // Push lazy updates down before moving nodes.
  PushDownLaziness(node);
  PushDownLaziness(new_root);

  AVLTreeNode* subtree1 = new_root->left_child;
  AVLTreeNode* subtree2 = new_root->right_child;
  AVLTreeNode* subtree3 = node->right_child;

  // Use the SetInteriorNode() function bottom-up so that merging of
  // MemoryUsageAndInstruction is done correctly.
  SetInteriorNode(node, /*left_child=*/subtree2, /*right_child=*/subtree3,
                  /*parent=*/new_root);
  SetInteriorNode(new_root, /*left_child=*/subtree1, /*right_child=*/node,
                  /*parent=*/parent_of_node);
  subtree1->parent = new_root;
  subtree2->parent = node;
  subtree3->parent = node;
  return new_root;
}

void AVLLazySegmentTree::PushDownLaziness(AVLTreeNode* absl_nonnull node) {
  if (node->left_child != nullptr) {
    node->left_child->memory_usage_and_instruction.memory_usage +=
        node->lazy_additive_memory_update;
    node->left_child->lazy_additive_memory_update +=
        node->lazy_additive_memory_update;
  }
  if (node->right_child != nullptr) {
    node->right_child->memory_usage_and_instruction.memory_usage +=
        node->lazy_additive_memory_update;
    node->right_child->lazy_additive_memory_update +=
        node->lazy_additive_memory_update;
  }
  node->lazy_additive_memory_update = 0;
}

absl::StatusOr<std::vector<AVLTreeNode* absl_nonnull>>
AVLLazySegmentTree::GetPathFromLeafToRoot(const HloInstruction* instruction) {
  if (!instruction_to_leaf_.contains(instruction)) {
    return absl::NotFoundError(
        absl::StrFormat("Tree does not contain instruction %p", instruction));
  }
  std::vector<AVLTreeNode*> path;
  path.push_back(instruction_to_leaf_[instruction]);
  while (path.back()->parent != nullptr) {
    path.push_back(path.back()->parent);
  }
  if (path.back() != root_) {
    return absl::InternalError(absl::StrFormat(
        "Path to root for instruction %p does not contain root.", instruction));
  }
  for (AVLTreeNode* node : path | std::views::reverse) {
    PushDownLaziness(node);
  }
  return path;
}

absl::StatusOr<std::deque<AVLTreeNode* absl_nonnull>>
AVLLazySegmentTree::GetNodesSpanningInterval(
    const HloInstruction* absl_nonnull first_instruction,
    const HloInstruction* absl_nonnull last_instruction) {
  std::deque<AVLTreeNode*> spanning_nodes;

  TF_ASSIGN_OR_RETURN(std::vector<AVLTreeNode*> first_leaf_to_root,
                      GetPathFromLeafToRoot(first_instruction));
  TF_ASSIGN_OR_RETURN(std::vector<AVLTreeNode*> last_leaf_to_root,
                      GetPathFromLeafToRoot(last_instruction));

  // We get this edge case out of the way first. When we are not in this edge
  // case, we know that the two paths must disagree with each other at some
  // point.
  if (first_instruction == last_instruction) {
    spanning_nodes.push_back(first_leaf_to_root[0]);
    return spanning_nodes;
  }

  // We search for that point of first disagreement. Note that paths from leaf
  // to root cannot have size zero, since that implies that the leaf doesn't
  // exist and we would have errored out from GetPathFromLeafToRoot() already.
  int index_into_first_path = first_leaf_to_root.size() - 1;
  int index_into_last_path = last_leaf_to_root.size() - 1;
  AVLTreeNode* least_common_ancestor = root_;

  while (first_leaf_to_root[index_into_first_path] ==
         last_leaf_to_root[index_into_last_path]) {
    least_common_ancestor = first_leaf_to_root[index_into_first_path];
    --index_into_first_path;
    --index_into_last_path;
    // This cannot happen since we got rid of the
    // first_instruction == last_instruction case.
    if (index_into_first_path < 0 || index_into_last_path < 0) {
      return absl::InternalError(absl::StrFormat(
          "Ran out of nodes when waiting for paths to instructions %p, %p to "
          "diverge",
          first_instruction, last_instruction));
    }
  }

  if (least_common_ancestor->left_child !=
          first_leaf_to_root[index_into_first_path] ||
      least_common_ancestor->right_child !=
          last_leaf_to_root[index_into_last_path]) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Queried instructions %p, %p are in incorrect order",
                        first_instruction, last_instruction));
  }

  for (; index_into_first_path >= 0; --index_into_first_path) {
    if (index_into_first_path == 0) {
      // We are at a leaf node.
      spanning_nodes.push_front(first_leaf_to_root[index_into_first_path]);
    } else {
      // We are at an internal node. If we are about to go left, we need to
      // add the right child.
      if (first_leaf_to_root[index_into_first_path - 1] ==
          first_leaf_to_root[index_into_first_path]->left_child) {
        AVLTreeNode* node =
            first_leaf_to_root[index_into_first_path]->right_child;
        PushDownLaziness(node);
        spanning_nodes.push_front(node);
      }
    }
  }
  for (; index_into_last_path >= 0; --index_into_last_path) {
    if (index_into_last_path == 0) {
      // We are at a leaf node.
      spanning_nodes.push_back(last_leaf_to_root[index_into_last_path]);
    } else {
      // We are at an internal node. If we are about to go right, we need to
      // add the left child.
      if (last_leaf_to_root[index_into_last_path - 1] ==
          last_leaf_to_root[index_into_last_path]->right_child) {
        AVLTreeNode* node = last_leaf_to_root[index_into_last_path]->left_child;
        PushDownLaziness(node);
        spanning_nodes.push_back(node);
      }
    }
  }
  if (spanning_nodes.empty()) {
    return absl::InternalError(
        absl::StrFormat("Could not find nodes representing interval %p,%p",
                        first_instruction, last_instruction));
  }

  return spanning_nodes;
}

absl::Status AVLLazySegmentTree::InsertHelper(
    const HloInstruction* absl_nonnull old_instruction,
    const MemoryUsageAndInstruction& new_memory_usage_and_instruction,
    bool insert_before) {
  TF_ASSIGN_OR_RETURN(std::vector<AVLTreeNode*> leaf_to_root,
                      GetPathFromLeafToRoot(old_instruction));
  AVLTreeNode* leaf = leaf_to_root[0];

  // We replace `leaf` with a three-node subtree. For example, when
  // `insert_before` is true, the situation looks like:
  //
  //   parent_of_leaf        parent_of_leaf
  //          |                     |
  //         leaf      -->  new_interior_node
  //                            /       \
  //                         new_leaf  leaf
  AVLTreeNode* parent_of_leaf = leaf->parent;
  AVLTreeNode* new_interior_node = AllocateNode();
  AVLTreeNode* new_leaf_node = AllocateNode();
  // After saving a pointer to the parent of leaf, we work our way up, fixing
  // the tree.
  SetLeafNode(new_leaf_node, new_memory_usage_and_instruction,
              /*parent=*/new_interior_node);
  instruction_to_leaf_[new_memory_usage_and_instruction.instruction] =
      new_leaf_node;
  leaf->parent = new_interior_node;
  if (insert_before) {
    SetInteriorNode(new_interior_node, new_leaf_node, leaf, parent_of_leaf);
  } else {
    SetInteriorNode(new_interior_node, leaf, new_leaf_node, parent_of_leaf);
  }
  if (parent_of_leaf == nullptr) {
    // `root_` used to point to leaf, update it
    root_ = new_interior_node;
    return absl::OkStatus();
  } else {
    // `parent_of_leaf` used to point to leaf, update it
    TF_ASSIGN_OR_RETURN(AVLTreeNode * *child_pointer,
                        FindChildPointer(parent_of_leaf, leaf));
    *child_pointer = new_interior_node;
  }
  // There's no need to balance `new_interior_node` since we just constructed
  // it to be completely balanced (score 0), but other interior nodes on the
  // path up might have become unbalanced.

  for (AVLTreeNode* node : leaf_to_root) {
    // We drop the first node of `leaf_to_root` (i.e. our leaf) and focus on the
    // remaining nodes (all internal).
    if (node == leaf) {
      continue;
    }
    // Recompute the memory_usage and height
    SetInteriorNode(node, node->left_child, node->right_child, node->parent);
    if (node->parent == nullptr) {
      root_ = Balance(node);
    } else {
      TF_ASSIGN_OR_RETURN(AVLTreeNode * *child_pointer,
                          FindChildPointer(node->parent, node));
      *child_pointer = Balance(node);
    }
  }

  return absl::OkStatus();
}
}  // namespace xla

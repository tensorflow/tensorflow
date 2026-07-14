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
#include <memory>
#include <optional>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
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
}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_REMATERIALIZATION_DATA_STRUCTURES_H_

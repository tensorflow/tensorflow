/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/hlo_rematerialization.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/map_util.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_dataflow_analysis.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"

namespace xla {

namespace {

using ::tsl::strings::HumanReadableNumBytes;

// Potential optimizations:
// . TODO(b/35244891): Avoid N^2 behavior by keeping a priority queue
//   of candidates.
// . Cache IsRematerializable in Item?  Only correct if control
//   predecessors and successors don't change.

// Returns true if the given instruction is rematerializable.
bool IsRematerializable(const HloInstruction* instruction) {
  if (instruction->opcode() == HloOpcode::kCopy) {
    if (LayoutUtil::Equal(instruction->shape().layout(),
                          instruction->operand(0)->shape().layout())) {
      // Don't rematerialize copies added by copy insertion (layout doesn't
      // change).
      return false;
    }
  }

  if (auto collective = DynCast<HloCollectiveInstruction>(instruction)) {
    return !collective->constrain_layout();
  }

  // Don't rematerialize instructions with side effects or instructions which
  // cannot be cloned safely.
  switch (instruction->opcode()) {
    case HloOpcode::kCall:
    case HloOpcode::kConstant:
    case HloOpcode::kConditional:
    case HloOpcode::kCustomCall:
    case HloOpcode::kParameter:
    case HloOpcode::kWhile:
      return false;
    default:
      return !instruction->HasSideEffect();
  }
}

// Checks whether an instruction can be rematerialized, by looking up the
// cache before, and eventually calling the IsRematerializable() API.
bool CanBeRematerialized(
    const HloInstruction* instruction,
    absl::flat_hash_map<const HloInstruction*, bool>* rematerializable_map) {
  auto it = rematerializable_map->find(instruction);
  if (it != rematerializable_map->end()) {
    return it->second;
  }
  bool rematerializable = IsRematerializable(instruction);
  (*rematerializable_map)[instruction] = rematerializable;
  return rematerializable;
}

// Return if this is an instruction that relays the buffers it uses to its own
// users and if this is one of these instructions we support the
// rematerialization of.
bool IsSupportedIndirectUser(const HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kBitcast ||
         instruction->opcode() == HloOpcode::kGetTupleElement;
}

// Type holding a unique identifier for each Buffer object.
using BufferId = int64_t;
using BufferIdList = absl::InlinedVector<BufferId, 3>;

struct RematStrategy {
  enum {
    // Recompute the node at a later program point.
    kRecompute,
    // Change the layout into a compact form and uncompress it back at a later
    // program point.
    kCompress,
    // Copy the data off the device to the host to be copied back later.
    kHostOffload,
  } kind;
  Shape compact_shape;
};

// We wrap HloInstruction* with an Item that holds auxiliary
// per-instruction state.
struct Item {
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
  friend class InstructionList;

  // Items are arranged in a doubly linked list.
  Item* next = nullptr;
  Item* prev = nullptr;

  Item* prev_skip_node = nullptr;
  Item* next_skip_node = nullptr;

  // List is ordered by position, which can however be duplicated as
  // new instructions are inserted.  See InsertBeforeInstructions
  // comment for details.
  int64_t position;
};

// Data structure meant to record the user of the buffer defined from an Item.
// It records also the operand_number from where such use derives, so that
// indirect uses can be better identified (like for example a buffer used
// through a bitcast).
struct ItemUse {
  Item* user;
  int64_t operand_number;
  std::optional<int64_t> index;

  ItemUse(Item* user, int64_t op_num, std::optional<int64_t> index)
      : user(user), operand_number(op_num), index(index) {}
  bool operator==(const ItemUse& other) const {
    return user == other.user && operand_number == other.operand_number &&
           index == other.index;
  }
};

using ItemList = absl::InlinedVector<Item*, 3>;
using UsesList = absl::InlinedVector<ItemUse, 3>;

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
class InstructionList {
 public:
  explicit InstructionList(const HloInstructionSequence& order) {
    int64_t position = 0;
    Item* last = nullptr;
    last_skip_node_ = nullptr;
    first_skip_node_ = nullptr;
    for (HloInstruction* inst : order.instructions()) {
      // Add a new item to the linked list.
      Item* item = new Item;
      item->next = nullptr;
      item->prev = last;
      if (last == nullptr) {
        first_ = item;
      } else {
        last->next = item;
      }
      last = item;

      // Initially position numbers are uniquely assigned in order. Later as
      // instructions are added with InsertBefore* methods, some instructions
      // may have duplicate position numbers, but the values will be guaranteed
      // to be monotonically increasing through the list, and so is still useful
      // for quickly(-ish) determining the order of arbitrary instructions in
      // the list.
      item->instruction = inst;
      item->position = position;
      position++;

      item_map_[inst] = item;
    }
  }

  ~InstructionList() {
    for (Item* item = first_; item != nullptr;) {
      Item* next = item->next;
      delete item;
      item = next;
    }
  }

  size_t size() const { return item_map_.size(); }

  // For ordered iteration over items.
  //    for (auto item = q.first(); item != nullptr; item = q.next(item)) {...}
  Item* first() const { return first_; }
  Item* next(Item* item) const { return item->next; }
  const Item* next(const Item* item) const { return item->next; }
  Item* prev(Item* item) const { return item->prev; }
  const Item* prev(const Item* item) const { return item->prev; }

  Item* first_skip_node() const { return first_skip_node_; }
  Item* next_skip_node(Item* item) const { return item->next_skip_node; }

  // Creates an Item for the given instruction, but doesn't add it to the list.
  // (Use InsertBeforeInstructions to add the Item to the list.)
  Item* CreateItem(HloInstruction* inst) {
    Item* item = new Item;
    item->instruction = inst;
    CHECK(item_map_.insert({inst, item}).second)
        << "inserting inst twice " << inst->name();
    return item;
  }

  // Return the Item corresponding to inst.
  Item* GetItem(const HloInstruction* inst) const {
    auto iter = item_map_.find(inst);
    CHECK(iter != item_map_.end()) << "Did not find " << inst->name();
    return iter->second;
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
  void InsertBeforeInstructions(Item* to_insert,
                                absl::Span<Item* const> before_instructions) {
    VLOG(3) << "InsertBeforeInstructions: " << to_insert->instruction->name()
            << " before {"
            << absl::StrJoin(before_instructions, ", ",
                             [](std::string* out, Item* item) {
                               absl::StrAppend(out, item->instruction->name());
                             })
            << "}";

    // Find the minimal position number of any instruction in
    // 'before_instructions'.
    CHECK(!before_instructions.empty());
    Item* min_position_item = nullptr;
    for (Item* item : before_instructions) {
      if (min_position_item == nullptr ||
          item->position < min_position_item->position) {
        min_position_item = item;
      }
    }

    // Because more than one instruction in 'before_instructions' may have a
    // position number of 'min_position_number', find the first such instruction
    // with position number 'min_position_number'.

    // First find first instruction with the min position.
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

  // Scan the list and promote nodes to express lane if should_promote(Item)
  // returns true;
  void PromoteNodesToSkip(absl::FunctionRef<bool(Item*)> should_promote) {
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

  void InsertAfterInstructions(Item* to_insert,
                               absl::Span<Item* const> after_instructions) {
    VLOG(3) << "InsertAfterInstructions: " << to_insert->instruction->name()
            << " after {"
            << absl::StrJoin(after_instructions, ", ",
                             [](std::string* out, Item* item) {
                               absl::StrAppend(out, item->instruction->name());
                             })
            << "}";

    // Find the max position number of any instruction in
    // 'after_instructions'.
    CHECK(!after_instructions.empty());
    Item* max_position_item = nullptr;
    for (Item* item : after_instructions) {
      if (max_position_item == nullptr ||
          item->position > max_position_item->position) {
        max_position_item = item;
      }
    }
    // No rematerializable instruction should be inserted at the end of the
    // computation.
    CHECK(max_position_item->next != nullptr);
    InsertBeforeInstructions(to_insert, {max_position_item->next});
  }

  void Denylist(const HloInstruction* inst) {
    GetItem(inst)->denylisted = true;
  }

 private:
  // Insert instruction 'item' immediately before 'before' in the list.
  void InsertBefore(Item* item, Item* before) {
    VLOG(3) << "InsertBefore: " << item->instruction->name() << " before "
            << before->instruction->name();
    // Always place new nodes on express lane for the ease of implementation.
    item->is_skip_node = true;
    // Find the next express node starting from 'before'. Set up the node's
    // express pointers.
    Item* cursor = before;
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

  Item* first_;

  // First skip node of this list.
  Item* first_skip_node_;

  // Last skip node of this list.
  Item* last_skip_node_;

  // Item for each instruction.
  absl::flat_hash_map<const HloInstruction*, Item*> item_map_;
};

// Return the items which use the given LogicalBuffer. Sets
// has_indirect_users to whether any of the uses is indirect. A use is indirect
// if the instruction defining logical_buffer is not an operand of the use. This
// can happen via buffer aliasing (eg, tuples).
UsesList GetUsers(const InstructionList& instruction_list,
                  const LogicalBuffer* logical_buffer,
                  const TuplePointsToAnalysis& points_to_analysis,
                  bool* has_indirect_users) {
  UsesList users;
  // To identify uses iterate through all HloInstruction users of the
  // BufferAliases of the logical buffer.
  *has_indirect_users = false;
  for (const BufferAlias& buffer_alias :
       points_to_analysis.GetBufferAliases(*logical_buffer)) {
    for (const HloInstruction* user : buffer_alias.instruction()->users()) {
      if (points_to_analysis.DoesNotUseOperandBuffer(
              buffer_alias.instruction(), buffer_alias.index(), user)) {
        // The alias may be an operand of 'user', but the LogicalBuffer cannot
        // possibly be used by the instruction so ignore 'user'. This is the
        // case, for example, for the tuple element buffers in a GetTupleElement
        // instruction (the GTE instruction only uses the pointer vector).
        continue;
      }
      if (buffer_alias.instruction() != logical_buffer->instruction() &&
          !IsSupportedIndirectUser(buffer_alias.instruction())) {
        *has_indirect_users = true;
      }
      // A buffer may be used by the instruction via more than one alias. For
      // example, a buffer which appears in more than one element of a tuple.
      Item* user_item = instruction_list.GetItem(user);
      std::optional<int64_t> user_index =
          logical_buffer->index().size() != 1
              ? std::nullopt
              : std::make_optional(logical_buffer->index().back());
      for (int64_t op_idx : user->OperandIndices(buffer_alias.instruction())) {
        if (!absl::c_linear_search(
                users,
                ItemUse{user_item, static_cast<int>(op_idx), user_index})) {
          users.push_back(
              ItemUse{user_item, static_cast<int>(op_idx), user_index});
        }
      }
    }
  }
  return users;
}

// Class for tracking memory usage of a computation as the instructions are
// placed sequentially. Memory usage is the sum of the sizes of live values
// (LogicalBuffers) at the current point in the instruction sequence.
class MemoryUsageTracker {
 public:
  MemoryUsageTracker(const HloRematerialization::Options& options,
                     const HloComputation* computation,
                     const TuplePointsToAnalysis& points_to_analysis,
                     const InstructionList& instruction_list);

  // Starts the placement of the given instruction. This adds the sizes of the
  // LogicalBuffers defined by the instruction to the current memory
  // usage. Placement is broken into two steps (BeginInstruction and
  // EndInstruction) to accurately model memory usage. At BeginInstruction the
  // memory for the output value(s) of the current instruction is allocated. At
  // EndInstruction memory for dead operand(s) is freed.
  absl::Status BeginInstruction(Item* item);

  int64_t RematerializationCost(const std::vector<Item*>& items,
                                int64_t memory_reduced,
                                int64_t memory_limit_bytes) const {
    // If none of the users of any 'item' have been placed in the
    // sequence (as tracked by memory_tracker), then rematerialization of
    // 'item' is a zero-cost move of 'item->instruction' in the sequence.
    bool zero_cost_move = true;
    for (auto* item : items) {
      auto* instruction = item->instruction;
      if (absl::c_any_of(
              instruction->users(),
              [this](const HloInstruction* inst) { return IsPlaced(inst); })) {
        zero_cost_move = false;
        break;
      }
    }
    if (zero_cost_move) {
      return 0;
    }

    CHECK_GT(memory_reduced, 0);
    // Return the inverse of the benefit of rematerialization.
    return memory_limit_bytes / memory_reduced;
  }

  // Finishes the placement of the current instruction. This frees any dead
  // operands or dead result of the instruction. This must be called after
  // each call to BeginInstruction.
  absl::Status EndInstruction();

  // Returns the number of bytes that the current memory usage will be reduced
  // if the given instruction is compact.
  int64_t MemoryReducedIfCompressed(const Item* item,
                                    const Shape& compact_shape) const;

  // Returns the number of bytes that the current memory usage will be reduced
  // by if the given sequence of instructions is rematerialized.
  int64_t MemoryReducedIfRematerialized(
      absl::Span<const Item* const> items) const;

  absl::Status AddCompressInstructions(Item* original_item,
                                       Item* compressed_item,
                                       Item* uncompressed_item);

  // Adjusts memory usage to account for the rematerialization of
  // original_item for all remaining unplaced uses. The rematerialization
  // is remat_item. This method should be called after the HLO graph has
  // been transformed (rematerialization instruction created and connected
  // to uses).
  absl::Status AddRematerializedInstruction(Item* original_item,
                                            Item* remat_item,
                                            absl::Span<Item*> indirect_users);

  // Given a list of uses return two lists where one is the ones which are
  // placed and the other is ones which are not yet placed.
  std::tuple<UsesList, UsesList> GetPlacedAndUnplacedUsers(
      const UsesList& uses) const;

 public:
  // Given the newly created instructions for host memory offload, create new
  // buffers, link their uses to their users, and update the current memory
  // usage.
  absl::Status AddHostOffloadCopyInstructions(Item* original_item,
                                              Item* copy_start_to_host_item,
                                              Item* copy_done_to_host_item,
                                              Item* copy_start_to_device_item,
                                              Item* copy_done_to_device_item);

  // Counts the bytes that this item occupies by summing up the buffers defined
  // by this item. If only_count_unplaced_users is true, only count users of
  // buffers which are not yet placed. This will represent the current memory
  // usage of the item. Otherwise, count all buffers. This will represent the
  // peak memory usage of the item.
  int64_t BytesUsedByBuffers(const Item* item,
                             bool only_count_unplaced_users) const;

  // Calculates the cost of compressing the candidate_item's output.
  std::optional<int64_t> GetCostOfCompression(const Item* candidate_item,
                                              int64_t memory_limit_bytes,
                                              int64_t peak_memory_bytes);

  // Calculates the cost of offloading the candidate_item's output to host
  // memory.
  std::optional<int64_t> GetCostOfHostOffload(const Item* candidate_item,
                                              int64_t memory_limit_bytes) const;

  // Calculates the cost of recomputing the candidate_item's output.
  std::optional<int64_t> GetCostOfRecompute(
      const std::vector<Item*>& candidate_items,
      int64_t memory_limit_bytes) const;

  // Selects and returns the best candidate instructions for rematerialization.
  // A sequence of candidate instructions of length between min_block_size and
  // max_block_size (both inclusive) with the lowest rematerialization cost is
  // selected among those candidates which reduce memory use at the program
  // point of the current instruction as indicated by memory_tracker. Returns an
  // empty vector if no candidates are found. Also returns an integer that
  // represents the amount of "effort" expended to find the candidate
  // instructions.
  std::tuple<std::vector<Item*>, RematStrategy, int>
  PickRematerializationCandidates(
      const InstructionList& instruction_list, int64_t memory_limit_bytes,
      absl::flat_hash_map<const HloInstruction*, bool>* rematerializable_map,
      int min_block_size, int max_block_size, int64_t peak_memory_bytes);

  // Returns whether the given instruction has been placed (BeginInstruction
  // has been called with 'instruction' as the argument).
  bool IsPlaced(const HloInstruction* instruction) const {
    return instruction_list_.GetItem(instruction)->placed;
  }

  // Returns whether 'item' has any unplaced users.
  bool HasUnplacedUsers(Item* item) const;

  // Returns the list of uses for a specific 'item'.
  UsesList GetItemUses(Item* item) const;

  // Returns whether 'item' is currently in progress.
  bool IsInProgressItem(Item* item) const { return item == in_progress_item_; }

  // Returns the current memory usage. This is the sum of sizes of all live
  // values.
  int64_t memory_usage() const { return memory_usage_; }

  //
  int64_t AllocatedSize(Item* item) const {
    int64_t size = 0;
    for (auto buffer_id : item->buffers_defined) {
      size += AllocatedSize(buffer_id);
    }
    return size;
  }

  const HloComputation* computation() const { return computation_; }

  const HloRematerialization::Options& options() const { return options_; }

  // Check invariants of the data structure. This is expensive to call.
  bool Check() const;

  std::string ToString() const;

 private:
  // A Buffer represents a single LogicalBuffer in the computation including
  // various metadata useful for tracking liveness of the value. A LogicalBuffer
  // is not used directly because the HLO graph is transformed and
  // TuplePointsToAnalysis which owns all LogicalBuffers cannot be updated after
  // HLO graph transformations.
  struct Buffer {
    // The unique id of this Buffer. This value is equal to the buffer's index
    // in the vector buffers_.
    const BufferId id;

    // The instruction which defines this buffer.
    Item* defining_instruction;

    // The materialized size of the buffer in bytes.
    const int64_t size;

    // Shape of the buffer.
    Shape shape;

    // Whether this buffer is live-out of the computation.
    bool live_out;

    // Whether this buffer has indirect uses. Ie, an instruction which is not a
    // user of defining_instruction uses this buffer. This can occur due to
    // buffer aliasing (eg, tuples).
    bool has_indirect_uses;

    // Position in the tuple this buffer definition lives in.
    ShapeIndex index;

    // The instructions which use this buffer.
    UsesList users;

    // The number of users (HloInstructions) of this buffer which have not yet
    // been placed in the sequence.
    int64_t unfinished_user_count;

    std::string ToString() const {
      return absl::StrCat("Buffer ", id, " (defined by ",
                          defining_instruction->instruction->name(), ", size ",
                          size, " bytes)");
    }
  };

  // Adjust our tracked memory usage as a result of this new item coming into
  // scope.
  void CountAllocatedMemory(Item* item);

  // Adjust our tracked memory usage as a result of this item going out of
  // scope.
  absl::Status CountFreedMemory(Item* item);

  // Buffers have users and users have buffers used, this function resolves
  // outstanding issues in that bidirectional dependency.
  void ReplaceUsesInUsersOfBuffer(Buffer& buffer, BufferId old_id) const;

  // Get the compact shape of given hlo instruction. An internal cache is used
  // to avoid computing the shape multiple times.
  absl::StatusOr<Shape> GetCompactShape(const HloInstruction* hlo);

  // Creates a Buffer representing the given logical buffer. The buffer is added
  // to buffers_ and a reference is returned.
  Buffer& CreateBufferFromLogicalBuffer(
      const LogicalBuffer* logical_buffer,
      const TuplePointsToAnalysis& points_to_analysis, bool live_out) {
    bool has_indirect_uses = false;
    UsesList users = GetUsers(instruction_list_, logical_buffer,
                              points_to_analysis, &has_indirect_uses);
    return NewBuffer(instruction_list_.GetItem(logical_buffer->instruction()),
                     logical_buffer->shape(), logical_buffer->index(),
                     std::move(users), live_out, has_indirect_uses);
  }

  // Create a new buffer representing a rematerialization of given buffer for
  // the given uses.
  Buffer& RematerializeBuffer(const Buffer& original_buffer, Item* remat_item,
                              UsesList&& rematerialized_uses) {
    CHECK(original_buffer.defining_instruction->placed)
        << original_buffer.defining_instruction->instruction->name();
    CHECK(!original_buffer.has_indirect_uses) << original_buffer.ToString();
    CHECK(!original_buffer.live_out) << original_buffer.ToString();
    for (ItemUse& use : rematerialized_uses) {
      CHECK(!use.user->placed) << use.user->instruction->name();
    }
    return NewBuffer(remat_item, original_buffer.shape, original_buffer.index,
                     std::move(rematerialized_uses), /*live_out=*/false,
                     /*has_indirect_uses=*/false);
  }

  // Return number of bytes allocated for the buffer with the given id. Buffers
  // allocated by the calling computation (eg, parameter and output buffers) are
  // considered to have zero bytes because the memory is accounted for in a
  // different computation.
  int64_t AllocatedSize(BufferId buffer_id) const {
    const Buffer& buffer = buffers_.at(buffer_id);
    HloInstruction* inst = buffer.defining_instruction->instruction;
    HloOpcode def_opcode = inst->opcode();
    if (buffer.live_out || def_opcode == HloOpcode::kParameter) {
      return 0;
    } else {
      if (options_.host_memory_offload_config && buffer.shape.has_layout() &&
          buffer.shape.layout().memory_space() ==
              options_.host_memory_offload_config->host_memory_space) {
        // Host memory counts for nothing.
        return 0;
      }
      return buffer.size;
    }
  }

  // Returns true if BeginInstruction and EndInstruction has been called for the
  // given instruction.
  bool IsFinished(Item* item) const {
    return item->placed && item != in_progress_item_;
  }

  // Returns whether the given buffer is being used by the in-progress
  // instruction.
  bool IsInUse(BufferId buffer_id) const {
    if (in_progress_item_ == nullptr) {
      return false;
    }
    const BufferIdList& in_progress_uses = in_progress_item_->buffers_used;
    return absl::c_linear_search(in_progress_uses, buffer_id);
  }

  bool IsCurrentlyLive(BufferId buffer_id) const {
    const Buffer& buffer = buffers_[buffer_id];
    return (buffer.defining_instruction->placed &&
            buffer.unfinished_user_count > 0);
  }

  // Returns whether the given instruction is live at the current program
  // point.
  bool IsInstructionCurrentlyLive(const Item* instruction) const {
    // If the instruction has not started yet, it is not alive.
    if (!IsPlaced(instruction->instruction)) {
      return false;
    }
    for (const HloInstruction* user : instruction->instruction->users()) {
      if (!IsPlaced(user)) {
        // If there is an unplaced user, consider this instruction currently
        // live.
        return true;
      }
    }
    return false;
  }

  // Create a new buffer, add it to buffers_, and return a reference.
  Buffer& NewBuffer(Item* defining_instruction, const Shape& shape,
                    const ShapeIndex& index, UsesList&& uses, bool live_out,
                    bool has_indirect_uses) {
    int buffer_id = buffers_.size();
    auto get_num_of_unique_users = [](const UsesList& uses) -> int64_t {
      absl::flat_hash_set<Item*> users_set;
      for (const ItemUse& use : uses) {
        users_set.insert(use.user);
      }
      return users_set.size();
    };
    buffers_.push_back(Buffer{buffer_id, defining_instruction,
                              options_.hlo_cost_analysis.GetShapeSize(shape),
                              shape, live_out, has_indirect_uses, index, uses,
                              get_num_of_unique_users(uses)});
    return buffers_.back();
  }

  const HloRematerialization::Options& options_;

  const HloComputation* computation_;

  // Instruction list containing the ordering of instructions in
  // computation_. This is the order in which instructions are placed
  // (BeginInstruction/EndInstruction calls).
  const InstructionList& instruction_list_;

  // A map that caches existing known compact shape for each instruction.
  absl::flat_hash_map<const HloInstruction*, Shape> compact_shape_;

  // Memory usage at the currently placed instruction.
  int64_t memory_usage_ = 0;

  // The instruction currently being placed. This value is non-null only
  // between the calling of BeginInstruction and EndInstruction.
  Item* in_progress_item_ = nullptr;

  // All buffers in the computation.
  std::vector<Buffer> buffers_;
};

MemoryUsageTracker::MemoryUsageTracker(
    const HloRematerialization::Options& options,
    const HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const InstructionList& instruction_list)
    : options_(options),
      computation_(computation),
      instruction_list_(instruction_list) {
  PointsToSet::BufferSet live_out_set =
      points_to_analysis.GetPointsToSet(computation_->root_instruction())
          .CreateFlattenedSet();
  absl::flat_hash_map<const LogicalBuffer*, BufferId>
      logical_buffer_to_buffer_id;
  for (auto* item = instruction_list_.first(); item != nullptr;
       item = instruction_list_.next(item)) {
    const HloInstruction* const instruction = item->instruction;
    for (const LogicalBuffer* logical_buffer :
         points_to_analysis.GetBuffersDefinedByInstruction(instruction)) {
      Buffer* buffer;
      if (instruction->opcode() == HloOpcode::kWhile) {
        // The while instruction defines no new buffers. Instead it reuses the
        // buffers of its operand. Find the Buffer of its operand at the
        // proper ShapeIndex.
        const PointsToSet& operand_points_to =
            points_to_analysis.GetPointsToSet(instruction->operand(0));
        CHECK_EQ(operand_points_to.element(logical_buffer->index()).size(), 1);
        const LogicalBuffer* source_logical_buffer =
            operand_points_to.element(logical_buffer->index())[0];
        buffer =
            &buffers_.at(logical_buffer_to_buffer_id.at(source_logical_buffer));

        // Mark buffer as has indirect use and live out.
        buffer->has_indirect_uses = true;
        buffer->live_out =
            buffer->live_out || ContainsKey(live_out_set, logical_buffer);

        // Add users of while to Buffer users.
        bool unused;
        for (ItemUse& user_item : GetUsers(instruction_list_, logical_buffer,
                                           points_to_analysis, &unused)) {
          auto existing_user_it = absl::c_find_if(
              buffer->users,
              [&](const ItemUse& use) { return user_item.user == use.user; });
          if (existing_user_it == buffer->users.end()) {
            buffer->unfinished_user_count++;
            user_item.user->buffers_used.push_back(buffer->id);
            buffer->users.push_back(user_item);
          }
        }
      } else {
        buffer = &CreateBufferFromLogicalBuffer(
            logical_buffer, points_to_analysis,
            ContainsKey(live_out_set, logical_buffer));
        item->buffers_defined.push_back(buffer->id);
        for (ItemUse& user : buffer->users) {
          if (!absl::c_linear_search(user.user->buffers_used, buffer->id)) {
            user.user->buffers_used.push_back(buffer->id);
          }
        }
      }

      logical_buffer_to_buffer_id[logical_buffer] = buffer->id;
    }

    // Trace the output of each instruction. This is so that we can properly
    // track which outputs does GTEs have.
    for (const LogicalBuffer* logical_buffer :
         points_to_analysis.GetPointsToSet(instruction).CreateFlattenedSet()) {
      item->buffers_output.push_back(
          logical_buffer_to_buffer_id[logical_buffer]);
    }
  }
  XLA_VLOG_LINES(10, ToString());
  DCHECK(Check());
}

void MemoryUsageTracker::CountAllocatedMemory(Item* item) {
  // All buffers defined by this instruction need memory.
  for (BufferId buffer_id : item->buffers_defined) {
    VLOG(3) << "  Buffer " << buffers_.at(buffer_id).ToString()
            << " is now live.";
    memory_usage_ += AllocatedSize(buffer_id);
  }
}

absl::Status MemoryUsageTracker::CountFreedMemory(Item* item) {
  for (BufferId buffer_id : item->buffers_used) {
    Buffer& buffer = buffers_.at(buffer_id);
    buffer.unfinished_user_count--;
    TF_RET_CHECK(buffer.unfinished_user_count >= 0)
        << buffer.ToString() << " has negative unfinished user count.";
    if (buffer.unfinished_user_count == 0) {
      // Buffer is now dead.
      VLOG(3) << "  " << buffer.ToString() << " is now dead.";
      memory_usage_ -= AllocatedSize(buffer_id);
      // The memory usage can become negative inside the computation as we can
      // free up the parameter space and reuse it for other tensors.
    }
  }

  // If any buffer defined by this instruction has no uses, then memory can be
  // reclaimed immediately.
  for (BufferId buffer_id : item->buffers_defined) {
    const Buffer& buffer = buffers_.at(buffer_id);
    if (buffer.unfinished_user_count == 0) {
      VLOG(3) << "  " << buffer.ToString() << " is immediately dead.";
      memory_usage_ -= AllocatedSize(buffer_id);
      // The memory usage can become negative inside the computation as we can
      // free up the parameter space and reuse it for other tensors.
    }
  }
  return absl::OkStatus();
}

absl::Status MemoryUsageTracker::BeginInstruction(Item* item) {
  const HloInstruction* instruction = item->instruction;
  VLOG(3) << "BeginInstruction " << instruction->name();
  TF_RET_CHECK(in_progress_item_ == nullptr);
  in_progress_item_ = item;

  item->placed = true;

  CountAllocatedMemory(item);

  // TODO(b/37686934): Elementwise instructions can share the buffer of a (dead)
  // operand. Account for this potential reuse here.

  VLOG(3) << "  memory usage = " << memory_usage_;
  VLOG(10) << ToString();

  if (VLOG_IS_ON(1)) {
    DCHECK(Check());
  }
  return absl::OkStatus();
}

absl::Status MemoryUsageTracker::EndInstruction() {
  TF_RET_CHECK(in_progress_item_ != nullptr);
  VLOG(3) << "EndInstruction " << in_progress_item_->instruction->name();

  TF_RETURN_IF_ERROR(CountFreedMemory(in_progress_item_));

  in_progress_item_ = nullptr;

  VLOG(3) << "  memory usage = " << memory_usage_;
  VLOG(10) << ToString();

  if (VLOG_IS_ON(1)) {
    DCHECK(Check());
  }
  return absl::OkStatus();
}

int64_t MemoryUsageTracker::MemoryReducedIfCompressed(
    const Item* item, const Shape& compact_shape) const {
  CHECK_NE(in_progress_item_, nullptr);
  if (!item->placed || item == in_progress_item_) {
    return 0;
  }

  int64_t memory_reduced = 0;

  // We only compress a single piece of an output at one time.
  CHECK_EQ(item->buffers_output.size(), 1);
  BufferId buffer_id = item->buffers_output[0];
  if (IsCurrentlyLive(buffer_id) && !IsInUse(buffer_id) &&
      IsInstructionCurrentlyLive(item)) {
    const Buffer& buffer = buffers_.at(buffer_id);
    memory_reduced += buffer.size;

    int64_t compact_shape_size =
        options_.hlo_cost_analysis.GetShapeSize(compact_shape);
    // Account for buffers that are compressed after instruction.
    memory_reduced -= compact_shape_size;
  }
  return memory_reduced;
}

int64_t MemoryUsageTracker::MemoryReducedIfRematerialized(
    absl::Span<const Item* const> items) const {
  CHECK_NE(in_progress_item_, nullptr);
  int64_t memory_reduced = 0;
  absl::flat_hash_set<const Item*> remat_candidates;

  for (const Item* item : items) {
    if (!item->placed || item == in_progress_item_) {
      LOG(WARNING) << "Unplaced item or in progress item being checked for "
                      "rematerialization.";
      return 0;
    }

    // Compute the amount of memory reduced (if any) by rematerializing
    // 'item->instruction'. The LogicalBuffers defined by 'item->instruction'
    // will no longer be live at this program point, so initially set
    // memory_reduced to the size of its defined values.
    for (BufferId buffer_id : item->buffers_defined) {
      const Buffer& buffer = buffers_.at(buffer_id);
      // Avoid rematerializing instructions with indirect uses as it is
      // difficult to reason about liveness after rematerializing the
      // instruction.
      // Avoid rematerializing instructions with live out buffers.
      // Avoid rematerializing buffers that are in nested tuples.
      // TODO(mpurohit): Check why live_out buffers are an issue here.
      if (buffer.has_indirect_uses || buffer.live_out ||
          buffer.index.size() > 1) {
        return 0;
      }
      if (IsInUse(buffer_id)) {
        return 0;
      }
      if (IsCurrentlyLive(buffer_id)) {
        memory_reduced += AllocatedSize(buffer_id);
      }
    }

    // Account for any logical buffers whose live range must be extended across
    // this program point.
    for (BufferId buffer_id : item->buffers_used) {
      if (!IsCurrentlyLive(buffer_id)) {
        // This logical buffer is used by 'item->instruction' but is not live at
        // this program point. Rematerializing 'item->instruction' will extend
        // the buffer's live range across this program point unless it is
        // defined by an instruction that is also being rematerialized.
        Item* defining_instruction =
            buffers_.at(buffer_id).defining_instruction;
        if (!remat_candidates.contains(defining_instruction)) {
          memory_reduced -= AllocatedSize(buffer_id);
        }
      }
    }
    remat_candidates.insert(item);
  }

  return memory_reduced;
}

std::tuple<UsesList, UsesList> MemoryUsageTracker::GetPlacedAndUnplacedUsers(
    const UsesList& uses) const {
  UsesList placed_users, unplaced_users;
  for (const ItemUse& use : uses) {
    if (use.user->placed) {
      DCHECK(IsFinished(use.user)) << use.user->instruction->name();
      placed_users.push_back(use);
    } else {
      unplaced_users.push_back(use);
    }
  }
  return {placed_users, unplaced_users};
}

void MemoryUsageTracker::ReplaceUsesInUsersOfBuffer(Buffer& buffer,
                                                    BufferId old_id) const {
  // Loop over the users of this buffer. For each of those users look at their
  // buffers used. If that buffer Id matches the passed in old_id, then replace
  // it with the Id of this current buffer.
  for (ItemUse& use : buffer.users) {
    BufferIdList& buffers_used = use.user->buffers_used;
    absl::c_replace(buffers_used, old_id, buffer.id);
  }
}

absl::Status MemoryUsageTracker::AddCompressInstructions(
    Item* original_item, Item* compressed_item, Item* uncompressed_item) {
  CHECK(original_item->placed)
      << "Compressing instruction, but the original is not yet placed.";
  CHECK_EQ(original_item->buffers_output.size(), 1)
      << "Only compressing items which have a single output buffer";

  // Update the memory usage by replacing the old instruction with the new one.
  // Original buffer is now dead.
  memory_usage_ -= options_.hlo_cost_analysis.GetShapeSize(
      original_item->instruction->shape());
  // Compressed buffer is now alive.
  memory_usage_ += options_.hlo_cost_analysis.GetShapeSize(
      compressed_item->instruction->shape());

  // Update the original item's only output buffer.
  BufferId original_buffer_id = original_item->buffers_output[0];
  Buffer& original_buffer = buffers_.at(original_buffer_id);
  auto [placed_users, unplaced_users] =
      GetPlacedAndUnplacedUsers(original_buffer.users);
  // Update the list of users to only be placed_users.
  original_buffer.users = std::move(placed_users);
  // Update to reflect that all users are finished, since any user after this
  // point will be using the uncompressed version.
  original_buffer.unfinished_user_count = 0;
  // Add the new compression instruction as a user of the original instruction.
  original_buffer.users.push_back(ItemUse{compressed_item, 0, std::nullopt});

  // We are reallocating the vector containing the buffers potentially,
  // invalidating the original_buffer reference, so copy the index that we need
  // across NewBuffer calls.
  ShapeIndex copied_index = original_buffer.index;

  // Create a new buffer which is the one that the new compress instruction will
  // define.
  Buffer& compressed_buffer =
      NewBuffer(compressed_item, compressed_item->instruction->shape(),
                copied_index, {ItemUse{uncompressed_item, 0, std::nullopt}},
                /*live_out=*/false,
                /*has_indirect_uses=*/false);
  // Update the compress item to only use the output buffer of the original
  // item.
  compressed_item->buffers_used = original_item->buffers_output;
  // Update the compress item to define & output this newly created buffer.
  compressed_item->buffers_output = {compressed_buffer.id};
  compressed_item->buffers_defined.push_back(compressed_buffer.id);

  // Create a new buffer which is the one that the new uncompress instruction
  // will define.
  Buffer& uncompressed_buffer =
      NewBuffer(uncompressed_item, uncompressed_item->instruction->shape(),
                copied_index, std::move(unplaced_users), /*live_out=*/false,
                /*has_indirect_uses=*/false);
  // Update the uncompressed item to only use the output buffer of the compress
  // item.
  uncompressed_item->buffers_used = {compressed_item->buffers_output[0]};
  // Update the uncompressed item to define & output this newly created buffer.
  uncompressed_item->buffers_output = {uncompressed_buffer.id};
  uncompressed_item->buffers_defined = {uncompressed_buffer.id};

  // uncompressed_buffer inherited its users as the unplaced users of the
  // original instruction. In each of these uses, replace the use of the
  // original buffer with the newly created final buffer.
  ReplaceUsesInUsersOfBuffer(uncompressed_buffer, original_buffer_id);

  return absl::OkStatus();
}

absl::Status MemoryUsageTracker::AddRematerializedInstruction(
    Item* original_item, Item* remat_item, absl::Span<Item*> indirect_users) {
  VLOG(3) << "AddRematerializedInstruction: original_instruction = "
          << original_item->instruction->name()
          << ", remat_instruction = " << remat_item->instruction->name();

  TF_RET_CHECK(in_progress_item_ != nullptr);
  TF_RET_CHECK(original_item->placed) << original_item->instruction->name();
  TF_RET_CHECK(!remat_item->placed) << remat_item->instruction->name();

  // Construct the list of buffers used and defined by the rematerialization.
  remat_item->buffers_used = original_item->buffers_used;

  // Account for the additional buffer uses created by the new rematerialization
  // instruction. Update memory usage if the rematerialization makes a dead
  // buffer live again.
  for (BufferId buffer_id : original_item->buffers_used) {
    Buffer& buffer = buffers_.at(buffer_id);
    if (buffer.unfinished_user_count == 0) {
      // Buffer used by this instruction was dead, now is alive.
      memory_usage_ += AllocatedSize(buffer.id);
    }
    buffer.unfinished_user_count++;
    absl::InlinedVector<ItemUse, 2> filtered_users;
    std::copy_if(buffer.users.begin(), buffer.users.end(),
                 std::back_inserter(filtered_users),
                 [&](const ItemUse& iu) { return iu.user == original_item; });
    for (ItemUse& u : filtered_users) {
      buffer.users.push_back(ItemUse{remat_item, u.operand_number, u.index});
    }
  }

  const absl::flat_hash_set<Item*> indirect_users_set(indirect_users.begin(),
                                                      indirect_users.end());
  // Create a new set of Buffers defined by the new rematerialization
  // instruction. Update the internal data structures and memory use to account
  // for them.
  for (BufferId old_buffer_id : original_item->buffers_defined) {
    Buffer& old_buffer = buffers_.at(old_buffer_id);

    UsesList placed_users;
    UsesList unplaced_users;
    for (ItemUse& user : old_buffer.users) {
      if (user.user->placed) {
        placed_users.push_back(user);
      } else {
        // We keep only the indirect users that are in the provided list.
        // We consider all the other dead and remove any buffer use they might
        // perform and remove it from the buffer user list.
        if (!IsSupportedIndirectUser(user.user->instruction) ||
            indirect_users_set.contains(user.user)) {
          unplaced_users.push_back(user);
        } else {
          CHECK(user.user->buffers_defined.empty())
              << "Buffers defined expected to be empty for use passthrough "
                 "instructions";
          user.user->buffers_output.clear();
          user.user->buffers_used.clear();
        }
      }
    }
    old_buffer.users = std::move(placed_users);
    old_buffer.unfinished_user_count = 0;

    // Buffer is now dead.
    memory_usage_ -= AllocatedSize(old_buffer.id);

    Buffer& new_buffer =
        RematerializeBuffer(old_buffer, remat_item, std::move(unplaced_users));

    remat_item->buffers_defined.push_back(new_buffer.id);
    remat_item->buffers_output.push_back(new_buffer.id);
    auto update_buffers = [old_buffer_id, new_buffer_id = new_buffer.id](
                              BufferIdList& to_update) {
      std::replace(to_update.begin(), to_update.end(), old_buffer_id,
                   new_buffer_id);
    };
    // Update users with the id of the new buffer.
    for (ItemUse& user : new_buffer.users) {
      update_buffers(user.user->buffers_used);
      update_buffers(user.user->buffers_output);
    }
  }

  // Update the indirect users with the id of the new buffers.
  for (Item* indirect_user : indirect_users) {
    // Source of the buffers that are gonna be passthrough.
    const Item* source_item =
        instruction_list_.GetItem(indirect_user->instruction->operand(0));
    switch (indirect_user->instruction->opcode()) {
      case HloOpcode::kBitcast: {
        // If the source is another indirect user then copy the output
        // in the used and output lists of the bitcast as they don't define any
        // buffer.
        if (IsSupportedIndirectUser(source_item->instruction)) {
          indirect_user->buffers_used = source_item->buffers_output;
          indirect_user->buffers_output = source_item->buffers_output;
        } else {
          // If it's a real instruction producing a buffer then copy the defined
          // buffers into used and output.
          indirect_user->buffers_used = source_item->buffers_defined;
          indirect_user->buffers_output = source_item->buffers_defined;
        }
        break;
      }
      case HloOpcode::kGetTupleElement: {
        // GTEs just use the tuple buffer and output the buffer they actually
        // extract from the tuple.
        const HloGetTupleElementInstruction* gte =
            Cast<HloGetTupleElementInstruction>(indirect_user->instruction);
        for (BufferId buffer_id : source_item->buffers_defined) {
          const Buffer& def_buffer = buffers_.at(buffer_id);
          if (def_buffer.index == ShapeIndex{gte->tuple_index()}) {
            indirect_user->buffers_output.push_back(buffer_id);
          }
          // This is the tuple buffer.
          if (def_buffer.index.empty()) {
            indirect_user->buffers_used.push_back(buffer_id);
          }
        }
        break;
      }
      default: {
        LOG(FATAL) << "Unsupported indirect instruction with opcode "
                   << indirect_user->instruction->opcode();
        break;
      }
    }
    // Fixup buffer users for the indirect instructions. For GTEs is only the
    // tuple buffer, while for bitcast is the buffer they pass through.
    for (BufferId buffer_id : indirect_user->buffers_used) {
      Buffer& buffer = buffers_.at(buffer_id);
      buffer.unfinished_user_count++;
      buffer.users.push_back(ItemUse{indirect_user, 0, std::nullopt});
    }
  }

  VLOG(3) << "  memory usage = " << memory_usage_;
  XLA_VLOG_LINES(10, ToString());

  DCHECK(Check());

  return absl::OkStatus();
}

absl::Status MemoryUsageTracker::AddHostOffloadCopyInstructions(
    Item* original_item, Item* copy_start_to_host_item,
    Item* copy_done_to_host_item, Item* copy_start_to_device_item,
    Item* copy_done_to_device_item) {
  CHECK_EQ(original_item->buffers_defined.size(), 1);

  // Split up the users of the original instruction into placed and unplaced.
  CHECK_EQ(original_item->buffers_output.size(), 1);
  BufferId original_buffer_id = original_item->buffers_output[0];
  Buffer& original_buffer = buffers_.at(original_buffer_id);
  auto [placed_users, unplaced_users] =
      GetPlacedAndUnplacedUsers(original_buffer.users);

  // Update the original item's buffer's users to be:
  //  1. The placed_users only.
  //  2. The newly created copy_start_to_host.
  original_buffer.users = std::move(placed_users);
  original_buffer.users.emplace_back(copy_start_to_host_item, 0, std::nullopt);
  // Set the only unfinished user as the newly created copy_to_host instruction.
  // We will later determine if that user is finished or not and update this
  // value if so.
  original_buffer.unfinished_user_count = 1;

  // Create new buffers for all of the newly created instructions.
  CHECK_EQ(copy_start_to_host_item->instruction->shape().tuple_shapes_size(), 3)
      << "copy_start_to_host_item's shape is "
      << copy_start_to_host_item->instruction->shape().ToString();
  CHECK_EQ(copy_start_to_device_item->instruction->shape().tuple_shapes_size(),
           3)
      << "copy_start_to_device_item's shape is "
      << copy_start_to_device_item->instruction->shape().ToString();

  // The first copy-start is a tuple of 3 elements: (host_buffer, device_buffer,
  // context). Since we're not tracking host memory, we'll only create buffers
  // for the other two.
  BufferId copy_start_to_host_device_buffer_id =
      NewBuffer(copy_start_to_host_item,
                copy_start_to_host_item->instruction->shape().tuple_shapes(1),
                ShapeIndex(),
                UsesList{ItemUse{copy_done_to_host_item, 0, std::nullopt}},
                /*live_out=*/false, /*has_indirect_uses=*/false)
          .id;
  BufferId copy_start_to_host_context_buffer_id =
      NewBuffer(copy_start_to_host_item,
                copy_start_to_host_item->instruction->shape().tuple_shapes(2),
                ShapeIndex(),
                UsesList{ItemUse{copy_done_to_host_item, 0, std::nullopt}},
                /*live_out=*/false, /*has_indirect_uses=*/false)
          .id;

  // The second copy-start is a tuple of 3 elements: (device_buffer,
  // host_buffer, context). Since we're not tracking host memory, we'll only
  // create buffers for the other two.
  BufferId copy_start_to_device_device_buffer_id =
      NewBuffer(copy_start_to_device_item,
                copy_start_to_device_item->instruction->shape().tuple_shapes(0),
                ShapeIndex(),
                UsesList{ItemUse{copy_done_to_device_item, 0, std::nullopt}},
                /*live_out=*/false, /*has_indirect_uses=*/false)
          .id;
  BufferId copy_start_to_device_context_buffer_id =
      NewBuffer(copy_start_to_device_item,
                copy_start_to_device_item->instruction->shape().tuple_shapes(2),
                ShapeIndex(),
                UsesList{ItemUse{copy_done_to_device_item, 0, std::nullopt}},
                /*live_out=*/false, /*has_indirect_uses=*/false)
          .id;

  // The final copy-done outputs the final device buffer that is the
  // rematerialized original buffer.
  BufferId copy_done_to_device_buffer_id =
      NewBuffer(copy_done_to_device_item,
                copy_done_to_device_item->instruction->shape(), ShapeIndex(),
                std::move(unplaced_users), /*live_out=*/false,
                /*has_indirect_uses=*/false)
          .id;

  // Update items of the newly created instructions to reference the newly
  // created buffers.
  copy_start_to_host_item->buffers_used = original_item->buffers_output;
  copy_start_to_host_item->buffers_output = {
      copy_start_to_host_device_buffer_id,
      copy_start_to_host_context_buffer_id};
  copy_start_to_host_item->buffers_defined = {
      copy_start_to_host_device_buffer_id,
      copy_start_to_host_context_buffer_id};

  copy_done_to_host_item->buffers_used =
      copy_start_to_host_item->buffers_output;
  // The only buffer that copy_done_to_host defines is a host buffer. Since
  // we're not tracking host memory, we're not going to bother with that buffer
  // for now.
  copy_done_to_host_item->buffers_output = {};
  copy_done_to_host_item->buffers_defined = {};

  copy_start_to_device_item->buffers_used =
      copy_done_to_host_item->buffers_output;
  copy_start_to_device_item->buffers_output = {
      copy_start_to_device_device_buffer_id,
      copy_start_to_device_context_buffer_id};
  copy_start_to_device_item->buffers_defined = {
      copy_start_to_device_device_buffer_id,
      copy_start_to_device_context_buffer_id};

  copy_done_to_device_item->buffers_used =
      copy_start_to_device_item->buffers_output;
  copy_done_to_device_item->buffers_output = {copy_done_to_device_buffer_id};
  copy_done_to_device_item->buffers_defined = {copy_done_to_device_buffer_id};

  Buffer& copy_done_to_device_buffer =
      buffers_.at(copy_done_to_device_buffer_id);
  ReplaceUsesInUsersOfBuffer(copy_done_to_device_buffer, original_buffer_id);

  // We know that the 4 newly created instructions are not in progress, so if
  // they're marked as placed, we can count the allocation and deallocation of
  // buffers. Calling these functions also does some user accounting. Since
  // these instructions have a strict order, if one isn't placed, the following
  // ones won't be either.
  if (copy_start_to_host_item->placed) {
    CountAllocatedMemory(copy_start_to_host_item);
    TF_RETURN_IF_ERROR(CountFreedMemory(copy_start_to_host_item));
    // This will account for the freed memory that is defined by the original
    // item.

    if (copy_done_to_host_item->placed) {
      CountAllocatedMemory(copy_done_to_host_item);
      TF_RETURN_IF_ERROR(CountFreedMemory(copy_done_to_host_item));

      if (copy_start_to_device_item->placed) {
        CountAllocatedMemory(copy_start_to_device_item);
        TF_RETURN_IF_ERROR(CountFreedMemory(copy_start_to_device_item));

        if (copy_done_to_device_item->placed) {
          CountAllocatedMemory(copy_done_to_device_item);
          TF_RETURN_IF_ERROR(CountFreedMemory(copy_done_to_device_item));
        }
      }
    }
  }

  return absl::OkStatus();
}

std::string MemoryUsageTracker::ToString() const {
  std::string output =
      absl::StrCat("MemoryUsageTracker for ", computation_->name(), "\n");
  absl::StrAppend(&output,
                  "Memory usage: ", HumanReadableNumBytes(memory_usage()), " (",
                  memory_usage(), " bytes)");
  for (auto* item = instruction_list_.first(); item != nullptr;
       item = instruction_list_.next(item)) {
    const HloInstruction* instruction = item->instruction;
    absl::string_view inprogress =
        item == in_progress_item_ ? " in-progress" : "";
    absl::string_view placed = item->placed ? " placed" : "";
    absl::StrAppend(&output, "  ", instruction->name(), inprogress, placed,
                    "\n    Defines:\n");
    for (BufferId buffer_id : item->buffers_defined) {
      const Buffer& buffer = buffers_[buffer_id];
      absl::string_view live = IsCurrentlyLive(buffer_id) ? " live" : "";
      absl::StrAppend(&output, "      ", buffer.ToString(), live, ", ",
                      buffer.unfinished_user_count, " unfinished uses\n");
    }
    absl::StrAppend(&output, "    Outputs:\n");
    for (BufferId buffer_id : item->buffers_output) {
      absl::StrAppend(&output, "      ", buffers_[buffer_id].ToString(), "\n");
    }
    absl::StrAppend(&output, "    Uses:\n");
    for (BufferId buffer_id : item->buffers_used) {
      absl::StrAppend(&output, "      ", buffers_[buffer_id].ToString(), "\n");
    }
  }
  return output;
}

absl::StatusOr<Shape> MemoryUsageTracker::GetCompactShape(
    const HloInstruction* hlo) {
  auto it = compact_shape_.find(hlo);
  if (it != compact_shape_.end()) {
    return it->second;
  }
  const Shape& original_shape = hlo->shape();
  TF_ASSIGN_OR_RETURN(Shape min_shape,
                      options_.compact_shape_function(original_shape));
  compact_shape_[hlo] = min_shape;
  return min_shape;
}

bool MemoryUsageTracker::Check() const {
  auto elements_are_unique = [](const BufferIdList& vec) {
    return vec.size() == std::set<BufferId>(vec.begin(), vec.end()).size();
  };

  // Verify buffers_defined per instruction.
  for (auto* instruction : computation_->instructions()) {
    const BufferIdList& defined_buffers =
        instruction_list_.GetItem(instruction)->buffers_defined;
    CHECK(elements_are_unique(defined_buffers))
        << "Instruction " << instruction->name()
        << " does not have unique defined buffers: "
        << absl::StrJoin(defined_buffers, ", ",
                         [this](std::string* out, BufferId buffer_id) {
                           absl::StrAppend(out,
                                           buffers_.at(buffer_id).ToString());
                         });

    for (const Buffer& buffer : buffers_) {
      if (buffer.defining_instruction->instruction == instruction) {
        CHECK(absl::c_linear_search(defined_buffers, buffer.id))
            << "Instruction " << instruction->name()
            << " defined buffers is missing: " << buffer.ToString();
      }
    }
  }

  // Verify buffers_used per instruction.
  for (auto* instruction : computation_->instructions()) {
    const BufferIdList& used_buffers =
        instruction_list_.GetItem(instruction)->buffers_used;
    CHECK(elements_are_unique(used_buffers))
        << "Instruction " << instruction->name()
        << " does not have unique used buffers: "
        << absl::StrJoin(used_buffers, ", ",
                         [this](std::string* out, BufferId buffer_id) {
                           absl::StrAppend(out,
                                           buffers_.at(buffer_id).ToString());
                         });
  }
  for (const Buffer& buffer : buffers_) {
    int64_t unfinished_uses = 0;
    absl::flat_hash_set<Item*> already_counted_user;
    for (const ItemUse& user : buffer.users) {
      const BufferIdList& used_buffers = user.user->buffers_used;
      CHECK(absl::c_linear_search(used_buffers, buffer.id))
          << "Instruction " << user.user->instruction->name()
          << " used buffers is missing " << buffer.ToString();
      if (!IsFinished(user.user) &&
          already_counted_user.insert(user.user).second) {
        unfinished_uses++;
      }
    }
    CHECK_EQ(buffer.unfinished_user_count, unfinished_uses)
        << "Incorrect unplaced use count for " << buffer.ToString();
  }
  return true;
}

// Returns a block of up to min_block_size consecutive candidate instructions
// from instruction_list starting from start_item. Returns fewer than
// min_block_size instructions if the block of unplaced instructions starting
// from start_item is smaller than min_block_size.
std::vector<Item*> GetInitialBlock(const InstructionList& instruction_list,
                                   const MemoryUsageTracker& tracker,
                                   Item* start_item, int min_block_size) {
  std::vector<Item*> item_block;
  Item* curr_item = start_item;
  for (int i = 0; i < min_block_size; ++i) {
    if (curr_item == nullptr || !curr_item->placed ||
        tracker.IsInProgressItem(curr_item)) {
      break;
    }
    item_block.push_back(curr_item);
    curr_item = instruction_list.next(curr_item);
  }
  return item_block;
}

// Returns whether any instruction in 'block' is denylisted or
// non-rematerializable.
bool AnyDenylistedOrNonRematerializable(
    const std::vector<Item*>& block,
    absl::flat_hash_map<const HloInstruction*, bool>* rematerializable_map) {
  for (auto* item : block) {
    if (item->denylisted) {
      return true;
    }
    if (!CanBeRematerialized(item->instruction, rematerializable_map)) {
      return true;
    }
  }
  return false;
}

int64_t MemoryUsageTracker::BytesUsedByBuffers(
    const Item* item, bool only_count_unplaced_users) const {
  int64_t bytes_used_by_buffers = 0;
  for (const auto& buffer_id : item->buffers_defined) {
    VLOG(3) << "  buffer " << buffer_id << "'s users are "
            << absl::StrJoin(buffers_.at(buffer_id).users, ", ",
                             [](std::string* str, const auto& use) {
                               str->append(use.user->instruction->name());
                             });
    for (const auto& use : buffers_.at(buffer_id).users) {
      if (!only_count_unplaced_users || !use.user->placed) {
        // Found a non-placed user
        bytes_used_by_buffers += AllocatedSize(buffer_id);
        // Don't count uses of this buffer multiple times.
        break;
      }
    }
  }
  return bytes_used_by_buffers;
}

std::optional<int64_t> MemoryUsageTracker::GetCostOfCompression(
    const Item* candidate_item, int64_t memory_limit_bytes,
    int64_t peak_memory_bytes) {
  CHECK(candidate_item != nullptr);

  // Only consider compressing single output instruction.
  if (candidate_item->buffers_output.size() != 1) {
    // TODO(b/291824123): Currently only handling single output buffers.
    HloInstruction* candidate_instruction = candidate_item->instruction;
    VLOG(2) << "  " << candidate_instruction->name()
            << " has more than one output buffer; cannot offload to host.";
    return {};
  }

  const Buffer& output_buffer = buffers_.at(candidate_item->buffers_output[0]);
  if (!candidate_item->placed || candidate_item == in_progress_item_ ||
      output_buffer.live_out) {
    return {};
  }

  const Shape& original_shape = candidate_item->instruction->shape();
  if (!original_shape.IsArray()) {
    return {};
  }

  Shape compact_shape = GetCompactShape(candidate_item->instruction).value();
  const int64_t memory_reduced =
      MemoryReducedIfCompressed(candidate_item, compact_shape);
  // Since the compressed and uncompressed buffers need to be alive
  // while performing the compression/uncompression, only perform
  // the compression if the sum of the two sizes is less than the
  // peak memory.
  const int64_t size = options_.hlo_cost_analysis.GetShapeSize(
      candidate_item->instruction->shape());
  const int64_t reduced_size =
      options_.hlo_cost_analysis.GetShapeSize(compact_shape);
  // TODO(victorstone): I don't think this size check is right.
  if (memory_reduced > 0 && size + reduced_size < peak_memory_bytes) {
    return memory_limit_bytes / memory_reduced;
  } else {
    return {};
  }
}

std::optional<int64_t> MemoryUsageTracker::GetCostOfHostOffload(
    const Item* candidate_item, int64_t memory_limit_bytes) const {
  CHECK(candidate_item != nullptr);
  HloInstruction* candidate_instruction = candidate_item->instruction;

  VLOG(2)
      << "Considering host offload as an option for remat. looking at instr "
      << candidate_instruction->name();

  if (candidate_item->buffers_output.size() != 1) {
    // TODO(b/291824123): Currently only handling single output buffers.
    VLOG(2) << "  " << candidate_instruction->name()
            << " has more than one output buffer; cannot offload to host.";
    return {};
  }

  // TODO(b/291823800): Bitcasts and tuples complicate things. Skip for now.
  for (auto buffer_id : candidate_item->buffers_defined) {
    for (auto use : buffers_.at(buffer_id).users) {
      if (use.user->instruction->opcode() == HloOpcode::kBitcast) {
        VLOG(3) << "  " << candidate_item->instruction->name()
                << " has a user which is a bitcast instruction("
                << use.user->instruction->name()
                << "); cannot offload "
                   "to host.";
        return {};
      } else if (use.user->instruction->opcode() == HloOpcode::kTuple) {
        VLOG(3) << "  " << candidate_item->instruction->name()
                << " has a user which is a tuple instruction("
                << use.user->instruction->name()
                << "); cannot offload "
                   "to host.";
        return {};
      }
    }
  }

  const Buffer& output_buffer = buffers_.at(candidate_item->buffers_output[0]);
  if (!candidate_item->placed || candidate_item == in_progress_item_ ||
      output_buffer.live_out) {
    VLOG(2) << "  " << candidate_instruction->name()
            << " is not yet placed, is in progress, or is \"live_out\"; cannot "
               "offload to host.";
    return {};
  }

  // If the current instruction uses this buffer, it doesn't make sense to
  // offload.
  const bool current_instruction_uses_this_item = [&]() {
    if (in_progress_item_ == nullptr) {
      // There is no current instruction.
      return false;
    }
    // Check if any of our output buffers' users are the current item.
    const auto& output_buffer_ids = candidate_item->buffers_output;
    for (const auto& output_buffer_id : output_buffer_ids) {
      const Buffer& output_buffer = buffers_.at(output_buffer_id);
      for (const auto& use : output_buffer.users) {
        if (use.user == in_progress_item_) {
          return true;
        }
      }
    }
    return false;
  }();
  if (current_instruction_uses_this_item) {
    VLOG(2) << "  " << candidate_instruction->name()
            << " is used by the current instruction in mem tracker ("
            << in_progress_item_->instruction->name()
            << "); cannot offload to host.";
    return {};
  }

  const int64_t bytes_used_by_buffers =
      BytesUsedByBuffers(candidate_item, /*only_count_unplaced_users=*/true);

  if (bytes_used_by_buffers == 0) {
    VLOG(2) << "  " << candidate_instruction->name()
            << " consumes no memory; no point in offloading.";
    return {};
  }

  // How much compute is between this candidate's last placed user and its first
  // non-placed user?
  const auto [placed_uses, unplaced_uses] =
      GetPlacedAndUnplacedUsers(output_buffer.users);
  const Item* last_placed_user = nullptr;
  const Item* first_unplaced_user = nullptr;
  for (const auto* item = instruction_list_.first(); item != nullptr;
       item = instruction_list_.next(item)) {
    if (absl::c_find_if(placed_uses, [&](const auto& use) {
          return use.user == item;
        }) != placed_uses.end()) {
      last_placed_user = item;
    }
    if (first_unplaced_user == nullptr &&
        absl::c_find_if(unplaced_uses, [&](const auto& use) {
          return use.user == item;
        }) != unplaced_uses.end()) {
      first_unplaced_user = item;
      break;
    }
  }

  if (last_placed_user == nullptr) {
    VLOG(3) << "  " << candidate_instruction->name()
            << " has no placed users, starting search at self.";
    last_placed_user = candidate_item;
  }
  CHECK(first_unplaced_user != nullptr)
      << "Didn't find any unplaced user for instruction \""
      << candidate_instruction->name()
      << "\". There must be a "
         "bug in how we calculate how much memory this item uses.";

  float time_spent_before_next_use = 0.0;
  for (auto* item = last_placed_user; item != first_unplaced_user;
       item = instruction_list_.next(item)) {
    time_spent_before_next_use += std::max(
        0.0f, options_.hlo_cost_analysis.optimal_seconds(*item->instruction));
  }

  if (time_spent_before_next_use <= 0.0) {
    // Instructions between take no time.
    return {};
  }

  const float time_spent_on_copies =
      bytes_used_by_buffers / options_.host_memory_offload_config
                                  ->bandwidth_to_host_bytes_per_second +
      bytes_used_by_buffers / options_.host_memory_offload_config
                                  ->bandwidth_from_host_bytes_per_second;
  if (time_spent_before_next_use < time_spent_on_copies) {
    // Host offload only considers cases where we can completely hide the copy
    // times. In this case, there is not enough compute time to hide offloading
    // and copying the data back in.
    return {};
  }
  VLOG(3) << "  " << candidate_instruction->name() << " has enough time ("
          << time_spent_before_next_use
          << ") between itself and next use. The memcpy out and back will take "
          << time_spent_on_copies << "s";
  // TODO(b/293323448): Properly calculate a cost; this cost metric is not
  // useful.
  return memory_limit_bytes / bytes_used_by_buffers;
}

std::optional<int64_t> MemoryUsageTracker::GetCostOfRecompute(
    const std::vector<Item*>& candidate_items,
    int64_t memory_limit_bytes) const {
  // If any of the candidate's control successor has been placed, we need
  // to skip this candidate. Otherwise we will violate control dependency.
  for (auto* item : candidate_items) {
    HloInstruction* candidate = item->instruction;
    if (std::any_of(
            candidate->control_successors().begin(),
            candidate->control_successors().end(),
            [this](const HloInstruction* inst) { return IsPlaced(inst); })) {
      return {};
    }
  }

  // Evaluate this block as a candidate for recompute rematerialization.
  VLOG(5) << "Block contains:";
  for (auto* hlo : candidate_items) {
    VLOG(5) << hlo->instruction->name();
  }
  const int64_t memory_reduced = MemoryReducedIfRematerialized(candidate_items);
  if (memory_reduced <= 0) {
    return {};
  }

  return RematerializationCost(candidate_items, memory_reduced,
                               memory_limit_bytes);
}

std::tuple<std::vector<Item*>, RematStrategy, int>
MemoryUsageTracker::PickRematerializationCandidates(
    const InstructionList& instruction_list, int64_t memory_limit_bytes,
    absl::flat_hash_map<const HloInstruction*, bool>* rematerializable_map,
    int min_block_size, int max_block_size, int64_t peak_memory_bytes) {
  // Keep track of the cost of each rematerialization option.
  // This cost is defined as:
  //
  // memory_limit_bytes / memory_reduced
  //
  // The idea is to choose the operation that will save the most memory for
  // rematerialization and do not worry about how much the compute costs since
  // running out of memory is more harmful than taking longer to get the answer.
  std::vector<Item*> best_items;
  int64_t best_cost = std::numeric_limits<int64_t>::max();
  RematStrategy best_strategy;
  int effort = 0;
  VLOG(5) << "Picking candidate block with size in [" << min_block_size << ", "
          << max_block_size << "]";

  for (auto* start_item = instruction_list.first_skip_node();
       start_item != nullptr;
       start_item = instruction_list.next_skip_node(start_item)) {
    std::vector<Item*> block =
        GetInitialBlock(instruction_list, *this, start_item, min_block_size);
    if (block.size() < min_block_size) {
      // There are no more blocks of size at least min_block_size with unplaced
      // instructions.
      break;
    }

    // If any item in the starting block are denylisted or non-rematable, then
    // break and move on to next start_item (we can actually move to the last
    // invalid item in this block, but let's ignore that optimization for now).
    if (AnyDenylistedOrNonRematerializable(block, rematerializable_map)) {
      continue;
    }

    // First, calculate the cost of compression rematerialziation for this
    // instruction.
    if (options_.remat_mode_config.compress && block.size() == 1) {
      auto cost =
          GetCostOfCompression(block[0], memory_limit_bytes, peak_memory_bytes);
      ++effort;
      if (cost && *cost < best_cost) {
        VLOG(1) << "Found new best cost; from " << best_cost << " to " << *cost
                << " with strategy kCompress on block of size " << block.size();
        best_strategy.kind = RematStrategy::kCompress;
        // TODO(b/293323448): This `best_strategy.compact_shape` is already
        // computed inside GetCostOfCompression, should we get it from there? Or
        // is it ok to recompute?
        best_strategy.compact_shape =
            GetCompactShape(block[0]->instruction).value();
        best_items = block;
        best_cost = *cost;
      }
    }

    // Second, calculate the cost of host offload rematerialization for this
    // instruction.
    if (options_.remat_mode_config.host_offload && block.size() == 1) {
      auto cost = GetCostOfHostOffload(block[0], memory_limit_bytes);
      ++effort;
      if (cost && *cost < best_cost) {
        VLOG(1) << "Found new best cost; from " << best_cost << " to " << *cost
                << " with strategy kHostOffload on block of size "
                << block.size();
        best_strategy.kind = RematStrategy::kHostOffload;
        best_items = block;
        best_cost = *cost;
      }
    }

    // Finally, calculate the cost of recompute rematerialization for this
    // instruction block. There is one difference between this rematerialization
    // strategy and the other two: recompute can rematerialize more than one
    // instruction at a time. Evaluate the cost of rematerializing the current
    // block, add the next instruction to the block, and then repeat until we
    // reach the configured max block size.
    if (!options_.remat_mode_config.recompute) {
      // Recompute is not enabled, nothing else to do for this block.
      continue;
    }

    while (block.size() <= max_block_size) {
      auto cost = GetCostOfRecompute(block, memory_limit_bytes);
      ++effort;
      if (cost && *cost < best_cost) {
        VLOG(1) << "Found new best cost; from " << best_cost << " to " << *cost
                << " with strategy kRecompute on block of size "
                << block.size();
        best_strategy.kind = RematStrategy::kRecompute;
        best_items = block;
        best_cost = *cost;
      }

      // Try to add the next instruction to this block to evaluate as a possibly
      // better candidate for rematerialization.
      auto* last_item = block[block.size() - 1];
      auto* next_item = instruction_list.next(last_item);
      if (next_item == nullptr || next_item->denylisted || !next_item->placed ||
          next_item == in_progress_item_ ||
          !CanBeRematerialized(next_item->instruction, rematerializable_map)) {
        break;
      }
      block.push_back(next_item);
    }
  }
  return {best_items, best_strategy, effort};
}

bool MemoryUsageTracker::HasUnplacedUsers(Item* item) const {
  for (BufferId buffer_id : item->buffers_defined) {
    const Buffer& buffer = buffers_.at(buffer_id);
    for (const ItemUse& user : buffer.users) {
      if (!user.user->placed) {
        return true;
      }
    }
  }
  return false;
}

UsesList MemoryUsageTracker::GetItemUses(Item* item) const {
  UsesList combined_users;
  for (BufferId buffer_id : item->buffers_defined) {
    const Buffer& buffer = buffers_.at(buffer_id);
    for (const ItemUse& user : buffer.users) {
      combined_users.push_back(user);
    }
  }
  return combined_users;
}

absl::StatusOr<int64_t> RematerializeInstructions(
    MemoryUsageTracker* memory_tracker, std::vector<Item*>* best_items,
    absl::flat_hash_set<const HloInstruction*>* remat_move_instructions,
    InstructionList* instruction_list, HloSchedule* schedule,
    HloRematerialization* rematerialization) {
  int64_t net_instructions_added = 0;
  std::vector<std::string> instruction_names(best_items->size());
  // Rematerialize the block of instructions in the reverse order to account for
  // dependencies between instructions in best_items.
  for (int i = best_items->size() - 1; i >= 0; --i) {
    Item* best_item = (*best_items)[i];
    HloInstruction* best = best_item->instruction;
    instruction_names[i] = best->name();
    HloComputation* computation = best->parent();

    // If the item to remat has no unplaced users, then skip the
    // rematerialization. Such an instruction can appear in best_items because
    // it is part of a good block, but does not itself add any benefit.
    if (!memory_tracker->HasUnplacedUsers(best_item)) {
      continue;
    }

    HloCloneContext context(computation->parent());
    HloInstruction* remat =
        computation->AddInstruction(best->Clone(/*suffix=*/"remat", &context));
    for (auto& cloned_computation_pair : context.cloned_computations()) {
      if (!schedule->is_computation_scheduled(cloned_computation_pair.first)) {
        continue;
      }
      HloInstructionSequence& sequence =
          schedule->GetOrCreateSequence(cloned_computation_pair.second);
      HloInstructionSequence& old_sequence =
          schedule->GetOrCreateSequence(cloned_computation_pair.first);
      for (HloInstruction* instr : old_sequence.instructions()) {
        sequence.push_back(instr);
      }
    }
    // Increment channel_id on channel instructions with a channel id.
    if (DynCast<HloChannelInstruction>(best) &&
        DynCast<HloChannelInstruction>(best)->channel_id()) {
      remat->set_channel_id(rematerialization->NextChannelId());
    }

    // Add control dependencies to the new operation.
    TF_RETURN_IF_ERROR(remat->CopyAllControlDepsFrom(best));

    Item* remat_item = instruction_list->CreateItem(remat);

    // Replace each remaining use of 'best' with the rematerialization.
    absl::InlinedVector<Item*, 4> indirect_users;
    absl::flat_hash_map<int64_t, HloInstruction*> gte_cache;
    for (auto& user : memory_tracker->GetItemUses(best_item)) {
      if (!memory_tracker->IsPlaced(user.user->instruction)) {
        VLOG(2) << "  Replacing use of " << best->name() << " in "
                << user.user->instruction->name() << " with " << remat->name();
        HloInstruction* remat_use = remat;
        HloInstruction* const user_operand =
            user.user->instruction->mutable_operand(user.operand_number);
        if (remat_use == user_operand) {
          continue;
        }
        // If the output of a multi-output fusion node is forwarded to one of
        // its users as is, all the element buffers are also treated as uses
        // by that user, which need to be skipped.
        if (user.index && remat_use->shape() != user_operand->shape()) {
          auto cached_gte = gte_cache.find(*user.index);
          if (cached_gte == gte_cache.end()) {
            remat_use = computation->AddInstruction(
                HloInstruction::CreateGetTupleElement(
                    ShapeUtil::GetTupleElementShape(remat_use->shape(),
                                                    *user.index),
                    remat_use, *user.index),
                /*new_name=*/"gte.remat");
            indirect_users.push_back(instruction_list->CreateItem(remat_use));
            gte_cache[*user.index] = remat_use;
          } else {
            remat_use = cached_gte->second;
          }
        }
        if (user_operand->shape() != remat_use->shape()) {
          remat_use = computation->AddInstruction(
              HloInstruction::CreateBitcast(user_operand->shape(), remat_use),
              /*new_name=*/"bitcast.remat");
          indirect_users.push_back(instruction_list->CreateItem(remat_use));
        }
        TF_RETURN_IF_ERROR(user.user->instruction->ReplaceOperandWith(
            user.operand_number, remat_use));
      }
    }

    // Account for the rematerialization in the memory tracker.
    TF_RETURN_IF_ERROR(memory_tracker->AddRematerializedInstruction(
        best_item, remat_item, absl::MakeSpan(indirect_users)));

    // Insert rematerialized instruction right before the earliest unplaced
    // use of the instruction *and* the earliest unplaced last use of any
    // operands of remat. Unplaced uses of the remat's operands are included
    // because we don't want to extend the live range of remat's operands as
    // this could increase memory usage.
    ItemList place_before;
    const absl::flat_hash_set<Item*> indirect_users_set(indirect_users.begin(),
                                                        indirect_users.end());
    for (auto user : remat->users()) {
      if (!indirect_users_set.contains(instruction_list->GetItem(user))) {
        place_before.push_back(instruction_list->GetItem(user));
      }
    }
    for (auto* indirect_user : indirect_users) {
      for (auto user : indirect_user->instruction->users()) {
        if (!indirect_users_set.contains(instruction_list->GetItem(user))) {
          place_before.push_back(instruction_list->GetItem(user));
        }
      }
    }
    for (auto* operand : remat->operands()) {
      for (auto* operand_user : operand->users()) {
        if (operand_user != remat) {
          Item* operand_user_item = instruction_list->GetItem(operand_user);
          if (!operand_user_item->placed) {
            place_before.push_back(operand_user_item);
          }
        }
      }
    }
    // Insert rematerialized instruction before any of its successors to
    // preserve ordering regarding control dependency.
    for (auto successor : remat->control_successors()) {
      Item* successor_item = instruction_list->GetItem(successor);
      // Assert to make sure we never remat an operation with control
      // successor already placed.
      CHECK(!successor_item->placed) << successor_item->instruction->name();
      place_before.push_back(successor_item);
    }
    instruction_list->InsertBeforeInstructions(remat_item, place_before);

    for (auto* bitcast : indirect_users) {
      instruction_list->InsertBeforeInstructions(bitcast, place_before);
    }
    // Helper function that looks through indirect users when determining if
    // there is an active user for an HloInstruction.
    std::function<bool(HloInstruction*)> uses_empty = [&](HloInstruction* i) {
      for (auto* u : i->users()) {
        if (!IsSupportedIndirectUser(u) || !uses_empty(u)) {
          return false;
        }
      }
      return true;
    };
    // If the rematerialized instruction is dead then rematerialization is
    // essentially a move. Don't delete the instruction now because we don't
    // want duplicate HloInstruction* values during the course of the
    // transformation because we keep maps with HloInstruction* values as
    // keys.
    if (uses_empty(best)) {
      VLOG(2) << best->name() << " is now dead";
      if (ContainsKey(*remat_move_instructions, best)) {
        // Previously, 'best' was a rematerialization which killed the
        // instruction it was a copying of. Now 'remat' is a rematerialization
        // of 'best' and kills 'best'. Stop rematerializing this instruction
        // to avoid an infinite loop.
        instruction_list->Denylist(remat);
      }
      remat_move_instructions->insert(remat);
      net_instructions_added += indirect_users.size();
    } else {
      net_instructions_added += indirect_users.size() + 1;
    }
    for (auto* indirect_user : indirect_users) {
      instruction_list->Denylist(indirect_user->instruction);
    }
    if (HloDataflowAnalysis::IsAsynchronousOperationStart(best->opcode()) ||
        HloDataflowAnalysis::IsAsynchronousOperationDone(best->opcode())) {
      VLOG(2) << "The old instruction " << best->name()
              << " is an async op. Removing to maintain one start to one done "
                 "invariant to keep the HLO valid.";
      // We need to remove all control dependencies from best before removing it
      // from the computation.  Its control dependencies were previously copied
      // to the remat instruction.
      TF_RETURN_IF_ERROR(best->DropAllControlDeps());
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(best));
    }
  }
  return net_instructions_added;
}

absl::StatusOr<int64_t> CompressInstruction(MemoryUsageTracker* memory_tracker,
                                            Item* best_item,
                                            const Shape& compact_shape,
                                            InstructionList* instruction_list) {
  HloInstruction* best = best_item->instruction;
  VLOG(5) << "Transposing instruction " << best->name() << " (saving "
          << HumanReadableNumBytes(memory_tracker->MemoryReducedIfCompressed(
                 best_item, compact_shape))
          << ") to" << compact_shape.ToString(true);

  HloComputation* computation = best->parent();
  HloInstruction* compressed = computation->AddInstruction(
      HloInstruction::CreateUnary(compact_shape, HloOpcode::kCopy, best),
      /*new_name=*/absl::StrCat(best->name(), ".remat_compressed"));

  HloInstruction* uncompressed = computation->AddInstruction(
      HloInstruction::CreateUnary(best->shape(), HloOpcode::kCopy, compressed),
      /*new_name=*/absl::StrCat(best->name(), ".remat_uncompressed"));

  Item* compressed_item = instruction_list->CreateItem(compressed);
  compressed_item->placed = true;

  Item* uncompressed_item = instruction_list->CreateItem(uncompressed);

  // Replace each remaining use of 'best' with the uncompressed.
  std::vector<HloInstruction*> best_users_copy = best->users();
  for (HloInstruction* user : best_users_copy) {
    if (!memory_tracker->IsPlaced(user)) {
      VLOG(5) << "  Replacing use of " << best->name() << " in " << user->name()
              << " with " << uncompressed->name();
      TF_RETURN_IF_ERROR(best->ReplaceUseWith(user, uncompressed));
    }
  }

  // Account for the rematerialization in the memory tracker.
  TF_RETURN_IF_ERROR(memory_tracker->AddCompressInstructions(
      best_item, compressed_item, uncompressed_item));

  // Insert rematerialized instruction right before the earliest unplaced
  // use of the instruction.
  ItemList place_before;
  for (auto user : uncompressed->users()) {
    place_before.push_back(instruction_list->GetItem(user));
  }

  instruction_list->Denylist(compressed_item->instruction);
  instruction_list->Denylist(uncompressed_item->instruction);
  instruction_list->InsertBeforeInstructions(uncompressed_item, place_before);
  instruction_list->InsertAfterInstructions(compressed_item, {best_item});

  return 2;
}

absl::StatusOr<int64_t> OffloadInstruction(MemoryUsageTracker* memory_tracker,
                                           Item* best_item,
                                           InstructionList* instruction_list) {
  HloInstruction* best_instruction = best_item->instruction;
  HloComputation* computation = best_instruction->parent();
  VLOG(2) << "Best_instruction's users: "
          << absl::StrJoin(best_instruction->users(), ", ",
                           [](std::string* str, const auto* x) {
                             return str->append(x->name());
                           });

  // Set up shapes for different memory locations.
  Shape instruction_shape_device = best_instruction->shape();
  Shape instruction_shape_host = best_instruction->shape();
  instruction_shape_host.mutable_layout()->set_memory_space(
      memory_tracker->options().host_memory_offload_config->host_memory_space);
  Shape context_shape = ShapeUtil::MakeShape(U32, {});

  // Create copy instructions to and from host memory.
  HloInstruction* copy_start_to_host =
      computation->AddInstruction(HloInstruction::CreateCopyStart(
          ShapeUtil::MakeTupleShape({instruction_shape_host,
                                     instruction_shape_device, context_shape}),
          best_instruction));
  HloInstruction* copy_done_to_host =
      computation->AddInstruction(HloInstruction::CreateUnary(
          instruction_shape_host, HloOpcode::kCopyDone, copy_start_to_host));

  HloInstruction* copy_start_to_device =
      computation->AddInstruction(HloInstruction::CreateCopyStart(
          ShapeUtil::MakeTupleShape({instruction_shape_device,
                                     instruction_shape_host, context_shape}),
          copy_done_to_host));
  HloInstruction* copy_done_to_device = computation->AddInstruction(
      HloInstruction::CreateUnary(instruction_shape_device,
                                  HloOpcode::kCopyDone, copy_start_to_device));
  VLOG(3) << "Created copy_start_to_host instr: "
          << copy_start_to_host->ToString();
  VLOG(3) << "Created copy_done_to_host instr: "
          << copy_done_to_host->ToString();
  VLOG(3) << "Created copy_start_to_device instr: "
          << copy_start_to_device->ToString();
  VLOG(3) << "Created copy_done_to_device instr: "
          << copy_done_to_device->ToString();

  // Update the HloCostAnalysis with the new instructions.
  TF_RETURN_IF_ERROR(
      copy_start_to_host->Visit(&memory_tracker->options().hlo_cost_analysis));
  TF_RETURN_IF_ERROR(
      copy_done_to_host->Visit(&memory_tracker->options().hlo_cost_analysis));
  TF_RETURN_IF_ERROR(copy_start_to_device->Visit(
      &memory_tracker->options().hlo_cost_analysis));
  TF_RETURN_IF_ERROR(
      copy_done_to_device->Visit(&memory_tracker->options().hlo_cost_analysis));

  // Create an Item for each instruction. These items will be inserted into the
  // InstructionList, which is essentially our schedule.
  Item* copy_start_to_host_item =
      instruction_list->CreateItem(copy_start_to_host);
  Item* copy_done_to_host_item =
      instruction_list->CreateItem(copy_done_to_host);
  Item* copy_start_to_device_item =
      instruction_list->CreateItem(copy_start_to_device);
  Item* copy_done_to_device_item =
      instruction_list->CreateItem(copy_done_to_device);

  // Add the newly created instructions to the deny list to prevent them from
  // becoming rematerialized later.
  instruction_list->Denylist(copy_start_to_host);
  instruction_list->Denylist(copy_done_to_host);
  instruction_list->Denylist(copy_start_to_device);
  instruction_list->Denylist(copy_done_to_device);

  Item* place_before{nullptr};
  // Find the first item that we need to place our final copy-done before. That
  // will be the first unplaced user of best_instruction.
  {
    ItemList place_before_list;
    for (auto user : best_instruction->users()) {
      if (user == copy_start_to_host) {
        // Skip the copy that we just added.
        continue;
      }
      auto item_of_user = instruction_list->GetItem(user);
      if (item_of_user->placed) {
        // Skip placed items.
        continue;
      }
      place_before_list.push_back(item_of_user);
    }
    CHECK(!place_before_list.empty()) << "Have nothing to place this before!";
    for (auto* item = instruction_list->first(); item != nullptr;
         item = instruction_list->next(item)) {
      if (absl::c_linear_search(place_before_list, item)) {
        place_before = item;
        break;
      }
    }
  }
  CHECK_NE(place_before, nullptr)
      << "Could not find an item to place this before.";

  // This function walks along the instruction list (schedule) and returns first
  // instruction which will be executed after `time_spent_on_copy` seconds of
  // compute has elapsed. Returns a result in the range [start_item, end_item).
  auto get_first_item_after_compute_time = [&](Item* start_item, Item* end_item,
                                               auto successor_func,
                                               float time_spent_on_copy) {
    // Do not count the computation time of the first item.
    // In the case of iterating forward in time, it is the output of this
    // item which we want to offload. In the case of iterating backward in
    // time, this buffer is a dependency of that start item.
    float time_so_far = 0.0;
    auto* current_item = start_item;
    // Walk the instruction list and accumulate the computation time.
    while (time_so_far < time_spent_on_copy) {
      auto next_item = successor_func(current_item);
      if (next_item == end_item) {
        // TODO(b/293323448): This is a bad thing, but not an error. Previously,
        // when evaluating whether or not to host offload this instruction we
        // checked how much compute there was between uses. We found that there
        // was enough total compute to cover the time required to copy the data
        // to the host and back. However, that check does not necessarily
        // guarantee that the compute is split in such a way that it will give
        // us enough compute to hide both copies in series. For example lets say
        // that the copies take this long: | <-------------  Copies take this
        // long
        // -------------->| Lets say the two copies take the same amount of
        // time: | <----- Copy to host -----> <---- Copy to device ----> |

        // And you have a compute sequence that looks like this:
        // +-----------+ +-----------+   +-----------+ +-----------+
        // | Compute-1 | | Compute-2 |   | Compute-3 | | Compute-4 |
        // +-----------+ +-----------+   +-----------+ +-----------+
        // It would make sense to insert the copy-start/done instructions
        // as follows:
        // ^ Copy-start to host
        //          Copy-done to host ^
        //        Copy-start to device ^
        //                                     Copy-done to device ^

        // However, if the compute sequence is not even, like this:
        // +-----------------------------------------+ +-----------+
        // |                Compute-1                | | Compute-2 |
        // +-----------------------------------------+ +-----------+
        // Then we would find enough compute to hide our copy on the forward
        // pass, but on the backward pass, there wouldn't be enough compute
        // remaining, even though we originally calculated that there was enough
        // total compute for the two copies.
        LOG(WARNING) << "Didn't find enough computation before end of window";
        break;
      }
      current_item = next_item;
      CHECK_NE(current_item, nullptr) << "current_item is null";
      CHECK_NE(current_item->instruction, nullptr)
          << "current_item's instruction is null";
      // TODO(b/293321321): HloCostAnalysis has no knowledge of any newly
      // rematerialized instructions via recompute or compression strategies.
      // This should be fixed.
      time_so_far += std::max(
          0.0f, memory_tracker->options().hlo_cost_analysis.optimal_seconds(
                    *current_item->instruction));
    }
    return current_item;
  };

  // Figure out how much time these copies will take.
  const int64_t bytes_used_by_buffers = memory_tracker->BytesUsedByBuffers(
      best_item, /*only_count_unplaced_users=*/false);
  const float copy_to_host_time_seconds =
      bytes_used_by_buffers /
      memory_tracker->options()
          .host_memory_offload_config->bandwidth_to_host_bytes_per_second;
  const float copy_from_host_time_seconds =
      bytes_used_by_buffers /
      memory_tracker->options()
          .host_memory_offload_config->bandwidth_from_host_bytes_per_second;
  VLOG(2) << "Item uses " << bytes_used_by_buffers << "B and will take "
          << copy_to_host_time_seconds << "s to copy to host and "
          << copy_from_host_time_seconds << "s to copy from host.";

  // Place the copy-start to host as early as possible.
  VLOG(2) << "Inserting " << copy_start_to_host_item->instruction->name()
          << " immediately after " << best_item->instruction->name();
  instruction_list->InsertAfterInstructions(copy_start_to_host_item,
                                            {best_item});

  // Place the copy-done to device as late as possible.
  VLOG(2) << "Inserting " << copy_done_to_device_item->instruction->name()
          << " immediately before " << place_before->instruction->name();
  instruction_list->InsertBeforeInstructions(copy_done_to_device_item,
                                             {place_before});

  // Place the first copy-done after enough runtime after the first copy-start
  // to hide the memory transfer.
  auto first_item_after_to_host_copy = get_first_item_after_compute_time(
      copy_start_to_host_item, copy_done_to_device_item,
      [&instruction_list](Item* item) { return instruction_list->next(item); },
      copy_to_host_time_seconds);
  VLOG(2) << "Inserting " << copy_done_to_host_item->instruction->name()
          << " immediately after "
          << first_item_after_to_host_copy->instruction->name();
  instruction_list->InsertAfterInstructions(copy_done_to_host_item,
                                            {first_item_after_to_host_copy});

  // Place the second copy-start early enough so that there is enough
  // runtime to hide the memory transfer before the second copy-done.
  auto first_item_before_from_host_copy = get_first_item_after_compute_time(
      copy_done_to_device_item, copy_done_to_host_item,
      [&instruction_list](Item* item) { return instruction_list->prev(item); },
      copy_from_host_time_seconds);
  VLOG(2) << "Inserting " << copy_start_to_device_item->instruction->name()
          << " immediately before "
          << first_item_before_from_host_copy->instruction->name();
  instruction_list->InsertBeforeInstructions(
      copy_start_to_device_item, {first_item_before_from_host_copy});

  // Once all of the items are in the proper place in the instruction list, mark
  // them as placed or not depending on which item is the current item in the
  // memory tracker.
  {
    auto item = instruction_list->first();
    while (item != nullptr) {
      if (item == copy_start_to_host_item || item == copy_done_to_host_item ||
          item == copy_start_to_device_item ||
          item == copy_done_to_device_item) {
        item->placed = true;
      } else if (memory_tracker->IsInProgressItem(item)) {
        // Our newly added items are defaulted as not placed, so breaking here
        // gives us our desired result.
        break;
      }
      item = instruction_list->next(item);
    }
  }

  // It is critical to only update the users after items have been marked as
  // placed, since we will only want to update non-placed items.

  // Replace uses of best_instruction with copy_done_to_device.
  // Note that items must be created before this point.
  std::vector<HloInstruction*> best_users_copy = best_instruction->users();
  for (HloInstruction* user : best_users_copy) {
    if (!memory_tracker->IsPlaced(user)) {
      VLOG(3) << "  Replacing use of " << best_instruction->name() << " in "
              << user->name() << " with " << copy_done_to_device->name();
      TF_RETURN_IF_ERROR(
          best_instruction->ReplaceUseWith(user, copy_done_to_device));
    } else {
      VLOG(3) << user->name() << " is placed, not going to update";
    }
  }

  // Finally, update the MemoryUsageTracker. This will update the tracking of
  // buffer creations and uses.
  TF_RETURN_IF_ERROR(memory_tracker->AddHostOffloadCopyInstructions(
      best_item, copy_start_to_host_item, copy_done_to_host_item,
      copy_start_to_device_item, copy_done_to_device_item));

  return 4;
}

// A simple struct to encapsulate the number of instructions added during
// rematerialization.
struct InstructionsAdded {
  // Total count of instructions rematerialized.
  int remat_count;
  // Total count of instructions rematerialized minus number of original
  // instructions that are now dead.
  int net_instructions_added;
  // Amount of effort expended to find the instructions to rematerialize.
  int effort;
};

// Rematerializes the best block of instructions of size between min_block_size
// and max_block_size (both inclusive) if at least one candidate block of
// instructions can be found. Returns number of instructions rematerialized.
absl::StatusOr<InstructionsAdded> RematerializeBestBlock(
    int min_block_size, int max_block_size, MemoryUsageTracker* memory_tracker,
    InstructionList* instruction_list, HloSchedule* schedule,
    int64_t memory_limit_bytes,
    absl::flat_hash_map<const HloInstruction*, bool>* rematerializable_map,
    absl::flat_hash_set<const HloInstruction*>* remat_move_instructions,
    HloRematerialization* rematerialization) {
  CHECK(min_block_size > 0) << "Negative block size.";

  std::vector<Item*> best_items;
  RematStrategy best_strategy;
  int effort;
  std::tie(best_items, best_strategy, effort) =
      memory_tracker->PickRematerializationCandidates(
          *instruction_list, memory_limit_bytes, rematerializable_map,
          min_block_size, max_block_size,
          rematerialization->ComputationPeakMemory(
              memory_tracker->computation()));
  InstructionsAdded num_instructions_added;
  num_instructions_added.remat_count = best_items.size();
  num_instructions_added.effort = effort;
  if (best_items.empty()) {
    num_instructions_added.net_instructions_added = 0;
    return num_instructions_added;
  }

  if (best_strategy.kind == RematStrategy::kCompress) {
    CHECK(best_items.size() == 1)
        << "More than one instruction compressed simultaneously.";
    HloInstruction* best = best_items[0]->instruction;
    VLOG(1) << "Remat via compression: " << best->name() << " (saving "
            << HumanReadableNumBytes(memory_tracker->MemoryReducedIfCompressed(
                   best_items[0], best_strategy.compact_shape))
            << ")";

    TF_ASSIGN_OR_RETURN(
        num_instructions_added.net_instructions_added,
        CompressInstruction(memory_tracker, best_items[0],
                            best_strategy.compact_shape, instruction_list));

  } else if (best_strategy.kind == RematStrategy::kHostOffload) {
    CHECK_EQ(best_items.size(), 1)
        << "More than one buffer offloaded simultaneously.";
    VLOG(1) << "Remat via offload: " << best_items[0]->instruction->name();
    TF_ASSIGN_OR_RETURN(
        num_instructions_added.net_instructions_added,
        OffloadInstruction(memory_tracker, best_items[0], instruction_list));
    VLOG(4) << "Offload done, hlo computation:\n"
            << memory_tracker->computation()->ToString();
    VLOG(6) << "Memory tracker:\n" << memory_tracker->ToString();
  } else {
    CHECK_EQ(best_strategy.kind, RematStrategy::kRecompute)
        << "Expecting strategy to be Recompute";
    VLOG(1) << "Remat via recomputation: {"
            << absl::StrJoin(best_items, ", ",
                             [](std::string* out, Item* item) {
                               absl::StrAppend(out, item->instruction->name());
                             })
            << '}';
    TF_ASSIGN_OR_RETURN(
        num_instructions_added.net_instructions_added,
        RematerializeInstructions(memory_tracker, &best_items,
                                  remat_move_instructions, instruction_list,
                                  schedule, rematerialization));
  }
  return num_instructions_added;
}
}  // namespace

absl::StatusOr<int64_t> HloRematerialization::ComputePeakMemory(
    const HloComputation* computation, const HloInstructionSequence& order,
    const absl::flat_hash_set<absl::string_view>& execution_threads) const {
  InstructionList instruction_list(order);
  MemoryUsageTracker tracker(options_, computation, *points_to_analysis_,
                             instruction_list);
  int64_t peak_memory = tracker.memory_usage();
  for (auto* item = instruction_list.first(); item != nullptr;
       item = instruction_list.next(item)) {
    const HloInstruction* instruction = item->instruction;
    TF_RETURN_IF_ERROR(tracker.BeginInstruction(item));
    TF_ASSIGN_OR_RETURN(
        int64_t callee_usage,
        CalledComputationsMemoryUsage(instruction, execution_threads));
    peak_memory =
        std::max<int64_t>(peak_memory, tracker.memory_usage() + callee_usage);
    TF_RETURN_IF_ERROR(tracker.EndInstruction());
  }
  VLOG(1) << "Peak memory for " << computation->name() << ": "
          << HumanReadableNumBytes(peak_memory);
  return peak_memory;
}

absl::StatusOr<int64_t> HloRematerialization::CalledComputationsMemoryUsage(
    const HloInstruction* instruction,
    const absl::flat_hash_set<absl::string_view>& execution_threads) const {
  const CallSite* callsite =
      call_graph_->GetNode(instruction->parent()).GetCallSite(instruction);
  if (callsite == nullptr || callsite->context() == CallContext::kEmbedded) {
    return 0;
  }
  int64_t callee_usage = 0;
  for (const HloComputation* computation : callsite->called_computations()) {
    if (!HloInstruction::IsThreadIncluded(computation->execution_thread(),
                                          execution_threads)) {
      continue;
    }
    TF_RET_CHECK(ContainsKey(computation_peak_memory_, computation));
    callee_usage += computation_peak_memory_.at(computation);
  }
  return callee_usage;
}

absl::StatusOr<bool> HloRematerialization::RematerializeComputation(
    HloComputation* computation, HloSchedule* schedule,
    int64_t memory_limit_bytes, int64_t min_remat_size,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  const auto peak_memory_usage = computation_peak_memory_.at(computation);
  if (peak_memory_usage <= memory_limit_bytes) {
    // Nothing to do.
    VLOG(1) << "Asked to rematerialize computation of size "
            << peak_memory_usage
            << " but it already fits within the given memory limit ("
            << memory_limit_bytes << ")";
    return false;
  }
  VLOG(1) << "Rematerializing computation " << computation->name()
          << " with limit " << HumanReadableNumBytes(memory_limit_bytes);
  VLOG(1) << "peak memory usage is "
          << HumanReadableNumBytes(peak_memory_usage);
  CHECK(!ContainsKey(rematerialized_computations_, computation));

  InstructionList instruction_list(schedule->sequence(computation));
  MemoryUsageTracker memory_tracker(options_, computation, *points_to_analysis_,
                                    instruction_list);

  instruction_list.PromoteNodesToSkip([&](Item* item) {
    return memory_tracker.AllocatedSize(item) >= min_remat_size;
  });
  bool changed = false;

  // If the rematerialization makes the source instruction dead, then the
  // rematerialization is added to 'remat_move_instructions' (the
  // rematerialization is essentially a move). If the next rematerialization of
  // the instruction is also a move then the rematerialization is added to the
  // denylist.
  absl::flat_hash_set<const HloInstruction*> remat_move_instructions;

  // The map from instructions to their rematerializable status.
  absl::flat_hash_map<const HloInstruction*, bool> rematerializable_map;

  // The peak memory of the computation at any point in the instruction
  // sequence.
  int64_t peak_memory = memory_tracker.memory_usage();

  // Total count of instructions rematerialized.
  int64_t remat_count = 0;
  // Total count of clones created minus number of original rematerialized
  // instructions which are dead.
  int64_t net_instructions_added = 0;

  const CallGraphNode& call_graph_node = call_graph_->GetNode(computation);

  // Iterate through all instructions in the sequence. At each instruction
  // (program point) if memory_usage exceeds the specified limit then
  // rematerialize HLO instructions until memory_usage is reduced.
  int64_t instruction_index = 0;
  for (auto* item = instruction_list.first(); item != nullptr;
       item = instruction_list.next(item)) {
    const HloInstruction* instruction = item->instruction;
    TF_ASSIGN_OR_RETURN(
        int64_t callee_usage,
        CalledComputationsMemoryUsage(instruction, execution_threads));
    TF_RETURN_IF_ERROR(memory_tracker.BeginInstruction(item));

    VLOG(2) << "Program point at " << instruction->name()
            << ", memory usage = " << memory_tracker.memory_usage()
            << ", callee usage = " << callee_usage << ", [" << instruction_index
            << "/" << instruction_list.size() << "]";
    instruction_index++;

    // Initialize both min_block_size and max_block_size to 1 so that only
    // single instruction rematerialization is considered first.
    int min_block_size = 1;
    int max_block_size = 1;
    // Only trigger rematerialization when the memory usage changes.
    if (memory_tracker.AllocatedSize(item) + callee_usage > 0) {
      // Finding larger blocks of instructions to rematerialize can be time
      // consuming. To limit the amount of time spent attempting to find such
      // large blocks, count the amount of effort expended to find single
      // instructions to rematerialize and then limit the total amount of effort
      // to at most a factor of block_rematerialization_factor_ more.
      bool is_first_phase = true;
      int64_t first_phase_effort = 0;
      int64_t second_phase_effort = 0;
      while (memory_tracker.memory_usage() + callee_usage >
             memory_limit_bytes) {
        VLOG(2) << "Over memory limit at instruction " << instruction->name()
                << ", using "
                << HumanReadableNumBytes(memory_tracker.memory_usage() +
                                         callee_usage)
                << ", limit is " << HumanReadableNumBytes(memory_limit_bytes);

        TF_ASSIGN_OR_RETURN(
            InstructionsAdded instructions_added,
            RematerializeBestBlock(min_block_size, max_block_size,
                                   &memory_tracker, &instruction_list, schedule,
                                   memory_limit_bytes, &rematerializable_map,
                                   &remat_move_instructions, this));
        net_instructions_added += instructions_added.net_instructions_added;
        remat_count += instructions_added.remat_count;
        if (is_first_phase) {
          first_phase_effort += instructions_added.effort;
        } else {
          second_phase_effort += instructions_added.effort;
        }
        if (instructions_added.net_instructions_added > 0) {
          VLOG(1) << "memory_usage after rematerialization = "
                  << HumanReadableNumBytes(memory_tracker.memory_usage());
        }
        if (instructions_added.remat_count == 0) {
          // Unable to find a block to rematerialize.
          // Consider doubling the block size.
          min_block_size = max_block_size + 1;
          max_block_size = 2 * max_block_size;
          is_first_phase = false;
        } else {
          // Found a valid block. Reset to start looking for single instructions
          // again.
          max_rematerialized_block_size_ =
              std::max(max_rematerialized_block_size_, max_block_size);
          changed = true;
          min_block_size = 1;
          max_block_size = 1;
        }
        if (max_block_size > options_.block_size_limit ||
            second_phase_effort >
                options_.block_rematerialization_factor * first_phase_effort) {
          break;
        }
      }
    }
    const CallSite* callsite = call_graph_node.GetCallSite(instruction);
    if (callsite != nullptr &&
        callsite->context() == CallContext::kControlFlow &&
        memory_tracker.memory_usage() + callee_usage > memory_limit_bytes) {
      // Memory usage exceeds the limit. Try to rematerialize any
      // subcomputation(s) that this instruction calls.
      VLOG(1) << "Memory usage still over the limit ("
              << (memory_tracker.memory_usage() + callee_usage) << " > "
              << memory_limit_bytes
              << "). Rematerializing computations called by "
              << instruction->name();

      // Recompute callee usage to account for any rematerialization performed
      // in the callee computations.
      for (HloComputation* called_computation :
           callsite->called_computations()) {
        if (!ContainsKey(rematerialized_computations_, called_computation) &&
            HloInstruction::IsThreadIncluded(
                called_computation->execution_thread(), execution_threads)) {
          // Memory limit for the subcomputation is the memory limit less the
          // amount of memory used at this point in the computation.
          int64_t subcomputation_memory_limit_bytes = std::max<int64_t>(
              0, memory_limit_bytes - memory_tracker.memory_usage());
          TF_ASSIGN_OR_RETURN(
              bool subcomputation_changed,
              RematerializeComputation(called_computation, schedule,
                                       subcomputation_memory_limit_bytes,
                                       min_remat_size, execution_threads));
          changed |= subcomputation_changed;
        }
      }

      TF_ASSIGN_OR_RETURN(callee_usage, CalledComputationsMemoryUsage(
                                            instruction, execution_threads));
    }

    peak_memory = std::max<int64_t>(
        peak_memory, memory_tracker.memory_usage() + callee_usage);
    VLOG(3) << "peak memory usage = " << HumanReadableNumBytes(peak_memory);

    TF_RETURN_IF_ERROR(memory_tracker.EndInstruction());
  }

  // Verify some invariants on the memory tracker.
  for (auto* instruction : computation->instructions()) {
    CHECK(memory_tracker.IsPlaced(instruction)) << instruction->name();
  }

  VLOG(1) << "In computation " << computation->name() << " rematerialized "
          << remat_count << " instructions; " << net_instructions_added
          << " net instructions added";
  VLOG(1) << "  peak memory usage now " << HumanReadableNumBytes(peak_memory)
          << " (was "
          << HumanReadableNumBytes(computation_peak_memory_.at(computation))
          << ")";

  // Update peak memory used by computation.
  computation_peak_memory_.at(computation) = peak_memory;

  // Update order to include rematerialized instructions.
  HloInstructionSequence& sequence = schedule->GetOrCreateSequence(computation);
  sequence.clear();
  for (auto* item = instruction_list.first(); item != nullptr;
       item = instruction_list.next(item)) {
    HloInstruction* instruction = item->instruction;
    sequence.push_back(instruction);
  }
  rematerialized_computations_.insert(computation);

  instructions_rematerialized_ += remat_count;
  net_instructions_added_ += net_instructions_added;

  return changed;
}

absl::StatusOr<bool> HloRematerialization::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (options_.remat_mode_config.host_offload) {
    CHECK(options_.host_memory_offload_config.has_value())
        << "Host memory config is required when host memory offload strategy "
           "is specified";
  }
  VLOG(1) << "HloRematerialization() with memory limit of "
          << HumanReadableNumBytes(options_.memory_limit_bytes);
  if (!options_.remat_mode_config.compress &&
      !options_.remat_mode_config.recompute &&
      !options_.remat_mode_config.host_offload) {
    // All rematerialization strategies are disabled; nothing to do.
    VLOG(1) << "All rematerialization strategies are disabled. Skipping.";
    return false;
  }
  VLOG(2) << "HloRemat mode: compress: " << options_.remat_mode_config.compress
          << ", host_offload: " << options_.remat_mode_config.host_offload
          << ", recompute: " << options_.remat_mode_config.recompute;
  XLA_VLOG_LINES(3, "Before HloRematerialization:\n" + module->ToString());

  // Initialize pass object state.
  computation_peak_memory_.clear();
  rematerialized_computations_.clear();
  instructions_rematerialized_ = 0;
  net_instructions_added_ = 0;

  TF_RET_CHECK(module->has_schedule());
  TF_ASSIGN_OR_RETURN(points_to_analysis_, TuplePointsToAnalysis::Run(module));
  next_channel_id_ = hlo_query::NextChannelId(*module);

  // Adjust memory limit to account for the output of the entry
  // computation. This is necessary because the per-computation accounting in
  // MemoryUsageTracker do not include output as these are typically allocated
  // by the caller.
  int64_t module_output_size = 0;
  ShapeUtil::ForEachSubshape(
      module->result_shape(),
      [&module_output_size, this](const Shape& subshape,
                                  const ShapeIndex& output_index) {
        module_output_size += options_.hlo_cost_analysis.GetShapeSize(subshape);
      });

  int64_t adjusted_memory_limit_bytes =
      std::max<int64_t>(0, options_.memory_limit_bytes - module_output_size);
  VLOG(1) << "Adjusted memory limit accounting for output ("
          << HumanReadableNumBytes(module_output_size)
          << "): " << HumanReadableNumBytes(adjusted_memory_limit_bytes);

  call_graph_ = CallGraph::Build(module);

  // Buffer assignment allocates a single stack for all asynchronous
  // computations of the same thread, which persists for the entire duration of
  // the program. We need to account for this by adjusting the memory limit.
  int64_t total_async_peak_memory = 0;
  if (!options_.async_computation_parallelism.empty()) {
    // We cannot compute memory usage for both the main and asynchronous threads
    // at the same time, as that will cause the asynchronous callee usage to be
    // added to the main thread callers usage. The callee's memory is
    // preallocated, so the caller doesn't pay for it.
    absl::flat_hash_set<std::string_view> async_threads;
    for (const auto& [computation, _] :
         options_.async_computation_parallelism) {
      async_threads.insert(computation->execution_thread());
    }
    TF_RETURN_IF_ERROR(call_graph_->VisitNodes(
        [this, module,
         &async_threads](const CallGraphNode& node) -> absl::Status {
          auto callee_thread = node.computation()->execution_thread();
          if (node.context() == CallContext::kControlFlow &&
              HloInstruction::IsThreadIncluded(callee_thread, async_threads)) {
            TF_ASSIGN_OR_RETURN(computation_peak_memory_[node.computation()],
                                ComputePeakMemory(node.computation(),
                                                  module->schedule().sequence(
                                                      node.computation()),
                                                  {callee_thread}));
          }
          return absl::OkStatus();
        },
        /*visit_unreachable_nodes=*/false));

    int64_t async_peak_memory = 0;
    // Only consider asynchronous computations invoked from the main thread.
    for (const auto [entry_computation, parallel_threads] :
         options_.async_computation_parallelism) {
      const int64_t peak_memory =
          computation_peak_memory_.at(entry_computation);
      // Adjust memory usage for parallel execution of the same computation
      // on different devices.
      const int64_t parallel_peak_memory = peak_memory * parallel_threads;
      async_peak_memory = std::max(async_peak_memory, parallel_peak_memory);
    }
    adjusted_memory_limit_bytes =
        std::max<int64_t>(0, adjusted_memory_limit_bytes - async_peak_memory);
    total_async_peak_memory += async_peak_memory;
    VLOG(1) << "Adjusted memory limit accounting for async computations ("
            << HumanReadableNumBytes(async_peak_memory)
            << "): " << HumanReadableNumBytes(adjusted_memory_limit_bytes);

    // Reset back to a clean state, since we don't expect to utilize the
    // async computation memory usage anymore.
    computation_peak_memory_.clear();
  }
  // Compute peak memory usage of all computations in the module called in a
  // sequential context.
  TF_RETURN_IF_ERROR(call_graph_->VisitNodes(
      [this, module,
       &execution_threads](const CallGraphNode& node) -> absl::Status {
        if (node.context() == CallContext::kControlFlow &&
            HloInstruction::IsThreadIncluded(
                node.computation()->execution_thread(), execution_threads)) {
          TF_ASSIGN_OR_RETURN(
              computation_peak_memory_[node.computation()],
              ComputePeakMemory(node.computation(),
                                module->schedule().sequence(node.computation()),
                                execution_threads));
        }
        return absl::OkStatus();
      },
      /*visit_unreachable_nodes=*/false));

  // The peak memory usage of the module equals the peak memory use of the entry
  // computation plus the output size of the computation plus memory use of
  // asynchronous computations. This is because the peak memory for a
  // computation does not include the output as this is typically accounted for
  // in the caller.
  const int64_t before_peak_memory =
      computation_peak_memory_.at(module->entry_computation()) +
      module_output_size + total_async_peak_memory;
  VLOG(1) << "Peak memory usage of module (before): "
          << HumanReadableNumBytes(before_peak_memory);

  // Initialize the HloCostAnalysis on this computation.
  for (auto* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    TF_RETURN_IF_ERROR(computation->Accept(&options_.hlo_cost_analysis));
  }

  // Subcomputations called by the entry computation will also be
  // rematerialized.
  TF_ASSIGN_OR_RETURN(
      bool changed,
      RematerializeComputation(module->entry_computation(), &module->schedule(),
                               adjusted_memory_limit_bytes,
                               options_.min_remat_size, execution_threads));
  // Rematerialization can introduce dead code. This occurs if all uses of an
  // instruction are replaced with rematerializations of the instruction.

  // Stash away the schedule during copy insertion, to avoid validation failures
  // while the module is in flux.
  HloSchedule saved_schedule = module->schedule();
  module->clear_schedule();
  TF_ASSIGN_OR_RETURN(bool dead_code_removed, HloDCE().Run(module));
  changed |= dead_code_removed;

  // After DCE, the module sequence may include instructions which no longer
  // exist. Update the schedule and restore it.
  TF_RETURN_IF_ERROR(saved_schedule.Update(execution_threads));
  TF_RETURN_IF_ERROR(module->set_schedule(std::move(saved_schedule)));
  VLOG(1) << "Rematerialized " << instructions_rematerialized_
          << " instructions in module " << module->name() << "; "
          << net_instructions_added_ << " net instructions added";
  const int64_t current_peak_memory =
      computation_peak_memory_.at(module->entry_computation()) +
      module_output_size + total_async_peak_memory;
  VLOG(1) << "Peak memory usage of module now "
          << HumanReadableNumBytes(current_peak_memory) << " ("
          << current_peak_memory << " bytes), was "
          << HumanReadableNumBytes(before_peak_memory) << " ("
          << before_peak_memory << " bytes)";
  const int64_t reduced_peak_memory = before_peak_memory - current_peak_memory;
  VLOG(1) << "Reduced peak memory by "
          << HumanReadableNumBytes(reduced_peak_memory) << " ("
          << reduced_peak_memory << " bytes)";

  sizes_.before_bytes = before_peak_memory;
  sizes_.after_bytes = current_peak_memory;

  XLA_VLOG_LINES(5, "After HloRematerialization:\n" + module->ToString());

  if (current_peak_memory > options_.memory_limit_bytes) {
    LOG(WARNING) << absl::StrFormat(
        "Can't reduce memory use below %s (%d bytes) by rematerialization; "
        "only reduced to %s (%d bytes), down from %s (%d bytes) originally",
        HumanReadableNumBytes(options_.memory_limit_bytes),
        options_.memory_limit_bytes, HumanReadableNumBytes(current_peak_memory),
        current_peak_memory, HumanReadableNumBytes(before_peak_memory),
        before_peak_memory);
  }
  return changed;
}

}  // namespace xla

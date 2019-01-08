/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_rematerialization.h"

#include <algorithm>
#include <memory>
#include <set>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/buffer_value.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace {

using ::tensorflow::strings::HumanReadableNumBytes;

// Potential optimizations:
// . TODO(b/35244891): Avoid N^2 behavior by keeping a priority queue
//   of candidates.
// . Cache IsRematerializable in Item?  Only correct if control
//   predecessors and successors don't change.

// Returns true if the given instruction is rematerializable.
bool IsRematerializable(const HloInstruction* instruction) {
  // Don't rematerialize instructions with side effects or instructions which
  // cannot be cloned safely.
  switch (instruction->opcode()) {
    case HloOpcode::kCall:
    case HloOpcode::kConstant:
    case HloOpcode::kConditional:
    case HloOpcode::kAllReduce:
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
    absl::flat_hash_map<const HloInstruction*, bool>* remat_able) {
  auto it = remat_able->find(instruction);
  if (it != remat_able->end()) {
    return it->second;
  }
  bool rematerializable = IsRematerializable(instruction);
  (*remat_able)[instruction] = rematerializable;
  return rematerializable;
}

// Type holding a unique identifier for each Buffer object.
using BufferId = int64;
using BufferIdList = absl::InlinedVector<BufferId, 3>;

// We wrap HloInstruction* with an Item that holds auxiliary
// per-instruction state.
struct Item {
  HloInstruction* instruction;

  // True once the instruction is marked as placed (when BeginInstruction
  // has been called for this instruction).
  bool placed = false;

  // To avoid an infinite loop rematerializing the same set of
  // instructions ad infinitum, keep a blacklist of instructions
  // which should not be rematerialized.
  bool blacklisted = false;

  // The buffers defined by this instruction.
  BufferIdList buffers_defined;

  // The buffers used by this instruction.
  BufferIdList buffers_used;

 private:
  friend class InstructionList;

  // Items are arranged in a doubly linked list.
  Item* next;
  Item* prev;

  // List is ordered by position, which can however be duplicated as
  // new instructions are inserted.  See InsertBeforeInstructions
  // comment for details.
  int64 position;
};

using ItemList = absl::InlinedVector<Item*, 3>;

// Class which maintains an ordered list of instructions with fast insertion
// before arbitrary elements.
class InstructionList {
 public:
  explicit InstructionList(const HloInstructionSequence& order) {
    int64 position = 0;
    Item* last = nullptr;
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

  // Creates an Item for the given instruction, but doesn't add it to the list.
  // (Use InsertBeforeInstructions to add the Item to the list.)
  Item* CreateItem(HloInstruction* inst) {
    Item* item = new Item;
    item->instruction = inst;
    CHECK(item_map_.insert({inst, item}).second) << "inserting inst twice";
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
                             [](string* out, Item* item) {
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

  void Blacklist(const HloInstruction* inst) {
    GetItem(inst)->blacklisted = true;
  }

 private:
  // Insert instruction 'item' immediately before 'before' in the list.
  void InsertBefore(Item* item, Item* before) {
    VLOG(3) << "InsertBefore: " << item->instruction->name() << " before "
            << before->instruction->name();
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

  // Item for each instruction.
  absl::flat_hash_map<const HloInstruction*, Item*> item_map_;
};

// Return the items which use the given LogicalBuffer. Sets
// has_indirect_users to whether any of the uses is indirect. A use is indirect
// if the instruction defining logical_buffer is not an operand of the use. This
// can happen via buffer aliasing (eg, tuples).
ItemList GetUsers(const InstructionList& instruction_list,
                  const LogicalBuffer* logical_buffer,
                  const TuplePointsToAnalysis& points_to_analysis,
                  bool* has_indirect_users) {
  ItemList users;
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
      if (buffer_alias.instruction() != logical_buffer->instruction()) {
        *has_indirect_users = true;
      }
      // A buffer may be used by the instruction via more than one alias. For
      // example, a buffer which appears in more than one element of a tuple.
      Item* user_item = instruction_list.GetItem(user);
      if (!absl::c_linear_search(users, user_item)) {
        users.push_back(user_item);
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
  MemoryUsageTracker(
      const HloComputation* computation,
      const HloRematerialization::ShapeSizeFunction& size_function,
      const TuplePointsToAnalysis& points_to_analysis,
      const InstructionList& instruction_list);

  // Starts the placement of the given instruction. This adds the sizes of the
  // LogicalBuffers defined by the instruction to the current memory
  // usage. Placement is broken into two steps (BeginInstruction and
  // EndInstruction) to accurately model memory usage. At BeginInstruction the
  // memory for the output value(s) of the current instruction is allocated. At
  // EndInstruction memory for dead operand(s) is freed.
  Status BeginInstruction(Item* item);

  // Finishes the placement of the current instruction. This frees any dead
  // operands or dead result of the instruction. This must be called after
  // each call to BeginInstruction.
  Status EndInstruction();

  // Returns the number of bytes that the current memory usage will be reduced
  // if the given instruction is rematerialized.
  int64 MemoryReducedIfRematerialized(Item* item) const;

  // Adjusts memory usage to account for the rematerialization of
  // original_item for all remaining unplaced uses. The rematerialization
  // is remat_item. This method should be called after the HLO graph has
  // been transformed (rematerialization instruction created and connected to
  // uses).
  Status AddRematerializedInstruction(Item* original_item, Item* remat_item);

  // Returns whether the given instruction has been placed (BeginInstruction
  // has been called with 'instruction' as the argument).
  bool IsPlaced(const HloInstruction* instruction) const {
    return instruction_list_.GetItem(instruction)->placed;
  }

  // Returns the current memory usage. This is the sum of sizes of all live
  // values.
  int64 memory_usage() const { return memory_usage_; }

  // Check invariants of the data structure. This is expensive to call.
  bool Check() const;

  string ToString() const;

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
    const int64 size;

    // Whether this buffer is live-out of the computation.
    bool live_out;

    // Whether this buffer has indirect uses. Ie, an instruction which is not a
    // user of defining_instruction uses this buffer. This can occur due to
    // buffer aliasing (eg, tuples).
    bool has_indirect_uses;

    // The instructions which use this buffer.
    ItemList users;

    // The number of users (HloInstructions) of this buffer which have not yet
    // been placed in the sequence.
    int64 unfinished_user_count;

    string ToString() const {
      return absl::StrCat("Buffer ", id, " (defined by ",
                          defining_instruction->instruction->name(), ", size ",
                          size, " bytes)");
    }
  };

  // Creates a Buffer representing the given logical buffer. The buffer is added
  // to buffers_ and a reference is returned.
  Buffer& CreateBufferFromLogicalBuffer(
      const LogicalBuffer* logical_buffer,
      const TuplePointsToAnalysis& points_to_analysis,
      const HloRematerialization::ShapeSizeFunction& size_function,
      bool live_out) {
    bool has_indirect_uses = false;
    ItemList users = GetUsers(instruction_list_, logical_buffer,
                              points_to_analysis, &has_indirect_uses);
    return NewBuffer(instruction_list_.GetItem(logical_buffer->instruction()),
                     size_function(logical_buffer->shape()), std::move(users),
                     live_out, has_indirect_uses);
  }

  // Create a new buffer representing a rematerialization of given buffer for
  // the given uses.
  Buffer& RematerializeBuffer(const Buffer& original_buffer, Item* remat_item,
                              ItemList&& rematerialized_uses) {
    CHECK(original_buffer.defining_instruction->placed);
    CHECK(!original_buffer.has_indirect_uses);
    CHECK(!original_buffer.live_out);
    for (Item* use : rematerialized_uses) {
      CHECK(!use->placed);
    }
    return NewBuffer(remat_item, original_buffer.size,
                     std::move(rematerialized_uses), /*live_out=*/false,
                     /*has_indirect_uses=*/false);
  }

  // Return number of bytes allocated for the buffer with the given id. Buffers
  // allocated by the calling computation (eg, parameter and output buffers) are
  // considered to have zero bytes because the memory is accounted for in a
  // different computation.
  int64 AllocatedSize(BufferId buffer_id) const {
    const Buffer& buffer = buffers_.at(buffer_id);
    HloOpcode def_opcode = buffer.defining_instruction->instruction->opcode();
    if (buffer.live_out || def_opcode == HloOpcode::kParameter) {
      return 0;
    } else {
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

  // Returns whether the given instruction is live at the current program
  // point.
  bool IsCurrentlyLive(BufferId buffer_id) const {
    const Buffer& buffer = buffers_[buffer_id];
    return (buffer.defining_instruction->placed &&
            buffer.unfinished_user_count > 0);
  }

  // Create a new buffer, add it to buffers_, and return a reference.
  Buffer& NewBuffer(Item* defining_instruction, int64 size, ItemList&& users,
                    bool live_out, bool has_indirect_uses) {
    int buffer_id = buffers_.size();
    buffers_.push_back(Buffer{buffer_id, defining_instruction, size, live_out,
                              has_indirect_uses, users,
                              static_cast<int64>(users.size())});
    return buffers_.back();
  }

  const HloComputation* computation_;

  // Instruction list containing the ordering of instructions in
  // computation_. This is the order in which instructions are placed
  // (BeginInstruction/EndInstruction calls).
  const InstructionList& instruction_list_;

  // Memory usage at the currently placed instruction.
  int64 memory_usage_ = 0;

  // The instruction currently being placed. This value is non-null only
  // between the calling of BeginInstruction and EndInstruction.
  Item* in_progress_item_ = nullptr;

  // All buffers in the computation.
  std::vector<Buffer> buffers_;
};

MemoryUsageTracker::MemoryUsageTracker(
    const HloComputation* computation,
    const HloRematerialization::ShapeSizeFunction& size_function,
    const TuplePointsToAnalysis& points_to_analysis,
    const InstructionList& instruction_list)
    : computation_(computation), instruction_list_(instruction_list) {
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
        for (Item* user_item : GetUsers(instruction_list_, logical_buffer,
                                        points_to_analysis, &unused)) {
          if (!absl::c_linear_search(buffer->users, user_item)) {
            buffer->users.push_back(user_item);
            buffer->unfinished_user_count++;
            user_item->buffers_used.push_back(buffer->id);
          }
        }
      } else {
        buffer = &CreateBufferFromLogicalBuffer(
            logical_buffer, points_to_analysis, size_function,
            ContainsKey(live_out_set, logical_buffer));
        item->buffers_defined.push_back(buffer->id);
        for (Item* user : buffer->users) {
          user->buffers_used.push_back(buffer->id);
        }
      }

      logical_buffer_to_buffer_id[logical_buffer] = buffer->id;
    }
  }
  XLA_VLOG_LINES(10, ToString());
  DCHECK(Check());
}

Status MemoryUsageTracker::BeginInstruction(Item* item) {
  const HloInstruction* instruction = item->instruction;
  VLOG(3) << "BeginInstruction " << instruction->name();
  TF_RET_CHECK(in_progress_item_ == nullptr);
  in_progress_item_ = item;

  item->placed = true;

  // All buffers defined by this instruction need memory.
  for (BufferId buffer_id : item->buffers_defined) {
    VLOG(3) << "  Buffer " << buffers_.at(buffer_id).ToString()
            << " is now live.";
    memory_usage_ += AllocatedSize(buffer_id);
  }

  // TODO(b/37686934): Elementwise instructions can share the buffer of a (dead)
  // operand. Account for this potential reuse here.

  VLOG(3) << "  memory usage = " << memory_usage_;
  VLOG(10) << ToString();

  if (VLOG_IS_ON(1)) {
    DCHECK(Check());
  }
  return Status::OK();
}

Status MemoryUsageTracker::EndInstruction() {
  TF_RET_CHECK(in_progress_item_ != nullptr);
  VLOG(3) << "EndInstruction " << in_progress_item_->instruction->name();

  for (BufferId buffer_id : in_progress_item_->buffers_used) {
    Buffer& buffer = buffers_.at(buffer_id);
    buffer.unfinished_user_count--;
    CHECK_GE(buffer.unfinished_user_count, 0)
        << buffer.ToString() << " has negative unfinished use count.";
    if (buffer.unfinished_user_count == 0) {
      // Buffer is now dead.
      VLOG(3) << "  " << buffer.ToString() << " is now dead.";
      memory_usage_ -= AllocatedSize(buffer_id);
      CHECK_GE(memory_usage_, 0);
    }
  }

  // If any buffer defined by this instruction has no uses, then memory can be
  // reclaimed immediately.
  for (BufferId buffer_id : in_progress_item_->buffers_defined) {
    const Buffer& buffer = buffers_.at(buffer_id);
    if (buffer.unfinished_user_count == 0) {
      VLOG(3) << "  " << buffer.ToString() << " is immediately dead.";
      memory_usage_ -= AllocatedSize(buffer_id);
      CHECK_GE(memory_usage_, 0);
    }
  }

  in_progress_item_ = nullptr;

  VLOG(3) << "  memory usage = " << memory_usage_;
  VLOG(10) << ToString();

  if (VLOG_IS_ON(1)) {
    DCHECK(Check());
  }
  return Status::OK();
}

int64 MemoryUsageTracker::MemoryReducedIfRematerialized(Item* item) const {
  CHECK_NE(in_progress_item_, nullptr);
  if (!item->placed || item == in_progress_item_) {
    return 0;
  }

  // TODO(b/37687140): Rematerialization can increase peak memory consumption at
  // an earlier point in the program if rematerialization extends the live range
  // of the operand of the instruction being rematerialized across the live
  // range of the value of instruction being rematerialized. Don't rematerialize
  // in this case (ie, return 0 here).

  // Compute the amount of memory reduced (if any) by rematerializing
  // 'instruction'. The LogicalBuffers defined by 'instruction' will no longer
  // be live at this program point, so initially set memory_reduced to the
  // size of its defined values.
  int64 memory_reduced = 0;
  for (BufferId buffer_id : item->buffers_defined) {
    // Avoid rematerializing instructions with indirect uses as it is difficult
    // to reason about liveness after rematerializing the instruction.
    // TODO(b/37714814): Consider rematerialzing instructions with indirect
    // uses.
    if (buffers_.at(buffer_id).has_indirect_uses) {
      return 0;
    }

    if (IsCurrentlyLive(buffer_id) && !IsInUse(buffer_id)) {
      memory_reduced += AllocatedSize(buffer_id);
    }
  }

  // Account for any logical buffers whose live range must be extended across
  // this program point.
  for (BufferId buffer_id : item->buffers_used) {
    if (!IsCurrentlyLive(buffer_id)) {
      // This logical buffer is used by 'instruction' but is not live at this
      // program point. Rematerializing 'instruction' will extend the buffer's
      // live range across this program point.
      memory_reduced -= AllocatedSize(buffer_id);
    }
  }

  return memory_reduced;
}

Status MemoryUsageTracker::AddRematerializedInstruction(Item* original_item,
                                                        Item* remat_item) {
  VLOG(3) << "AddRematerializedInstruction: original_instruction = "
          << original_item->instruction->name()
          << ", remat_instruction = " << remat_item->instruction->name();

  TF_RET_CHECK(in_progress_item_ != nullptr);
  TF_RET_CHECK(original_item->placed);
  TF_RET_CHECK(!remat_item->placed);

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
    buffer.users.push_back(remat_item);
  }

  // Create a new set of Buffers defined by the new rematerialization
  // instruction. Update the internal data structures and memory use to account
  // for them.
  for (BufferId old_buffer_id : original_item->buffers_defined) {
    Buffer& old_buffer = buffers_.at(old_buffer_id);

    ItemList placed_users;
    ItemList unplaced_users;
    for (Item* user : old_buffer.users) {
      if (user->placed) {
        CHECK(IsFinished(user));
        placed_users.push_back(user);
      } else {
        unplaced_users.push_back(user);
      }
    }
    old_buffer.users = std::move(placed_users);
    old_buffer.unfinished_user_count = 0;

    // Buffer is now dead.
    memory_usage_ -= AllocatedSize(old_buffer.id);

    Buffer& new_buffer =
        RematerializeBuffer(old_buffer, remat_item, std::move(unplaced_users));

    remat_item->buffers_defined.push_back(new_buffer.id);
    for (Item* user : new_buffer.users) {
      BufferIdList& buffers_used = user->buffers_used;
      std::replace(buffers_used.begin(), buffers_used.end(), old_buffer_id,
                   new_buffer.id);
    }
  }

  VLOG(3) << "  memory usage = " << memory_usage_;
  XLA_VLOG_LINES(10, ToString());

  DCHECK(Check());

  return Status::OK();
}

string MemoryUsageTracker::ToString() const {
  string output =
      absl::StrCat("MemoryUsageTracker for ", computation_->name(), "\n");
  absl::StrAppend(&output,
                  "Memory usage: ", HumanReadableNumBytes(memory_usage()), " (",
                  memory_usage(), " bytes)");
  for (auto* item = instruction_list_.first(); item != nullptr;
       item = instruction_list_.next(item)) {
    const HloInstruction* instruction = item->instruction;
    string inprogress = item == in_progress_item_ ? " in-progress" : "";
    string placed = item->placed ? " placed" : "";
    absl::StrAppend(&output, "  ", instruction->name(), inprogress, placed,
                    "\n    Defines:\n");
    for (BufferId buffer_id : item->buffers_defined) {
      const Buffer& buffer = buffers_[buffer_id];
      string live = IsCurrentlyLive(buffer_id) ? " live" : "";
      absl::StrAppend(&output, "      ", buffer.ToString(), live, ", ",
                      buffer.unfinished_user_count, " unfinished uses\n");
    }
    absl::StrAppend(&output, "    Uses:\n");
    for (BufferId buffer_id : item->buffers_used) {
      absl::StrAppend(&output, "      ", buffers_[buffer_id].ToString(), "\n");
    }
  }
  return output;
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
        << absl::StrJoin(
               defined_buffers, ", ", [this](string* out, BufferId buffer_id) {
                 absl::StrAppend(out, buffers_.at(buffer_id).ToString());
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
        << absl::StrJoin(
               used_buffers, ", ", [this](string* out, BufferId buffer_id) {
                 absl::StrAppend(out, buffers_.at(buffer_id).ToString());
               });
  }
  for (const Buffer& buffer : buffers_) {
    int64 unfinished_uses = 0;
    for (Item* user : buffer.users) {
      const BufferIdList& used_buffers = user->buffers_used;
      CHECK(absl::c_linear_search(used_buffers, buffer.id))
          << "Instruction " << user->instruction->name()
          << " used buffers is missing " << buffer.ToString();
      if (!IsFinished(user)) {
        unfinished_uses++;
      }
    }
    CHECK_EQ(buffer.unfinished_user_count, unfinished_uses)
        << "Incorrect unplaced use count for " << buffer.ToString();
  }
  return true;
}

// Computes and returns the cost of rematerializing the given instruction.
// Cost per rematerialized instruction is defined as:
//
// memory_limit_bytes / memory_reduced
//
// The idea is to choose the operation that will save the most memory for
// rematerialization and do not worry about how much the compute costs since
// running out of memory is more harmful than taking longer to get the answer.
int64 RematerializationCost(const HloInstruction* instruction,
                            const MemoryUsageTracker& memory_tracker,
                            int64 memory_reduced, int64 memory_limit_bytes) {
  // If none of the users of 'instruction' have been placed in the sequence (as
  // tracked by memory_tracker), then rematerialization of 'instruction' is a
  // zero-cost move of 'instruction' in the sequence.
  if (!absl::c_any_of(instruction->users(),
                      [&memory_tracker](const HloInstruction* inst) {
                        return memory_tracker.IsPlaced(inst);
                      })) {
    return 0;
  }

  CHECK_GT(memory_reduced, 0);
  // Return the inverse of the benefit of rematerialization.
  return memory_limit_bytes / memory_reduced;
}

// Selects and returns the best candidate instruction for rematerialization.
// The instruction with lowest rematerialization cost is selected among those
// candidate which reduce memory use at the program point of the current
// instruction as indicated by memory_tracker. nullptr is returned if no
// candidate can be found.
Item* PickRematerializationCandidate(
    const MemoryUsageTracker& memory_tracker,
    const InstructionList& instruction_list, int64 memory_limit_bytes,
    absl::flat_hash_map<const HloInstruction*, bool>* remat_able) {
  Item* best_item = nullptr;
  int64 best_cost = 0;

  // TODO(b/35244891): This is currently quadratic in the number of HLO
  // instructions.
  for (auto* item = instruction_list.first(); item != nullptr;
       item = instruction_list.next(item)) {
    if (!item->placed) {
      // Only iterate up to the currently placed instruction.
      // We are trying to reduce memory usage at the placed
      // instruction so rematerializing later values is of no benefit.
      break;
    }
    HloInstruction* candidate = item->instruction;
    VLOG(5) << "considering rematerialization candidate " << candidate->name();

    if (item->blacklisted) {
      // Skip instructions on the blacklist to avoid infinite loops of
      // rematerializing the same instruction(s) repeatedly.
      VLOG(5) << "candidate " << candidate->name()
              << " is excluded from rematerialization";
      continue;
    }
    if (!CanBeRematerialized(candidate, remat_able)) {
      VLOG(5) << "candidate " << candidate->name()
              << " not viable: is not rematerializable";
      continue;
    }

    // If any of the candidate's control successor has been placed, we need to
    // skip this candidate. Otherwise we will violate control dependency.
    bool control_successor_placed =
        std::any_of(candidate->control_successors().begin(),
                    candidate->control_successors().end(),
                    [&memory_tracker](const HloInstruction* inst) {
                      return memory_tracker.IsPlaced(inst);
                    });

    if (control_successor_placed) {
      continue;
    }

    const int64 memory_reduced =
        memory_tracker.MemoryReducedIfRematerialized(item);

    if (memory_reduced <= 0) {
      VLOG(5) << "candidate " << candidate->name()
              << " memory reduced = " << memory_reduced << " <=  0";
      continue;
    }

    const int cost = RematerializationCost(candidate, memory_tracker,
                                           memory_reduced, memory_limit_bytes);

    VLOG(5) << "candidate " << candidate->name() << ", memory reduced "
            << memory_reduced << ", cost per byte " << cost;

    if (best_item == nullptr || cost < best_cost) {
      VLOG(5) << "candidate " << candidate->name() << " now best";
      best_item = item;
      best_cost = cost;
    }
  }
  return best_item;
}

}  // namespace

StatusOr<int64> HloRematerialization::ComputePeakMemory(
    const HloComputation* computation,
    const HloInstructionSequence& order) const {
  InstructionList instruction_list(order);
  MemoryUsageTracker tracker(computation, size_function_, *points_to_analysis_,
                             instruction_list);
  int64 peak_memory = tracker.memory_usage();
  for (auto* item = instruction_list.first(); item != nullptr;
       item = instruction_list.next(item)) {
    const HloInstruction* instruction = item->instruction;
    TF_RETURN_IF_ERROR(tracker.BeginInstruction(item));
    TF_ASSIGN_OR_RETURN(int64 callee_usage,
                        CalledComputationsMemoryUsage(instruction));
    peak_memory =
        std::max<int64>(peak_memory, tracker.memory_usage() + callee_usage);
    TF_RETURN_IF_ERROR(tracker.EndInstruction());
  }
  VLOG(1) << "Peak memory for " << computation->name() << ": "
          << HumanReadableNumBytes(peak_memory);
  return peak_memory;
}

StatusOr<int64> HloRematerialization::CalledComputationsMemoryUsage(
    const HloInstruction* instruction) const {
  const CallSite* callsite =
      call_graph_->GetNode(instruction->parent()).GetCallSite(instruction);
  if (callsite == nullptr || callsite->context() == CallContext::kParallel) {
    return 0;
  }
  int64 callee_usage = 0;
  for (const HloComputation* computation : callsite->called_computations()) {
    TF_RET_CHECK(ContainsKey(computation_peak_memory_, computation));
    callee_usage += computation_peak_memory_.at(computation);
  }
  return callee_usage;
}

StatusOr<bool> HloRematerialization::RematerializeComputation(
    HloComputation* computation, HloSchedule* schedule,
    int64 memory_limit_bytes) {
  VLOG(1) << "Rematerializing computation " << computation->name()
          << " with limit " << HumanReadableNumBytes(memory_limit_bytes);
  VLOG(1) << "peak memory usage is "
          << HumanReadableNumBytes(computation_peak_memory_.at(computation));
  CHECK(!ContainsKey(rematerialized_computations_, computation));

  InstructionList instruction_list(schedule->sequence(computation));
  MemoryUsageTracker memory_tracker(computation, size_function_,
                                    *points_to_analysis_, instruction_list);
  bool changed = false;

  // If the rematerialization makes the source instruction dead, then the
  // rematerialization is added to 'remat_move_instructions' (the
  // rematerialization is essentially a move). If the next rematerialization of
  // the instruction is also a move then the rematerialization is added to the
  // blacklist.
  absl::flat_hash_set<const HloInstruction*> remat_move_instructions;

  // The map from instructions to their rematerializable status.
  absl::flat_hash_map<const HloInstruction*, bool> remat_able;

  // The peak memory of the computation at any point in the instruction
  // sequence.
  int64 peak_memory = memory_tracker.memory_usage();

  // Total count of instructions rematerialized.
  int64 remat_count = 0;
  // Total count of clones created minus number of original rematerialized
  // instructions which are dead.
  int64 net_instructions_added = 0;

  const CallGraphNode& call_graph_node = call_graph_->GetNode(computation);

  // Iterate through all instructions in the sequence. At each instruction
  // (program point) if memory_usage exceeds the specified limit then
  // rematerialize HLO instructions until memory_usage is reduced.
  int64 instruction_index = 0;
  for (auto* item = instruction_list.first(); item != nullptr;
       item = instruction_list.next(item)) {
    const HloInstruction* instruction = item->instruction;
    TF_ASSIGN_OR_RETURN(int64 callee_usage,
                        CalledComputationsMemoryUsage(instruction));
    TF_RETURN_IF_ERROR(memory_tracker.BeginInstruction(item));

    VLOG(2) << "Program point at " << instruction->name()
            << ", memory usage = " << memory_tracker.memory_usage()
            << ", callee usage = " << callee_usage << ", [" << instruction_index
            << "/" << instruction_list.size() << "]";
    instruction_index++;

    while (memory_tracker.memory_usage() + callee_usage > memory_limit_bytes) {
      VLOG(2) << "Over memory limit at instruction " << instruction->name()
              << ", using "
              << HumanReadableNumBytes(memory_tracker.memory_usage() +
                                       callee_usage)
              << ", limit is " << HumanReadableNumBytes(memory_limit_bytes);

      Item* best_item = PickRematerializationCandidate(
          memory_tracker, instruction_list, memory_limit_bytes, &remat_able);

      if (best_item == nullptr) {
        VLOG(3) << "Unable to find rematerialization candidate at program "
                   "point "
                << instruction->name() << ". Memory usage = "
                << HumanReadableNumBytes(memory_tracker.memory_usage() +
                                         callee_usage);
        break;
      }

      HloInstruction* best = best_item->instruction;
      VLOG(1) << "Rematerializing instruction " << best->name() << " (saving "
              << HumanReadableNumBytes(
                     memory_tracker.MemoryReducedIfRematerialized(best_item))
              << ")";
      changed = true;
      remat_count++;

      HloInstruction* remat =
          computation->AddInstruction(best->Clone(/*suffix=*/"remat"));

      // Add control dependencies to the new operation.
      for (auto successor : best->control_successors()) {
        TF_RETURN_IF_ERROR(remat->AddControlDependencyTo(successor));
      }
      for (auto predecessor : best->control_predecessors()) {
        TF_RETURN_IF_ERROR(predecessor->AddControlDependencyTo(remat));
      }

      Item* remat_item = instruction_list.CreateItem(remat);

      // Replace each remaining use of 'best' with the rematerialization.
      std::vector<HloInstruction*> best_users_copy = best->users();
      for (HloInstruction* user : best_users_copy) {
        if (!memory_tracker.IsPlaced(user)) {
          VLOG(2) << "  Replacing use of " << best->name() << " in "
                  << user->name() << " with " << remat->name();
          TF_RETURN_IF_ERROR(best->ReplaceUseWith(user, remat));
        }
      }

      // Account for the rematerialization in the memory tracker.
      TF_RETURN_IF_ERROR(
          memory_tracker.AddRematerializedInstruction(best_item, remat_item));

      // Insert rematerialized instruction right before the earliest unplaced
      // use of the instruction *and* the earliest unplaced last use of any
      // operands of remat. Unplaced uses of the remat's operands are included
      // because we don't want to extend the live range of remat's operands as
      // this could increase memory usage.
      ItemList place_before;
      for (auto user : remat->users()) {
        place_before.push_back(instruction_list.GetItem(user));
      }
      for (auto* operand : remat->operands()) {
        for (auto* operand_user : operand->users()) {
          if (operand_user != remat) {
            Item* operand_user_item = instruction_list.GetItem(operand_user);
            if (!operand_user_item->placed) {
              place_before.push_back(operand_user_item);
            }
          }
        }
      }
      // Insert rematerialized instruction before any of its successors to
      // preserve ordering regarding control dependency.
      for (auto successor : remat->control_successors()) {
        Item* successor_item = instruction_list.GetItem(successor);
        // Assert to make sure we never remat an operation with control
        // successor already placed.
        CHECK(!successor_item->placed);
        place_before.push_back(successor_item);
      }
      instruction_list.InsertBeforeInstructions(remat_item, place_before);

      // If the rematerialized instruction is dead then rematerialization is
      // essentially a move. Don't delete the instruction now because we don't
      // want duplicate HloInstruction* values during the course of the
      // transformation because we keep maps with HloInstruction* values as
      // keys.
      if (best->users().empty()) {
        VLOG(2) << best->name() << " is now dead";
        if (ContainsKey(remat_move_instructions, best)) {
          // Previously, 'best' was a rematerialization which killed the
          // instruction it was a copying of. Now 'remat' is a rematerialization
          // of 'best' and kills 'best'. Stop rematerializing this instruction
          // to avoid an infinite loop.
          instruction_list.Blacklist(remat);
        }
        remat_move_instructions.insert(remat);
      } else {
        net_instructions_added++;
      }

      VLOG(1) << "memory_usage after rematerialization = "
              << HumanReadableNumBytes(memory_tracker.memory_usage());
    }

    const CallSite* callsite = call_graph_node.GetCallSite(instruction);
    if (callsite != nullptr &&
        callsite->context() == CallContext::kSequential &&
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
        if (!ContainsKey(rematerialized_computations_, called_computation)) {
          // Memory limit for the subcomputation is the memory limit less the
          // amount of memory used at this point in the computation.
          int64 subcomputation_memory_limit_bytes = std::max<int64>(
              0, memory_limit_bytes - memory_tracker.memory_usage());
          TF_ASSIGN_OR_RETURN(
              bool subcomputation_changed,
              RematerializeComputation(called_computation, schedule,
                                       subcomputation_memory_limit_bytes));
          changed |= subcomputation_changed;
        }
      }
      TF_ASSIGN_OR_RETURN(callee_usage,
                          CalledComputationsMemoryUsage(instruction));
    }

    peak_memory = std::max<int64>(peak_memory,
                                  memory_tracker.memory_usage() + callee_usage);
    VLOG(3) << "peak memory usage = " << HumanReadableNumBytes(peak_memory);

    TF_RETURN_IF_ERROR(memory_tracker.EndInstruction());
  }

  // Verify some invariants on the memory tracker.
  CHECK_EQ(memory_tracker.memory_usage(), 0);
  for (auto* instruction : computation->instructions()) {
    CHECK(memory_tracker.IsPlaced(instruction));
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

StatusOr<bool> HloRematerialization::Run(HloModule* module) {
  VLOG(1) << "HloRematerialization() with memory limit of "
          << HumanReadableNumBytes(memory_limit_bytes_);
  XLA_VLOG_LINES(3, "Before HloRematerialization:\n" + module->ToString());

  // Initialize pass object state.
  computation_peak_memory_.clear();
  rematerialized_computations_.clear();
  instructions_rematerialized_ = 0;
  net_instructions_added_ = 0;

  TF_RET_CHECK(module->has_schedule());
  TF_ASSIGN_OR_RETURN(points_to_analysis_, TuplePointsToAnalysis::Run(module));

  // Adjust memory limit to account for the output of the entry
  // computation. This is necessary because the per-computation accounting in
  // MemoryUsageTracker do not include output as these are typically allocated
  // by the caller.
  int64 module_output_size = 0;
  ShapeUtil::ForEachSubshape(
      module->result_shape(),
      [&module_output_size, this](const Shape& subshape,
                                  const ShapeIndex& /*index*/) {
        module_output_size += size_function_(subshape);
      });

  const int64 adjusted_memory_limit_bytes =
      memory_limit_bytes_ - module_output_size;
  VLOG(1) << "Adjusted memory limit accounting for output ("
          << HumanReadableNumBytes(module_output_size)
          << "): " << HumanReadableNumBytes(adjusted_memory_limit_bytes);

  // Compute peak memory usage of all computations in the module called in a
  // sequential context.
  call_graph_ = CallGraph::Build(module);
  TF_RETURN_IF_ERROR(call_graph_->VisitNodes(
      [this, module](const CallGraphNode& node) -> Status {
        if (node.context() == CallContext::kSequential) {
          TF_ASSIGN_OR_RETURN(
              computation_peak_memory_[node.computation()],
              ComputePeakMemory(node.computation(), module->schedule().sequence(
                                                        node.computation())));
        }
        return Status::OK();
      },
      /*visit_unreachable_nodes=*/false));

  // The peak memory usage of the module equals the peak memory use of the entry
  // computation plus the output size of the computation. This is because the
  // peak memory for a computation does not include the output as this is
  // typically accounted for in the caller.
  const int64 before_peak_memory =
      computation_peak_memory_.at(module->entry_computation()) +
      module_output_size;
  VLOG(1) << "Peak memory usage of module (before): "
          << HumanReadableNumBytes(before_peak_memory);

  // Subcomputations called by the entry computation will also be
  // rematerialized.
  TF_ASSIGN_OR_RETURN(
      bool changed,
      RematerializeComputation(module->entry_computation(), &module->schedule(),
                               adjusted_memory_limit_bytes));

  // Rematerialization can introduce dead code. This occurs if all uses of an
  // instruction are replaced with rematerializations of the instruction.
  TF_ASSIGN_OR_RETURN(bool dead_code_removed, HloDCE().Run(module));
  changed |= dead_code_removed;

  // After DCE, the module sequence may include instructions which no longer
  // exist.
  TF_RETURN_IF_ERROR(module->schedule().Update());
  VLOG(1) << "Rematerialized " << instructions_rematerialized_
          << " instructions in module " << module->name() << "; "
          << net_instructions_added_ << " net instructions added";
  const int64 current_peak_memory =
      computation_peak_memory_.at(module->entry_computation()) +
      module_output_size;
  VLOG(1) << "Peak memory usage of module now "
          << HumanReadableNumBytes(current_peak_memory) << " ("
          << current_peak_memory << " bytes), was "
          << HumanReadableNumBytes(before_peak_memory) << " ("
          << before_peak_memory << " bytes)";
  const int64 reduced_peak_memory = before_peak_memory - current_peak_memory;
  VLOG(1) << "Reduced peak memory by "
          << HumanReadableNumBytes(reduced_peak_memory) << " ("
          << reduced_peak_memory << " bytes)";

  if (sizes_ != nullptr) {
    sizes_->before_bytes = before_peak_memory;
    sizes_->after_bytes = current_peak_memory;
  }

  XLA_VLOG_LINES(3, "After HloRematerialization:\n" + module->ToString());

  if (current_peak_memory > memory_limit_bytes_) {
    LOG(WARNING) << absl::StrFormat(
        "Can't reduce memory use below %s (%d bytes) by rematerialization; "
        "only reduced to %s (%d bytes)",
        HumanReadableNumBytes(memory_limit_bytes_), memory_limit_bytes_,
        HumanReadableNumBytes(current_peak_memory), current_peak_memory);
  }

  return changed;
}

}  // namespace xla

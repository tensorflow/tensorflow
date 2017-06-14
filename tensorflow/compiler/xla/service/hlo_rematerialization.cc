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

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/liveness_util.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"

using ::tensorflow::strings::HumanReadableNumBytes;

namespace xla {

namespace {

// Returns true if the given instruction is rematerializable.
bool IsRematerializable(const HloInstruction* instruction) {
  // Conservatively, don't rematerialize instruction with control
  // dependencies. For one, control dependencies are added to prevent
  // interference of aliased buffers (say, in while bodies) and
  // rematerialization is ignorant of liveness and may break the intended
  // ordering.
  if (!instruction->control_predecessors().empty() ||
      !instruction->control_successors().empty()) {
    return false;
  }

  // Don't rematerialize instructions with side effects, those with a cost that
  // might not be captured by HloCostAnalysis, or instructions which cannot be
  // cloned safely.
  switch (instruction->opcode()) {
    case HloOpcode::kCall:
    case HloOpcode::kConstant:
    case HloOpcode::kCrossReplicaSum:
    case HloOpcode::kCustomCall:
    case HloOpcode::kOutfeed:
    case HloOpcode::kInfeed:
    case HloOpcode::kParameter:
    case HloOpcode::kRecv:
    case HloOpcode::kSend:
    case HloOpcode::kTrace:
    case HloOpcode::kWhile:
      return false;
    default:
      return true;
  }
}

// Class which maintains an ordered list of instructions with fast insertion
// before arbitrary elements.
class InstructionList {
 public:
  explicit InstructionList(const std::vector<const HloInstruction*> order) {
    int64 position = 0;
    for (const HloInstruction* inst : order) {
      instructions_.push_back(const_cast<HloInstruction*>(inst));
      instruction_iterators_.insert({const_cast<HloInstruction*>(inst),
                                     std::next(instructions_.end(), -1)});
      // Initially position numbers are uniquely assigned in order. Later as
      // instructions are added with InsertBefore* methods, some instructions
      // may have duplicate position numbers, but the values will be guaranteed
      // to be monotonically increasing through the list, and so is still useful
      // for quickly(-ish) determining the order of arbitrary instructions in
      // the list.
      position_number_[inst] = position;
      first_at_position_[position] = inst;
      position++;
    }
  }

  // Returns the list of instructions.
  const std::list<HloInstruction*>& instructions() const {
    return instructions_;
  }

  // Insert instruction 'to_insert' immediately before instruction 'before' in
  // the list.
  void InsertBefore(HloInstruction* to_insert, HloInstruction* before) {
    VLOG(3) << "InsertBefore: " << to_insert->name() << " before "
            << before->name();
    auto it = instruction_iterators_.find(before);
    CHECK(it != instruction_iterators_.end());
    instruction_iterators_.insert(
        {to_insert, instructions_.insert(it->second, to_insert)});
    // Assign the same position number to the newly added instruction as
    // 'before'. This guarantees monotonicity of the position numbers, but not
    // uniqueness.
    int64 pos = position_number_.at(before);
    position_number_[to_insert] = pos;
    if (first_at_position_.at(pos) == before) {
      first_at_position_[pos] = to_insert;
    }
  }

  // Insert instruction 'to_insert' immediately before the earliest instruction
  // in 'before_instructions'.
  void InsertBeforeInstructions(
      HloInstruction* to_insert,
      tensorflow::gtl::ArraySlice<HloInstruction*> before_instructions) {
    VLOG(3) << "InsertBeforeInstructions: " << to_insert->name() << " before {"
            << tensorflow::str_util::Join(
                   before_instructions, ", ",
                   [](string* out, HloInstruction* inst) {
                     tensorflow::strings::StrAppend(out, inst->name());
                   })
            << "}";

    // Find the minimal position number of any instruction in
    // 'before_instructions'.
    CHECK(!before_instructions.empty());
    int64 min_position_number = std::numeric_limits<int64>::max();
    for (const HloInstruction* instruction : before_instructions) {
      min_position_number =
          std::min(min_position_number, position_number_.at(instruction));
    }

    // Because more than one instruction in 'before_instructions' may have a
    // position number of 'min_position_number', find the first such instruction
    // with position number 'min_position_number'.
    for (auto it = instruction_iterators_.at(
             first_at_position_.at(min_position_number));
         it != instructions_.end() &&
         position_number_.at(*it) == min_position_number;
         ++it) {
      if (std::find(before_instructions.begin(), before_instructions.end(),
                    *it) != before_instructions.end()) {
        return InsertBefore(to_insert, *it);
      }
    }
    LOG(FATAL) << "Expected to find instruction in before_instructions with "
                  "position number "
               << min_position_number;
  }

 private:
  // List of instructions.
  std::list<HloInstruction*> instructions_;

  // Iterators for each instruction in the list.
  tensorflow::gtl::FlatMap<const HloInstruction*,
                           std::list<HloInstruction*>::iterator>
      instruction_iterators_;

  // A number assigned to each instruction which increases monotonically through
  // 'instructions_'. Used to facilitate fast insertion of an instruction before
  // the earliest instruction in a set of instructions
  // (InsertBeforeInstructions) by enabling fast-ish ordering queries between
  // instructions. If position_number_[a] < position_number_[b] then 'a' comes
  // before 'b' in the list. If the position numbers are the same then nothing
  // can be said about their order without examining the list.
  //
  // On object construction this value is precisely the instruction's ordinal
  // position in the list. Instructions inserted via InsertBefore receive
  // duplicate values. However, monotonicity is preserved.
  tensorflow::gtl::FlatMap<const HloInstruction*, int64> position_number_;

  // The first instruction in the list assigned a particular position number.
  tensorflow::gtl::FlatMap<int64, const HloInstruction*> first_at_position_;
};

// Return the HloInstructions which use the given LogicalBuffer. Sets
// has_indirect_users to whether any of the uses is indirect. A use is indirect
// if the instruction defining logical_buffer is not an operand of the use. This
// can happen via buffer aliasing (eg, tuples).
std::vector<const HloInstruction*> GetUsers(
    const LogicalBuffer* logical_buffer,
    const TuplePointsToAnalysis& points_to_analysis, bool* has_indirect_users) {
  std::vector<const HloInstruction*> users;
  // To identify uses iterate through all HloInstruction users of the
  // BufferAliases of the logical buffer.
  *has_indirect_users = false;
  for (const BufferAlias& buffer_alias :
       points_to_analysis.GetBufferAliases(*logical_buffer)) {
    for (const HloInstruction* user : buffer_alias.instruction()->users()) {
      if (DoesNotUseOperandBuffer(buffer_alias.instruction(),
                                  buffer_alias.index(), user,
                                  points_to_analysis)) {
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
      if (std::find(users.begin(), users.end(), user) == users.end()) {
        users.push_back(user);
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
  Status BeginInstruction(const HloInstruction* instruction);

  // Finishes the placement of the current instruction. This frees any dead
  // operands or dead result of the instruction. This must be called after
  // each call to BeginInstruction.
  Status EndInstruction();

  // Returns the number of bytes that the current memory usage will be reduced
  // if the given instruction is rematerialized.
  int64 MemoryReducedIfRematerialized(const HloInstruction* instruction) const;

  // Adjusts memory usage to account for the rematerialization of
  // original_instruction for all remaining unplaced uses. The rematerialization
  // is remat_instruction. This method should be called after the HLO graph has
  // been transformed (rematerialization instruction created and connected to
  // uses).
  Status AddRematerializedInstruction(HloInstruction* original_instruction,
                                      HloInstruction* remat_instruction);

  // Returns whether the given instruction has been placed (BeginInstruction
  // has been called with 'instruction' as the argument).
  bool IsPlaced(const HloInstruction* instruction) const {
    return ContainsKey(placed_instructions_, instruction);
  }

  // Returns the current memory usage. This is the sum of sizes of all live
  // values.
  int64 memory_usage() const { return memory_usage_; }

  // Returns the current instruction being placed.
  const HloInstruction* in_progress_instruction() const {
    return in_progress_instruction_;
  }

  // Check invariants of the data structure. This is expensive to call.
  bool Check() const;

  string ToString() const;

 private:
  // Type holding a unique identifier for each Buffer object.
  using BufferId = int64;

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
    const HloInstruction* defining_instruction;

    // The materialized size of the buffer in bytes.
    const int64 size;

    // Whether this buffer is live-out of the computation.
    bool live_out;

    // Whether this buffer has indirect uses. Ie, an instruction which is not a
    // user of defining_instruction uses this buffer. This can occur due to
    // buffer aliasing (eg, tuples).
    bool has_indirect_uses;

    // The instructions which use this buffer.
    std::vector<const HloInstruction*> users;

    // The number of users (HloInstructions) of this buffer which have not yet
    // been placed in the sequence.
    int64 unfinished_user_count;

    string ToString() const {
      return tensorflow::strings::StrCat("Buffer ", id, " (defined by ",
                                         defining_instruction->name(),
                                         ", size ", size, " bytes)");
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
    std::vector<const HloInstruction*> users =
        GetUsers(logical_buffer, points_to_analysis, &has_indirect_uses);
    return NewBuffer(logical_buffer->instruction(),
                     size_function(logical_buffer->shape()), std::move(users),
                     live_out, has_indirect_uses);
  }

  // Create a new buffer representing a rematerialization of given buffer for
  // the given uses.
  Buffer& RematerializeBuffer(
      const Buffer& original_buffer, const HloInstruction* remat_instruction,
      std::vector<const HloInstruction*>&& rematerialized_uses) {
    CHECK(IsPlaced(original_buffer.defining_instruction));
    CHECK(!original_buffer.has_indirect_uses);
    CHECK(!original_buffer.live_out);
    for (const HloInstruction* use : rematerialized_uses) {
      CHECK(!IsPlaced(use));
    }
    return NewBuffer(remat_instruction, original_buffer.size,
                     std::move(rematerialized_uses), /*live_out=*/false,
                     /*has_indirect_uses=*/false);
  }

  // Return number of bytes allocated for the buffer with the given id. Buffers
  // allocated by the calling computation (eg, parameter and output buffers) are
  // considered to have zero bytes because the memory is accounted for in a
  // different computation.
  int64 AllocatedSize(BufferId buffer_id) const {
    const Buffer& buffer = buffers_.at(buffer_id);
    HloOpcode def_opcode = buffer.defining_instruction->opcode();
    if (buffer.live_out || def_opcode == HloOpcode::kParameter) {
      return 0;
    } else {
      return buffer.size;
    }
  }

  // Returns true if BeginInstruction and EndInstruction has been called for the
  // given instruction.
  bool IsFinished(const HloInstruction* instruction) const {
    return IsPlaced(instruction) && instruction != in_progress_instruction_;
  }

  // Returns whether the given buffer is being used by the in-progress
  // instruction.
  bool IsInUse(BufferId buffer_id) const {
    if (in_progress_instruction_ == nullptr) {
      return false;
    }
    const std::vector<BufferId>& in_progress_uses =
        buffers_used_by_instruction_.at(in_progress_instruction_);
    return std::find(in_progress_uses.begin(), in_progress_uses.end(),
                     buffer_id) != in_progress_uses.end();
  }

  // Returns whether the given instruction is live at the current program
  // point.
  bool IsCurrentlyLive(BufferId buffer_id) const {
    const Buffer& buffer = buffers_[buffer_id];
    return (IsPlaced(buffer.defining_instruction) &&
            buffer.unfinished_user_count > 0);
  }

  // Create a new buffer, add it to buffers_, and return a reference.
  Buffer& NewBuffer(const HloInstruction* defining_instruction, int64 size,
                    std::vector<const HloInstruction*>&& users, bool live_out,
                    bool has_indirect_uses) {
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
  const HloInstruction* in_progress_instruction_ = nullptr;

  // The buffers defined by each instruction.
  std::unordered_map<const HloInstruction*, std::vector<BufferId>>
      buffers_defined_by_instruction_;

  // The buffers used by each instruction.
  std::unordered_map<const HloInstruction*, std::vector<BufferId>>
      buffers_used_by_instruction_;

  // The set of instructions which have been placed. That is, BeginInstruction
  // has been called with the instruction as an argument.
  tensorflow::gtl::FlatSet<const HloInstruction*> placed_instructions_;

  // All buffers in the computation.
  std::vector<Buffer> buffers_;
};

MemoryUsageTracker::MemoryUsageTracker(
    const HloComputation* computation,
    const HloRematerialization::ShapeSizeFunction& size_function,
    const TuplePointsToAnalysis& points_to_analysis,
    const InstructionList& instruction_list)
    : computation_(computation), instruction_list_(instruction_list) {
  // Iterate through all LogicalBuffers in the computation and gather the
  // instructions which define them in buffers_defined_by_instruction_ and the
  // instructions which use them in buffers_used_by_instruction_.
  for (auto& instruction : computation_->instructions()) {
    // Initialize empty vectors for defs and uses of each instruction.
    buffers_used_by_instruction_[instruction.get()];
    buffers_defined_by_instruction_[instruction.get()];
  }

  tensorflow::gtl::FlatSet<const LogicalBuffer*> live_out_set =
      points_to_analysis.GetPointsToSet(computation_->root_instruction())
          .CreateFlattenedSet();
  tensorflow::gtl::FlatMap<const LogicalBuffer*, BufferId>
      logical_buffer_to_buffer_id;

  for (const HloInstruction* instruction : instruction_list_.instructions()) {
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
        for (const HloInstruction* user :
             GetUsers(logical_buffer, points_to_analysis, &unused)) {
          if (std::find(buffer->users.begin(), buffer->users.end(), user) ==
              buffer->users.end()) {
            buffer->users.push_back(user);
            buffer->unfinished_user_count++;
            buffers_used_by_instruction_.at(user).push_back(buffer->id);
          }
        }
      } else {
        buffer = &CreateBufferFromLogicalBuffer(
            logical_buffer, points_to_analysis, size_function,
            ContainsKey(live_out_set, logical_buffer));
        buffers_defined_by_instruction_.at(instruction).push_back(buffer->id);
        for (const HloInstruction* user : buffer->users) {
          buffers_used_by_instruction_.at(user).push_back(buffer->id);
        }
      }

      logical_buffer_to_buffer_id[logical_buffer] = buffer->id;
    }
  }
  XLA_VLOG_LINES(10, ToString());
  DCHECK(Check());
}

Status MemoryUsageTracker::BeginInstruction(const HloInstruction* instruction) {
  VLOG(3) << "BeginInstruction " << instruction->name();
  TF_RET_CHECK(in_progress_instruction_ == nullptr);
  in_progress_instruction_ = instruction;

  placed_instructions_.insert(in_progress_instruction_);

  // All buffers defined by this instruction need memory.
  for (BufferId buffer_id : buffers_defined_by_instruction_.at(instruction)) {
    VLOG(3) << "  Buffer " << buffers_.at(buffer_id).ToString()
            << " is now live.";
    memory_usage_ += AllocatedSize(buffer_id);
  }

  // TODO(b/37686934): Elementwise instructions can share the buffer of a (dead)
  // operand. Account for this potential reuse here.

  VLOG(3) << "  memory usage = " << memory_usage_;
  VLOG(10) << ToString();

  DCHECK(Check());
  return Status::OK();
}

Status MemoryUsageTracker::EndInstruction() {
  TF_RET_CHECK(in_progress_instruction_ != nullptr);
  VLOG(3) << "EndInstruction " << in_progress_instruction_->name();

  for (BufferId buffer_id :
       buffers_used_by_instruction_.at(in_progress_instruction_)) {
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
  for (BufferId buffer_id :
       buffers_defined_by_instruction_.at(in_progress_instruction_)) {
    const Buffer& buffer = buffers_.at(buffer_id);
    if (buffer.unfinished_user_count == 0) {
      VLOG(3) << "  " << buffer.ToString() << " is immediately dead.";
      memory_usage_ -= AllocatedSize(buffer_id);
      CHECK_GE(memory_usage_, 0);
    }
  }

  in_progress_instruction_ = nullptr;

  VLOG(3) << "  memory usage = " << memory_usage_;
  VLOG(10) << ToString();

  DCHECK(Check());

  return Status::OK();
}

int64 MemoryUsageTracker::MemoryReducedIfRematerialized(
    const HloInstruction* instruction) const {
  CHECK_NE(in_progress_instruction_, nullptr);
  if (!IsPlaced(instruction) || instruction == in_progress_instruction_) {
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
  for (BufferId buffer_id : buffers_defined_by_instruction_.at(instruction)) {
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
  for (BufferId buffer_id : buffers_used_by_instruction_.at(instruction)) {
    if (!IsCurrentlyLive(buffer_id)) {
      // This logical buffer is used by 'instruction' but is not live at this
      // program point. Rematerializing 'instruction' will extend the buffer's
      // live range across this program point.
      memory_reduced -= AllocatedSize(buffer_id);
    }
  }

  return memory_reduced;
}

Status MemoryUsageTracker::AddRematerializedInstruction(
    HloInstruction* original_instruction, HloInstruction* remat_instruction) {
  VLOG(3) << "AddRematerializedInstruction: original_instruction = "
          << original_instruction->name()
          << ", remat_instruction = " << remat_instruction->name();

  TF_RET_CHECK(in_progress_instruction_ != nullptr);
  TF_RET_CHECK(IsPlaced(original_instruction));
  TF_RET_CHECK(!IsPlaced(remat_instruction));
  CHECK(!ContainsKey(buffers_defined_by_instruction_, remat_instruction));
  CHECK(!ContainsKey(buffers_used_by_instruction_, remat_instruction));

  // Construct the list of buffers used and defined by the rematerialization.
  buffers_defined_by_instruction_[remat_instruction];
  buffers_used_by_instruction_[remat_instruction] =
      buffers_used_by_instruction_.at(original_instruction);

  // Account for the additional buffer uses created by the new rematerialization
  // instruction. Update memory usage if the rematerialization makes a dead
  // buffer live again.
  for (BufferId buffer_id :
       buffers_used_by_instruction_.at(original_instruction)) {
    Buffer& buffer = buffers_.at(buffer_id);
    if (buffer.unfinished_user_count == 0) {
      // Buffer used by this instruction was dead, now is alive.
      memory_usage_ += AllocatedSize(buffer.id);
    }

    buffer.unfinished_user_count++;
    buffer.users.push_back(remat_instruction);
  }

  // Create a new set of Buffers defined by the new rematerialization
  // instruction. Update the internal data structures and memory use to account
  // for them.
  for (BufferId old_buffer_id :
       buffers_defined_by_instruction_.at(original_instruction)) {
    Buffer& old_buffer = buffers_.at(old_buffer_id);

    std::vector<const HloInstruction*> placed_users;
    std::vector<const HloInstruction*> unplaced_users;
    for (const HloInstruction* user : old_buffer.users) {
      if (IsPlaced(user)) {
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

    Buffer& new_buffer = RematerializeBuffer(old_buffer, remat_instruction,
                                             std::move(unplaced_users));

    buffers_defined_by_instruction_.at(remat_instruction)
        .push_back(new_buffer.id);
    for (const HloInstruction* user : new_buffer.users) {
      std::vector<BufferId>& buffers_used =
          buffers_used_by_instruction_.at(user);
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
  string output = tensorflow::strings::StrCat("MemoryUsageTracker for ",
                                              computation_->name(), "\n");
  tensorflow::strings::StrAppend(
      &output, "Memory usage: ", HumanReadableNumBytes(memory_usage()), " (",
      memory_usage(), " bytes)");
  for (const HloInstruction* instruction : instruction_list_.instructions()) {
    string inprogress =
        instruction == in_progress_instruction_ ? " in-progress" : "";
    string placed = IsPlaced(instruction) ? " placed" : "";
    tensorflow::strings::StrAppend(&output, "  ", instruction->name(),
                                   inprogress, placed, "\n    Defines:\n");
    for (BufferId buffer_id : buffers_defined_by_instruction_.at(instruction)) {
      const Buffer& buffer = buffers_[buffer_id];
      string live = IsCurrentlyLive(buffer_id) ? " live" : "";
      tensorflow::strings::StrAppend(&output, "      ", buffer.ToString(), live,
                                     ", ", buffer.unfinished_user_count,
                                     " unfinished uses\n");
    }
    tensorflow::strings::StrAppend(&output, "    Uses:\n");
    for (BufferId buffer_id : buffers_used_by_instruction_.at(instruction)) {
      tensorflow::strings::StrAppend(&output, "      ",
                                     buffers_[buffer_id].ToString(), "\n");
    }
  }
  return output;
}

bool MemoryUsageTracker::Check() const {
  auto elements_are_unique = [](const std::vector<BufferId>& vec) {
    return vec.size() == std::set<BufferId>(vec.begin(), vec.end()).size();
  };

  // Verify buffers_defined_by_instruction_.
  for (auto& instruction : computation_->instructions()) {
    const std::vector<BufferId>& defined_buffers =
        buffers_defined_by_instruction_.at(instruction.get());
    CHECK(elements_are_unique(defined_buffers))
        << "Instruction " << instruction->name()
        << " does not have unique defined buffers: "
        << tensorflow::str_util::Join(
               defined_buffers, ", ", [this](string* out, BufferId buffer_id) {
                 tensorflow::strings::StrAppend(
                     out, buffers_.at(buffer_id).ToString());
               });

    for (const Buffer& buffer : buffers_) {
      if (buffer.defining_instruction == instruction.get()) {
        CHECK(std::find(defined_buffers.begin(), defined_buffers.end(),
                        buffer.id) != defined_buffers.end())
            << "Instruction " << instruction->name()
            << " defined buffers is missing: " << buffer.ToString();
      }
    }
  }

  // Verify buffers_used_by_instruction_.
  for (auto& instruction : computation_->instructions()) {
    const std::vector<BufferId>& used_buffers =
        buffers_used_by_instruction_.at(instruction.get());
    CHECK(elements_are_unique(used_buffers))
        << "Instruction " << instruction->name()
        << " does not have unique used buffers: "
        << tensorflow::str_util::Join(
               used_buffers, ", ", [this](string* out, BufferId buffer_id) {
                 tensorflow::strings::StrAppend(
                     out, buffers_.at(buffer_id).ToString());
               });
  }
  for (const Buffer& buffer : buffers_) {
    int64 unfinished_uses = 0;
    for (const HloInstruction* user : buffer.users) {
      const std::vector<BufferId>& used_buffers =
          buffers_used_by_instruction_.at(user);
      CHECK(std::find(used_buffers.begin(), used_buffers.end(), buffer.id) !=
            used_buffers.end())
          << "Instruction " << user->name() << " used buffers is missing "
          << buffer.ToString();
      if (!IsFinished(user)) {
        unfinished_uses++;
      }
    }
    CHECK_EQ(buffer.unfinished_user_count, unfinished_uses)
        << "Incorrect unplaced use count for " << buffer.ToString();
  }

  // Verify live set size against memory_usage_.
  int64 live_size = 0;
  for (const Buffer& buffer : buffers_) {
    // The while instruction reuses its input buffers as output buffers so
    // don't double count its buffers if it is currently executing.
    if (IsCurrentlyLive(buffer.id) &&
        !(buffer.defining_instruction == in_progress_instruction_ &&
          in_progress_instruction_->opcode() == HloOpcode::kWhile)) {
      live_size += AllocatedSize(buffer.id);
    }
  }
  CHECK(live_size == memory_usage_)
      << "Live set size " << live_size << " is not same as memory usage "
      << memory_usage_
      << ". This could happen if some nodes defined in the "
         "computation are not being used/executed.";

  return true;
}

// Computes and returns the cost of rematerializing the given instruction.
// Cost per rematerialized instruction is defined as:
//
// (flop_count + transcendental_count + element_count) / memory_reduced
//
//   flop_count: from HloCostAnalysis
//   transcendental_count: from HloCostAnalysis
//   element_count: number of elements accessed in operands and output of
//     instruction
//   memory_reduced: The memory usage reduced by rematerializing the
//     instruction.
//
// This is a rough estimate of the extra execution time per byte saved by
// rematerializing this instruction for its remaining uses. In general, we
// want the most memory saving for the least latency penalty which is captured
// by this heuristic.
int64 RematerializationCost(const HloInstruction* instruction,
                            const MemoryUsageTracker& memory_tracker,
                            const HloCostAnalysis& cost_analysis,
                            int64 memory_reduced) {
  // If none of the users of 'instruction' have been placed in the sequence (as
  // tracked by memory_tracker), then rematerialization of 'instruction' is a
  // zero-cost move of 'instruction' in the sequence.
  if (!std::any_of(instruction->users().begin(), instruction->users().end(),
                   [&memory_tracker](const HloInstruction* inst) {
                     return memory_tracker.IsPlaced(inst);
                   })) {
    return 0;
  }

  CHECK_GT(memory_reduced, 0);
  const int64 bytes_accessed = cost_analysis.bytes_accessed(*instruction);
  const int64 elements_accessed =
      ShapeUtil::IsTuple(instruction->shape())
          ? bytes_accessed
          : bytes_accessed / ShapeUtil::ByteSizeOfPrimitiveType(
                                 instruction->shape().element_type());

  // Multiply by 256 to improve precision of cost. Without this factor,
  // many instructions such as many elementwise instructions would have
  // zero cost because the bytes reduced can be several times greater than
  // the element count.
  return 256 *
         (cost_analysis.flop_count(*instruction) +
          cost_analysis.transcendental_count(*instruction) +
          elements_accessed) /
         memory_reduced;
}

// Selects and returns the best candidate instruction for rematerialization.
// The instruction with lowest rematerialization cost is selected among those
// candidate which reduce memory use at the program point of the current
// instruction as indicated by memory_tracker. nullptr is returned if no
// candidate can be found.
HloInstruction* PickRematerializationCandidate(
    const MemoryUsageTracker& memory_tracker,
    const InstructionList& instruction_list,
    const HloCostAnalysis& cost_analysis,
    const tensorflow::gtl::FlatSet<const HloInstruction*>& blacklist) {
  HloInstruction* best = nullptr;
  int64 best_cost = 0;

  // TODO(b/35244891): This is currently quadratic in the number of HLO
  // instructions.
  for (HloInstruction* candidate : instruction_list.instructions()) {
    if (!memory_tracker.IsPlaced(candidate)) {
      // Only iterate up to the currently placed instruction as indicated by
      // memory_tracker. We are trying to reduce memory usage at the placed
      // instruction so rematerializing later values is of no benefit.
      break;
    }
    VLOG(5) << "considering rematerialization candidate " << candidate->name();

    if (ContainsKey(blacklist, candidate)) {
      // Skip instructions on the blacklist to avoid infinite loops of
      // rematerializing the same instruction(s) repeatedly.
      VLOG(5) << "candidate " << candidate->name()
              << " is excluded from rematerialization";
      continue;
    }

    if (!IsRematerializable(candidate)) {
      VLOG(5) << "candidate " << candidate->name()
              << " not viable: is not rematerializable";
      continue;
    }

    const int64 memory_reduced =
        memory_tracker.MemoryReducedIfRematerialized(candidate);

    if (memory_reduced <= 0) {
      VLOG(5) << "candidate " << candidate->name()
              << " memory reduced = " << memory_reduced << " <= 0";
      continue;
    }

    const int cost = RematerializationCost(candidate, memory_tracker,
                                           cost_analysis, memory_reduced);

    VLOG(5) << "candidate " << candidate->name() << ", memory reduced "
            << memory_reduced << ", cost per byte " << cost;

    if (best == nullptr || cost < best_cost) {
      VLOG(5) << "candidate " << candidate->name() << " now best";
      best = candidate;
      best_cost = cost;
    }
  }
  return best;
}

}  // namespace

StatusOr<int64> HloRematerialization::ComputePeakMemory(
    const HloComputation* computation,
    const std::vector<const HloInstruction*>& order) const {
  InstructionList instruction_list(order);
  MemoryUsageTracker tracker(computation, size_function_, *points_to_analysis_,
                             instruction_list);
  int64 peak_memory = tracker.memory_usage();
  for (const HloInstruction* instruction : order) {
    TF_RETURN_IF_ERROR(tracker.BeginInstruction(instruction));
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
    HloComputation* computation,
    SequentialHloOrdering::HloModuleSequence* sequence,
    int64 memory_limit_bytes) {
  VLOG(1) << "Rematerializing computation " << computation->name()
          << " with limit " << HumanReadableNumBytes(memory_limit_bytes);
  VLOG(1) << "peak memory usage is "
          << HumanReadableNumBytes(computation_peak_memory_.at(computation));
  CHECK(!ContainsKey(rematerialized_computations_, computation));

  InstructionList instruction_list(sequence->at(computation));
  MemoryUsageTracker memory_tracker(computation, size_function_,
                                    *points_to_analysis_, instruction_list);
  bool changed = false;

  // To avoid an infinite loop rematerializing the same set of instructions ad
  // infinitum, keep a blacklist of instructions which should not be
  // rematerialized.
  tensorflow::gtl::FlatSet<const HloInstruction*> blacklist;

  // If the rematerialization makes the source instruction dead, then the
  // rematerialization is added to 'remat_move_instructions' (the
  // rematerialization is essentially a move). If the next rematerialization of
  // the instruction is also a move then the rematerialization is added to the
  // blacklist.
  tensorflow::gtl::FlatSet<const HloInstruction*> remat_move_instructions;

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
  for (auto list_it = instruction_list.instructions().begin();
       list_it != instruction_list.instructions().end(); ++list_it) {
    HloInstruction* instruction = *list_it;
    TF_ASSIGN_OR_RETURN(int64 callee_usage,
                        CalledComputationsMemoryUsage(instruction));
    TF_RETURN_IF_ERROR(memory_tracker.BeginInstruction(instruction));

    VLOG(2) << "Program point at " << instruction->name()
            << ", memory usage = " << memory_tracker.memory_usage()
            << ", callee usage = " << callee_usage << ", [" << instruction_index
            << "/" << instruction_list.instructions().size() << "]";
    instruction_index++;

    while (memory_tracker.memory_usage() + callee_usage > memory_limit_bytes) {
      VLOG(2) << "Over memory limit at instruction " << instruction->name()
              << ", using "
              << HumanReadableNumBytes(memory_tracker.memory_usage() +
                                       callee_usage)
              << ", limit is " << HumanReadableNumBytes(memory_limit_bytes);

      HloInstruction* best = PickRematerializationCandidate(
          memory_tracker, instruction_list, cost_analysis_, blacklist);

      if (best == nullptr) {
        VLOG(3) << "Unable to find rematerialization candidate at program "
                   "point "
                << instruction->name() << ". Memory usage = "
                << HumanReadableNumBytes(memory_tracker.memory_usage() +
                                         callee_usage);
        break;
      }

      VLOG(1) << "Rematerializing instruction " << best->name() << " (saving "
              << memory_tracker.MemoryReducedIfRematerialized(best) << ")";
      changed = true;
      remat_count++;

      HloInstruction* remat =
          computation->AddInstruction(best->Clone(/*suffix=*/"remat"));

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
          memory_tracker.AddRematerializedInstruction(best, remat));

      // Insert rematerialized instruction right before the earliest unplaced
      // use of the instruction *and* the earliest unplaced last use of any
      // operands of remat. Unplaced uses of the remat's operands are included
      // because we don't want to extend the live range of remat's operands as
      // this could increase memory usage.
      std::vector<HloInstruction*> place_before = remat->users();
      for (auto* operand : remat->operands()) {
        for (auto* operand_user : operand->users()) {
          if (!memory_tracker.IsPlaced(operand_user) && operand_user != remat) {
            place_before.push_back(operand_user);
          }
        }
      }
      instruction_list.InsertBeforeInstructions(remat, place_before);

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
          blacklist.insert(remat);
        }
        remat_move_instructions.insert(remat);
      } else {
        net_instructions_added++;
      }

      VLOG(3) << "memory_usage after rematerialization = "
              << memory_tracker.memory_usage();
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
              RematerializeComputation(called_computation, sequence,
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
  for (auto& instruction : computation->instructions()) {
    CHECK(memory_tracker.IsPlaced(instruction.get()));
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
  sequence->at(computation)
      .assign(instruction_list.instructions().begin(),
              instruction_list.instructions().end());

  rematerialized_computations_.insert(computation);

  instructions_rematerialized_ += remat_count;
  net_instructions_added_ += net_instructions_added;

  return changed;
}

StatusOr<bool> HloRematerialization::Run(
    HloModule* module, SequentialHloOrdering::HloModuleSequence* sequence,
    int64 memory_limit_bytes) {
  // The sequence is constructed entirely by this method.
  TF_RET_CHECK(sequence->empty());

  VLOG(1) << "HloRematerialization() with memory limit of "
          << HumanReadableNumBytes(memory_limit_bytes);

  TF_ASSIGN_OR_RETURN(points_to_analysis_, TuplePointsToAnalysis::Run(module));

  // Adjust memory limit to account for the output of the entry
  // computation. This is necessary because the per-computation accounting in
  // MemoryUsageTracker do not include output as these are typically allocated
  // by the caller.
  int64 module_output_size = 0;
  ShapeUtil::ForEachSubshape(
      module->entry_computation()->root_instruction()->shape(),
      [&module_output_size, this](const Shape& subshape,
                                  const ShapeIndex& /*index*/) {
        module_output_size += size_function_(subshape);
      });

  const int64 adjusted_memory_limit_bytes =
      memory_limit_bytes - module_output_size;
  VLOG(1) << "Adjusted memory limit accounting for output ("
          << HumanReadableNumBytes(module_output_size)
          << "): " << HumanReadableNumBytes(adjusted_memory_limit_bytes);

  XLA_VLOG_LINES(3, "Before HloRematerialization:\n" + module->ToString());
  // Create initial sequence of HLO instructions.
  TF_ASSIGN_OR_RETURN(*sequence,
                      CreateMemoryMinimizingSequence(
                          *module, [this](const LogicalBuffer& buffer) {
                            return size_function_(buffer.shape());
                          }));
  // Compute peak memory usage of all computations in the module called in a
  // sequential context.
  call_graph_ = CallGraph::Build(module);
  TF_RETURN_IF_ERROR(call_graph_->VisitNodes(
      [this, sequence](const CallGraphNode& node) -> Status {
        if (node.context() == CallContext::kSequential) {
          TF_ASSIGN_OR_RETURN(
              computation_peak_memory_[node.computation()],
              ComputePeakMemory(node.computation(),
                                sequence->at(node.computation())));
        }
        return Status::OK();
      }));

  // The peak memory usage of the module equals the peak memory use of the entry
  // computation plus the output size of the computation. This is because the
  // peak memory for a computation does not include the output as this is
  // typically accounted for in the caller.
  const int64 before_peak_memory =
      computation_peak_memory_.at(module->entry_computation()) +
      module_output_size;
  VLOG(1) << "Peak memory usage of module (before): "
          << HumanReadableNumBytes(before_peak_memory);

  // Run cost analysis. Operation cost is used in the heuristic for selecting
  // instructions for rematerialization.
  TF_RETURN_IF_ERROR(
      module->entry_computation()->root_instruction()->Accept(&cost_analysis_));

  // Subcomputations called by the entry computation will also be
  // rematerialized.
  TF_ASSIGN_OR_RETURN(bool changed, RematerializeComputation(
                                        module->entry_computation(), sequence,
                                        adjusted_memory_limit_bytes));

  // Rematerialization can introduce dead code. This occurs if all uses of an
  // instruction are replaced with rematerializations of the instruction.
  TF_ASSIGN_OR_RETURN(bool dead_code_removed, HloDCE().Run(module));
  changed |= dead_code_removed;

  // After DCE, the module sequence may include instructions which no longer
  // exist.
  for (const auto& computation : module->computations()) {
    if (sequence->at(computation.get()).size() !=
        computation->instruction_count()) {
      // A size mismatch between the computation instruction count and the size
      // of the ordering of instructions can only be caused by DCE. Rebuild the
      // order by removing the deleted instructions from the order.
      tensorflow::gtl::FlatSet<const HloInstruction*> instruction_set;
      for (const auto& instruction : computation->instructions()) {
        instruction_set.insert(instruction.get());
      }
      // Move the old order into a temporary vector, then build new order
      // inplace.
      std::vector<const HloInstruction*>& order =
          sequence->at(computation.get());
      std::vector<const HloInstruction*> old_order;
      using std::swap;
      swap(order, old_order);
      std::copy_if(old_order.begin(), old_order.end(),
                   std::back_inserter(order),
                   [&instruction_set](const HloInstruction* instruction) {
                     return ContainsKey(instruction_set, instruction);
                   });
      TF_RET_CHECK(sequence->at(computation.get()).size() ==
                   computation->instruction_count());
    }
  }
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

  XLA_VLOG_LINES(3, "After HloRematerialization:\n" + module->ToString());

  if (current_peak_memory > memory_limit_bytes) {
    LOG(WARNING) << "Can't reduce memory use below "
                 << HumanReadableNumBytes(memory_limit_bytes)
                 << " by rematerialization (only reduced to "
                 << HumanReadableNumBytes(current_peak_memory) << ")";
  }

  return changed;
}

/* static */ StatusOr<bool> HloRematerialization::RematerializeAndSchedule(
    const HloRematerialization::ShapeSizeFunction& size_function,
    int64 memory_limit_bytes, HloModule* hlo_module,
    SequentialHloOrdering::HloModuleSequence* sequence) {
  HloRematerialization remat(size_function);
  return remat.Run(hlo_module, sequence, memory_limit_bytes);
}

}  // namespace xla

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
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"
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

// Returns a vector of the operands of 'instruction' with repeated elements
// removed.
std::vector<HloInstruction*> UniqueOperands(const HloInstruction* instruction) {
  std::vector<HloInstruction*> unique_operands;
  for (HloInstruction* operand : instruction->operands()) {
    if (std::find(unique_operands.begin(), unique_operands.end(), operand) ==
        unique_operands.end()) {
      unique_operands.push_back(operand);
    }
  }
  return unique_operands;
}

// Returns true if the given instruction is rematerializable.
bool IsRematerializable(const HloInstruction* instruction) {
  // Don't rematerialize instructions with side effects, those with a cost that
  // might not be captured by HloCostAnalysis, or instructions which cannot be
  // cloned safely.
  switch (instruction->opcode()) {
    case HloOpcode::kCall:
    case HloOpcode::kCrossReplicaSum:
    case HloOpcode::kCustomCall:
    case HloOpcode::kOutfeed:
    case HloOpcode::kInfeed:
    case HloOpcode::kRecv:
    case HloOpcode::kSend:
    case HloOpcode::kTrace:
    case HloOpcode::kWhile:
      return false;
    default:
      break;
  }

  // Skip tuple shapes because we do not currently account for buffer aliasing
  // properly which results in improperly accounting of rematerialization cost
  // for these shapes.
  if (ShapeUtil::IsTuple(instruction->shape())) {
    return false;
  }
  for (auto* operand : instruction->operands()) {
    if (ShapeUtil::IsTuple(operand->shape())) {
      return false;
    }
  }

  return true;
}

// Class which maintains an ordered list of instructions with fast insertion and
// removal of arbitrary elements.
class InstructionList {
 public:
  explicit InstructionList(const std::vector<const HloInstruction*> order) {
    for (const HloInstruction* inst : order) {
      instructions_.push_back(const_cast<HloInstruction*>(inst));
      instruction_iterators_.insert({const_cast<HloInstruction*>(inst),
                                     std::next(instructions_.end(), -1)});
    }
  }

  // Returns the list of instructions.
  const std::list<HloInstruction*>& instructions() const {
    return instructions_;
  }

  // Insert instruction 'to_insert' before instruction 'before' in the list.
  Status InsertBefore(HloInstruction* to_insert, HloInstruction* before) {
    auto it = instruction_iterators_.find(before);
    TF_RET_CHECK(it != instruction_iterators_.end());
    instruction_iterators_.insert(
        {to_insert, instructions_.insert(it->second, to_insert)});
    return Status::OK();
  }

  // Removes instruction from the list.
  Status Remove(HloInstruction* instruction) {
    auto it = instruction_iterators_.find(instruction);
    TF_RET_CHECK(it != instruction_iterators_.end());
    instructions_.erase(it->second);
    instruction_iterators_.erase(it);
    return Status::OK();
  }

 private:
  // List of instructions.
  std::list<HloInstruction*> instructions_;

  // Iterators for each instruction in the list.
  tensorflow::gtl::FlatMap<const HloInstruction*,
                           std::list<HloInstruction*>::iterator>
      instruction_iterators_;
};

// Class for tracking memory usage of a computation as the instructions are
// placed sequentially. Memory usage is the sum of live values at the current
// point in the instruction sequence.
class MemoryUsageTracker {
 public:
  MemoryUsageTracker(
      const HloComputation* computation,
      const HloRematerialization::ShapeSizeFunction& size_function)
      : computation_(computation), size_function_(size_function) {
    for (const std::unique_ptr<HloInstruction>& instruction :
         computation->instructions()) {
      // Initially only live-in values occupy memory.
      if (IsLiveIn(instruction.get())) {
        memory_usage_ += TotalSizeBytes(instruction->shape());
      }
    }
  }

  // Starts the placement of the given instruction. This adds the output size of
  // the instruction to the current memory usage. Placement is broken into two
  // steps (BeginInstruction and EndInstruction) to accurately model memory
  // usage. At BeginInstruction the memory for the output value of the current
  // instruction is allocated. At EndInstruction memory for dead operands is
  // freed.
  Status BeginInstruction(const HloInstruction* instruction) {
    VLOG(3) << "BeginInstruction " << instruction->name();
    TF_RET_CHECK(in_progress_instruction_ == nullptr);
    in_progress_instruction_ = instruction;

    // Add instruction to remaining_uses_.
    TF_RET_CHECK(!ContainsKey(remaining_uses_, instruction));
    std::vector<HloInstruction*>& instruction_uses =
        remaining_uses_[instruction];
    instruction_uses.insert(instruction_uses.begin(),
                            instruction->users().begin(),
                            instruction->users().end());

    if (!IsLiveIn(instruction)) {
      // Instruction was not previously live so add output size to memory usage.
      memory_usage_ += TotalSizeBytes(instruction->shape());
    }

    VLOG(3) << "  memory usage = " << memory_usage_;
    VLOG(10) << ToString();
    return Status::OK();
  }

  // Finishes the placement of the current instruction. This frees any dead
  // operands or dead result of the instruction. This must be called after each
  // call to BeginInstruction.
  Status EndInstruction() {
    TF_RET_CHECK(in_progress_instruction_ != nullptr);
    VLOG(3) << "EndInstruction " << in_progress_instruction_->name();

    for (HloInstruction* operand : UniqueOperands(in_progress_instruction_)) {
      TF_RET_CHECK(ContainsKey(remaining_uses_, operand));
      std::vector<HloInstruction*>& uses = remaining_uses_.at(operand);
      auto it = std::find(uses.begin(), uses.end(), in_progress_instruction_);
      TF_RET_CHECK(it != uses.end());
      uses.erase(it);

      if (uses.empty()) {
        // Operand is dead.
        int64 operand_size = TotalSizeBytes(operand->shape());
        if (!IsLiveOut(operand)) {
          VLOG(4) << operand->name() << " ("
                  << HumanReadableNumBytes(operand_size) << ") is dead";
          memory_usage_ -= operand_size;
          TF_RET_CHECK(memory_usage_ >= 0);
        }
      }
    }

    // Value is dead if the instruction has no uses and is not live out.
    if (in_progress_instruction_->users().empty() &&
        !IsLiveOut(in_progress_instruction_)) {
      memory_usage_ -= TotalSizeBytes(in_progress_instruction_->shape());
      TF_RET_CHECK(memory_usage_ >= 0);
    }

    in_progress_instruction_ = nullptr;

    VLOG(3) << "  memory usage = " << memory_usage_;
    VLOG(10) << ToString();
    return Status::OK();
  }

  // Adjusts memory usage to account for the rematerialization of
  // original_instruction for the given use. The rematerialization is
  // remat_instruction. This method should be called after the HLO graph has
  // been transformed (rematerialization instruction created and connected to
  // its use).
  Status RematerializeInstructionForUse(HloInstruction* original_instruction,
                                        HloInstruction* remat_instruction,
                                        HloInstruction* use) {
    VLOG(3) << "RematerializeInstructionForUse: original_instruction = "
            << original_instruction->name()
            << ", remat_instruction = " << remat_instruction->name()
            << ", use = " << use->name();

    TF_RET_CHECK(in_progress_instruction_ != nullptr);
    TF_RET_CHECK(IsPlaced(original_instruction));
    TF_RET_CHECK(!IsPlaced(remat_instruction));
    TF_RET_CHECK(!IsPlaced(use));
    TF_RET_CHECK(IsCurrentlyLive(original_instruction));

    // Remove 'use' from remaining uses of original_instruction.
    auto it = std::find(remaining_uses_[original_instruction].begin(),
                        remaining_uses_[original_instruction].end(), use);
    TF_RET_CHECK(it != remaining_uses_[original_instruction].end());
    remaining_uses_[original_instruction].erase(it);

    // If original_instruction is no longer live ('use' was its last use) then
    // deduct original_instruction's memory usage.
    if (!IsCurrentlyLive(original_instruction)) {
      memory_usage_ -= TotalSizeBytes(original_instruction->shape());
      TF_RET_CHECK(memory_usage_ >= 0);
    }

    // Add the new remat_instruction to the remaining uses of its operands.
    for (auto* operand : UniqueOperands(remat_instruction)) {
      // Rematerialization may extend the lifetime of the operand so account for
      // this in memory_usage_.
      TF_RET_CHECK(IsPlaced(operand));
      if (!IsCurrentlyLive(operand)) {
        memory_usage_ += TotalSizeBytes(operand->shape());
      }
      remaining_uses_.at(operand).push_back(remat_instruction);
    }

    VLOG(3) << "  memory usage = " << memory_usage_;
    VLOG(10) << ToString();
    return Status::OK();
  }

  // Returns the number of bytes that the current memory usage will be reduced
  // if the given instruction is rematerialized.
  int64 MemoryReducedIfRematerialized(const HloInstruction* instruction) const {
    // To reduce memory consumption 'instruction' must be currently live and
    // rematerialization must make 'instruction' not live.
    if (IsLiveIn(instruction) || IsLiveOut(instruction) ||
        !IsCurrentlyLive(instruction)) {
      return 0;
    }

    // If the in-progress instruction is a user of 'instruction' (or
    // 'instruction' itself) then rematerializing 'instruction' cannot reduce
    // memory usage because the value is required to be live at this program
    // point.
    if (in_progress_instruction_ == instruction ||
        in_progress_instruction_->IsUserOf(instruction)) {
      return 0;
    }

    // Compute the amount of memory reduced (if any) by rematerializing
    // 'instruction'. 'instruction' will no longer be live at this program
    // point, so initially set memory_reduced to the size of its output value.
    int64 memory_reduced = TotalSizeBytes(instruction->shape());

    // Account for any operands whose live range must be extended across this
    // program point.
    for (const HloInstruction* operand : UniqueOperands(instruction)) {
      if (!IsCurrentlyLive(operand)) {
        // This operand of candidate is not live at this program
        // point. Rematerializing 'instruction' will extend the operand's live
        // range across this program point.
        memory_reduced -= TotalSizeBytes(operand->shape());
      }
    }
    return memory_reduced;
  }

  // Returns the remaining unplaced uses of the given instruction.
  const std::vector<HloInstruction*>& RemainingUses(
      const HloInstruction* instruction) const {
    return remaining_uses_.at(instruction);
  }

  // Returns whether the given instruction has been placed (BeginInstruction has
  // been called with 'instruction' as the argument).
  bool IsPlaced(const HloInstruction* instruction) const {
    return ContainsKey(remaining_uses_, instruction);
  }

  // Returns whether the given instruction is live at the current program point.
  bool IsCurrentlyLive(const HloInstruction* instruction) const {
    return (!IsPlaced(instruction) && IsLiveIn(instruction)) ||
           (IsPlaced(instruction) &&
            (!RemainingUses(instruction).empty() || IsLiveOut(instruction)));
  }

  string ToString() const {
    string output = tensorflow::strings::StrCat("MemoryUsageTracker for ",
                                                computation_->name(), "\n");
    tensorflow::strings::StrAppend(&output, "memory usage = ", memory_usage(),
                                   "\n");
    tensorflow::strings::StrAppend(&output, "Live values:\n");
    for (const auto& pair : remaining_uses_) {
      const HloInstruction* instruction = pair.first;
      const std::vector<HloInstruction*>& uses = pair.second;
      tensorflow::strings::StrAppend(
          &output, "  ", instruction->name(), "; remaining uses: ",
          tensorflow::str_util::Join(uses, ", ",
                                     [](string* out, HloInstruction* use) {
                                       tensorflow::strings::StrAppend(
                                           out, use->name());
                                     }),
          "\n");
    }
    return output;
  }

  // Returns the current memory usage. This is the sum of sizes of all live
  // values.
  int64 memory_usage() const { return memory_usage_; }

  // Returns the current instruction being placed.
  const HloInstruction* in_progress_instruction() const {
    return in_progress_instruction_;
  }

 private:
  // Returns the total size of the shape (including nested elements) in bytes.
  int64 TotalSizeBytes(const Shape& shape) const {
    int64 total_size = 0;
    ShapeUtil::ForEachSubshape(
        shape,
        [this, &total_size](const Shape& subshape,
                            const ShapeIndex& /*index*/) {
          total_size += size_function_(subshape);
          return Status::OK();
        })
        .IgnoreError();
    return total_size;
  }

  // Returns true if the value of given instruction is live into the
  // computation.
  bool IsLiveIn(const HloInstruction* instruction) const {
    return instruction->opcode() == HloOpcode::kConstant ||
           instruction->opcode() == HloOpcode::kParameter;
  }

  // Returns true if the value of given instruction is live out of the
  // computation.
  bool IsLiveOut(const HloInstruction* instruction) const {
    return instruction->opcode() == HloOpcode::kConstant ||
           instruction->opcode() == HloOpcode::kParameter ||
           instruction == instruction->parent()->root_instruction();
  }

  const HloComputation* computation_;

  // Function which computes the size of the top-level buffer of a shape.
  const HloRematerialization::ShapeSizeFunction size_function_;

  // Memory usage at the currently placed instruction.
  int64 memory_usage_ = 0;

  // The instruction currently being placed. This value is non-null only between
  // the calling of BeginInstruction and EndInstruction.
  const HloInstruction* in_progress_instruction_ = nullptr;

  // remaining_uses is a vector of uses of the HLO instruction's value which
  // have not yet been visited by in the rematerialization loop. Use to track
  // liveness of HLO instructions.
  // TODO(b/35212854): Track values using logical buffers rather than HLO
  // instructions. Using HLO instructions over-estimates memory usage because
  // buffer aliasing is ignored.
  tensorflow::gtl::FlatMap<const HloInstruction*, std::vector<HloInstruction*>>
      remaining_uses_;
};

// Computes and returns the cost of rematerializing the given instruction. Cost
// per rematerialized instruction is defined as:
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
// rematerializing this instruction for its remaining uses. In general, we want
// the most memory saving for the least latency penalty which is captured by
// this heuristic.
int64 RematerializationCost(const HloInstruction* instruction,
                            const MemoryUsageTracker& memory_tracker,
                            const HloCostAnalysis& cost_analysis,
                            int64 memory_reduced) {
  const int64 bytes_accessed = cost_analysis.bytes_accessed(*instruction);
  const int64 elements_accessed =
      bytes_accessed /
      ShapeUtil::ByteSizeOfPrimitiveType(instruction->shape().element_type());

  // A duplicate of the rematerialized instruction will be created at each
  // remaining use.
  int64 duplication = memory_tracker.RemainingUses(instruction).size();
  if (duplication == instruction->users().size()) {
    // All remaining uses of instruction are after this point so we can remove
    // the original instruciton after rematerialization.
    duplication -= 1;
  }
  CHECK_GT(memory_reduced, 0);

  // Multiply by 256 to improve precision of cost. Without this factor,
  // many instructions such as many elementwise instructions would have
  // zero cost because the bytes reduced can be several times greater than
  // the element count.
  return 256 * duplication *
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
    const tensorflow::gtl::FlatSet<const HloInstruction*>& remat_instructions) {
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

    if (ContainsKey(remat_instructions, candidate)) {
      // Skip instructions which are rematerialization clones to avoid infinite
      // loops of rematerializing the same instruction(s) repeatedly.
      VLOG(5) << "candidate " << candidate->name()
              << " not viable: is a rematerialized instruction";
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
  MemoryUsageTracker tracker(computation, size_function_);
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
  TF_ASSIGN_OR_RETURN(const CallGraphNode* node,
                      call_graph_->GetNode(instruction->parent()));
  const CallSite* callsite = node->GetCallSite(instruction);
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

  InstructionList instruction_list(sequence->at(computation));
  MemoryUsageTracker memory_tracker(computation, size_function_);
  bool changed = false;

  // Set of instruction clones (not the originals) created during
  // rematerialization. A record is kept to avoid rematerializing an instruction
  // more than once to avoid looping infinitely during rematerialization.
  tensorflow::gtl::FlatSet<const HloInstruction*> remat_instructions;

  // The peak memory of the computation at any point in the instruction
  // sequence.
  int64 peak_memory = memory_tracker.memory_usage();

  // Total count of instructions rematerialized.
  int64 remat_count = 0;
  // Total count of clones created minus number of original rematerialized
  // instructions which are dead.
  int64 net_instructions_added = 0;

  TF_ASSIGN_OR_RETURN(const CallGraphNode* call_graph_node,
                      call_graph_->GetNode(computation));

  // Iterate through all instructions in the sequence. At each instruction
  // (program point) if memory_usage exceeds the specified limit then
  // rematerialize HLO instructions until memory_usage is reduced.
  for (auto list_it = instruction_list.instructions().begin();
       list_it != instruction_list.instructions().end(); ++list_it) {
    HloInstruction* instruction = *list_it;
    TF_ASSIGN_OR_RETURN(int64 callee_usage,
                        CalledComputationsMemoryUsage(instruction));
    TF_RETURN_IF_ERROR(memory_tracker.BeginInstruction(instruction));

    VLOG(2) << "Program point at " << instruction->name()
            << ", memory usage = " << memory_tracker.memory_usage()
            << ", callee usage = " << callee_usage;

    while (memory_tracker.memory_usage() + callee_usage > memory_limit_bytes) {
      VLOG(2) << "Over memory limit at instruction " << instruction->name()
              << ", using "
              << HumanReadableNumBytes(memory_tracker.memory_usage() +
                                       callee_usage)
              << ", limit is " << HumanReadableNumBytes(memory_limit_bytes);

      HloInstruction* best = PickRematerializationCandidate(
          memory_tracker, instruction_list, cost_analysis_, remat_instructions);

      if (best == nullptr) {
        VLOG(3) << "Unable to find rematerialization candidate at program "
                   "point "
                << instruction->name() << ". Memory usage = "
                << HumanReadableNumBytes(memory_tracker.memory_usage() +
                                         callee_usage);
        break;
      }

      VLOG(1) << "Rematerializing instruction " << best->name();
      changed = true;
      remat_count++;

      // Create a rematerialized copy of the candidate at each remaining use.
      // Make a copy of remaining uses because RematerializeInstructionForUse
      // modifies the remaining uses vector in memory_tracker.
      // TODO(b/35213652): It may be profitable to share one rematerialized copy
      // amongst more than one use.
      std::vector<HloInstruction*> remaining_uses_copy =
          memory_tracker.RemainingUses(best);
      for (HloInstruction* use : remaining_uses_copy) {
        // Create a new rematerialized instruction in the HLO graph.
        HloInstruction* remat =
            computation->AddInstruction(best->Clone(/*suffix=*/"remat"));

        VLOG(3) << "Replacing use of " << best->name() << " in " << use->name()
                << " with rematerialization " << remat->name();

        TF_RETURN_IF_ERROR(best->ReplaceUseWith(use, remat));

        // Account for the rematerialization in the memory tracker.
        TF_RETURN_IF_ERROR(
            memory_tracker.RematerializeInstructionForUse(best, remat, use));

        // Insert rematerialized instruction right before its use.
        TF_RETURN_IF_ERROR(instruction_list.InsertBefore(remat, use));

        // Add rematerialized instruction to remat_instructions so the
        // rematerialized instruction is not rematerialized again.
        remat_instructions.insert(remat);

        net_instructions_added++;
      }

      // Original instruction should no longer be live at this point. All
      // of its remaining uses are fed by rematerialized instructions.
      TF_RET_CHECK(!memory_tracker.IsCurrentlyLive(best));

      // If the rematerialized instruction is dead then rematerialization is
      // essentially a move. Don't delete the instruction now because we don't
      // want duplicate HloInstruction* values during the course of the
      // transformation because we keep maps with HloInstruction* values as
      // keys.
      if (best->users().empty()) {
        VLOG(3) << best->name() << " is now dead";
        net_instructions_added--;
      }

      VLOG(3) << "memory_usage after rematerialization = "
              << memory_tracker.memory_usage();
    }

    const CallSite* callsite = call_graph_node->GetCallSite(instruction);
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
      callee_usage = 0;
      for (HloComputation* called_computation :
           callsite->called_computations()) {
        // Memory limit for the subcomputation is the memory limit less the
        // amount of memory used at this point in the computation.
        int64 subcomputation_memory_limit_bytes = std::max<int64>(
            0, memory_limit_bytes - memory_tracker.memory_usage());
        TF_ASSIGN_OR_RETURN(
            bool subcomputation_changed,
            RematerializeComputation(called_computation, sequence,
                                     subcomputation_memory_limit_bytes));
        changed |= subcomputation_changed;

        callee_usage += computation_peak_memory_.at(called_computation);
      }
    }

    peak_memory = std::max<int64>(peak_memory,
                                  memory_tracker.memory_usage() + callee_usage);
    VLOG(3) << "peak memory usage = " << HumanReadableNumBytes(peak_memory);

    TF_RETURN_IF_ERROR(memory_tracker.EndInstruction());
  }

  if (peak_memory > memory_limit_bytes) {
    LOG(WARNING) << "Can't reduce memory use of computation "
                 << computation->name() << " below "
                 << HumanReadableNumBytes(memory_limit_bytes)
                 << " by rematerialization (only reduced to "
                 << HumanReadableNumBytes(peak_memory) << ")";
  }

  // Verify that there are no more remaining uses.
  for (auto& instruction : computation->instructions()) {
    auto& remaining_uses = memory_tracker.RemainingUses(instruction.get());
    CHECK(remaining_uses.empty())
        << instruction->name() << " has remaining uses: "
        << tensorflow::str_util::Join(
               remaining_uses, ", ", [](string* out, HloInstruction* inst) {
                 tensorflow::strings::StrAppend(out, inst->name());
               });
  }

  VLOG(1) << "Rematerialized " << remat_count << " instructions; "
          << net_instructions_added << " net instructions added";
  VLOG(1) << "peak memory usage now " << HumanReadableNumBytes(peak_memory);

  // Update peak memory used by computation.
  computation_peak_memory_[computation] = peak_memory;

  // Update order to include rematerialized instructions.
  sequence->at(computation)
      .assign(instruction_list.instructions().begin(),
              instruction_list.instructions().end());

  return changed;
}

StatusOr<bool> HloRematerialization::Run(
    HloModule* module, SequentialHloOrdering::HloModuleSequence* sequence,
    int64 memory_limit_bytes) {
  // The sequence is constructed entirely by this method.
  TF_RET_CHECK(sequence->empty());

  VLOG(1) << "HloRematerialization() with memory limit of "
          << HumanReadableNumBytes(memory_limit_bytes);

  XLA_VLOG_LINES(3, "Before HloRematerialization:\n" + module->ToString());
  // Create initial sequence of HLO instructions.
  TF_ASSIGN_OR_RETURN(*sequence,
                      CreateMemoryMinimizingSequence(
                          *module, [this](const LogicalBuffer& buffer) {
                            return size_function_(buffer.shape());
                          }));

  // Compute peak memory usage of all computations in the module called in a
  // sequential context.
  TF_ASSIGN_OR_RETURN(call_graph_, CallGraph::Build(module));
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

  VLOG(1) << "Peak memory usage of module (before): "
          << HumanReadableNumBytes(
                 computation_peak_memory_[module->entry_computation()]);

  // Run cost analysis. Operation cost is used in the heuristic for selecting
  // instructions for rematerialization.
  TF_RETURN_IF_ERROR(
      module->entry_computation()->root_instruction()->Accept(&cost_analysis_));

  // Subcomputations called by the entry computation will also be
  // rematerialized.
  TF_ASSIGN_OR_RETURN(bool changed,
                      RematerializeComputation(module->entry_computation(),
                                               sequence, memory_limit_bytes));

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

  VLOG(1) << "Peak memory usage of module (after): "
          << HumanReadableNumBytes(
                 computation_peak_memory_[module->entry_computation()]);

  XLA_VLOG_LINES(3, "After HloRematerialization:\n" + module->ToString());

  return changed;
}

/* static */ StatusOr<bool> HloRematerialization::RematerializeAndSchedule(
    const ShapeSizeFunction& size_function, int64 memory_limit_bytes,
    HloModule* hlo_module, SequentialHloOrdering::HloModuleSequence* sequence) {
  HloRematerialization remat(size_function);
  return remat.Run(hlo_module, sequence, memory_limit_bytes);
}

}  // namespace xla

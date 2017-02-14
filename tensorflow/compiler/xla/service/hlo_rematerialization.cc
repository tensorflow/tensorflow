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

#include <memory>
#include <set>
#include <string>

#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
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

// Returns whether the value produced by the instruction is always
// live. Parameters and constants have memory permanently allocated and are
// effectively always live throughout the computation.
bool AlwaysLive(const HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kConstant ||
         instruction->opcode() == HloOpcode::kParameter;
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

  // Rematerializing instruction with infinite lifetimes is not profitable.
  if (AlwaysLive(instruction)) {
    return false;
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
  std::list<HloInstruction*>& instructions() { return instructions_; }

  // Insert instruction 'to_insert' before instruction 'before' in the list.
  Status InsertBefore(HloInstruction* to_insert, HloInstruction* before) {
    auto it = instruction_iterators_.find(before);
    TF_RET_CHECK(it != instruction_iterators_.end());
    instruction_iterators_.insert(
        {to_insert, instructions_.insert(it->second, to_insert)});
    return Status::OK();
  }

  // Remove instruction from the list.
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
  std::unordered_map<HloInstruction*, std::list<HloInstruction*>::iterator>
      instruction_iterators_;
};

}  // namespace

int64 HloRematerialization::TotalSizeBytes(const Shape& shape) {
  int64 total_size = 0;
  ShapeUtil::ForEachSubshape(
      shape,
      [this, &total_size](const Shape& subshape, const ShapeIndex& /*index*/) {
        total_size += size_function_(subshape);
        return Status::OK();
      })
      .IgnoreError();
  return total_size;
}

StatusOr<bool> HloRematerialization::RematerializeComputation(
    HloComputation* computation, std::vector<const HloInstruction*>* order) {
  InstructionList instruction_list(*order);

  HloCostAnalysis cost_analysis(size_function_);
  TF_RETURN_IF_ERROR(computation->Accept(&cost_analysis));

  // remaining_uses is a vector of uses of the HLO instruction's value which
  // have not yet been visited by in the rematerialization loop. Use to track
  // liveness of HLO instructions.
  // TODO(b/35212854): Track values using logical buffers rather than HLO
  // instructions. Using HLO instructions over-estimates memory usage because
  // buffer aliasing is ignored.
  std::unordered_map<HloInstruction*, std::vector<HloInstruction*>>
      remaining_uses;
  for (HloInstruction* instruction : instruction_list.instructions()) {
    remaining_uses[instruction].insert(remaining_uses[instruction].begin(),
                                       instruction->users().begin(),
                                       instruction->users().end());
  }

  // memory_usage is the total number of bytes in all live HLO values at the
  // current program point in the loop below. Initially this value contains
  // always-live values (constants and parameters).
  int64 memory_usage = 0;
  for (HloInstruction* instruction : instruction_list.instructions()) {
    if (AlwaysLive(instruction)) {
      memory_usage += TotalSizeBytes(instruction->shape());
    }
  }
  const int64 initial_memory_usage = memory_usage;
  int64 max_usage = memory_usage;

  // Set of instruction clones (not the originals) created during
  // rematerialization. A record is kept to avoid rematerializing an instruction
  // more than once.
  std::unordered_set<HloInstruction*> remat_instructions;

  VLOG(2) << "Rematerializing computation " << computation->name();
  VLOG(2) << "starting memory usage = " << memory_usage;
  VLOG(2) << "limit = " << memory_limit_bytes_;
  bool changed = false;

  int64 remat_count = 0;

  // Iterate through all instructions in the sequence. At each instruction
  // (program point) if memory_usage exceeds the specified limit then
  // rematerialize HLO instructions until memory_usage is reduced.
  int64 program_point_index = 0;
  for (auto list_it = instruction_list.instructions().begin();
       list_it != instruction_list.instructions().end(); ++list_it) {
    HloInstruction* instruction = *list_it;
    VLOG(4) << "Program point: after " << instruction->name() << ", ("
            << program_point_index << "/"
            << instruction_list.instructions().size() << ")";
    program_point_index++;

    // Update remaining_uses. Remove instruction from the remaining uses vector
    // of its operands. Simultaneously update the current memory usage. Subtract
    // the size of last uses, and add the size of the value defined by the
    // current instruction.
    for (HloInstruction* operand : UniqueOperands(instruction)) {
      TF_RET_CHECK(remaining_uses.count(operand) >= 1);
      std::vector<HloInstruction*>& uses = remaining_uses.at(operand);
      auto it = std::find(uses.begin(), uses.end(), instruction);
      TF_RET_CHECK(it != uses.end());
      uses.erase(it);

      if (uses.empty()) {
        // Operand is dead.
        int64 operand_size = TotalSizeBytes(operand->shape());
        if (!AlwaysLive(operand)) {
          VLOG(4) << operand->name() << " (" << operand_size
                  << " bytes) is dead";
          memory_usage -= operand_size;
          TF_RET_CHECK(memory_usage >= 0);
        }
      }
    }
    // If instruction value is live, add its size to memory_usage.
    if (!AlwaysLive(instruction) && !remaining_uses.at(instruction).empty()) {
      memory_usage += TotalSizeBytes(instruction->shape());
    }
    VLOG(3) << "memory_usage = " << memory_usage;

    // Iterate up to but not including last instruction. Rematerialization is
    // not possible after last instruction.
    if (std::next(list_it) == instruction_list.instructions().end()) {
      break;
    }
    HloInstruction* next_instruction = *std::next(list_it);

    // Rematerialize values if possible until memory usage drops to or below the
    // specified limit.
    while (memory_usage > memory_limit_bytes_) {
      VLOG(4) << "memory usage = " << memory_usage
              << " > limit = " << memory_limit_bytes_;
      // memory_usage is the sum of the sizes of values which are live at the
      // program point between 'instruction' and the next instruction in the
      // sequence. Find an instruction before 'instruction' which can be
      // rematerialized *after* 'instruction' such that memory_usage is
      // reduced. A rematerialization candate has the following properties:
      //
      // (0) is a rematerializable instruction type.
      //
      // (1) is before 'instruction' in the sequence.
      //
      // (2) has uses after 'instruction'.
      //
      // (3) is not used by the next instruction in the sequence after
      //     'instruction'. (Rematerializing a value used by the next
      //     instruction, call it N, would result in identical memory usage at
      //     the program point immediate before N which is not helpful)
      //
      // (4) the sum of the sizes of the operands of the candidate which are not
      //     live at this program point is less than the size of the value
      //     defined by the candidate. This difference is the amount by which
      //     memory_usage is reduced at this program point by rematerialization.
      //
      // Amongst all candidates we choose the one with the lowest cost per byte
      // reduced defined as:
      //
      //  (flop_count + transcendental_count + element_count) / bytes_reduced
      //
      //   flop_count: from HloCostAnalysis
      //   transcendental_count: from HloCostAnalysis
      //   element_count: number of elements accessed in operands and output of
      //     instruction
      //   bytes_reduced: value (4) from above, always positive
      //
      // To rematerialize we create a clone of the instruction for each
      // remaining use. The clones are placed immediate before the remaining use
      // in the sequence.
      HloInstruction* best = nullptr;
      int64 best_cost = 0;
      int64 best_bytes_reduced = 0;
      // TODO(b/35244891): This is currently quadratic in the number of HLO
      // instructions.
      for (HloInstruction* candidate : instruction_list.instructions()) {
        if (candidate == instruction) {
          // Condition (1) failed.
          break;
        }
        VLOG(5) << "considering rematerialization candidate "
                << candidate->name();

        if (remat_instructions.count(candidate) > 0) {
          VLOG(5) << "candidate " << candidate->name()
                  << " not viable: is a rematerialized instruction";
          continue;
        }

        if (!IsRematerializable(candidate)) {
          // Condition (0) failed.
          VLOG(5) << "candidate " << candidate->name()
                  << " not viable: is not rematerializable";
          continue;
        }

        std::vector<HloInstruction*> candidate_remaining_uses =
            remaining_uses.at(candidate);
        if (candidate_remaining_uses.empty()) {
          // Condition (2) failed.
          VLOG(5) << "candidate " << candidate->name()
                  << " not viable: is dead";
          continue;
        }
        if (std::find(candidate_remaining_uses.begin(),
                      candidate_remaining_uses.end(),
                      next_instruction) != candidate_remaining_uses.end()) {
          // Condition (3) failed.
          VLOG(5) << "candidate " << candidate->name()
                  << " not viable: is used by next instruction in sequence ("
                  << next_instruction->name() << ")";
          continue;
        }

        // Compute the amount of memory reduced (if any) by rematerializing
        // candidate. The candidate value will no longer be live at this program
        // point, so initially set memory_reduced to the size of candidate
        // value.
        TF_RET_CHECK(!AlwaysLive(candidate));
        int64 memory_reduced = TotalSizeBytes(candidate->shape());
        for (auto* operand : UniqueOperands(candidate)) {
          if (remaining_uses.at(operand).empty() && !AlwaysLive(operand)) {
            // This operand of candidate is not live at this program
            // point. Rematerializing candidate will extend live range across
            // this program point.
            memory_reduced -= TotalSizeBytes(operand->shape());
          }
        }

        if (memory_reduced <= 0) {
          // Condition (4) failed.
          VLOG(5) << "candidate " << candidate->name()
                  << " memory reduced = " << memory_reduced << " <= 0";
          continue;
        }

        // Compute the cost of rematerializing the candidate. Cost is sum of
        // flops, transcendental ops, and number of elements accessed in memory
        // divided by the memory reduced. This is a rough estimate of the extra
        // execution time per byte saved by rematerializing this instruction. In
        // general, we want the most memory saving for the least latency penalty
        // which is captured by this heuristic.
        int64 bytes_accessed = cost_analysis.bytes_accessed(*candidate);
        int64 net_memory_reduced =
            std::min(memory_reduced, memory_usage - memory_limit_bytes_);
        int64 elements_accessed =
            bytes_accessed / ShapeUtil::ByteSizeOfPrimitiveType(
                                 candidate->shape().element_type());
        TF_RET_CHECK(net_memory_reduced > 0);
        // Multiply by 256 to improve precision of cost. Without this factor,
        // many instructions such as many elementwise instructions would have
        // zero cost because the bytes reduced can be several times greater than
        // the element count.
        int64 candidate_cost = 256 *
                               (cost_analysis.flop_count(*candidate) +
                                cost_analysis.transcendental_count(*candidate) +
                                elements_accessed) /
                               net_memory_reduced;
        VLOG(5) << "candidate " << candidate->name() << " cost per byte "
                << candidate_cost;

        if (best == nullptr || candidate_cost < best_cost) {
          VLOG(5) << "candidate " << candidate->name() << " now best";
          best = candidate;
          best_cost = candidate_cost;
          best_bytes_reduced = memory_reduced;
        }
      }

      if (best == nullptr) {
        VLOG(3) << "Unable to find rematerialization candidate at program "
                   "point after "
                << instruction->name()
                << ". Memory usage = " << HumanReadableNumBytes(memory_usage);
        break;
      }

      changed = true;

      VLOG(2) << "rematerializing " << best->name() << ", saving "
              << best_bytes_reduced << ", cost per byte " << best_cost;

      remat_count++;

      // Create a rematerialized copy of the candidate at each remaining use.
      // TODO(b/35213652): It may be profitable to share one rematerialized copy
      // amongst more than one use.
      for (HloInstruction* use : remaining_uses.at(best)) {
        HloInstruction* remat = computation->AddInstruction(best->Clone());
        // The rematerialized instruction has only one use.
        remaining_uses.insert({remat, {use}});
        TF_RETURN_IF_ERROR(best->ReplaceUseWith(use, remat));
        VLOG(3) << "Replacing use of " << best->name() << " in " << use->name()
                << " with rematerialization " << remat->name();

        // Update remaining uses of each operand to include the new
        // rematerialized instruction.
        for (auto* operand : UniqueOperands(remat)) {
          TF_RET_CHECK(remaining_uses.count(operand) == 1);
          remaining_uses[operand].push_back(remat);
        }

        // Insert rematerialized instruction right before its use.
        TF_RETURN_IF_ERROR(instruction_list.InsertBefore(remat, use));

        // Add rematerialized instruction to remat_instructions so the
        // rematerialized instruction is not rematerialized again.
        remat_instructions.insert(remat);
      }

      // The instruction which was rematerialized ('best') has no remaining
      // uses. All uses have been replaced with rematerializations.
      remaining_uses[best].clear();

      // If the rematerialized instruction is now dead, then remove it from the
      // computation and instruction list.
      if (best->users().empty()) {
        VLOG(3) << best->name() << " is now dead, removing";
        TF_RETURN_IF_ERROR(instruction_list.Remove(best));
        remaining_uses.erase(best);
        TF_RETURN_IF_ERROR(computation->RemoveInstruction(best));
      }

      memory_usage -= best_bytes_reduced;
      VLOG(3) << "memory_usage after rematerialization = " << memory_usage;

      TF_RET_CHECK(memory_usage >= 0);
    }
    max_usage = std::max<int64>(max_usage, memory_usage);
    VLOG(3) << "max memory usage = " << max_usage;
  }

  if (max_usage > memory_limit_bytes_) {
    return ResourceExhausted(
        "Can't reduce memory use below %s by rematerialization (only "
        "reduced to %s)",
        HumanReadableNumBytes(memory_limit_bytes_).c_str(),
        HumanReadableNumBytes(max_usage).c_str());
  }

  // Verify that there are no more remaining uses.
  for (auto instruction_uses : remaining_uses) {
    HloInstruction* instruction = instruction_uses.first;
    const std::vector<HloInstruction*>& uses = instruction_uses.second;
    DCHECK(uses.empty()) << instruction->name() << " has remaining uses: "
                         << tensorflow::str_util::Join(
                                uses, ", ",
                                [](string* out, HloInstruction* inst) {
                                  tensorflow::strings::StrAppend(out,
                                                                 inst->name());
                                });
  }

  // Memory usage should be exactly back to its initial value.
  TF_RET_CHECK(memory_usage == initial_memory_usage);

  VLOG(2) << "Rematerialized " << remat_count << " instructions";
  VLOG(2) << "maximum memory usage now " << HumanReadableNumBytes(max_usage);

  // Update order to include rematerialized instructions.
  order->clear();
  for (HloInstruction* instruction : instruction_list.instructions()) {
    order->push_back(instruction);
  }

  return changed;
}

StatusOr<bool> HloRematerialization::Run(
    HloModule* module, SequentialHloOrdering::HloModuleSequence* sequence) {
  // The sequence is constructed entirely by this method.
  TF_RET_CHECK(sequence->empty());

  VLOG(1) << "HloRematerialization() with memory limit of "
          << HumanReadableNumBytes(memory_limit_bytes_);

  XLA_VLOG_LINES(3, "Before HloRematerialization:\n" + module->ToString());

  // Create initial sequence of HLO instructions.
  TF_ASSIGN_OR_RETURN(*sequence,
                      CreateMemoryMinimizingSequence(
                          *module, [this](const LogicalBuffer& buffer) {
                            return size_function_(buffer.shape());
                          }));

  // TODO(b/35213508): Rematerialize more than the entry computation such as
  // while bodies.
  TF_ASSIGN_OR_RETURN(
      bool changed,
      RematerializeComputation(module->entry_computation(),
                               &(*sequence)[module->entry_computation()]));

  XLA_VLOG_LINES(3, "After HloRematerialization:\n" + module->ToString());

  return changed;
}

/* static */ StatusOr<bool> HloRematerialization::RematerializeAndSchedule(
    const ShapeSizeFunction& size_function, int64 memory_limit_bytes,
    HloModule* hlo_module, SequentialHloOrdering::HloModuleSequence* sequence) {
  HloRematerialization remat(size_function, memory_limit_bytes);
  return remat.Run(hlo_module, sequence);
}

}  // namespace xla

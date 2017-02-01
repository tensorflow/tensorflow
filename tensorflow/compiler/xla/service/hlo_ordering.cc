/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_ordering.h"

#include <set>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

PredecessorHloOrdering::PredecessorHloOrdering(const HloModule* module)
    : module_(module) {}

bool PredecessorHloOrdering::ExecutesBefore(const HloInstruction* a,
                                            const HloInstruction* b) const {
  // Instructions in different computations are unordered.
  if (a->parent() != b->parent()) {
    return false;
  }
  // 'a' executes before 'b' if 'a' is in the strict predecessor set of 'b'.
  return strict_predecessors_.at(b->parent())->IsReachable(b, a);
}

string PredecessorHloOrdering::ToStringHelper(const string& name) const {
  std::vector<string> pieces;
  pieces.push_back(name);
  for (auto& computation : module_->computations()) {
    pieces.push_back(tensorflow::strings::Printf("computation %s:",
                                                 computation->name().c_str()));
    const auto all = computation->MakeInstructionPostOrder();
    for (auto instruction : all) {
      pieces.push_back(tensorflow::strings::Printf(
          "  %s strict predecessors:", instruction->name().c_str()));
      for (auto predecessor : all) {
        if (strict_predecessors_.at(computation.get())
                ->IsReachable(instruction, predecessor)) {
          pieces.push_back(
              tensorflow::strings::Printf("  %s", predecessor->name().c_str()));
        }
      }
    }
  }
  return tensorflow::str_util::Join(pieces, "\n");
}

DependencyHloOrdering::DependencyHloOrdering(const HloModule* module)
    : PredecessorHloOrdering(module) {
  // Compute predecessor relationships between all instructions to determine
  // ordering based on dependencies. ExecutesBefore will return true iff there
  // exists a path in the HLO computation graph from 'a' to 'b'.
  for (auto& computation : module->computations()) {
    strict_predecessors_.emplace(computation.get(),
                                 computation->ComputeTransitiveOperands());
  }
}

string DependencyHloOrdering::ToString() const {
  return ToStringHelper("DependencyHloOrdering");
}

SequentialHloOrdering::SequentialHloOrdering(
    const HloModule* module, const HloModuleSequence& module_sequence)
    : module_(module) {
  // Create a map from instruction to its order position.
  for (auto computation_order : module_sequence) {
    const std::vector<const HloInstruction*>& order = computation_order.second;
    for (int i = 0; i < order.size(); ++i) {
      DCHECK_EQ(0, order_position_.count(order[i]));
      order_position_.emplace(order[i], i);
    }
  }
}

bool SequentialHloOrdering::ExecutesBefore(const HloInstruction* a,
                                           const HloInstruction* b) const {
  // Instructions in different computations are unordered.
  if (a->parent() != b->parent()) {
    return false;
  }
  // If either instruction is not in the order, then 'a' and 'b' are unordered.
  if (order_position_.count(a) == 0 || order_position_.count(b) == 0) {
    return false;
  }
  return order_position_.at(a) < order_position_.at(b);
}

string SequentialHloOrdering::ToString() const {
  std::vector<string> pieces;
  pieces.push_back("SequentialHloOrdering");
  for (auto& computation : module_->computations()) {
    pieces.push_back(tensorflow::strings::Printf("computation %s order:",
                                                 computation->name().c_str()));
    // Gather all instructions in the module sequence for this computation and
    // sort them by their position.
    std::vector<const HloInstruction*> instructions;
    for (auto& instruction_position : order_position_) {
      const HloInstruction* instruction = instruction_position.first;
      if (instruction->parent() == computation.get()) {
        instructions.push_back(instruction);
      }
    }
    std::sort(instructions.begin(), instructions.end(),
              [this](const HloInstruction* a, const HloInstruction* b) {
                return order_position_.at(a) < order_position_.at(b);
              });
    for (auto instruction : instructions) {
      pieces.push_back(
          tensorflow::strings::Printf("  %s", instruction->name().c_str()));
    }
  }
  return tensorflow::str_util::Join(pieces, "\n");
}

namespace {

// Class implementing a list scheduler of HLO instructions which produces a
// sequence which minimizes memory usage.
class ListScheduler {
 public:
  // Construct and return a memory-minimizing sequence of HLO instructions
  // containing the given HLO computation.
  static StatusOr<std::vector<const HloInstruction*>> Run(
      const HloComputation& computation,
      const TuplePointsToAnalysis& points_to_analysis,
      const LogicalBuffer::SizeFunction& size_function) {
    ListScheduler scheduler(computation, points_to_analysis, size_function);
    return scheduler.CreateSchedule();
  }

 private:
  // The scheduling priority of an instruction is first the number of bytes
  // freed by scheduling the instruction, and second (tie-breaker) by the number
  // of users. This is represented as a std::pair containing these two values
  // (first element is the bytes freed). std::pair provides the necessary
  // comparison operators.
  using Priority = std::pair<int64, int64>;

  ListScheduler(const HloComputation& computation,
                const TuplePointsToAnalysis& points_to_analysis,
                const LogicalBuffer::SizeFunction& size_function)
      : computation_(computation),
        points_to_analysis_(points_to_analysis),
        size_function_(size_function) {
    // Create a map containing the LogicalBuffer uses for each HLO
    // instruction. An HLO instruction "uses" a LogicalBuffer if the
    // LogicalBuffer is in an operand of the instruction as indicated by
    // points-to analysis.
    for (auto& instruction : computation.instructions()) {
      buffer_uses_.insert(
          {instruction.get(), std::unordered_set<const LogicalBuffer*>()});
      for (auto* operand : instruction->operands()) {
        for (const LogicalBuffer* buffer :
             points_to_analysis.GetBuffersDefinedByInstruction(operand)) {
          buffer_uses_[instruction.get()].insert(buffer);
        }
      }
    }

    // Create map containing the number of unscheduled uses (hlo instructions)
    // of each logical buffer.
    for (auto& instruction : computation.instructions()) {
      for (auto* buffer : points_to_analysis.GetBuffersDefinedByInstruction(
               instruction.get())) {
        unscheduled_use_count_[buffer] = 0;
      }
    }
    for (auto& instruction : computation.instructions()) {
      for (const LogicalBuffer* buffer : buffer_uses_.at(instruction.get())) {
        ++unscheduled_use_count_[buffer];
      }
    }

    // Buffers live out of the computation have an implicit use at the end of
    // the computation.
    for (const LogicalBuffer* live_out_buffer :
         points_to_analysis.GetPointsToSet(computation.root_instruction())
             .CreateFlattenedSet()) {
      ++unscheduled_use_count_[live_out_buffer];
    }
  }

  // Returns whether the memory used by the given buffer should be ignored by
  // the scheduling heuristic.
  bool IgnoreBuffer(const LogicalBuffer& buffer) {
    return buffer.instruction()->opcode() == HloOpcode::kParameter ||
           buffer.instruction()->opcode() == HloOpcode::kConstant;
  }

  // Return the number of bytes freed if the HLO instruction is scheduled.
  int64 BytesFreedIfScheduled(const HloInstruction* instruction) {
    int64 freed_bytes = 0;
    // Sum the total size of the values last used by this instruction.
    for (auto* buffer : buffer_uses_.at(instruction)) {
      if (IgnoreBuffer(*buffer)) {
        continue;
      }
      CHECK_GE(unscheduled_use_count_.at(buffer), 1);
      if (unscheduled_use_count_.at(buffer) == 1) {
        // This is the last use of the logical buffer.
        freed_bytes += size_function_(*buffer);
      }
    }
    // Then subtract the size of the value(s) defined by this instruction.
    for (auto* buffer :
         points_to_analysis_.GetBuffersDefinedByInstruction(instruction)) {
      if (!IgnoreBuffer(*buffer)) {
        freed_bytes -= size_function_(*buffer);
      }
    }
    return freed_bytes;
  }

  // Construct the scheduling priority of the given instruciton.
  Priority GetPriority(const HloInstruction* instruction) {
    return {BytesFreedIfScheduled(instruction), instruction->user_count()};
  }

  std::vector<const HloInstruction*> CreateSchedule() {
    std::vector<const HloInstruction*> schedule;

    // Populate the ready list with instructions which have no operands.
    std::list<const HloInstruction*> ready_list;
    for (auto& instruction : computation_.instructions()) {
      if (instruction->operand_count() == 0 &&
          instruction->control_predecessors().empty()) {
        ready_list.push_back(instruction.get());
      }
    }

    while (!ready_list.empty()) {
      // Select the highest priority HLO instruction from the ready list.
      auto best_it = ready_list.begin();
      Priority best_priority = GetPriority(*best_it);
      for (auto ready_it = std::next(ready_list.begin());
           ready_it != ready_list.end(); ++ready_it) {
        Priority priority = GetPriority(*ready_it);
        if (priority > best_priority) {
          best_it = ready_it;
          best_priority = priority;
        }
      }

      // Remove the selected instruction from the ready list and add it to the
      // schedule.
      const HloInstruction* best = *best_it;
      ready_list.erase(best_it);
      schedule.push_back(best);
      scheduled_instructions_.insert(best);

      // Update the unscheduled uses of the logical buffers.
      for (const LogicalBuffer* buffer : buffer_uses_.at(best)) {
        CHECK_GT(unscheduled_use_count_.at(buffer), 0);
        --unscheduled_use_count_[buffer];
      }

      // Add new instructions to ready list.
      // TODO(b/34466113): Replace this with successors()/predecessors() when
      // predecessor/successor methods are added to HloInstruction. This also
      // will resolve the nondeterminism of using a set here assuming
      // predecessors/successors is a vector.
      std::set<HloInstruction*> successors = best->users();
      successors.insert(best->control_successors().begin(),
                        best->control_successors().end());
      for (auto* successor : successors) {
        std::set<HloInstruction*> predecessors(successor->operands().begin(),
                                               successor->operands().end());
        predecessors.insert(successor->control_predecessors().begin(),
                            successor->control_predecessors().end());
        bool is_ready = true;
        for (auto* predecessor : predecessors) {
          if (scheduled_instructions_.count(predecessor) == 0) {
            is_ready = false;
            break;
          }
        }
        if (is_ready) {
          ready_list.push_back(successor);
        }
      }
    }
    CHECK_EQ(schedule.size(), computation_.instructions().size());
    CHECK_EQ(scheduled_instructions_.size(),
             computation_.instructions().size());

    return schedule;
  }

  const HloComputation& computation_;
  const TuplePointsToAnalysis& points_to_analysis_;
  const LogicalBuffer::SizeFunction& size_function_;

  // A map containing the LogicalBuffers that each instruction uses.
  std::unordered_map<const HloInstruction*,
                     std::unordered_set<const LogicalBuffer*>>
      buffer_uses_;

  // A map containing the count of unscheduled HLOs which using a particular
  // LogicalBuffer.
  std::unordered_map<const LogicalBuffer*, int64> unscheduled_use_count_;

  // Set of instructions which have been scheduled.
  std::unordered_set<const HloInstruction*> scheduled_instructions_;
};

}  // namespace

StatusOr<SequentialHloOrdering::HloModuleSequence>
CreateMemoryMinimizingSequence(
    const HloModule& module, const LogicalBuffer::SizeFunction& size_function) {
  SequentialHloOrdering::HloModuleSequence sequence;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<TuplePointsToAnalysis> points_to_analysis,
                      TuplePointsToAnalysis::Run(&module));

  for (auto& computation : module.computations()) {
    TF_ASSIGN_OR_RETURN(
        sequence[computation.get()],
        ListScheduler::Run(*computation, *points_to_analysis, size_function));
  }

  return sequence;
}

std::ostream& operator<<(
    std::ostream& out,
    const SequentialHloOrdering::HloModuleSequence& module_sequence) {
  for (auto computation_pair : module_sequence) {
    const HloComputation* computation = computation_pair.first;
    const std::vector<const HloInstruction*>& computation_sequence =
        computation_pair.second;
    out << "Computation " << computation->name() << ":\n";
    for (auto* instruction : computation_sequence) {
      out << "  " << instruction->name() << "\n";
    }
  }
  return out;
}

}  // namespace xla

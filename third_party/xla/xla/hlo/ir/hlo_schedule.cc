/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/hlo/ir/hlo_schedule.h"

#include <algorithm>
#include <cstdint>
#include <ostream>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/map_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/status_macros.h"
#include "xla/tsl/lib/gtl/map_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"

namespace xla {

/* static */ absl::StatusOr<HloSchedule> HloSchedule::CreateFromProto(
    const HloModule* module, const HloScheduleProto& proto,
    const absl::flat_hash_map<int64_t, absl::flat_hash_map<int64_t, int64_t>>*
        computation_id_to_instruction_id_remap) {
  absl::flat_hash_map<int64_t, const HloComputation*> id_to_computation;
  for (const HloComputation* computation : module->computations()) {
    id_to_computation[computation->unique_id()] = computation;
  }

  HloSchedule schedule(module);
  for (const auto& id_sequence : proto.sequences()) {
    int64_t computation_id = id_sequence.first;

    auto comp_it = id_to_computation.find(computation_id);
    // Computation could have been removed if unused, so
    // skip if not found.
    if (comp_it == id_to_computation.end()) {
      continue;
    }
    const HloComputation* computation = comp_it->second;

    absl::flat_hash_map<int64_t, HloInstruction*> id_to_instruction;
    for (HloInstruction* instruction : computation->instructions()) {
      id_to_instruction[instruction->unique_id()] = instruction;
    }

    HloInstructionSequence& sequence =
        schedule.GetOrCreateSequence(computation);
    if (computation_id_to_instruction_id_remap != nullptr) {
      TF_RET_CHECK(
          computation_id_to_instruction_id_remap->contains(computation_id))
          << "Computation id " << computation_id
          << " not found in computation_id_to_instruction_id_remap";
    }

    for (const int64_t instruction_id : id_sequence.second.instruction_ids()) {
      int64_t corrected_instruction_id = instruction_id;
      if (computation_id_to_instruction_id_remap != nullptr) {
        TF_RET_CHECK(computation_id_to_instruction_id_remap->at(computation_id)
                         .contains(instruction_id))
            << "Instruction id " << instruction_id
            << " not found in its computation's proto_id_to_instruction_id_map";
        corrected_instruction_id =
            computation_id_to_instruction_id_remap->at(computation_id)
                .at(instruction_id);
      }
      int64_t complete_unique_id = HloInstruction::CalculateUniqueId(
          computation->unique_id(), corrected_instruction_id);
      auto instr_it = id_to_instruction.find(complete_unique_id);
      TF_RET_CHECK(instr_it != id_to_instruction.end())
          << "No instruction exists in HLO computation " << computation->name()
          << " with unique id " << corrected_instruction_id
          << " (complete unique id " << complete_unique_id << ")";
      sequence.push_back(instr_it->second);
    }
  }
  ABSL_RETURN_IF_ERROR(schedule.Verify());
  return schedule;
}

absl::StatusOr<HloScheduleProto> HloSchedule::ToProto() const {
  ABSL_RETURN_IF_ERROR(Verify());
  HloScheduleProto proto;
  for (const auto& id_sequence : sequences_) {
    int64_t computation_id = id_sequence.first;
    const HloInstructionSequence& sequence = id_sequence.second;
    HloScheduleProto::InstructionSequence& proto_sequence =
        (*proto.mutable_sequences())[computation_id];
    proto_sequence.mutable_instruction_ids()->Reserve(sequence.size());
    for (const int64_t id : sequence.ids()) {
      proto_sequence.add_instruction_ids(id);
    }
  }
  return proto;
}

void HloSchedule::set_sequence(const HloComputation* computation,
                               absl::Span<HloInstruction* const> sequence) {
  set_sequence(computation, HloInstructionSequence(sequence));
}

void HloSchedule::set_sequence(const HloComputation* computation,
                               HloInstructionSequence sequence) {
  CHECK(computation->parent() == module_);
  sequences_[computation->unique_id()] = std::move(sequence);
  execution_threads_[computation->unique_id()] =
      std::string(computation->execution_thread());
}

HloInstructionSequence& HloSchedule::GetOrCreateSequence(
    const HloComputation* computation) {
  auto it = sequences_.find(computation->unique_id());
  if (it == sequences_.end()) {
    // No sequence found for computation. Create and return an empty one.
    CHECK(computation->parent() == module_);
    execution_threads_[computation->unique_id()] =
        std::string(computation->execution_thread());
    return sequences_[computation->unique_id()];
  }
  return it->second;
}

const HloInstructionSequence& HloSchedule::sequence(
    const HloComputation* computation) const {
  return sequences_.at(computation->unique_id());
}

namespace {

// Tracks in-degrees and successors for dependency-driven schedule
// reconstruction.
struct DependencyTracker {
  // Number of unscheduled operands and control predecessors for each
  // instruction in the computation.
  absl::flat_hash_map<const HloInstruction*, int> in_degree;

  // Map from HloInstruction X to instructions in the computation which depend
  // on X.
  absl::flat_hash_map<const HloInstruction*, std::vector<HloInstruction*>>
      successors;

  // Worklist of instructions whose dependencies are fully satisfied and are
  // ready to be placed in the schedule.
  std::queue<HloInstruction*> ready_worklist;
};

// Builds in-degree counts and successor lists for all instructions in the
// computation, and seeds the ready worklist with newly added 0-predecessor
// instructions.
absl::StatusOr<DependencyTracker> BuildDependencyTracker(
    const HloComputation* computation,
    const absl::flat_hash_set<int64_t>& ids_in_schedule) {
  DependencyTracker tracker;
  absl::flat_hash_set<const HloInstruction*> predecessors;

  for (HloInstruction* instruction : computation->instructions()) {
    predecessors.clear();
    for (const HloInstruction* operand : instruction->operands()) {
      if (predecessors.insert(operand).second) {
        tracker.successors[operand].push_back(instruction);
      }
    }
    for (HloInstruction* control_pred : instruction->control_predecessors()) {
      if (control_pred == instruction) {
        ABSL_RETURN_IF_ERROR(control_pred->RemoveControlDependencyTo(instruction));
      } else if (predecessors.insert(control_pred).second) {
        tracker.successors[control_pred].push_back(instruction);
      }
    }

    tracker.in_degree[instruction] = predecessors.size();

    // Newly added instructions with zero predecessors are ready immediately.
    if (predecessors.empty() &&
        !ids_in_schedule.contains(instruction->unique_id())) {
      tracker.ready_worklist.push(instruction);
    }
  }
  return tracker;
}

// Reconstructs the schedule by replaying existing_sequence and interleaving
// ready instructions as their dependencies are satisfied.
HloInstructionSequence BuildUpdatedSequence(
    const HloInstructionSequence& existing_sequence,
    const absl::flat_hash_map<int64_t, HloInstruction*>& id_to_instruction,
    const absl::flat_hash_set<int64_t>& ids_in_schedule,
    DependencyTracker& tracker) {
  HloInstructionSequence new_sequence;
  absl::flat_hash_set<const HloInstruction*> scheduled;
  absl::flat_hash_set<const HloInstruction*> deferred_instructions;

  auto schedule_instruction = [&](HloInstruction* instruction) {
    if (!scheduled.insert(instruction).second) {
      return;
    }
    new_sequence.push_back(instruction);

    auto it = tracker.successors.find(instruction);
    if (it != tracker.successors.end()) {
      for (HloInstruction* successor : it->second) {
        int& in_degree = tracker.in_degree.at(successor);
        in_degree--;
        CHECK_GE(in_degree, 0);
        if (in_degree == 0) {
          // Push newly added instructions and previously deferred instructions
          // to the ready worklist. Instructions still awaiting their turn in
          // existing_sequence are not pushed early.
          if (!ids_in_schedule.contains(successor->unique_id()) ||
              deferred_instructions.contains(successor)) {
            tracker.ready_worklist.push(successor);
          }
        }
      }
    }
  };

  auto drain_ready_worklist = [&]() {
    while (!tracker.ready_worklist.empty()) {
      HloInstruction* ready_instruction = tracker.ready_worklist.front();
      tracker.ready_worklist.pop();
      schedule_instruction(ready_instruction);
    }
  };

  // 1. Schedule any 0-predecessor newly added instructions first.
  drain_ready_worklist();

  // 2. Iterate through the existing sequence in order.
  for (int64_t id : existing_sequence.ids()) {
    auto it = id_to_instruction.find(id);
    if (it == id_to_instruction.end()) {
      // Instruction was removed from computation; skip.
      continue;
    }
    HloInstruction* instruction = it->second;
    if (scheduled.contains(instruction)) {
      // Already scheduled via the ready worklist.
      continue;
    }
    if (tracker.in_degree.at(instruction) == 0) {
      schedule_instruction(instruction);
      drain_ready_worklist();
    } else {
      // Instruction has unsatisfied dependencies and cannot be scheduled yet;
      // defer it until its in-degree reaches 0.
      deferred_instructions.insert(instruction);
    }
  }

  // 3. Drain any remaining ready instructions.
  drain_ready_worklist();

  return new_sequence;
}

}  // namespace

absl::Status HloSchedule::UpdateComputationSchedule(
    const HloComputation* computation) {
  // Map from unique ID to HloInstruction pointer for instructions in the
  // computation.
  absl::flat_hash_map<int64_t, HloInstruction*> id_to_instruction;
  for (HloInstruction* instruction : computation->instructions()) {
    InsertOrDie(&id_to_instruction, instruction->unique_id(), instruction);
  }

  // Set of all HloInstructions in the schedule.
  HloInstructionSequence& current_sequence =
      sequences_.at(computation->unique_id());
  absl::flat_hash_set<int64_t> ids_in_schedule(current_sequence.ids().begin(),
                                               current_sequence.ids().end());

  // Build in-degree dependency tracking for all instructions.
  ABSL_ASSIGN_OR_RETURN(DependencyTracker tracker,
                   BuildDependencyTracker(computation, ids_in_schedule));

  // Reconstruct schedule in topological order.
  set_sequence(computation,
               BuildUpdatedSequence(current_sequence, id_to_instruction,
                                    ids_in_schedule, tracker));
  return absl::OkStatus();
}

absl::Status HloSchedule::Update(
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // The schedule must contain a sequence for every non-fusion computation in
  // the module for the specified threads, but can have sequences for
  // computations which no longer exist (these are removed).
  std::vector<HloComputation*> nonfusion_computations =
      module_->MakeNonfusionComputations(execution_threads);
  for (const HloComputation* computation : nonfusion_computations) {
    if (!is_computation_scheduled(computation)) {
      GetOrCreateSequence(computation);
      ABSL_RETURN_IF_ERROR(UpdateComputationSchedule(computation));
    }
  }
  auto sum_of_sequences_for_threads = [&]() -> int64_t {
    if (execution_threads.empty()) {
      return sequences_.size();
    }
    int64_t sequences_num_for_threads = 0;
    for (const auto& [thread_name, sequence_num] :
         num_sequences_by_execution_thread()) {
      sequences_num_for_threads +=
          execution_threads.contains(thread_name) ? sequence_num : 0;
    }
    return sequences_num_for_threads;
  };
  int64_t sequence_sum = sum_of_sequences_for_threads();
  if (sequence_sum > nonfusion_computations.size()) {
    // Schedule contains some computations which have been removed from the
    // HloModule. Remove them from the schedule as well.
    absl::flat_hash_set<int64_t> nonfusion_computations_ids;
    for (const HloComputation* computation : nonfusion_computations) {
      nonfusion_computations_ids.insert(computation->unique_id());
    }
    for (auto it = sequences_.begin(); it != sequences_.end();) {
      std::string sequence_thread_name = tsl::gtl::FindWithDefault(
          execution_threads_, it->first, HloInstruction::kMainExecutionThread);
      bool is_thread_included =
          execution_threads.empty() ||
          execution_threads.contains(sequence_thread_name);
      if (!nonfusion_computations_ids.contains(it->first) &&
          is_thread_included) {
        execution_threads_.erase(it->first);
        sequences_.erase(it++);
      } else {
        ++it;
      }
    }
  }
  sequence_sum = sum_of_sequences_for_threads();
  CHECK_EQ(sequence_sum, nonfusion_computations.size());

  for (const HloComputation* computation : nonfusion_computations) {
    ABSL_RETURN_IF_ERROR(UpdateComputationSchedule(computation));
  }

  ABSL_RETURN_IF_ERROR(Verify());
  return absl::OkStatus();
}

absl::flat_hash_map<std::string, int64_t>
HloSchedule::num_sequences_by_execution_thread() const {
  absl::flat_hash_map<std::string, int64_t> sequence_num_by_execution_threads;
  for (const auto& id_sequence_item : sequences_) {
    ++sequence_num_by_execution_threads[tsl::gtl::FindWithDefault(
        execution_threads_, id_sequence_item.first,
        HloInstruction::kMainExecutionThread)];
  }
  return sequence_num_by_execution_threads;
}

absl::Status HloSchedule::Verify() const {
  VLOG(2) << "VerifySchedule()";
  XLA_VLOG_LINES(2, ToString());

  // Verify schedule contains exactly the same set of non-fusion computations as
  // module currently does for each thread that has schedule.
  absl::flat_hash_map<std::string, int64_t> sequence_num_by_execution_threads =
      num_sequences_by_execution_thread();
  for (const auto& [thread_name, sequence_size] :
       sequence_num_by_execution_threads) {
    std::vector<HloComputation*> nonfusion_computations =
        module_->MakeNonfusionComputations({thread_name});

    // TODO(dasenov): Replace with std::erase_if after XLA uses C++20.
    auto remove_it = std::remove_if(nonfusion_computations.begin(),
                                    nonfusion_computations.end(),
                                    [](const HloComputation* computation) {
                                      return computation->IsDeadComputation();
                                    });
    nonfusion_computations.erase(remove_it, nonfusion_computations.end());

    // It's possible to have more sequences than non_fusion_computations.
    // This is because in some cases computations that have schedules are
    // actually dead. The important thing to check is that each live non-fusion
    // computation has a sequence.
    //
    // TODO(b/418034360): Consider strenghtening this check to equality. That
    // would require cleaning up dead computations and/or recomputing the
    // schedule in a number of tests. In its present state (using less or equal)
    // this check is subsumed by the next one.
    TF_RET_CHECK(nonfusion_computations.size() <= sequence_size)
        << "For thread " << thread_name << ", schedule has " << sequence_size
        << " sequences, but module has " << nonfusion_computations.size()
        << " non-fusion computations for thread " << thread_name;
    for (const HloComputation* computation : nonfusion_computations) {
      TF_RET_CHECK(sequences_.contains(computation->unique_id()))
          << "Computation " << computation->name()
          << " missing from HLO schedule.";
    }

    // For each computation verify the set of instructions is the same and
    // that each dependency and control edge is honored.
    for (const HloComputation* computation : nonfusion_computations) {
      ABSL_RETURN_IF_ERROR(Verify(computation));
    }
  }

  return absl::OkStatus();
}

absl::Status HloSchedule::Verify(const HloComputation* computation) const {
  absl::flat_hash_map<const HloInstruction*, int> instruction_position;
  int pos = 0;
  for (const HloInstruction* instruction :
       sequence(computation).instructions()) {
    TF_RET_CHECK(instruction_position.insert({instruction, pos}).second)
        << "Instruction " << instruction->name()
        << " appears more than once in the schedule";
    pos++;
  }
  TF_RET_CHECK(instruction_position.size() == computation->instruction_count())
      << "Schedule for computation " << computation->name() << " has "
      << instruction_position.size() << " instructions, expected "
      << computation->instruction_count();
  for (const HloInstruction* instruction : computation->instructions()) {
    TF_RET_CHECK(instruction_position.contains(instruction))
        << "Instruction " << instruction->name() << " is not in schedule";
  }

  for (const HloInstruction* instruction : computation->instructions()) {
    for (const HloInstruction* operand : instruction->operands()) {
      TF_RET_CHECK(instruction_position.at(operand) <
                   instruction_position.at(instruction))
          << "Instruction " << instruction->name()
          << " is not scheduled after its operand " << operand->name();
    }

    for (const HloInstruction* pred : instruction->control_predecessors()) {
      TF_RET_CHECK(instruction_position.at(pred) <
                   instruction_position.at(instruction))
          << "Instruction " << instruction->name()
          << " is not scheduled after its control predecessor " << pred->name();
    }
  }
  return absl::OkStatus();
}

namespace {

// Returns the computation in the given module with the given unique ID. Returns
// nullptr if no such computation exists.
const HloComputation* IdToComputation(const HloModule* module, int64_t id) {
  for (const HloComputation* computation : module->computations()) {
    if (computation->unique_id() == id) {
      return computation;
    }
  }
  return nullptr;
}

}  // namespace

std::string HloSchedule::ToString() const {
  std::vector<std::string> pieces;

  pieces.push_back("HloSchedule");
  std::vector<int64_t> sorted_ids;
  for (const auto& id_sequence : sequences_) {
    sorted_ids.push_back(id_sequence.first);
  }
  absl::c_sort(sorted_ids);

  for (const int64_t id : sorted_ids) {
    const HloComputation* computation = IdToComputation(module_, id);
    const HloInstructionSequence& sequence = sequences_.at(id);
    if (computation == nullptr) {
      // The computation is not in the module and may have been deleted so it is
      // not safe to dereference any HLO pointers. Just use the HLO unique ids
      // stored in this object.
      pieces.push_back(absl::StrFormat(
          "computation with id %d (no longer in HLO module):", id));
      for (int64_t id : sequence.ids()) {
        pieces.push_back(absl::StrCat("  ", id));
      }
    } else {
      pieces.push_back(
          absl::StrFormat("computation %s (thread %s):", computation->name(),
                          computation->execution_thread()));
      for (const HloInstruction* instruction : sequence.instructions()) {
        pieces.push_back(absl::StrCat("  ", instruction->name()));
      }
    }
  }
  return absl::StrJoin(pieces, "\n");
}

std::ostream& operator<<(std::ostream& out, const HloSchedule& schedule) {
  return out << schedule.ToString();
}

}  // namespace xla

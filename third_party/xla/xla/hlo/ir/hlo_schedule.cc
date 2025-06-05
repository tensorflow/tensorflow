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
    const HloModule* module, const HloScheduleProto& proto) {
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
      id_to_instruction[instruction->unique_id_64_bits()] = instruction;
    }

    HloInstructionSequence& sequence =
        schedule.GetOrCreateSequence(computation);
    for (const int64_t instruction_id : id_sequence.second.instruction_ids()) {
      auto instr_it = id_to_instruction.find(instruction_id);
      TF_RET_CHECK(instr_it != id_to_instruction.end())
          << "No instruction exists in HLO computation " << computation->name()
          << " with id " << instruction_id;
      sequence.push_back(instr_it->second);
    }
  }
  TF_RETURN_IF_ERROR(schedule.Verify());
  return schedule;
}

absl::StatusOr<HloScheduleProto> HloSchedule::ToProto() const {
  TF_RETURN_IF_ERROR(Verify());
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

absl::Status HloSchedule::UpdateComputationSchedule(
    const HloComputation* computation) {
  // Map from unique ID to HloInstruction pointer for instructions in the
  // computation.
  absl::flat_hash_map<int64_t, HloInstruction*> id_to_instruction;
  for (HloInstruction* instruction : computation->instructions()) {
    InsertOrDie(&id_to_instruction, instruction->unique_id_64_bits(),
                instruction);
  }

  // Set of all HloInstructions in the schedule.
  absl::flat_hash_set<int64_t> ids_in_schedule;
  for (int id : sequences_.at(computation->unique_id()).ids()) {
    InsertOrDie(&ids_in_schedule, id);
  }

  // Map from HloInstruction X to newly added instructions (instruction is in
  // computation, but not in schedule) which use X. If an instruction is not in
  // the map, then it has no users which are newly added instructions.
  absl::flat_hash_map<const HloInstruction*, std::vector<HloInstruction*>>
      new_instruction_uses;

  // For each newly added instruction, this is the count of the instruction's
  // operands that have not yet been scheduled. When this value reaches zero,
  // then the instruction may be placed in the schedule.
  absl::flat_hash_map<const HloInstruction*, int> unscheduled_operand_count;

  // Create a worklist of newly added instructions which are ready to be added
  // to the schedule. Initialize worklist with those that have zero operands.
  std::queue<HloInstruction*> worklist;

  for (HloInstruction* instruction : computation->instructions()) {
    if (!ids_in_schedule.contains(instruction->unique_id_64_bits())) {
      // This is a newly added instruction which is not in the schedule.
      if (instruction->operands().empty()) {
        worklist.push(instruction);
      } else {
        for (const HloInstruction* operand : instruction->operands()) {
          new_instruction_uses[operand].push_back(instruction);
        }
        unscheduled_operand_count[instruction] = instruction->operand_count();
      }
    }
  }

  // Update the schedule with the newly added instructions, and remove any
  // instructions no longer in the graph.
  HloInstructionSequence new_sequence;

  // Lambda which schedules all instructions on the worklist.
  auto schedule_worklist = [&]() {
    while (!worklist.empty()) {
      HloInstruction* instruction = worklist.front();
      worklist.pop();
      new_sequence.push_back(instruction);
      std::vector<HloInstruction*>* new_users =
          tsl::gtl::FindOrNull(new_instruction_uses, instruction);
      if (new_users != nullptr) {
        // This just-scheduled instruction has users which are newly added to
        // the module. Update the number of unscheduled operands and push the
        // newly added instruction to the worklist if it is ready to
        // schedule.
        for (HloInstruction* new_user : *new_users) {
          unscheduled_operand_count.at(new_user)--;
          CHECK_GE(unscheduled_operand_count.at(new_user), 0);
          if (unscheduled_operand_count.at(new_user) == 0) {
            worklist.push(new_user);
          }
        }
      }
    }
  };

  schedule_worklist();
  for (int64_t id : sequences_.at(computation->unique_id()).ids()) {
    auto it = id_to_instruction.find(id);
    if (it == id_to_instruction.end()) {
      // This instruction in the schedule is no longer in the module. Do not add
      // it to the new schedule.
      continue;
    }
    worklist.push(it->second);
    schedule_worklist();
  }

  set_sequence(computation, std::move(new_sequence));
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
      TF_RETURN_IF_ERROR(UpdateComputationSchedule(computation));
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
    TF_RETURN_IF_ERROR(UpdateComputationSchedule(computation));
  }

  TF_RETURN_IF_ERROR(Verify());
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
    TF_RET_CHECK(nonfusion_computations.size() == sequence_size)
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
      absl::flat_hash_map<const HloInstruction*, int> instruction_position;
      int pos = 0;
      for (const HloInstruction* instruction :
           sequence(computation).instructions()) {
        TF_RET_CHECK(instruction_position.insert({instruction, pos}).second)
            << "Instruction " << instruction->name()
            << " appears more than once in the schedule";
        pos++;
      }

      TF_RET_CHECK(instruction_position.size() ==
                   computation->instruction_count())
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
              << " is not scheduled after its control predecessor "
              << pred->name();
        }
      }
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
      for (int id : sequence.ids()) {
        pieces.push_back(absl::StrCat("  ", id));
      }
    } else {
      pieces.push_back(absl::StrFormat("computation %s:", computation->name()));
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

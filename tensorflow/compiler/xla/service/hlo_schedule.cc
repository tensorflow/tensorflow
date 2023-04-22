/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_schedule.h"

#include <queue>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/gtl/map_util.h"

namespace xla {

/* static */ StatusOr<HloSchedule> HloSchedule::CreateFromProto(
    const HloModule* module, const HloScheduleProto& proto) {
  absl::flat_hash_map<int64, const HloComputation*> id_to_computation;
  for (const HloComputation* computation : module->computations()) {
    id_to_computation[computation->unique_id()] = computation;
  }

  HloSchedule schedule(module);
  for (const auto& id_sequence : proto.sequences()) {
    int64 computation_id = id_sequence.first;

    auto comp_it = id_to_computation.find(computation_id);
    TF_RET_CHECK(comp_it != id_to_computation.end())
        << "No computation exists in HLO module with id " << computation_id;
    const HloComputation* computation = comp_it->second;

    absl::flat_hash_map<int64, HloInstruction*> id_to_instruction;
    for (HloInstruction* instruction : computation->instructions()) {
      id_to_instruction[instruction->unique_id()] = instruction;
    }

    HloInstructionSequence& sequence =
        schedule.GetOrCreateSequence(computation);
    for (const int64 instruction_id : id_sequence.second.instruction_ids()) {
      auto instr_it = id_to_instruction.find(instruction_id);
      TF_RET_CHECK(instr_it != id_to_instruction.end())
          << "No instruction exists in HLO computation " << computation->name()
          << " with id " << instruction_id;
      sequence.push_back(instr_it->second);
    }
  }
  TF_RETURN_IF_ERROR(schedule.Verify());
  return std::move(schedule);
}

StatusOr<HloScheduleProto> HloSchedule::ToProto() const {
  TF_RETURN_IF_ERROR(Verify());
  HloScheduleProto proto;
  for (const auto& id_sequence : sequences_) {
    int64 computation_id = id_sequence.first;
    const HloInstructionSequence& sequence = id_sequence.second;
    HloScheduleProto::InstructionSequence& proto_sequence =
        (*proto.mutable_sequences())[computation_id];
    proto_sequence.mutable_instruction_ids()->Reserve(sequence.size());
    for (const int64 id : sequence.ids()) {
      proto_sequence.add_instruction_ids(id);
    }
  }
  return std::move(proto);
}

void HloSchedule::set_sequence(const HloComputation* computation,
                               absl::Span<HloInstruction* const> sequence) {
  set_sequence(computation, HloInstructionSequence(sequence));
}

void HloSchedule::set_sequence(const HloComputation* computation,
                               HloInstructionSequence sequence) {
  CHECK(computation->parent() == module_);
  sequences_[computation->unique_id()] = std::move(sequence);
}

HloInstructionSequence& HloSchedule::GetOrCreateSequence(
    const HloComputation* computation) {
  auto it = sequences_.find(computation->unique_id());
  if (it == sequences_.end()) {
    // No sequence found for computation. Create and return an empty one.
    CHECK(computation->parent() == module_);
    return sequences_[computation->unique_id()];
  } else {
    return it->second;
  }
}

const HloInstructionSequence& HloSchedule::sequence(
    const HloComputation* computation) const {
  return sequences_.at(computation->unique_id());
}

Status HloSchedule::UpdateComputationSchedule(
    const HloComputation* computation) {
  // Map from unique ID to HloInstruction pointer for instructions in the
  // computation.
  absl::flat_hash_map<int, HloInstruction*> id_to_instruction;
  for (HloInstruction* instruction : computation->instructions()) {
    InsertOrDie(&id_to_instruction, instruction->unique_id(), instruction);
  }

  // Set of all HloInstructions in the schedule.
  absl::flat_hash_set<int> ids_in_schedule;
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
    if (!ids_in_schedule.contains(instruction->unique_id())) {
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
          tensorflow::gtl::FindOrNull(new_instruction_uses, instruction);
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
  for (int id : sequences_.at(computation->unique_id()).ids()) {
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
  return Status::OK();
}

Status HloSchedule::Update() {
  // The schedule must contain a sequence for every non-fusion computation in
  // the module, but can have sequences for computations which no longer exist
  // (these are removed).
  std::vector<HloComputation*> nonfusion_computations =
      module_->MakeNonfusionComputations();
  for (const HloComputation* computation : nonfusion_computations) {
    TF_RET_CHECK(sequences_.contains(computation->unique_id()))
        << "Computation " << computation->name() << " not in HloSchedule.";
  }
  if (sequences_.size() > nonfusion_computations.size()) {
    // Schedule contains some computations which have been removed from the
    // HloModule. Remove them from the schedule as well.
    absl::flat_hash_set<int64> nonfusion_computations_ids;
    for (const HloComputation* computation : nonfusion_computations) {
      nonfusion_computations_ids.insert(computation->unique_id());
    }
    for (auto it = sequences_.begin(); it != sequences_.end();) {
      if (!nonfusion_computations_ids.contains(it->first)) {
        sequences_.erase(it++);
      } else {
        ++it;
      }
    }
  }
  CHECK_EQ(sequences_.size(), nonfusion_computations.size());

  for (const HloComputation* computation : nonfusion_computations) {
    TF_RETURN_IF_ERROR(UpdateComputationSchedule(computation));
  }

  TF_RETURN_IF_ERROR(Verify());
  return Status::OK();
}

Status HloSchedule::Verify() const {
  VLOG(2) << "VerifySchedule()";
  XLA_VLOG_LINES(2, ToString());

  // Verify schedule contains exactly the same set of non-fusion computations as
  // module currently does.
  std::vector<HloComputation*> nonfusion_computations =
      module_->MakeNonfusionComputations();
  TF_RET_CHECK(nonfusion_computations.size() == sequences_.size())
      << "Schedule has " << sequences_.size() << " sequences, but module has "
      << nonfusion_computations.size() << " non-fusion computations";
  for (const HloComputation* computation : nonfusion_computations) {
    TF_RET_CHECK(sequences_.contains(computation->unique_id()))
        << "Computation " << computation->name()
        << " missing from HLO schedule.";
  }

  // For each computation verify the set of instructions is the same and that
  // each dependency and control edge is honored.
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

  return Status::OK();
}

namespace {

// Returns the computation in the given module with the given unique ID. Returns
// nullptr if no such computation exists.
const HloComputation* IdToComputation(const HloModule* module, int64 id) {
  for (const HloComputation* computation : module->computations()) {
    if (computation->unique_id() == id) {
      return computation;
    }
  }
  return nullptr;
}

}  // namespace

string HloSchedule::ToString() const {
  std::vector<string> pieces;

  pieces.push_back("HloSchedule");
  for (const auto& id_sequence : sequences_) {
    const HloComputation* computation =
        IdToComputation(module_, id_sequence.first);
    if (computation == nullptr) {
      // The computation is not in the module and may have been deleted so it is
      // not safe to dereference any HLO pointers. Just use the HLO unique ids
      // stored in this object.
      pieces.push_back(
          absl::StrFormat("computation with id %d (no longer in HLO module):",
                          id_sequence.first));
      for (int id : id_sequence.second.ids()) {
        pieces.push_back(absl::StrCat("  ", id));
      }
    } else {
      pieces.push_back(absl::StrFormat("computation %s:", computation->name()));
      for (const HloInstruction* instruction :
           id_sequence.second.instructions()) {
        pieces.push_back(absl::StrCat("  ", instruction->name()));
      }
    }
  }
  return absl::StrJoin(pieces, "\n");
}

std::ostream& operator<<(std::ostream& out, const HloSchedule& schedule) {
  out << schedule.ToString();
  return out;
}

}  // namespace xla

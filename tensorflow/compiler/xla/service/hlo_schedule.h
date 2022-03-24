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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SCHEDULE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SCHEDULE_H_

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status.h"

namespace xla {

class HloModule;

// Class representing a sequence of HLO instructions such as the sequential
// execution order of an HLO computation.
class HloInstructionSequence {
 public:
  HloInstructionSequence() = default;
  explicit HloInstructionSequence(
      absl::Span<HloInstruction* const> instructions) {
    for (HloInstruction* instruction : instructions) {
      push_back(instruction);
    }
  }

  // Adds the instruction to the end of the sequence.
  void push_back(HloInstruction* instruction) {
    instruction_sequence_.push_back(instruction);
    id_sequence_.push_back(instruction->unique_id());
  }

  // Removes the instruction from the sequence.
  void remove_instruction(HloInstruction* instruction) {
    auto instruction_it = std::find(instruction_sequence_.begin(),
                                    instruction_sequence_.end(), instruction);
    if (instruction_it != instruction_sequence_.end()) {
      auto id_it = std::find(id_sequence_.begin(), id_sequence_.end(),
                             instruction->unique_id());
      instruction_sequence_.erase(instruction_it);
      id_sequence_.erase(id_it);
    }
  }

  // Replaces the old instruction with the new instruction in the sequence.
  void replace_instruction(HloInstruction* old_instruction,
                           HloInstruction* new_instruction) {
    auto instruction_it =
        std::find(instruction_sequence_.begin(), instruction_sequence_.end(),
                  old_instruction);
    auto id_it = std::find(id_sequence_.begin(), id_sequence_.end(),
                           old_instruction->unique_id());
    CHECK(instruction_it != instruction_sequence_.end())
        << "Do not find instruction id " << old_instruction->unique_id();
    CHECK(id_it != id_sequence_.end());
    *instruction_it = new_instruction;
    *id_it = new_instruction->unique_id();
  }

  // Clears the sequence of all instructions.
  void clear() {
    instruction_sequence_.clear();
    id_sequence_.clear();
  }

  int64_t size() const { return instruction_sequence_.size(); }

  // Returns the sequence of HLO instructions.
  const std::vector<HloInstruction*>& instructions() const {
    return instruction_sequence_;
  }

  // Returns the unique IDs of the instructions in the sequence (in order).
  const std::vector<int>& ids() const { return id_sequence_; }

 private:
  // The sequence as HloInstructions.
  std::vector<HloInstruction*> instruction_sequence_;

  // The sequence of HLO instructions, represented by their unique IDs. The
  // sequence is stored as both HloInstructions and unique IDs because the
  // sequence may be referenced after transformations to the HLO graph and HLO
  // pointers can be invalidated or recycled in this process (see
  // HloSchedule::Update).
  std::vector<int> id_sequence_;
};

// A class representing a sequential schedule of instructions for an HLO
// module. A complete HLO schedule contains an instruction sequence for every
// non-fusion computation in the HLO module.
class HloSchedule {
 public:
  explicit HloSchedule(const HloModule* module) : module_(module) {}

  // (De)Serialize an HloSchedule to/from a HloScheduleProto.
  static StatusOr<HloSchedule> CreateFromProto(const HloModule* module,
                                               const HloScheduleProto& proto);
  StatusOr<HloScheduleProto> ToProto() const;

  // Returns a reference to the sequence for the given computation.
  const HloInstructionSequence& sequence(
      const HloComputation* computation) const;

  // Returns the sequence for the given computation. An empty sequence is
  // created if none exists for the computation.
  HloInstructionSequence& GetOrCreateSequence(
      const HloComputation* computation);

  // Sets the sequence for the given computation to the given sequence.
  void set_sequence(const HloComputation* computation,
                    absl::Span<HloInstruction* const> sequence);
  void set_sequence(const HloComputation* computation,
                    HloInstructionSequence sequence);

  // Returns a map from HloComputation unique ID to instruction sequence. The
  // map contains all sequences in the schedule.
  const absl::flat_hash_map<int64_t, HloInstructionSequence>& sequences()
      const {
    return sequences_;
  }

  // Returns true if the schedule has a sequence for the given computation.
  bool is_computation_scheduled(const HloComputation* computation) const {
    return sequences_.contains(computation->unique_id());
  }

  // Removes the computation from the sequences.
  void remove_computation(const HloComputation* computation) {
    auto it = sequences_.find(computation->unique_id());
    CHECK(it != sequences_.end());
    sequences_.erase(it);
  }

  // Removes the instruction from the computation's sequence.
  void remove_instruction(const HloComputation* computation,
                          HloInstruction* instruction) {
    sequences_[computation->unique_id()].remove_instruction(instruction);
  }

  // Replaces the old instruction with the new instruction in the computation's
  // sequence.
  void replace_instruction(const HloComputation* computation,
                           HloInstruction* old_instruction,
                           HloInstruction* new_instruction) {
    sequences_[computation->unique_id()].replace_instruction(old_instruction,
                                                             new_instruction);
  }

  // Updates the schedule such that it is (again) a valid schedule for the
  // module. This is used to update a schedule after the HLO module has been
  // transformed in some way. In general, the only transformations to the module
  // for which a schedule can be updated is the addition or removal of
  // instructions and removal of computations. Updating the schedule after new
  // dependencies between existing instructions in the module is not supported
  // and may result in an error status returned.
  //
  // Instructions in the module which also exist in the given schedule will
  // remain in the same order in the updated schedule. Instructions which exist
  // in the module but not in the given schedule will be placed as early as
  // possible in the updated schedule.
  Status Update();

  // Verifies that the given schedule is valid for the given module.
  // Specifically, the schedule contains exactly the instructions in the
  // non-fusion computations in the module and every dependency in the module is
  // satisfied in the schedule.
  Status Verify() const;

  std::string ToString() const;

  bool empty() const { return sequences_.empty(); }

  const HloModule* module() const { return module_; }

 private:
  // Updates the instruction sequence for the given computation.
  Status UpdateComputationSchedule(const HloComputation* computation);

  const HloModule* module_;

  // A map from computation unique ID to instruction sequence. Unique IDs are
  // used rather than HloComputation pointers because HLO pointers are not
  // unique across HLO transformations because pointers may be recycled.
  absl::flat_hash_map<int64_t, HloInstructionSequence> sequences_;
};

std::ostream& operator<<(std::ostream& out, const HloSchedule& schedule);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SCHEDULE_H_

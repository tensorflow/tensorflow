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

#ifndef XLA_HLO_IR_HLO_CLONE_CONTEXT_H_
#define XLA_HLO_IR_HLO_CLONE_CONTEXT_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/map_util.h"

namespace xla {

class HloInstruction;
class HloComputation;
class HloModule;

// Data structure used to track the cloning of HloInstruction and HloComputation
// objects.
class HloCloneContext {
 public:
  // Creates a new HloCloneContext object to clone HloInstruction and
  // HloComputation objects to be added to the module specified as argument.
  // The suffix string will be appended to computation names.
  explicit HloCloneContext(HloModule* module, absl::string_view suffix = "")
      : module_(module), suffix_(suffix) {}

  HloModule* module() const { return module_; }

  const std::string& suffix() const { return suffix_; }

  void MapInstruction(const HloInstruction* old_instruction,
                      HloInstruction* new_instruction);

  void MapComputation(const HloComputation* old_computation,
                      HloComputation* new_computation);

  // Finds the new instruction mapped to its old copy, or return nullptr in case
  // it is not found.
  HloInstruction* FindInstruction(const HloInstruction* old_instruction) const {
    return FindOrDefault(instructions_, old_instruction, nullptr);
  }

  // Finds the new instruction mapped to the old instruction named `old_name`,
  // or return nullptr in case it is not found.
  HloInstruction* FindInstruction(absl::string_view old_name) const {
    return FindOrDefault(instructions_by_name_, std::string(old_name), nullptr);
  }

  // Finds the new computation mapped to its old copy, or return nullptr in case
  // it is not found.
  HloComputation* FindComputation(const HloComputation* old_computation) const {
    return FindOrDefault(computations_, old_computation, nullptr);
  }

  // Finds the new computation mapped to the old computation named `old_name`,
  // or return nullptr in case it is not found.
  HloComputation* FindComputation(absl::string_view old_name) const {
    return FindOrDefault(computations_by_name_, std::string(old_name), nullptr);
  }

  // Retrieves the new instruction mapped to its old copy, or fail if not found.
  HloInstruction* GetInstruction(const HloInstruction* old_instruction) const {
    return FindOrDie(instructions_, old_instruction);
  }

  // Retrieves the new instruction mapped to the old instruction named
  // `old_name`, or fail if not found.
  HloInstruction* GetInstruction(absl::string_view old_name) const {
    return FindOrDie(instructions_by_name_, std::string(old_name));
  }

  // Retrieves the new computation mapped to its old copy, or fail if not found.
  HloComputation* GetComputation(const HloComputation* old_computation) const {
    return FindOrDie(computations_, old_computation);
  }

  // Retrieves the new computation mapped to the old computation named
  // `old_name`, or fail if not found.
  HloComputation* GetComputation(absl::string_view old_name) const {
    return FindOrDie(computations_by_name_, std::string(old_name));
  }

  const absl::flat_hash_map<const HloInstruction*, HloInstruction*>&
  cloned_instructions() const {
    return instructions_;
  }

  const absl::flat_hash_map<std::string, HloInstruction*>&
  cloned_instructions_by_name() const {
    return instructions_by_name_;
  }

  const absl::flat_hash_map<const HloComputation*, HloComputation*>&
  cloned_computations() const {
    return computations_;
  }

  const absl::flat_hash_map<std::string, HloComputation*>&
  cloned_computations_by_name() const {
    return computations_by_name_;
  }

 private:
  HloCloneContext(const HloCloneContext&) = delete;
  const HloCloneContext& operator=(const HloCloneContext&) = delete;

  HloModule* module_;
  std::string suffix_;
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> instructions_;
  absl::flat_hash_map<std::string, HloInstruction*> instructions_by_name_;
  absl::flat_hash_map<const HloComputation*, HloComputation*> computations_;
  absl::flat_hash_map<std::string, HloComputation*> computations_by_name_;
};

}  // namespace xla

#endif  // XLA_HLO_IR_HLO_CLONE_CONTEXT_H_

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_CLONE_CONTEXT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_CLONE_CONTEXT_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/map_util.h"

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
  explicit HloCloneContext(HloModule* module, const string& suffix = "")
      : module_(module), suffix_(suffix) {}

  HloModule* module() const { return module_; }

  const string& suffix() const { return suffix_; }

  void MapInstruction(const HloInstruction* old_instruction,
                      HloInstruction* new_instruction) {
    instructions_[old_instruction] = new_instruction;
  }

  void MapComputation(const HloComputation* old_computation,
                      HloComputation* new_computation) {
    computations_[old_computation] = new_computation;
  }

  // Finds the new instruction mapped to its old copy, or return nullptr in case
  // it is not found.
  HloInstruction* FindInstruction(const HloInstruction* old_instruction) const {
    return FindOrDefault(instructions_, old_instruction, nullptr);
  }

  // Finds the new computation mapped to its old copy, or return nullptr in case
  // it is not found.
  HloComputation* FindComputation(const HloComputation* old_computation) const {
    return FindOrDefault(computations_, old_computation, nullptr);
  }

  // Retrieves the new instruction mapped to its old copy, or fail if not found.
  HloInstruction* GetInstruction(const HloInstruction* old_instruction) const {
    return FindOrDie(instructions_, old_instruction);
  }

  // Retrieves the new computation mapped to its old copy, or fail if not found.
  HloComputation* GetComputation(const HloComputation* old_computation) const {
    return FindOrDie(computations_, old_computation);
  }

  const absl::flat_hash_map<const HloInstruction*, HloInstruction*>&
  cloned_instructions() const {
    return instructions_;
  }

  const absl::flat_hash_map<const HloComputation*, HloComputation*>&
  cloned_computations() const {
    return computations_;
  }

 private:
  HloModule* module_;
  string suffix_;
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> instructions_;
  absl::flat_hash_map<const HloComputation*, HloComputation*> computations_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_CLONE_CONTEXT_H_

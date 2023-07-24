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

#ifndef TENSORFLOW_COMPILER_XLA_TOOLS_HLO_SLICER_H_
#define TENSORFLOW_COMPILER_XLA_TOOLS_HLO_SLICER_H_

#include <functional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"

namespace xla {

// Define FrontierSelector, which is a lambda that, given an HLO
// instruction, returns true if we should continue propagation from this
// instruction, otherwise return false.
using FrontierSelector = std::function<bool(const HloInstruction*)>;

// The data structure capturing the outputs of forward/backward slicing.
class SliceOutput {
 public:
  SliceOutput(absl::flat_hash_map<const HloComputation*,
                                  absl::flat_hash_set<const HloInstruction*>>
                  sliced_instructions,
              absl::flat_hash_map<const HloComputation*,
                                  absl::flat_hash_set<const HloInstruction*>>
                  frontier_instructions)
      : sliced_instructions_(sliced_instructions),
        frontier_instructions_(frontier_instructions) {}

  const absl::flat_hash_map<const HloComputation*,
                            absl::flat_hash_set<const HloInstruction*>>&
  sliced_instructions() const {
    return sliced_instructions_;
  }

  const absl::flat_hash_map<const HloComputation*,
                            absl::flat_hash_set<const HloInstruction*>>&
  frontier_instructions() const {
    return frontier_instructions_;
  }

  // Return the total number of the sliced instructions
  int NumSlicedInstructions() const {
    return CountMapOfSet(sliced_instructions_);
  }

  // Return the total number of the frontier instructions
  int NumFrontierInstructions() const {
    return CountMapOfSet(frontier_instructions_);
  }

 private:
  // A map that maps from sliced HLO computation to sliced HLO
  // instructions (excluding the parts of the HLO computations/instructions that
  // are irrelevant).
  absl::flat_hash_map<const HloComputation*,
                      absl::flat_hash_set<const HloInstruction*>>
      sliced_instructions_;

  // A map that maps from the computations to the instructions that form the
  // slicing frontier.
  absl::flat_hash_map<const HloComputation*,
                      absl::flat_hash_set<const HloInstruction*>>
      frontier_instructions_;

  int CountMapOfSet(
      const absl::flat_hash_map<const HloComputation*,
                                absl::flat_hash_set<const HloInstruction*>>&
          to_count) const {
    int count = 0;
    for (const auto& [key, set] : to_count) {
      count += set.size();
    }
    return count;
  }
};

// Conduct inter-computation program slicing.
//
// `slice_starting_instructions`: the starting HLO instructions of slicing.
// `ignore_control_dependency`: if set as true, control dependency will be
// ignored during slicing.
// `frontier_selector`: a lambda function that dictates the ending points (i.e.,
// frontier) of the slicing.
//
// 'forward_slice': conduct forward slicing (i.e., in the direction that from
// the `slice_starting_instructions` to the ROOT) if set as true, conduct
// backward slicing (i.e., from the `slice_starting_instructions` to the leaf
// nodes) otherwise.
SliceOutput SliceModule(
    const HloModule* hlo_module,
    absl::Span<const HloInstruction*> slice_starting_instructions,
    FrontierSelector frontier_selector, bool ignore_control_dependency = false,
    bool forward_slice = true);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TOOLS_HLO_SLICER_H_

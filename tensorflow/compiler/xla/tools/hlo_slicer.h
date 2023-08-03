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
                  frontier_instructions,
              const HloInstruction* nearest_common_ancestor_root = nullptr)
      : sliced_instructions_(sliced_instructions),
        frontier_instructions_(frontier_instructions),
        nearest_common_ancestor_root_(nearest_common_ancestor_root) {}

  // Returns all the instructions that are sliced, grouped by their parent
  // computation.
  const absl::flat_hash_map<const HloComputation*,
                            absl::flat_hash_set<const HloInstruction*>>&
  sliced_instructions() const {
    return sliced_instructions_;
  }

  // Returns all the instructions that are determined to be at the frontier,
  // grouped by their parent computation.
  const absl::flat_hash_map<const HloComputation*,
                            absl::flat_hash_set<const HloInstruction*>>&
  frontier_instructions() const {
    return frontier_instructions_;
  }

  // Returns the total number of the sliced instructions
  int NumSlicedInstructions() const {
    return CountMapOfSet(sliced_instructions_);
  }

  // Returns the total number of the frontier instructions
  int NumFrontierInstructions() const {
    return CountMapOfSet(frontier_instructions_);
  }

  // If forward slicing and "nearest_common_ancestor_as_new_root" are specified,
  // return the nearest common ancestor as an HLO instruction. Otherwise, return
  // nullptr.
  const HloInstruction* nearest_common_ancestor_root() const {
    return nearest_common_ancestor_root_;
  }

  // Computes the intersection of the sliced instructions
  // from two SliceOutput.
  static absl::flat_hash_map<const HloComputation*,
                             absl::flat_hash_set<const HloInstruction*>>
  IntersectSlicedInstructions(SliceOutput slice_a, SliceOutput slice_b) {
    absl::flat_hash_map<const HloComputation*,
                        absl::flat_hash_set<const HloInstruction*>>
        intersect_sliced_instructions;
    auto& sliced_instructions_a = slice_a.sliced_instructions();
    auto& sliced_instructions_b = slice_b.sliced_instructions();
    for (auto& [computation, instructions] : sliced_instructions_a) {
      for (auto& instruction : instructions) {
        if (sliced_instructions_b.contains(computation) &&
            sliced_instructions_b.at(computation).contains(instruction)) {
          intersect_sliced_instructions[computation].insert(instruction);
        }
      }
    }
    return intersect_sliced_instructions;
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

  // The computed nearest common ancestor.
  const HloInstruction* nearest_common_ancestor_root_;

  // Counts the number of HloInstruction in the data structure
  // `map<HloComputation, set<HloInstruction>>`.
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
//
// `nearest_common_ancestor_as_root`: this option is only available when
// `forward_slice` is true and `FrontierSelector` is not specified. When
// enabled, this function would compute one of the nearest common ancestor (NCA)
// as an HloInstructionm with `slice_starting_instructions` as the starting
// points, and only slice down to the NCA . Note that there could be multiple
// NCAs in the DAG. We would use the first NCA we encounter during the
// traversal as the root, and return it as `nearest_common_ancestor_root` in
// `SliceOutput`. Please check the test
// `HloSlicerTest.ForwardSlicingNearestCommonAncestor` and
// `MultipleComputationForwardSlicingNearestCommonAncestor` for detailed
// examples. Use the original root in the entry computation as the forward
// slicing root if this option is not enabled.
SliceOutput SliceModule(
    const HloModule* hlo_module,
    absl::Span<const HloInstruction*> slice_starting_instructions,
    FrontierSelector frontier_selector = nullptr,
    bool ignore_control_dependency = false, bool forward_slice = true,
    bool nearest_common_ancestor_as_root = false);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TOOLS_HLO_SLICER_H_

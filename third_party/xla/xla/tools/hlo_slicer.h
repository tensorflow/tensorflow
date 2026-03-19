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

#ifndef XLA_TOOLS_HLO_SLICER_H_
#define XLA_TOOLS_HLO_SLICER_H_

#include <functional>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"

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

  explicit SliceOutput(
      absl::flat_hash_map<const HloComputation*,
                          absl::flat_hash_set<const HloInstruction*>>
          sliced_instructions)
      : sliced_instructions_(sliced_instructions) {}

  // Default constructor.
  SliceOutput() = default;

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

  // Computes the intersection of the sliced instructions from two SliceOutput.
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

  // Computes the union of the sliced instructions from two SliceOutput.
  static absl::flat_hash_map<const HloComputation*,
                             absl::flat_hash_set<const HloInstruction*>>
  UnionSlicedInstructions(SliceOutput slice_a, SliceOutput slice_b) {
    absl::flat_hash_map<const HloComputation*,
                        absl::flat_hash_set<const HloInstruction*>>
        union_sliced_instructions;
    auto& sliced_instructions_a = slice_a.sliced_instructions();
    auto& sliced_instructions_b = slice_b.sliced_instructions();

    for (auto& sliced_instructions :
         {sliced_instructions_a, sliced_instructions_b}) {
      for (auto& [computation, instructions] : sliced_instructions) {
        for (auto& instruction : instructions) {
          union_sliced_instructions[computation].insert(instruction);
        }
      }
    }
    return union_sliced_instructions;
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

// Specifies slicing configurations.
//
// `forward_slicing`: how forward slicing is conducted from the
// the hlo instructions we are starting slicing from.
//    kRoot: slice to the root instruction of the entry computation.
//    kNca: slice to the nearest common ancestors of the starting hlo
//    instructions.
//
// `backward_slicing`: if backward slicing is conducted from the hlo
// instructions we are starting slicing from.
//
// `remove_sharding`: if the custom call to Sharding should be removed. If
// specified as true, the custom call instruction to sharding (e.g.,
// %custom-call = bf16[8] custom-call(bf16[8] %multiply),
// custom_call_target="Sharding", sharding={replicated}) will be removed./
//
// `reduce_tuple_parameter`: If specified as true, we will try to reduce the
// size of parameters of entry computation if they are tuple. Specifically, for
// each parameters of entry computation, if it is of tuple type, we will remove
// the elements that are not used by any other instructions. This is useful when
// slicing from a large module.
//
// `slicing_group`: `SliceModuleAndExtract` groups
// `slicing_starting_instructions` into multiple non-overlapping groups, and
// for each group of `slicing_starting_instructions`, slice/extract an HLO
// module. The `slicing_group` specifies the number of
// `slicing_starting_instructions` each group contains. For example,
// say `slicing_start_instructions` = {a, b, c ,d}. If `slicing_group` = 1,
// there would be 4 sliced/extracted HLO modules, sliced from {a}, {b}, {c},
// {d}, respectively. If `slicing_group` = 2, there would be 2 sliced/extracted
// HLO modules, sliced from {a, b}, {c, d}, respectively. The
// `slicing_starting_instructions` are grouped accoding to order in the
// absl::Span. When `slicing_group` = -1, there would be only one group which
// contains all the `slice_starting_instructions`, so there would be only 1
// sliced/extracted module. `slicing_group` can only be -1 or positive integer.
struct SlicingConfiguration {
  enum class ForwardSlicingConfig { kRoot, kNca };
  ForwardSlicingConfig forward_slicing = ForwardSlicingConfig::kRoot;
  bool backward_slicing = false;
  bool remove_sharding = false;
  bool reduce_tuple_parameter = false;
  int slicing_group = -1;
};

// Slices from the `hlo_module` from the `slicing_starting_instructions`,
// following configurations specified by `slicing_configuration`, and return
// (multiple) sliced hlo modules.
//
// `slice_starting_instructions`: the starting HLO instructions of slicing.
//
// `slicing_configuration`: specifies how the slicing is conducted. Please
// check more details at the comments of `SlicingConfiguration`.
std::vector<std::unique_ptr<HloModule>> SliceModuleAndExtract(
    const HloModule* hlo_module,
    absl::Span<const HloInstruction*> slice_starting_instructions,
    const SlicingConfiguration& slicing_configuration);

}  // namespace xla

#endif  // XLA_TOOLS_HLO_SLICER_H_

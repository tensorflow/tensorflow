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

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"

namespace xla {

// Conduct inter-computation forward program slicing, with  the provided
// HLO instructions as the starting points and ROOT instruction in the ENTRY
// computation as the ending point. It will return a map that maps from relevant
// HLO computation to relevant HLO instructions (excluding the parts of the HLO
// computations/instructions that are irrelevant).
absl::flat_hash_map<const HloComputation*,
                    absl::flat_hash_set<const HloInstruction*>>
SliceModule(const HloModule* hlo_module,
            std::vector<const HloInstruction*>& relevant_instructions,
            bool ignore_control_predecessors = false);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TOOLS_HLO_SLICER_H_

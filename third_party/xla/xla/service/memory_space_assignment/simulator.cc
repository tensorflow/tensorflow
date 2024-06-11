/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/memory_space_assignment/simulator.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/hlo_value.h"
#include "xla/service/memory_space_assignment/allocation.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace xla {
namespace memory_space_assignment {

float RuntimeSimulator::ComputeEstimatedElapsedTime(
    const HloLiveRange& hlo_live_range, const AllocationSequence& allocations) {
  absl::flat_hash_map<const HloInstruction*, std::vector<ShapeIndex>>
      outputs_in_alternate_memory_map;
  absl::flat_hash_map<const HloInstruction*,
                      std::vector<std::pair<int64_t, ShapeIndex>>>
      operands_in_alternate_memory_map;

  for (auto& allocation : allocations) {
    if (!allocation->is_copy_allocation()) {
      if (allocation->memory_space() == MemorySpace::kAlternate) {
        const HloInstruction* defining_instruction =
            allocation->defining_position().instruction;
        outputs_in_alternate_memory_map[defining_instruction].push_back(
            allocation->defining_position().index);
      }
    }
    for (auto& hlo_use : allocation->uses()) {
      const HloInstruction* use_instruction = hlo_use.instruction;
      operands_in_alternate_memory_map[use_instruction].push_back(
          std::make_pair(hlo_use.operand_number, hlo_use.operand_index));
    }
  }

  const auto& instruction_sequence =
      hlo_live_range.flattened_instruction_sequence().instructions();
  float total_elapsed = 0.0;
  for (const HloInstruction* instruction : instruction_sequence) {
    if (instruction->opcode() == HloOpcode::kWhile) {
      continue;
    }
    std::vector<ShapeIndex> outputs_in_alternate_memory;
    auto output_it = outputs_in_alternate_memory_map.find(instruction);
    if (output_it != outputs_in_alternate_memory_map.end()) {
      outputs_in_alternate_memory = output_it->second;
    }
    std::vector<std::pair<int64_t, ShapeIndex>> operands_in_alternate_memory;
    auto operand_it = operands_in_alternate_memory_map.find(instruction);
    if (operand_it != operands_in_alternate_memory_map.end()) {
      operands_in_alternate_memory = operand_it->second;
    }

    float instruction_elapsed_per_invoke =
        cost_analysis_->GetInstructionElapsedInAlternateMemory(
            *instruction, operands_in_alternate_memory,
            outputs_in_alternate_memory);
    float total_trip_count = cost_analysis_->CalculateNestTripCount(
        instruction, &cost_analysis_cache_);
    // Calculate total elapsed time by summing up the overall elapsed time of
    // each instruction.
    total_elapsed += total_trip_count * instruction_elapsed_per_invoke;
  }
  return total_elapsed;
}
}  // namespace memory_space_assignment
}  // namespace xla

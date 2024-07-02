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
#include <queue>
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

float RuntimeSimulator::SimulateElapsedTimeWithoutAsyncCopies(
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

float RuntimeSimulator::SimulateElapsedTime(
    const HloModule* hlo_module, const HloLiveRange& hlo_live_range,
    const AllocationSequence& allocations) {
  // Prepare the auxiliary data structures to calculate the elapsed time.
  const float kDefaultMemoryBandwidth =
      cost_analysis_->base_costs().BytesPerSecond();
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

  float total_elapsed = 0.0;
  for (auto computation : hlo_module->computations()) {
    if (hlo_module->has_schedule() &&
        hlo_module->schedule().sequences().contains(computation->unique_id())) {
      float computation_elapsed = 0.0;
      auto issued_async_copy_instructions =
          std::make_unique<std::queue<const HloInstruction*>>();
      auto remaining_size_of_buffers = std::make_unique<
          absl::flat_hash_map<const HloInstruction*, int64_t>>();
      const auto& instruction_sequence =
          hlo_module->schedule().sequence(computation);
      for (HloInstruction* instruction : instruction_sequence.instructions()) {
        float inst_elapsed = 0.0;
        if (instruction->opcode() == HloOpcode::kWhile) {
          // Since the instructions in the while body are calculated separately,
          // we can skip the while instruction.
          continue;
        } else if (instruction->opcode() == HloOpcode::kCopyStart) {
          issued_async_copy_instructions->push(instruction);
          remaining_size_of_buffers->insert(
              {instruction,
               cost_analysis_->base_costs().BytesAccessed(*instruction)});
        } else if (instruction->opcode() == HloOpcode::kCopyDone) {
          // Get corresponding copy start instruction.
          const HloInstruction* copy_start_instruction =
              instruction->operand(0);
          while (remaining_size_of_buffers->find(copy_start_instruction) !=
                 remaining_size_of_buffers->end()) {
            // Keep flushing the buffer, until complete the async copy
            const HloInstruction* front_async_copy =
                issued_async_copy_instructions->front();
            issued_async_copy_instructions->pop();
            inst_elapsed += remaining_size_of_buffers->at(front_async_copy) /
                            kDefaultMemoryBandwidth;
            remaining_size_of_buffers->erase(front_async_copy);
          }
        } else {
          // TODO(hanruobing): Plan to add another branch to handle async
          // copy instructions caused by slicing.
          std::vector<ShapeIndex> outputs_in_alternate_memory;
          auto output_it = outputs_in_alternate_memory_map.find(instruction);
          if (output_it != outputs_in_alternate_memory_map.end()) {
            outputs_in_alternate_memory = output_it->second;
          }
          std::vector<std::pair<int64_t, ShapeIndex>>
              operands_in_alternate_memory;
          auto operand_it = operands_in_alternate_memory_map.find(instruction);
          if (operand_it != operands_in_alternate_memory_map.end()) {
            operands_in_alternate_memory = operand_it->second;
          }
          // Calculate the elapsed time of the instruction.
          const float instruction_elapsed =
              cost_analysis_->GetInstructionElapsedInAlternateMemory(
                  *instruction, operands_in_alternate_memory,
                  outputs_in_alternate_memory);
          inst_elapsed = instruction_elapsed;
          float idleBandwidthTime =
              cost_analysis_->GetDefaultMemoryBandwidthIdleTime(
                  *instruction, operands_in_alternate_memory,
                  outputs_in_alternate_memory);
          while (idleBandwidthTime > 0) {
            // Try to execute as many async copies as possible
            if (issued_async_copy_instructions->empty()) {
              break;
            }
            const HloInstruction* copy_start_instruction =
                issued_async_copy_instructions->front();
            int64_t remaining_size =
                remaining_size_of_buffers->at(copy_start_instruction);
            if (remaining_size > idleBandwidthTime * kDefaultMemoryBandwidth) {
              // Can only transfer part of the copy.
              computation_elapsed += idleBandwidthTime;
              remaining_size_of_buffers->at(copy_start_instruction) -=
                  idleBandwidthTime * kDefaultMemoryBandwidth;
              idleBandwidthTime = 0;
            } else {
              // Can transfer the whole copy.
              computation_elapsed += remaining_size / kDefaultMemoryBandwidth;
              idleBandwidthTime -= remaining_size / kDefaultMemoryBandwidth;
              remaining_size_of_buffers->erase(copy_start_instruction);
              issued_async_copy_instructions->pop();
            }
          }
        }
        if (inst_elapsed > 0) {
          float total_trip_count = cost_analysis_->CalculateNestTripCount(
              instruction, &cost_analysis_cache_);
          total_elapsed += inst_elapsed * total_trip_count;
        }
      }
    }
  }
  return total_elapsed;
}
}  // namespace memory_space_assignment
}  // namespace xla

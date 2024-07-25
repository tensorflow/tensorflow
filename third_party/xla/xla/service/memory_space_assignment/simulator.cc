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

#include <algorithm>
#include <cstdint>
#include <list>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/layout.h"
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

MemoryTransferDirection GetAsyncCopyDirection(
    const HloInstruction* async_copy_start, int64_t alternate_memory_space) {
  CHECK_EQ(async_copy_start->opcode(), HloOpcode::kCopyStart);

  int64_t operand_memory_space =
      async_copy_start->operand(0)->shape().layout().memory_space();

  // Get all users
  std::optional<int64_t> output_memory_space;
  for (const HloInstruction* user : async_copy_start->users()) {
    if (user->opcode() == HloOpcode::kCopyDone) {
      output_memory_space.emplace(user->shape().layout().memory_space());
      break;
    }
  }
  if (!output_memory_space.has_value()) {
    return MemoryTransferDirection::kUnsupported;
  }

  if (operand_memory_space == xla::Layout::kDefaultMemorySpace &&
      output_memory_space == alternate_memory_space) {
    return MemoryTransferDirection::kDefaultToAlternate;
  }
  if (operand_memory_space == alternate_memory_space &&
      output_memory_space == xla::Layout::kDefaultMemorySpace) {
    return MemoryTransferDirection::kAlternateToDefault;
  }
  return MemoryTransferDirection::kUnsupported;
}

const std::list<OutstandingAsyncCopy>&
RuntimeSimulator::GetOutstandingReadDefaultQueue() const {
  return outstanding_read_default_queue_;
}

const std::list<OutstandingAsyncCopy>&
RuntimeSimulator::GetOutstandingWriteDefaultQueue() const {
  return outstanding_write_default_queue_;
}

const HloInstruction* RuntimeSimulator::RemoveBytesFromQueueIfNotEmpty(
    std::list<OutstandingAsyncCopy>& async_copy_queue, float processed_bytes) {
  if (async_copy_queue.empty()) return nullptr;
  CHECK_GE(async_copy_queue.front().remaining_bytes_to_transfer,
           processed_bytes);
  async_copy_queue.front().remaining_bytes_to_transfer -= processed_bytes;
  if (async_copy_queue.front().remaining_bytes_to_transfer == 0.0) {
    const HloInstruction* retired_instruction =
        async_copy_queue.front().copy_start_inst;
    async_copy_queue.pop_front();
    return retired_instruction;
  }
  return nullptr;
}

float RuntimeSimulator::SimulateAsyncCopyDone(
    const HloInstruction* copy_done_instruction) {
  const HloInstruction* copy_start_instruction =
      copy_done_instruction->operand(0);
  MemoryTransferDirection direction =
      GetAsyncCopyDirection(copy_start_instruction, alternate_memory_space_);
  if (direction == MemoryTransferDirection::kUnsupported) {
    // The memory access is not a default <-> alternate memory copy.
    LOG(WARNING) << "Unsupported memory transfer direction for copy-done: "
                 << copy_done_instruction->ToString();
    return 0.0;
  }
  std::list<OutstandingAsyncCopy>& same_direction_queue =
      direction == MemoryTransferDirection::kDefaultToAlternate
          ? outstanding_read_default_queue_
          : outstanding_write_default_queue_;
  std::list<OutstandingAsyncCopy>& opposite_direction_queue =
      direction == MemoryTransferDirection::kDefaultToAlternate
          ? outstanding_write_default_queue_
          : outstanding_read_default_queue_;

  if (absl::c_find_if(
          same_direction_queue, [&](const OutstandingAsyncCopy& async_copy) {
            return async_copy.copy_start_inst == copy_start_instruction;
          }) == same_direction_queue.end()) {
    // The copy has already finished; thus, the copy-done takes no time.
    return 0.0;
  }

  // Each iteration of the while loop simulates transferring a number of
  // bytes from each queue that is equal to the smaller of the two elements
  // at the front of each queue. If that causes us to finish a copy in the
  // same_direction_queue, and that copy is the copy_done_instruction, we
  // break the loop.
  float elapsed_time = 0.0;
  const HloInstruction* retired_instruction_in_same_direction_queue = nullptr;
  // Loop until we process the copy start instruction that the copy-done
  // instruction is waiting for.
  do {
    float bytes_to_process =
        same_direction_queue.front().remaining_bytes_to_transfer;
    float available_bandwidth = cost_analysis_->base_costs().BytesPerSecond();

    if (!opposite_direction_queue.empty()) {
      // Need to share the bandwidth with the opposite direction queue.
      available_bandwidth *= 0.5;
      bytes_to_process = std::min(
          bytes_to_process,
          opposite_direction_queue.front().remaining_bytes_to_transfer);
    }

    elapsed_time += bytes_to_process / available_bandwidth;

    RemoveBytesFromQueueIfNotEmpty(opposite_direction_queue, bytes_to_process);
    retired_instruction_in_same_direction_queue =
        RemoveBytesFromQueueIfNotEmpty(same_direction_queue, bytes_to_process);
  } while (retired_instruction_in_same_direction_queue !=
           copy_start_instruction);
  return elapsed_time;
};

float RuntimeSimulator::SimulateComputeInstruction(
    const HloInstruction* instruction,
    absl::Span<const std::pair<int64_t, ShapeIndex>>
        operands_in_alternate_memory,
    absl::Span<const ShapeIndex> outputs_in_alternate_memory) {
  // Calculate the time in which the instruction does not access the default
  // memory.
  float default_memory_idle_time =
      cost_analysis_->GetDefaultMemoryBandwidthIdleTime(
          *instruction, operands_in_alternate_memory,
          outputs_in_alternate_memory);

  // Execute the outstanding async copy in the idle time.
  ProcessAsyncCopiesInIdleTime(default_memory_idle_time);

  float inst_elapsed = cost_analysis_->GetInstructionElapsedInAlternateMemory(
      *instruction, operands_in_alternate_memory, outputs_in_alternate_memory);
  return inst_elapsed;
}

void RuntimeSimulator::ProcessAsyncCopiesInIdleTime(float time) {
  if (time <= 0.0) {
    return;
  }
  float remaining_simulation_time = time;
  // This loop simulates the execution of the front memory requests in the
  // read and/or write queues. The loop terminates when the remaining time is
  // exhausted or there are no more outstanding async copies.
  while ((!outstanding_read_default_queue_.empty() ||
          !outstanding_write_default_queue_.empty()) &&
         remaining_simulation_time > 0.0) {
    float available_bandwidth = cost_analysis_->base_costs().BytesPerSecond();
    if (!outstanding_read_default_queue_.empty() &&
        !outstanding_write_default_queue_.empty()) {
      // Need to share the bandwidth
      available_bandwidth *= 0.5;
    }
    float bytes_to_process = available_bandwidth * remaining_simulation_time;
    if (!outstanding_read_default_queue_.empty()) {
      bytes_to_process = std::min(
          bytes_to_process,
          outstanding_read_default_queue_.front().remaining_bytes_to_transfer);
    }
    if (!outstanding_write_default_queue_.empty()) {
      bytes_to_process = std::min(
          bytes_to_process,
          outstanding_write_default_queue_.front().remaining_bytes_to_transfer);
    }

    float real_elapsed_time = bytes_to_process / available_bandwidth;
    remaining_simulation_time -= real_elapsed_time;
    RemoveBytesFromQueueIfNotEmpty(outstanding_read_default_queue_,
                                   bytes_to_process);
    RemoveBytesFromQueueIfNotEmpty(outstanding_write_default_queue_,
                                   bytes_to_process);
  }
}

}  // namespace memory_space_assignment
}  // namespace xla

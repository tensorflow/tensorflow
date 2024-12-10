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
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/layout.h"
#include "xla/service/memory_space_assignment/allocation.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace xla {
namespace memory_space_assignment {

void RuntimeSimulator::InitializeAlternateMemoryMap(
    const AllocationSequence& allocations) {
  outputs_in_alternate_memory_map_.clear();
  operands_in_alternate_memory_map_.clear();
  for (auto& allocation : allocations) {
    if (!allocation->is_copy_allocation()) {
      if (allocation->memory_space() == MemorySpace::kAlternate) {
        const HloInstruction* defining_instruction =
            allocation->defining_position().instruction;
        outputs_in_alternate_memory_map_[defining_instruction].push_back(
            allocation->defining_position().index);
      }
    }
    for (auto& hlo_use : allocation->uses()) {
      const HloInstruction* use_instruction = hlo_use.instruction;
      operands_in_alternate_memory_map_[use_instruction].push_back(
          std::make_pair(hlo_use.operand_number, hlo_use.operand_index));
    }
  }
}

float RuntimeSimulator::SimulateElapsedTimeWithoutAsyncCopyLikes(
    const HloLiveRange& hlo_live_range, const AllocationSequence& allocations) {
  InitializeAlternateMemoryMap(allocations);
  const auto& instruction_sequence =
      hlo_live_range.flattened_instruction_sequence().instructions();
  float total_elapsed = 0.0;
  for (const HloInstruction* instruction : instruction_sequence) {
    if (instruction->opcode() == HloOpcode::kWhile) {
      continue;
    }

    absl::Span<const ShapeIndex> outputs_in_alternate_memory;
    auto output_it = outputs_in_alternate_memory_map_.find(instruction);
    if (output_it != outputs_in_alternate_memory_map_.end()) {
      outputs_in_alternate_memory = absl::MakeSpan(output_it->second);
    }

    absl::Span<const std::pair<int64_t, ShapeIndex>>
        operands_in_alternate_memory;
    auto operand_it = operands_in_alternate_memory_map_.find(instruction);
    if (operand_it != operands_in_alternate_memory_map_.end()) {
      operands_in_alternate_memory = absl::MakeSpan(operand_it->second);
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

bool IsAsyncCopyLikeStart(const HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kCopyStart ||
         (instruction->opcode() == HloOpcode::kAsyncStart &&
          instruction->async_wrapped_instruction()->opcode() ==
              HloOpcode::kSlice);
}

bool IsAsyncCopyLikeDone(const HloInstruction* instruction) {
  return (instruction->opcode() == HloOpcode::kCopyDone ||
          (instruction->opcode() == HloOpcode::kAsyncDone &&
           instruction->async_wrapped_instruction()->opcode() ==
               HloOpcode::kSlice));
}

MemoryTransferDirection GetAsyncCopyLikeDirection(
    const HloInstruction* async_copy_like_start,
    int64_t alternate_memory_space) {
  CHECK(IsAsyncCopyLikeStart(async_copy_like_start));

  int64_t operand_memory_space =
      async_copy_like_start->operand(0)->shape().layout().memory_space();
  // Get all users
  std::optional<int64_t> output_memory_space;
  for (const HloInstruction* user : async_copy_like_start->users()) {
    if (user->opcode() == HloOpcode::kCopyDone ||
        user->opcode() == HloOpcode::kAsyncDone) {
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

const std::list<OutstandingAsyncCopyLike>&
RuntimeSimulator::GetOutstandingReadDefaultQueue() const {
  return outstanding_read_default_queue_;
}

const std::list<OutstandingAsyncCopyLike>&
RuntimeSimulator::GetOutstandingWriteDefaultQueue() const {
  return outstanding_write_default_queue_;
}

const HloInstruction* RuntimeSimulator::RemoveBytesFromQueueIfNotEmpty(
    std::list<OutstandingAsyncCopyLike>& async_copy_like_queue,
    float processed_bytes) {
  if (async_copy_like_queue.empty()) return nullptr;
  CHECK_GE(async_copy_like_queue.front().remaining_bytes_to_transfer,
           processed_bytes);
  async_copy_like_queue.front().remaining_bytes_to_transfer -= processed_bytes;
  if (async_copy_like_queue.front().remaining_bytes_to_transfer == 0.0) {
    const HloInstruction* retired_instruction =
        async_copy_like_queue.front().copy_like_start_inst;
    async_copy_like_queue.pop_front();
    return retired_instruction;
  }
  return nullptr;
}

float RuntimeSimulator::SimulateAsyncCopyLikeDone(
    const HloInstruction* copy_like_done_instruction) {
  const HloInstruction* copy_like_start_instruction =
      copy_like_done_instruction->operand(0);
  MemoryTransferDirection direction = GetAsyncCopyLikeDirection(
      copy_like_start_instruction, alternate_memory_space_);
  if (direction == MemoryTransferDirection::kUnsupported) {
    // The memory access is not a default <-> alternate memory copy.
    VLOG(1) << "Unsupported memory transfer direction for copy-done: "
            << copy_like_done_instruction->ToString();
    return 0.0;
  }
  std::list<OutstandingAsyncCopyLike>& same_direction_queue =
      direction == MemoryTransferDirection::kDefaultToAlternate
          ? outstanding_read_default_queue_
          : outstanding_write_default_queue_;
  std::list<OutstandingAsyncCopyLike>& opposite_direction_queue =
      direction == MemoryTransferDirection::kDefaultToAlternate
          ? outstanding_write_default_queue_
          : outstanding_read_default_queue_;

  if (absl::c_find_if(same_direction_queue,
                      [&](const OutstandingAsyncCopyLike& async_copy_like) {
                        return async_copy_like.copy_like_start_inst ==
                               copy_like_start_instruction;
                      }) == same_direction_queue.end()) {
    // The copy has already finished; thus, the copy-done takes no time.
    return 0.0;
  }

  // Each iteration of the while loop simulates transferring a number of
  // bytes from each queue that is equal to the smaller of the two elements
  // at the front of each queue. If that causes us to finish a copy in the
  // same_direction_queue, and that copy is the copy_like_done_instruction, we
  // break the loop.
  float elapsed_time = 0.0;
  const HloInstruction* retired_instruction_in_same_direction_queue = nullptr;
  // Loop until we process the copy start instruction that the copy-done
  // instruction is waiting for.
  do {
    float bytes_to_process =
        same_direction_queue.front().remaining_bytes_to_transfer;
    float available_bandwidth =
        cost_analysis_->DefaultMemBandwidthBytesPerSecond();

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
           copy_like_start_instruction);
  return elapsed_time;
};

RuntimeSimulator::ElapsedAndIdleTimes
RuntimeSimulator::SimulateComputeInstruction(
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

  // Execute the outstanding async copy likes in the idle time.
  default_memory_idle_time =
      ProcessAsyncCopyLikesInIdleTime(default_memory_idle_time);

  float inst_elapsed = cost_analysis_->GetInstructionElapsedInAlternateMemory(
      *instruction, operands_in_alternate_memory, outputs_in_alternate_memory);
  return {inst_elapsed, default_memory_idle_time};
}

float RuntimeSimulator::ProcessAsyncCopyLikesInIdleTime(float time) {
  if (time <= 0.0) {
    return 0.0;
  }

  double available_bandwidth =
      cost_analysis_->DefaultMemBandwidthBytesPerSecond();

  float remaining_simulation_time = time;
  // This loop simulates the execution of the front memory requests in the
  // read and/or write queues. The loop terminates when the remaining time is
  // exhausted or there are no more outstanding async copy likes.
  while ((!outstanding_read_default_queue_.empty() ||
          !outstanding_write_default_queue_.empty()) &&
         remaining_simulation_time > 0.0) {
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
    if (remaining_simulation_time <= 0.0) {
      // This can happen due to floating point errors.
      remaining_simulation_time = 0.0;
    }

    RemoveBytesFromQueueIfNotEmpty(outstanding_read_default_queue_,
                                   bytes_to_process);
    RemoveBytesFromQueueIfNotEmpty(outstanding_write_default_queue_,
                                   bytes_to_process);
  }

  return remaining_simulation_time;
}

namespace {

float GetUnusedDefaultMemBandwidthBytes(float bytes_per_second, float seconds) {
  CHECK_GE(bytes_per_second, 0.0);

  return bytes_per_second * seconds;
}

}  // namespace

float RuntimeSimulator::SimulateElapsedTime(
    const HloModule* hlo_module, const AllocationSequence& allocations,
    const std::vector<int64_t>* alt_mem_bytes_occupied) {
  InitializeAlternateMemoryMap(allocations);

  std::unique_ptr<xla::HloAliasAnalysis> alias_analysis =
      HloAliasAnalysis::Run(hlo_module).value();
  std::unique_ptr<HloLiveRange> hlo_live_range =
      HloLiveRange::Run(hlo_module->schedule(), *alias_analysis,
                        hlo_module->entry_computation())
          .value();

  // Cannot provide a valid result if the bandwidth is invalid.
  CHECK_GT(cost_analysis_->DefaultMemBandwidthBytesPerSecond(), 0.0);

  float total_elapsed = 0.0;
  // The number of additional bytes that could be transferred between default
  // and alternate memory.
  float cumulative_available_transfer_bytes = 0.0;

  if (alt_mem_bytes_occupied) {
    CHECK_EQ(
        alt_mem_bytes_occupied->size(),
        hlo_live_range->flattened_instruction_sequence().instructions().size());
  }
  const auto& instruction_sequence =
      hlo_live_range->flattened_instruction_sequence().instructions();
  for (int time = 0; time < instruction_sequence.size(); ++time) {
    const HloInstruction* instruction = instruction_sequence[time];
    float inst_elapsed = 0.0;
    float idle_default_memory_bandwidth_time = 0.0;
    if (instruction->opcode() == HloOpcode::kWhile) {
      // Since the instructions in the while body are calculated
      // separately, we can skip the while instruction.
      continue;
    }
    if (instruction->parent()->IsAsyncComputation()) {
      // We assume the overhead of async computations can be hidden perfectly.
      continue;
    }
    if (IsAsyncCopyLikeStart(instruction)) {
      // Try to categorize the async copy instruction into
      // read-from-default and write-to-default queues.
      MemoryTransferDirection direction =
          GetAsyncCopyLikeDirection(instruction, alternate_memory_space_);
      const Shape& transfer_shape =
          (instruction->opcode() == HloOpcode::kCopyStart)
              ? instruction->operand(0)->shape()
              : ShapeUtil::GetSubshape(instruction->shape(),
                                       /*index=*/{1});
      float transfer_bytes =
          static_cast<float>(cost_analysis_->GetShapeSizeBytes(transfer_shape));
      if (direction == MemoryTransferDirection::kDefaultToAlternate) {
        outstanding_read_default_queue_.push_back(
            OutstandingAsyncCopyLike{instruction, transfer_bytes});
      } else if (direction == MemoryTransferDirection::kAlternateToDefault) {
        outstanding_write_default_queue_.push_back(
            OutstandingAsyncCopyLike{instruction, transfer_bytes});
      } else {
        // The copy does not involve default memory.
      }
    } else if (IsAsyncCopyLikeDone(instruction)) {
      inst_elapsed = SimulateAsyncCopyLikeDone(instruction);
    } else {
      // This branch is for the compute instructions.
      absl::Span<const ShapeIndex> outputs_in_alternate_memory;
      auto output_it = outputs_in_alternate_memory_map_.find(instruction);
      if (output_it != outputs_in_alternate_memory_map_.end()) {
        outputs_in_alternate_memory = absl::MakeSpan(output_it->second);
      }

      absl::Span<const std::pair<int64_t, ShapeIndex>>
          operands_in_alternate_memory;
      auto operand_it = operands_in_alternate_memory_map_.find(instruction);
      if (operand_it != operands_in_alternate_memory_map_.end())
        operands_in_alternate_memory = absl::MakeSpan(operand_it->second);

      ElapsedAndIdleTimes elapsed_and_idle =
          SimulateComputeInstruction(instruction, operands_in_alternate_memory,
                                     outputs_in_alternate_memory);
      inst_elapsed = elapsed_and_idle.elapsed_time;
      idle_default_memory_bandwidth_time =
          elapsed_and_idle.idle_default_memory_bandwidth_time;
    }
    float total_trip_count = 0.0;
    if (inst_elapsed > 0.0) {
      // The calculation assumes all instructions are executed independently.
      // Thus, the execution time is the same for each invocation. This property
      // is not hold for all cases. For example, if an async copies are
      // outstanding before the loop, and there are other async copies inside
      // the loop body. In this case, the first async copy in the first
      // iteration will be slower than other iterations, since it needs to wait
      // for the async copies issued before the loop.
      total_trip_count = cost_analysis_->CalculateNestTripCount(
          instruction, &cost_analysis_cache_);
      total_elapsed += inst_elapsed * total_trip_count;
    }

    cumulative_available_transfer_bytes +=
        (GetUnusedDefaultMemBandwidthBytes(
             cost_analysis_->DefaultMemBandwidthBytesPerSecond(),
             idle_default_memory_bandwidth_time) *
         total_trip_count);
    VLOG(2) << [&]() {
      std::string instruction_name(instruction->name());
      if (instruction->opcode() == HloOpcode::kCopyStart &&
          instruction->cross_program_prefetch_index().has_value()) {
        absl::StrAppend(&instruction_name, " (xprogram prefetch)");
      }
      std::string alt_mem_bytes_occupied_str = "";
      if (alt_mem_bytes_occupied) {
        alt_mem_bytes_occupied_str =
            absl::StrCat("; alt mem usage: ", alt_mem_bytes_occupied->at(time));
      }

      return absl::StrCat(time, ": instruction: ", instruction_name,
                          "; elapsed: ", inst_elapsed,
                          "; cumulative available transfer bytes: ",
                          cumulative_available_transfer_bytes,
                          "; trip count: ", total_trip_count,
                          alt_mem_bytes_occupied_str);
    }();
  }

  return total_elapsed;
}

}  // namespace memory_space_assignment
}  // namespace xla

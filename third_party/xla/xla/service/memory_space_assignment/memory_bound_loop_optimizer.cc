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

#include "xla/service/memory_space_assignment/memory_bound_loop_optimizer.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/buffer_value.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_alias_analysis.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_value.h"
#include "xla/service/memory_space_assignment/allocation.h"
#include "xla/service/memory_space_assignment/cost_analysis.h"
#include "xla/service/memory_space_assignment/memory_space_assignment.pb.h"
#include "xla/service/memory_space_assignment/options.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace memory_space_assignment {
namespace {

std::optional<int64_t> GetInstructionIndex(
    const HloInstruction* instruction,
    const absl::flat_hash_map<const HloInstruction*, int64_t>&
        instructions_to_index) {
  auto it = instructions_to_index.find(instruction);
  return it == instructions_to_index.end() ? std::nullopt
                                           : std::optional<int64_t>(it->second);
}

}  // namespace

/*static*/ absl::StatusOr<std::unique_ptr<MemoryBoundLoopOptimizer>>
MemoryBoundLoopOptimizer::Create(
    int loop_start, int loop_end, uint64_t alternate_memory_size,
    const MemoryBoundLoopOptimizerOptions& options,
    const HloLiveRange& hlo_live_range, const HloAliasAnalysis& alias_analysis,
    const CostAnalysis& cost_analysis,
    const BufferValue::SizeFunction& size_function,
    const ReservedScopedMemoryFunction& reserved_scoped_memory_fn) {
  std::unique_ptr<MemoryBoundLoopOptimizer> optimizer =
      absl::WrapUnique(new MemoryBoundLoopOptimizer(
          loop_start, loop_end, alternate_memory_size, options, hlo_live_range,
          alias_analysis, cost_analysis, size_function,
          reserved_scoped_memory_fn));
  TF_RETURN_IF_ERROR(optimizer->Initialize());
  return std::move(optimizer);
}

MemoryBoundLoopOptimizer::MemoryBoundLoopOptimizer(
    int loop_start, int loop_end, uint64_t alternate_memory_size,
    const MemoryBoundLoopOptimizerOptions& options,
    const HloLiveRange& hlo_live_range, const HloAliasAnalysis& alias_analysis,
    const CostAnalysis& cost_analysis,
    const BufferValue::SizeFunction& size_function,
    const ReservedScopedMemoryFunction& reserved_scoped_memory_fn)
    : loop_start_(loop_start),
      loop_end_(loop_end),
      loop_size_(loop_end - loop_start),
      alternate_memory_size_(alternate_memory_size),
      options_(options),
      hlo_live_range_(hlo_live_range),
      alias_analysis_(alias_analysis),
      cost_analysis_(cost_analysis),
      size_function_(size_function),
      reserved_scoped_memory_fn_(reserved_scoped_memory_fn) {}

absl::Status MemoryBoundLoopOptimizer::Initialize() {
  const auto& instruction_sequence =
      hlo_live_range_.flattened_instruction_sequence().instructions();
  VLOG(3) << "MemoryBoundLoopOptimizer::Initialize, loop start: " << loop_start_
          << ", loop end: " << loop_end_ << ", loop size: " << loop_size_;
  const HloComputation* loop_computation = nullptr;
  // Initialize the remaining memory array with the size of the alternate
  // memory. Also populate instructions_in_loop_ and
  // instructions_in_{prev,next}_iterations_ data structures to help find the
  // loop values.
  int prev_iteration_start = loop_start_ - loop_size_;
  int next_iteration_start = loop_start_ + loop_size_;
  for (int i = 0; i < loop_size_; ++i) {
    const HloInstruction* loop_inst = instruction_sequence[loop_start_ + i];
    instructions_in_loop_[loop_inst] = i;
    const HloInstruction* prev_iteration_inst =
        instruction_sequence[prev_iteration_start + i];
    instructions_in_prev_iteration_[prev_iteration_inst] = i;
    const HloInstruction* next_iteration_inst =
        instruction_sequence[next_iteration_start + i];
    instructions_in_next_iteration_[next_iteration_inst] = i;

    VLOG(3) << "  inst in loop [" << (i) << "]: " << loop_inst->name();
    if (!loop_computation) {
      loop_computation = loop_inst->parent();
    } else {
      TF_RET_CHECK(loop_computation == loop_inst->parent());
    }
    remaining_memory_.push_back(
        alternate_memory_size_ -
        reserved_scoped_memory_fn_(loop_inst,
                                   /*operands_in_alternate_memory=*/{},
                                   /*outputs_in_alternate_memory=*/{}));
  }

  // Create a tree set to keep track of all the values that the loop
  // instructions produce and consume. We use a tree set instead of a hash set
  // to ensure the iteration order is the same as insertion order. Since we
  // traverse the program in instruction order, the buffers would be inserted in
  // a deterministic order, so we'll be able to iterate over these buffers in a
  // deterministic order.
  std::set<const HloBuffer*> buffers_to_process;
  for (const auto& [instruction, idx] : instructions_in_loop_) {
    auto maybe_add_buffer = [&](const HloInstruction* instruction) {
      return [this, &buffers_to_process, instruction](const Shape& subshape,
                                                      const ShapeIndex& index) {
        if (!subshape.IsArray()) {
          return;
        }
        const HloBuffer& buffer =
            alias_analysis_.GetUniqueBufferAt(instruction, index);
        if (buffers_to_process.find(&buffer) == buffers_to_process.end()) {
          buffers_to_process.insert(&buffer);
        }
      };
    };
    ShapeUtil::ForEachSubshape(instruction->shape(),
                               maybe_add_buffer(instruction));
    for (const HloInstruction* operand : instruction->operands()) {
      ShapeUtil::ForEachSubshape(operand->shape(), maybe_add_buffer(operand));
    }
  }

  // Process the buffers and decide if they should be added as LoopValues.
  for (const HloBuffer* buffer : buffers_to_process) {
    MaybeCreateLoopValue(*buffer, loop_computation);
  }
  return absl::OkStatus();
}

void MemoryBoundLoopOptimizer::MaybeCreateLoopValue(
    const HloBuffer& buffer, const HloComputation* loop_computation) {
  loop_values_.push_back({});
  LoopValue& loop_value = loop_values_.back();
  float pos_bytes = 0;
  float use_bytes = 0;
  bool has_footer_consumer = false;
  for (const HloValue* value : buffer.values()) {
    // For each position and use of the value, populate the respective position
    // and use fields for the current, previous, and next iterations along with
    // the loop indices.
    for (const HloPosition& position : value->positions()) {
      if (position.instruction->opcode() == HloOpcode::kGetTupleElement) {
        continue;
      }
      std::optional<int64_t> loop_index =
          GetInstructionIndex(position.instruction, instructions_in_loop_);
      std::optional<int64_t> prev_iteration_index;
      if (loop_index) {
        loop_value.loop_positions.push_back({*loop_index, position});
        VLOG(3) << "Pos match: " << position.instruction->name() << " at "
                << *loop_index;
      } else if ((prev_iteration_index = GetInstructionIndex(
                      position.instruction, instructions_in_prev_iteration_))) {
        loop_value.prev_iteration_positions.push_back(
            {*prev_iteration_index, position});
        VLOG(3) << "Pos match (prev iteration): "
                << position.instruction->name() << " at "
                << *prev_iteration_index;
      } else if (loop_value.prev_iteration_positions.empty() &&
                 loop_value.loop_positions.empty() &&
                 position.instruction->parent() == loop_computation &&
                 !loop_value.header_position) {
        loop_value.header_position = position;
      }

      // Keep track of bytes accessed by this value.
      if (loop_index || prev_iteration_index) {
        float bytes_accessed = cost_analysis_.base_costs().OutputBytesAccessed(
            *position.instruction, position.index);
        pos_bytes += bytes_accessed;
        VLOG(3) << " accessed: " << bytes_accessed;
      }
    }

    for (const HloUse& use : value->GetUses()) {
      if (use.instruction->opcode() == HloOpcode::kGetTupleElement) {
        continue;
      }
      std::optional<int64_t> loop_index =
          GetInstructionIndex(use.instruction, instructions_in_loop_);
      std::optional<int64_t> next_iteration_index;
      if (loop_index) {
        loop_value.loop_uses.push_back({*loop_index, use});
        VLOG(3) << "Use match: " << use.instruction->name() << " at "
                << *loop_index;
      } else if ((next_iteration_index = GetInstructionIndex(
                      use.instruction, instructions_in_next_iteration_))) {
        loop_value.next_iteration_uses.push_back({*next_iteration_index, use});
        VLOG(3) << "Use match (next iteration): " << use.instruction->name()
                << " at " << *next_iteration_index;
      } else if (!loop_value.loop_positions.empty() ||
                 !loop_value.loop_uses.empty()) {
        has_footer_consumer = true;
      }

      // Keep track of bytes accessed by this value.
      if (loop_index || next_iteration_index) {
        float bytes_accessed = cost_analysis_.base_costs().OperandBytesAccessed(
            *use.instruction, use.operand_number, use.operand_index);
        use_bytes += bytes_accessed;
        VLOG(3) << " accessed: " << bytes_accessed;
      }
    }
  }

  // We only add the loop position if it has a position or use in the current
  // iteration and its previous iteration positions are empty. The reason why we
  // disallow values with previous iteration positions is because there will be
  // a different value that corresponds to the same value but one iteration
  // later, so we will add that one instead.
  if ((!loop_value.loop_positions.empty() || !loop_value.loop_uses.empty()) &&
      loop_value.prev_iteration_positions.empty()) {
    loop_value.size = size_function_(**buffer.values().begin());
    VLOG(3) << "Size: " << loop_value.size;
    // Classify the type of allocation. See the comment in LoopValue definition.
    loop_value.allocation_type = LoopValue::AllocationType::kUnsupported;
    auto position_compare = [](const std::pair<int64_t, HloPosition>& a,
                               const std::pair<int64_t, HloPosition>& b) {
      return a.first < b.first;
    };
    auto use_compare = [](const std::pair<int64_t, HloUse>& a,
                          const std::pair<int64_t, HloUse>& b) {
      return a.first < b.first;
    };
    absl::c_sort(loop_value.loop_positions, position_compare);
    absl::c_sort(loop_value.prev_iteration_positions, position_compare);
    absl::c_sort(loop_value.loop_uses, use_compare);
    absl::c_sort(loop_value.next_iteration_uses, use_compare);
    if (!loop_value.loop_positions.empty()) {
      if (loop_value.next_iteration_uses.empty() &&
          !loop_value.loop_uses.empty()) {
        loop_value.allocation_type = LoopValue::AllocationType::kTemporary;
      } else if (!loop_value.next_iteration_uses.empty()) {
        if (loop_value.next_iteration_uses.back().first >=
            loop_value.loop_positions.front().first) {
          loop_value.allocation_type =
              LoopValue::AllocationType::kLoopCarriedDependence;
        } else {
          loop_value.allocation_type = LoopValue::AllocationType::kTemporary;
        }
      }
    } else if (loop_value.header_position && !loop_value.loop_uses.empty()) {
      if (loop_value.loop_uses.size() ==
              loop_value.next_iteration_uses.size() &&
          loop_value.loop_uses.front().first ==
              loop_value.next_iteration_uses.front().first) {
        loop_value.allocation_type = LoopValue::AllocationType::kPinned;
      } else if (loop_value.next_iteration_uses.empty() ||
                 loop_value.next_iteration_uses.back().first <
                     loop_value.loop_uses.front().first) {
        loop_value.allocation_type = LoopValue::AllocationType::kPrefetch;
      }
    }

    VLOG(3) << "Allocation type "
            << LoopValue::AllocationTypeToString(loop_value.allocation_type);
    VLOG(3) << "Pos bytes: " << pos_bytes << " use bytes: " << use_bytes;

    // We calculate the savings of allocating this buffer in the alternate
    // memory.
    float savings = pos_bytes + use_bytes;
    if (loop_value.header_position) {
      savings -= loop_value.size;
    }
    if (!loop_value.loop_positions.empty() && has_footer_consumer) {
      savings -= loop_value.size;
    }
    loop_value.savings = savings;
    loop_value.savings_per_byte = savings / loop_value.size;
    VLOG(3) << "Savings: " << loop_value.savings;
    VLOG(3) << "Savings per byte: " << loop_value.savings_per_byte;
    for (const HloValue* value : buffer.values()) {
      VLOG(3) << value->ToString();
    }
    loop_value.hlo_values = buffer.values();
  } else {
    loop_values_.pop_back();
  }
}

void MemoryBoundLoopOptimizer::Optimize() {
  SortLoopValues();
  AllocateLoopValues();
  PostProcess();
}

float MemoryBoundLoopOptimizer::CalculateExecutionTime() const {
  // First populate the list of prefetches.
  std::vector<std::pair<const CopyAllocation*, float>> prefetches;
  for (const LoopValue& value : loop_values_) {
    if (!value.allocations.empty() &&
        value.allocations.back()->is_copy_allocation()) {
      prefetches.push_back(
          {static_cast<const CopyAllocation*>(value.allocations.back().get()),
           cost_analysis_.GetAsyncCopyElapsed(
               value.hlo_values.front()->shape())});
    }
  }

  // Returns the effective prefetch completion time. The effective time is a
  // value that will be larger than loop size for prefetches that start in this
  // iteration but complete in the next iteration.
  auto get_effective_done_time =
      [&](int64_t copy_start_schedule_after,
          int64_t copy_done_schedule_before) -> int64_t {
    if (copy_start_schedule_after == loop_size_ - 1 &&
        copy_done_schedule_before == 0) {
      return 2 * loop_size_;
    }
    if (copy_start_schedule_after + 1 >= copy_done_schedule_before) {
      return copy_done_schedule_before + loop_size_;
    }
    return copy_done_schedule_before;
  };

  // Sort the prefetches by first the start time, then the effective done time.
  absl::c_sort(
      prefetches, [&](const std::pair<const CopyAllocation*, float>& a,
                      const std::pair<const CopyAllocation*, float>& b) {
        return std::forward_as_tuple(
                   a.first->copy_start_schedule_after(),
                   get_effective_done_time(
                       a.first->copy_start_schedule_after(),
                       a.first->copy_done_schedule_before())) <
               std::forward_as_tuple(b.first->copy_start_schedule_after(),
                                     get_effective_done_time(
                                         b.first->copy_start_schedule_after(),
                                         b.first->copy_done_schedule_before()));
      });
  // Populate the required prefetch completions array. For each instruction in
  // the loop, this vector holds the index of the latest-issued prefetch that
  // needs to be completed before the instruction executes, or nullopt if there
  // is no prefetch that needs to finish by this instruction. To represent
  // prefetches that started in the previous iteration, we use negative numbers.
  std::vector<std::optional<int>> required_prefetch_completions(loop_size_);
  for (int i = 0; i < prefetches.size(); ++i) {
    const auto& [prefetch, elapsed] = prefetches[i];
    int required_prefetch_completion = i;
    if (prefetch->copy_start_schedule_after() == loop_size_ - 1 &&
        prefetch->copy_done_schedule_before() == 0) {
      required_prefetch_completion -= 2 * prefetches.size();
    } else if (prefetch->copy_start_schedule_after() + 1 >=
               prefetch->copy_done_schedule_before()) {
      required_prefetch_completion -= prefetches.size();
    }
    VLOG(3) << "Prefetch #" << i << " (elapsed " << elapsed
            << "): " << prefetch->ToString();
    if (required_prefetch_completions[prefetch->copy_done_schedule_before()]) {
      required_prefetch_completions[prefetch->copy_done_schedule_before()] =
          std::max(
              *required_prefetch_completions[prefetch
                                                 ->copy_done_schedule_before()],
              required_prefetch_completion);
    } else {
      required_prefetch_completions[prefetch->copy_done_schedule_before()] =
          required_prefetch_completion;
    }
    VLOG(4)
        << "Required completion at " << prefetch->copy_done_schedule_before()
        << " = "
        << *required_prefetch_completions[prefetch
                                              ->copy_done_schedule_before()];
  }

  // Populate the elapsed times of instructions and bandwidth idle times at each
  // point.
  float result;
  std::vector<float> bandwidth_idle_times;
  std::vector<float> instructions_elapsed;
  bandwidth_idle_times.reserve(loop_size_);
  instructions_elapsed.reserve(loop_size_);
  for (int i = 0; i < loop_size_; ++i) {
    bandwidth_idle_times.push_back(GetBandwidthIdleTime(i));
    instructions_elapsed.push_back(GetInstructionElapsed(i));
  }
  // We simulate the loop for three iterations to measure the steady state.
  const int kNumIterations = 3;
  // This data structure keeps track of the elapsed time remaining of each
  // prefetch. Note that there is a separate entry for each prefetch in each
  // iteration simulated.
  std::vector<float> prefetch_remaining_elapsed_times(prefetches.size() *
                                                      kNumIterations);
  int prefetch_start_index = 0;
  int prefetch_done_index = 0;
  int prefetch_completed_index = 0;

  for (int iteration = 0; iteration < kNumIterations; ++iteration) {
    float total_elapsed = 0;
    float total_bandwidth_idle_time = 0;
    float total_critical_prefetch = 0;
    for (int i = 0; i < loop_size_; ++i) {
      // If any prefetches are expected to be completed, check if they have any
      // remaining elapsed time associated with them, and if so add this to
      // critical prefetch time.
      std::optional<int> required_prefetch_completion =
          required_prefetch_completions[i];
      if (required_prefetch_completion) {
        int required_prefetch_done_index =
            iteration * static_cast<int>(prefetches.size()) +
            *required_prefetch_completion;
        VLOG(4) << "Prefetch #"
                << ((*required_prefetch_completion + prefetches.size()) %
                    prefetches.size())
                << " (" << required_prefetch_done_index
                << ") is required to be completed at " << i;
        for (; prefetch_done_index <= required_prefetch_done_index;
             ++prefetch_done_index) {
          CHECK_LE(prefetch_done_index, prefetch_start_index);
          if (prefetch_done_index == prefetch_completed_index) {
            float& prefetch_remaining =
                prefetch_remaining_elapsed_times[prefetch_done_index];
            VLOG(4) << "Prefetch #" << (prefetch_done_index % prefetches.size())
                    << " (" << prefetch_done_index
                    << ") did not complete, remaining elapsed = "
                    << prefetch_remaining;
            total_critical_prefetch += prefetch_remaining;
            prefetch_remaining = 0;
            ++prefetch_completed_index;
          }
        }
      }

      float elapsed = instructions_elapsed[i];
      total_elapsed += elapsed;
      float bandwidth_idle_time = bandwidth_idle_times[i];
      // Find the outstanding prefetches during this instruction, and if any of
      // them have remaining time, spend some or all of the bandwidth idle time
      // to satisfy them.
      for (; prefetch_completed_index < prefetch_start_index;
           ++prefetch_completed_index) {
        float& prefetch_remaining =
            prefetch_remaining_elapsed_times[prefetch_completed_index];
        if (bandwidth_idle_time < prefetch_remaining) {
          prefetch_remaining -= bandwidth_idle_time;
          bandwidth_idle_time = 0;
          VLOG(4) << "Prefetch #"
                  << (prefetch_completed_index % prefetches.size()) << " ("
                  << prefetch_completed_index << ") still ongoing at " << i
                  << ", remaining elapsed = " << prefetch_remaining;
          break;
        }
        bandwidth_idle_time -= prefetch_remaining;
        prefetch_remaining = 0;
        VLOG(4) << "Prefetch #"
                << (prefetch_completed_index % prefetches.size()) << " ("
                << prefetch_completed_index << ") completed at " << i
                << ", bandwidth idle time = " << bandwidth_idle_time;
      }
      if (bandwidth_idle_time > 0) {
        VLOG(4) << "Bandwidth idle time at " << i << " = "
                << bandwidth_idle_time;
        total_bandwidth_idle_time += bandwidth_idle_time;
      }

      // Start new prefetches that are scheduled to start after this
      // instruction.
      for (; prefetch_start_index < (iteration + 1) * prefetches.size() &&
             prefetches[prefetch_start_index % prefetches.size()]
                     .first->copy_start_schedule_after() == i;
           ++prefetch_start_index) {
        float& prefetch_remaining =
            prefetch_remaining_elapsed_times[prefetch_start_index];
        prefetch_remaining =
            prefetches[prefetch_start_index % prefetches.size()].second;
        VLOG(4) << "Prefetch #" << (prefetch_start_index % prefetches.size())
                << " (" << prefetch_start_index << ") started at " << i
                << ", remaining elapsed = " << prefetch_remaining;
      }
    }
    VLOG(3) << "Iteration " << iteration;
    VLOG(3) << "Total elapsed: " << total_elapsed
            << ", total critical prefetch: " << total_critical_prefetch
            << ", total bandwidth idle time: " << total_bandwidth_idle_time;
    result = total_elapsed + total_critical_prefetch;
  }
  return result;
}

/*static*/ std::string
MemoryBoundLoopOptimizer::LoopValue::AllocationTypeToString(
    LoopValue::AllocationType allocation_type) {
  switch (allocation_type) {
    case AllocationType::kTemporary:
      return "temporary";
    case AllocationType::kLoopCarriedDependence:
      return "loop-carried dependence";
    case AllocationType::kPinned:
      return "pinned";
    case AllocationType::kPrefetch:
      return "prefetch";
    default:
      CHECK(allocation_type == AllocationType::kUnsupported);
      return "unsupported";
  }
}

std::string MemoryBoundLoopOptimizer::LoopValue::ToString() const {
  std::string values_str;
  absl::StrAppend(&values_str, "Values:");
  for (const HloValue* hlo_value : hlo_values) {
    absl::StrAppend(&values_str, "\n  - ", hlo_value->ToShortString());
  }
  std::string allocations_str;
  if (!allocations.empty()) {
    absl::StrAppend(&allocations_str, "Allocations:");
  }
  for (const auto& allocation : allocations) {
    absl::StrAppend(&allocations_str, "\n  - ", allocation->ToString());
  }
  return absl::StrCat(
      "Size: ", size, " savings: ", savings,
      " savings per byte: ", savings_per_byte,
      " allocation type: ", AllocationTypeToString(allocation_type), "\n",
      values_str, "\n", allocations_str);
}

bool MemoryBoundLoopOptimizer::LoopValue::IsAllocationTypeSupported() const {
  return allocation_type == AllocationType::kTemporary ||
         allocation_type == AllocationType::kPinned ||
         allocation_type == AllocationType::kPrefetch;
}

void MemoryBoundLoopOptimizer::SortLoopValues() {
  absl::c_stable_sort(loop_values_, [](const LoopValue& a, const LoopValue& b) {
    return a.savings_per_byte > b.savings_per_byte;
  });
}

void MemoryBoundLoopOptimizer::AllocateLoopValues() {
  // This function allocates loop values.
  std::vector<LoopValue*> prefetch_values;
  VLOG(3) << "Pre optimization execution time: " << CalculateExecutionTime();
  for (LoopValue& value : loop_values_) {
    switch (value.allocation_type) {
      case LoopValue::AllocationType::kTemporary:
        AllocateTemporary(value);
        break;
      case LoopValue::AllocationType::kPinned:
        if (value.savings > 0) {
          AllocatePinned(value);
        }
        break;
      case LoopValue::AllocationType::kPrefetch:
        prefetch_values.push_back(&value);
        break;
      case LoopValue::AllocationType::kLoopCarriedDependence:
      case LoopValue::AllocationType::kUnsupported:
        VLOG(1) << "Unsupported allocation: " << value.ToString();
    }
  }
  VLOG(3) << "Execution time after allocating temporaries: "
          << CalculateExecutionTime();
  AllocatePrefetches(absl::MakeSpan(prefetch_values));
  VLOG(3) << "Execution time after allocating prefetches:  "
          << CalculateExecutionTime();
}

void MemoryBoundLoopOptimizer::PostProcess() {
  // At the end, ensure that all loop uses have a corresponding Allocation and
  // create one in the default memory space if they don't.
  for (LoopValue& value : loop_values_) {
    absl::flat_hash_set<HloUse> allocated_uses;
    for (const auto& allocation : value.allocations) {
      for (const HloUse& use : allocation->uses()) {
        allocated_uses.insert(use);
      }
    }
    std::vector<HloUse> unallocated_uses;
    absl::flat_hash_set<int> use_indices;
    for (const auto& [idx, use] : value.loop_uses) {
      use_indices.insert(idx);
      if (!allocated_uses.contains(use)) {
        unallocated_uses.push_back(use);
      }
    }
    for (const auto& [next_iteration_idx, use] : value.next_iteration_uses) {
      if (use_indices.contains(next_iteration_idx)) {
        continue;
      }
      HloInstruction* loop_instruction =
          hlo_live_range_.flattened_instruction_sequence().instructions().at(
              loop_start_ + next_iteration_idx);
      HloUse loop_use{loop_instruction, use.operand_number, use.operand_index};
      if (!allocated_uses.contains(loop_use)) {
        unallocated_uses.push_back(loop_use);
      }
    }
    if (!unallocated_uses.empty()) {
      // TODO(b/281582241): We should find the correct position. For now, we're
      // using the defining position on the first HLO value.
      value.allocations.push_back(std::make_unique<PinnedAllocation>(
          value.hlo_values.front()->defining_position(), MemorySpace::kDefault,
          std::nullopt, 0, loop_size_, /*is_scoped_allocation=*/false));
      for (const HloUse& use : unallocated_uses) {
        value.allocations.back()->AddUse(use);
      }
    }
  }
}

bool MemoryBoundLoopOptimizer::AllocateBetween(int64_t begin_idx,
                                               int64_t end_idx, int64_t size) {
  int64_t end_idx_sentinel = end_idx;
  if (end_idx < begin_idx) {
    end_idx_sentinel += loop_size_;
  }
  for (int64_t i = begin_idx; i <= end_idx_sentinel; ++i) {
    if (remaining_memory_[i % loop_size_] < size) {
      return false;
    }
  }
  for (int64_t i = begin_idx; i <= end_idx_sentinel; ++i) {
    remaining_memory_[i % loop_size_] -= size;
  }
  return true;
}

bool MemoryBoundLoopOptimizer::AllocateTemporary(LoopValue& value) {
  VLOG(3) << "AllocateTemporary: " << value.ToString();
  if (value.hlo_values.size() > 1) {
    VLOG(3) << "LoopValue has more than one hlo value associated.";
    return false;
  }
  int64_t definition_idx = value.loop_positions.front().first;
  int64_t max_use_idx;
  if (!value.next_iteration_uses.empty()) {
    max_use_idx = value.next_iteration_uses.back().first;
    // If max_use_idx >= definition_idx, then this is a loop carried dependence
    // and we should not have called this function.
    CHECK_LT(max_use_idx, definition_idx);
  } else {
    max_use_idx = value.loop_uses.back().first;
  }
  bool success = AllocateBetween(definition_idx, max_use_idx, value.size);
  if (success) {
    VLOG(3) << "Pos: " << value.loop_positions[0].second;
    value.allocations.push_back(std::make_unique<PinnedAllocation>(
        value.loop_positions[0].second, MemorySpace::kAlternate, std::nullopt,
        definition_idx, max_use_idx,
        /*is_scoped_allocation=*/false));
    AddAllLoopPositionsAndUses(value, /*allocate_next_iteration_uses=*/true);
  }
  return success;
}

bool MemoryBoundLoopOptimizer::AllocatePinned(LoopValue& value) {
  bool success = AllocateBetween(0, loop_size_ - 1, value.size);
  if (success) {
    CHECK(value.header_position);
    value.allocations.push_back(std::make_unique<PinnedAllocation>(
        *value.header_position, MemorySpace::kAlternate, std::nullopt, 0,
        loop_size_,
        /*is_scoped_allocation=*/false));
    AddAllLoopPositionsAndUses(value, /*allocate_next_iteration_uses=*/false);
  }
  return success;
}

bool MemoryBoundLoopOptimizer::AllocatePrefetches(
    absl::Span<LoopValue*> values) {
  VLOG(3) << "Allocating prefetches num values: " << values.size();
  AllocatePrefetchesContext context;
  context.values = values;
  // Populate value_indices, which is a list of indices into values array sorted
  // by the start time of the first use.
  context.value_indices.resize(values.size());
  absl::c_iota(context.value_indices, 0);
  absl::c_stable_sort(context.value_indices, [&](int a, int b) {
    return std::forward_as_tuple(
               values[a]->loop_uses.begin()->first,
               values[a]->loop_uses.begin()->second.operand_number) >
           std::forward_as_tuple(
               values[b]->loop_uses.begin()->first,
               values[b]->loop_uses.begin()->second.operand_number);
  });

  // Populate the data structures that contain additional positions and uses
  // that would get alternate memory allocations if all of the prefetches were
  // successful.
  absl::flat_hash_map<const HloInstruction*,
                      std::vector<std::pair<int64_t, ShapeIndex>>>
      additional_uses_in_alternate_mem;
  absl::flat_hash_map<const HloInstruction*, std::vector<ShapeIndex>>
      additional_positions_in_alternate_mem;
  for (const LoopValue* value : values) {
    VLOG(3) << "  prefetch value: " << value->ToString();
    for (const auto& [idx, use] : value->loop_uses) {
      additional_uses_in_alternate_mem[use.instruction].push_back(
          {use.operand_number, use.operand_index});
    }
    for (const auto& [idx, position] : value->loop_positions) {
      additional_positions_in_alternate_mem[position.instruction].push_back(
          position.index);
    }
  }
  // Calculate the default-memory remaining bandwidths assuming all prefetches
  // succeed.
  for (int i = 0; i < loop_size_; ++i) {
    context.bandwidth_idle_times.push_back(
        GetBandwidthIdleTime(i, additional_uses_in_alternate_mem,
                             additional_positions_in_alternate_mem));
    VLOG(3) << "Remaining bandwidth at " << i << " = "
            << *context.bandwidth_idle_times.rbegin();
  }

  context.additional_memory_used.resize(loop_size_, 0);

  // Allocate prefetches by traversing the loop values in reverse order of
  // the first uses.
  for (int value_index : context.value_indices) {
    AllocatePrefetch(value_index, context);
  }

  for (int i = 0; i < loop_size_; ++i) {
    remaining_memory_[i] -= context.additional_memory_used[i];
    VLOG(3) << "Additional memory [" << i
            << "]: " << context.additional_memory_used[i];
    VLOG(3) << "Remaining memory [" << i << "]: " << remaining_memory_[i];
    VLOG(3) << "Remaining bandwidth [" << i
            << "] : " << context.bandwidth_idle_times[i];
  }
  return true;
}

bool MemoryBoundLoopOptimizer::AllocatePrefetch(
    int value_index, AllocatePrefetchesContext& context) {
  LoopValue* value = context.values.at(value_index);
  VLOG(3) << "Allocating value: " << value->ToString();
  int first_use_idx = value->loop_uses.front().first;
  int last_use_idx = value->loop_uses.back().first;
  int last_use_idx_sentinel = last_use_idx;
  if (!value->next_iteration_uses.empty()) {
    last_use_idx = value->next_iteration_uses.back().first;
    last_use_idx_sentinel = last_use_idx + loop_size_;
    CHECK_LT(last_use_idx, first_use_idx);
  }
  bool out_of_memory = false;
  for (int i = first_use_idx; i <= last_use_idx_sentinel; ++i) {
    int loop_idx = i % loop_size_;
    if (context.additional_memory_used[loop_idx] + value->size >
        remaining_memory_[loop_idx]) {
      VLOG(3) << "Ran out of memory allocating for uses.";
      out_of_memory = true;
    }
  }
  if (out_of_memory) {
    return false;
  }
  float copy_resource =
      cost_analysis_.GetAsyncCopyElapsed(value->hlo_values.front()->shape());
  VLOG(3) << "First use: " << value->loop_uses.begin()->second
          << " use idx: " << first_use_idx
          << " copy resource: " << copy_resource;
  std::optional<int> copy_start_time;
  // The general allocation algorithm for prefetches is to first calculate the
  // default-memory bandwidth idle times at each point (assuming all prefetches
  // succeeded).  We show this pictorially below. We also show the previous
  // iteration for clarity. The algorithm solves allocation for one iteration
  // and this will be used for all iterations.
  //
  //               idx:  0  1  2  3  4  5| 0  1  2  3  4  5|
  //      bw idle time:  2  2  1  2  3  1| 2  2  1  2  3  1|
  // additional memory:  0  0  0  0  0  0| 0  0  0  0  0  0|
  //         iteration:       prev       |      current    |
  //
  // Now, let's assume there are two prefetches that need to be scheduled. For
  // the sake of the example, assume 1 MiB of prefetch uses 1 memory bandwidth
  // resource:
  //   - Prefetch 1 is 4 MiB and is first used at index 5.
  //   - Prefetch 2 is 5 MiB and is first used at index 1.
  //
  // We first order these prefetches by their first use from latest to earliest.
  // Then starting from the prefetch completion time (i.e. the first use time),
  // move the prefetch start time earlier until the copy resource is satisfied
  // (or reaching another resource satisfaction criteria explained below) by
  // consuming the bandwidth idle time of the overlapped instructions. We also
  // keep track of the additional memory required. Note that index 5 also
  // accounts for the additional 4 MiB consumed since the data needs to reside
  // during the execution of the instruction at index 5.  Below is the updated
  // state after scheduling prefetch 1:
  //
  //        prefetch 1:          +====+            +====+
  //               idx:  0  1  2  3  4  5| 0  1  2  3  4  5|
  //      bw idle time:  2  2  1  1  0  1| 2  2  1  1  0  1|
  // additional memory:  0  0  0  4  4  4| 0  0  0  4  4  4|
  //         iteration:       prev       |      current    |
  //
  // To schedule prefetch 2, we similarly start the same way, from its first use
  // and bring the prefetch start earlier. We first reach index 0 with still an
  // unsatisfied copy resource of 3:
  //
  //        prefetch 2: +=+               +=+                unsat res: 3
  //        prefetch 1:          +====+            +====+
  //               idx:  0  1  2  3  4  5| 0  1  2  3  4  5|
  //      bw idle time:  0  2  1  1  0  1| 0  2  1  1  0  1|
  // additional memory:  5  5  0  4  4  4| 5  5  0  4  4  4|
  //         iteration:       prev       |      current    |
  //
  // We continue onto the previous iteration:
  //
  //        prefetch 2:===+            +====+            +== unsat res: 2
  //        prefetch 1:          +====+            +====+
  //               idx:  0  1  2  3  4  5| 0  1  2  3  4  5|
  //      bw idle time:  0  2  1  1  0  0| 0  2  1  1  0  0|
  // additional memory:  5  5  0  4  4  9| 5  5  0  4  4  9|
  //         iteration:       prev       |      current    |
  //
  // As we bring the start time of prefetch 2 earlier, it starts overlapping
  // with prefetch 1:
  //
  //        prefetch 2:===+      +==========+      +======== unsat res: 1
  //        prefetch 1:          +====+            +====+
  //               idx:  0  1  2  3  4  5| 0  1  2  3  4  5|
  //      bw idle time:  0  2  1  0  0  0| 0  2  1  0  0  0|
  // additional memory:  5  5  0  9  9  9| 5  5  0  9  9  9|
  //         iteration:       prev       |      current    |
  //
  // The prefetch resource is still unsatisfied at this point. We can bring the
  // prefetch earlier. However, the first prefetch's end time is earlier than
  // the second and we need to maintain FIFO order with regard to prefetches. In
  // order to maintain this FIFO order, we "early force" prefetches that are
  // already scheduled by moving the start time earlier along with prefetch 2:
  //
  //        prefetch 2:===+   +=============+   +===========
  //        prefetch 1:       +=======+         +=======+
  //               idx:  0  1  2  3  4  5| 0  1  2  3  4  5|
  //      bw idle time:  0  2  0  0  0  0| 0  2  0  0  0  0|
  // additional memory:  5  5  9  9  9  9| 5  5  9  9  9  9|
  //         iteration:       prev       |      current    |
  //
  // Depending on the options provided, we can use alternative resource
  // satisfaction criteria. One option is to specify a percentage of the copy
  // resource that needs to be satisfied instead of the complete amount (100%).
  // This is called the "desired copy ratio". The reason why desired copy ratio
  // can be less than 100% is that in a memory-bound loop, we probably do not
  // have enough aggregate bandwidth resources to satisfy all of the prefetches,
  // but using up all of the default-memory bandwidth is more important than
  // having some prefetches with unsatisfied resources. In a similar vein,
  // another option is to accept prefetches that are fully pipelined, i.e.
  // their copy start time is scheduled the same time as the copy done time in
  // the previous iteration, regardless of how much of its copy resources are
  // actually satisfied. To illustrate a fully pipelined prefetch, consider
  // prefetch 3 (assume no prefetch 1 or 2 in this example) which is 15 MiB and
  // its first use is at index 4:
  //
  //        prefetch 3:=============+=================+===== unsat res: 4
  //               idx:  0  1  2  3  4  5| 0  1  2  3  4  5|
  //      bw idle time:  0  0  0  0  0  0| 0  0  0  0  0  0|
  // additional memory: 15 15 15 15 30 15|15 15 15 15 30 15|
  //         iteration:       prev       |      current    |
  //
  // Note that the additional memory consumption at index 4 is actually twice
  // the size of the prefetch as we are effectively double buffering. Also note
  // that the prefetch has an unsatisfied copy resource of 4 meaning the copy
  // will be in the critical path, but this actually will be faster than not
  // scheduling this particular prefetch in the first place since the bandwidth
  // idle time resource would go unused.
  float accumulated_copy_resource = 0;
  std::vector<int> early_forced_prefetch_value_indices;
  int early_forced_prefetch_value_search_index = 0;
  float early_forced_prefetch_additional_memory = 0;
  for (int i = first_use_idx - 1; i >= last_use_idx_sentinel - loop_size_;
       --i) {
    int loop_idx = (i + loop_size_) % loop_size_;
    // Check if this prefetch rolls over to the previous iteration, check if any
    // already-scheduled prefetches would violate the FIFO order, and if so,
    // "early-force" them to be co-scheduled with this prefetch to maintain the
    // FIFO order. This of course increases the required memory, so also keep
    // track of additional memory that would be consumed.
    if (i < 0) {
      for (; context.value_indices[early_forced_prefetch_value_search_index] !=
             value_index;
           ++early_forced_prefetch_value_search_index) {
        VLOG(3) << "Searching for early forced: "
                << early_forced_prefetch_value_search_index;
        LoopValue* early_forced_value = context.values.at(
            context.value_indices[early_forced_prefetch_value_search_index]);
        if (early_forced_value->allocations.empty()) {
          continue;
        }
        const CopyAllocation* early_forced_prefetch =
            static_cast<const CopyAllocation*>(
                early_forced_value->allocations.back().get());
        VLOG(3) << "Prefetch: " << early_forced_prefetch->ToString();

        // If the prefetch is already a roll-around prefetch, no need to further
        // early force it.
        if (early_forced_prefetch->copy_done_schedule_before() <=
                early_forced_prefetch->copy_start_schedule_after() + 1 ||
            (early_forced_prefetch->copy_start_schedule_after() ==
                 loop_size_ - 1 &&
             early_forced_prefetch->copy_done_schedule_before() == 0)) {
          break;
        }
        if (early_forced_prefetch->copy_start_schedule_after() != loop_idx) {
          break;
        }
        early_forced_prefetch_value_indices.push_back(
            early_forced_prefetch_value_search_index);
        early_forced_prefetch_additional_memory += early_forced_value->size;
        VLOG(3) << "Found early-forced prefetch value: "
                << early_forced_value->ToString();
        VLOG(3) << "Early forced prefetch additional memory: "
                << early_forced_prefetch_additional_memory;
      }
    }

    // Overlap memory overhead only happens if the copy start overlaps with the
    // first use (i.e. fully pipelined), so we'd need to account for 2X the
    // buffer at this time.
    int64_t overlap_memory_overhead = 0;
    if (loop_idx == last_use_idx) {
      overlap_memory_overhead = value->size;
      VLOG(3) << "Loop idx == last use idx (" << loop_idx
              << "), overlap memory overhead = " << overlap_memory_overhead;
    }

    // OOM; give up prefetch.
    if (context.additional_memory_used[loop_idx] + value->size +
            overlap_memory_overhead + early_forced_prefetch_additional_memory >
        remaining_memory_[loop_idx]) {
      VLOG(3) << "Ran out of memory. Accumulated copy resource "
              << accumulated_copy_resource << " out of " << copy_resource
              << " at " << loop_idx;
      break;
    }

    // We ideally find a time to overlap the prefetch fully where the previous
    // iteration's memory use is disjoint from this iteration. If that is not
    // possible, there are two compromises we could pick:
    //   - Find a prefetch time that satisfies a desired ratio < 1 of the
    //      prefetch elapsed time. This means the prefetch will be critical.
    //   - Overlap the prefetch with the previous iteration's buffer use, i.e.
    //     full pipelining. This would increase the peak memory consumption.
    float bandwidth_idle_time = context.bandwidth_idle_times[loop_idx];
    VLOG(3) << "Idx " << loop_idx
            << " bandwidth_idle_time: " << bandwidth_idle_time
            << " copy resource remaining: "
            << (copy_resource - accumulated_copy_resource) << " diff: "
            << (bandwidth_idle_time -
                (copy_resource - accumulated_copy_resource));
    if (bandwidth_idle_time >= copy_resource - accumulated_copy_resource) {
      accumulated_copy_resource = copy_resource;
      copy_start_time = loop_idx;
      VLOG(3) << "Found the complete copy ratio and updated accumulated copy "
                 "resource: "
              << accumulated_copy_resource;
      break;
    } else if (!copy_start_time &&
               accumulated_copy_resource + bandwidth_idle_time >=
                   copy_resource * options_.desired_copy_ratio()) {
      accumulated_copy_resource += bandwidth_idle_time;
      copy_start_time = loop_idx;
      VLOG(3) << "Found the desired copy ratio and updated accumulated copy "
                 "resource: "
              << accumulated_copy_resource;
    } else if (options_.allow_unsatisfied_fully_pipelined_prefetch() &&
               loop_idx == last_use_idx) {
      // Even if desired resource isn't reached, and if the options allow it,
      // allow a fully pipelined prefetch.
      accumulated_copy_resource += bandwidth_idle_time;
      copy_start_time = loop_idx;
      VLOG(3) << "Could not reach the desired copy ratio but scheduling "
                 "fully pipelined prefetch anyway: "
              << accumulated_copy_resource;
      break;
    } else {
      accumulated_copy_resource += bandwidth_idle_time;
      VLOG(3) << "Updated accumulated copy resource: "
              << accumulated_copy_resource;
    }
  }

  // Could not find a suitable copy start time.
  if (!copy_start_time) {
    return false;
  }

  VLOG(3) << "Success: copy_start_time: " << *copy_start_time
          << " leftover copy resource: "
          << (copy_resource - accumulated_copy_resource);
  auto update_additional_memory_used = [&](int loop_idx, int64_t addition) {
    VLOG(4) << "Updating additional memory used at " << loop_idx << ". "
            << context.additional_memory_used[loop_idx] << " + " << addition
            << " => " << (context.additional_memory_used[loop_idx] + addition)
            << " (remaining: " << remaining_memory_[loop_idx] << ")";
    context.additional_memory_used[loop_idx] += addition;
    CHECK_LE(context.additional_memory_used[loop_idx],
             remaining_memory_[loop_idx]);
  };
  for (int i = first_use_idx; i <= last_use_idx_sentinel; ++i) {
    int loop_idx = i % loop_size_;
    update_additional_memory_used(loop_idx, value->size);
  }
  // We reset accumulated copy resource and then reuse it to accumulate copy
  // resource time in order to replay the previous for loop. It is important
  // that we use the same arithmetic operations (as opposed to subtracting from
  // copy_resource) because floating point operations aren't commutative.
  accumulated_copy_resource = 0.0;
  for (int i = first_use_idx - 1; i >= last_use_idx_sentinel - loop_size_;
       --i) {
    int loop_idx = (i + loop_size_) % loop_size_;
    float& bandwidth_idle_time = context.bandwidth_idle_times[loop_idx];
    // Overlap memory overhead only happens if the copy start overlaps with the
    // first use (i.e. fully pipelined), so we'd need to account for 2X the
    // buffer at this time.
    int64_t overlap_memory_overhead = 0;
    update_additional_memory_used(loop_idx,
                                  value->size + overlap_memory_overhead);
    if (bandwidth_idle_time < copy_resource - accumulated_copy_resource) {
      accumulated_copy_resource += bandwidth_idle_time;
      bandwidth_idle_time = 0;
      if (loop_idx == *copy_start_time) {
        VLOG(3) << "Remaining copy resource: "
                << (copy_resource - accumulated_copy_resource);
        break;
      }
    } else {
      bandwidth_idle_time -= copy_resource - accumulated_copy_resource;
      CHECK_EQ(loop_idx, *copy_start_time);
      break;
    }
  }

  // Create the Allocation objects that correspond to the scheduled prefetch.
  CHECK(value->header_position);
  value->allocations.push_back(std::make_unique<PinnedAllocation>(
      *value->header_position, MemorySpace::kDefault, std::nullopt, 0,
      loop_size_, /*is_scoped_allocation=*/false));
  value->allocations.push_back(std::make_unique<CopyAllocation>(
      *value->allocations.back(), MemorySpace::kAlternate, std::nullopt,
      ((*copy_start_time - 1) + loop_size_) % loop_size_, first_use_idx,
      last_use_idx_sentinel));
  AddAllLoopPositionsAndUses(*value, /*allocate_next_iteration_uses=*/true);

  // Account for the additional memory used by early forcing the already
  // scheduled prefetches. Also modify the start times of these to this
  // prefetch's copy start time.
  for (int early_forced_prefetch_value_index :
       early_forced_prefetch_value_indices) {
    LoopValue* early_forced_value = context.values.at(
        context.value_indices[early_forced_prefetch_value_index]);
    CHECK(!early_forced_value->allocations.empty());
    CopyAllocation* early_forced_prefetch = static_cast<CopyAllocation*>(
        early_forced_value->allocations.back().get());
    for (int index = early_forced_prefetch->copy_start_schedule_after();
         index >= *copy_start_time; --index) {
      update_additional_memory_used(index, early_forced_value->size);
      VLOG(3) << "Additional memory used: " << index << " "
              << context.additional_memory_used[index];
    }
    early_forced_prefetch->set_copy_start_schedule_after(
        ((*copy_start_time - 1) + loop_size_) % loop_size_);
    VLOG(3) << "Updated prefetch: " << early_forced_prefetch->ToString();
  }
  return true;
}

void MemoryBoundLoopOptimizer::AddAllLoopPositionsAndUses(
    LoopValue& value, bool allocate_next_iteration_uses) {
  CHECK_GE(value.allocations.size(), 1);
  Allocation& allocation = *value.allocations.back();
  for (const auto& [idx, position] : value.loop_positions) {
    positions_in_alternate_mem_[position.instruction].push_back(position.index);
  }
  for (const auto& [idx, use] : value.loop_uses) {
    uses_in_alternate_mem_[use.instruction].push_back(
        {use.operand_number, use.operand_index});
    allocation.AddUse(use);
  }
  if (allocate_next_iteration_uses) {
    for (const auto& [next_iteration_idx, use] : value.next_iteration_uses) {
      HloInstruction* loop_instruction =
          hlo_live_range_.flattened_instruction_sequence().instructions().at(
              loop_start_ + next_iteration_idx);
      uses_in_alternate_mem_[loop_instruction].push_back(
          {use.operand_number, use.operand_index});
      allocation.AddUse(
          {loop_instruction, use.operand_number, use.operand_index});
    }
  }
}

float MemoryBoundLoopOptimizer::GetBandwidthIdleTime(int idx) const {
  const HloInstruction* inst =
      hlo_live_range_.flattened_instruction_sequence().instructions().at(
          loop_start_ + idx);
  std::vector<std::pair<int64_t, ShapeIndex>> empty_operands;
  std::vector<ShapeIndex> empty_outputs;
  const std::vector<std::pair<int64_t, ShapeIndex>>* operands_in_alternate_mem =
      &empty_operands;
  const std::vector<ShapeIndex>* outputs_in_alternate_mem = &empty_outputs;
  auto uses_it = uses_in_alternate_mem_.find(inst);
  if (uses_it != uses_in_alternate_mem_.end()) {
    operands_in_alternate_mem = &uses_it->second;
  }
  auto positions_it = positions_in_alternate_mem_.find(inst);
  if (positions_it != positions_in_alternate_mem_.end()) {
    outputs_in_alternate_mem = &positions_it->second;
  }
  return cost_analysis_.GetDefaultMemoryBandwidthIdleTime(
      *inst, *operands_in_alternate_mem, *outputs_in_alternate_mem);
}

float MemoryBoundLoopOptimizer::GetBandwidthIdleTime(
    int idx,
    const absl::flat_hash_map<const HloInstruction*,
                              std::vector<std::pair<int64_t, ShapeIndex>>>&
        additional_uses_in_alternate_mem,
    const absl::flat_hash_map<const HloInstruction*, std::vector<ShapeIndex>>&
        additional_positions_in_alternate_mem) const {
  const HloInstruction* inst =
      hlo_live_range_.flattened_instruction_sequence().instructions().at(
          loop_start_ + idx);
  std::vector<std::pair<int64_t, ShapeIndex>> operands_in_alternate_mem;
  std::vector<ShapeIndex> outputs_in_alternate_mem;
  auto uses_it = uses_in_alternate_mem_.find(inst);
  if (uses_it != uses_in_alternate_mem_.end()) {
    operands_in_alternate_mem = uses_it->second;
  }
  auto additional_uses_it = additional_uses_in_alternate_mem.find(inst);
  if (additional_uses_it != additional_uses_in_alternate_mem.end()) {
    absl::c_copy(additional_uses_it->second,
                 std::back_inserter(operands_in_alternate_mem));
  }
  auto positions_it = positions_in_alternate_mem_.find(inst);
  if (positions_it != positions_in_alternate_mem_.end()) {
    outputs_in_alternate_mem = positions_it->second;
  }
  auto additional_positions_it =
      additional_positions_in_alternate_mem.find(inst);
  if (additional_positions_it != additional_positions_in_alternate_mem.end()) {
    absl::c_copy(additional_positions_it->second,
                 std::back_inserter(outputs_in_alternate_mem));
  }
  return cost_analysis_.GetDefaultMemoryBandwidthIdleTime(
      *inst, operands_in_alternate_mem, outputs_in_alternate_mem);
}

float MemoryBoundLoopOptimizer::GetInstructionElapsed(int idx) const {
  const HloInstruction* inst =
      hlo_live_range_.flattened_instruction_sequence().instructions().at(
          loop_start_ + idx);
  std::vector<std::pair<int64_t, ShapeIndex>> empty_operands;
  std::vector<ShapeIndex> empty_outputs;
  const std::vector<std::pair<int64_t, ShapeIndex>>* operands_in_alternate_mem =
      &empty_operands;
  const std::vector<ShapeIndex>* outputs_in_alternate_mem = &empty_outputs;
  auto uses_it = uses_in_alternate_mem_.find(inst);
  if (uses_it != uses_in_alternate_mem_.end()) {
    operands_in_alternate_mem = &uses_it->second;
  }
  auto positions_it = positions_in_alternate_mem_.find(inst);
  if (positions_it != positions_in_alternate_mem_.end()) {
    outputs_in_alternate_mem = &positions_it->second;
  }
  return cost_analysis_.GetInstructionElapsedInAlternateMemory(
      *inst, *operands_in_alternate_mem, *outputs_in_alternate_mem);
}

}  // namespace memory_space_assignment
}  // namespace xla

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

#include "xla/service/memory_space_assignment/cost_analysis.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/call_graph.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo_alias_analysis.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_value.h"
#include "xla/service/while_loop_analysis.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace memory_space_assignment {

HloCostAnalysisCosts::HloCostAnalysisCosts(
    const HloCostAnalysis& hlo_cost_analysis)
    : hlo_cost_analysis_(hlo_cost_analysis) {}

int64_t HloCostAnalysisCosts::GetShapeSize(const Shape& shape) {
  return hlo_cost_analysis_.GetShapeSize(shape);
}

float HloCostAnalysisCosts::BytesAccessed(const HloInstruction& instruction) {
  return static_cast<float>(hlo_cost_analysis_.bytes_accessed(instruction));
}

float HloCostAnalysisCosts::OperandBytesAccessed(
    const HloInstruction& instruction, int64_t operand_num,
    const ShapeIndex& shape_index) {
  return static_cast<float>(hlo_cost_analysis_.operand_bytes_accessed(
      instruction, operand_num, shape_index));
}

float HloCostAnalysisCosts::OutputBytesAccessed(
    const HloInstruction& instruction, const ShapeIndex& shape_index) {
  return static_cast<float>(
      hlo_cost_analysis_.output_bytes_accessed(instruction, shape_index));
}

float HloCostAnalysisCosts::BytesPerSecond() {
  return hlo_cost_analysis_.per_second_rate(HloCostAnalysis::kBytesAccessedKey);
}

float HloCostAnalysisCosts::ComputeSeconds(const HloInstruction& instruction) {
  return std::max(
      static_cast<float>(hlo_cost_analysis_.flop_count(instruction)) /
          hlo_cost_analysis_.per_second_rate(HloCostAnalysis::kFlopsKey),
      static_cast<float>(hlo_cost_analysis_.transcendental_count(instruction)) /
          hlo_cost_analysis_.per_second_rate(
              HloCostAnalysis::kTranscendentalsKey));
}

/*static*/ absl::StatusOr<std::unique_ptr<CostAnalysis>> CostAnalysis::Create(
    BaseCosts& base_costs, const CostAnalysisOptions& options,
    const HloModule& module) {
  TF_ASSIGN_OR_RETURN(auto alias_analysis, HloAliasAnalysis::Run(&module));
  TF_ASSIGN_OR_RETURN(auto hlo_live_range,
                      HloLiveRange::Run(module.schedule(), *alias_analysis,
                                        module.entry_computation()));
  auto call_graph = CallGraph::Build(&module);
  // Using `new` to access a non-public constructor.
  return absl::WrapUnique(
      new CostAnalysis(base_costs, options, std::move(alias_analysis),
                       std::move(hlo_live_range), std::move(call_graph)));
}

float CostAnalysis::GetAlternateMemoryBenefit(
    const HloInstruction& instruction, float elapsed_time_due_to_alternate_mem,
    CostAnalysis::Cache* cache) const {
  float elapsed_time_due_to_compute =
      GetInstructionElapsedDueToCompute(instruction);
  float elapsed_time_due_to_memory =
      GetInstructionElapsedDueToMemory(instruction);
  if (elapsed_time_due_to_memory > elapsed_time_due_to_compute) {
    // Memory bound, return how much alternate memory is better.
    float while_nest_multiplier;
    if (cache) {
      // If there is a cache provided, memoize the while nest multiplier.
      auto it = cache->while_nest_multiplier.find(&instruction);
      if (it != cache->while_nest_multiplier.end()) {
        while_nest_multiplier = it->second;
      } else {
        while_nest_multiplier = GetWhileNestMultiplier(
            CalculateComputationNestLevel(&instruction,
                                          /*while_only=*/true));
        cache->while_nest_multiplier[&instruction] = while_nest_multiplier;
      }
    } else {
      while_nest_multiplier = GetWhileNestMultiplier(
          CalculateComputationNestLevel(&instruction,
                                        /*while_only=*/true));
    }
    return (elapsed_time_due_to_memory - elapsed_time_due_to_alternate_mem) *
           while_nest_multiplier;
  } else {
    // Compute bound, return how far off are we to memory boundedness.
    return elapsed_time_due_to_memory - elapsed_time_due_to_compute;
  }
}

float CostAnalysis::GetMemoryBoundedness(
    const GlobalDecreasingSizeBestFitHeap<HloValue>::BufferInterval& interval,
    CostAnalysis::Cache* cache) const {
  if (cache) {
    auto it =
        cache->memory_boundedness.find(interval.buffer->defining_position());
    if (it != cache->memory_boundedness.end()) {
      return it->second;
    }
  }
  float alternate_mem_benefit =
      GetAlternateMemoryBenefit(interval.buffer->defining_position(), cache);

  for (const HloBuffer* buffer : alias_analysis_->ComputeBuffersAt(
           interval.buffer->defining_position().instruction,
           interval.buffer->defining_position().index)) {
    for (const HloValue* value : buffer->values()) {
      for (const HloUse& use : value->GetUses()) {
        // We look inside the called computations of while and conditional, so
        // don't use the benefit of while and conditional directly.
        if (use.instruction->opcode() == HloOpcode::kWhile ||
            use.instruction->opcode() == HloOpcode::kConditional) {
          continue;
        }
        float use_alternate_mem_benefit = GetAlternateMemoryBenefit(use, cache);
        // If the benefit is positive (memory bound), add it to this buffer's
        // benefit. If the benefit is negative (compute bound), calculate the
        // maximum.
        if (alternate_mem_benefit > 0 && use_alternate_mem_benefit > 0) {
          alternate_mem_benefit += use_alternate_mem_benefit;
        } else {
          alternate_mem_benefit =
              std::max(alternate_mem_benefit, use_alternate_mem_benefit);
        }
      }
    }
  }

  // Penalize larger buffers by dividing the benefit by the square root of
  // the size. Empirically, we observed this resulted in better performance
  // compared to dividing by the size.
  float memory_boundedness = 1;
  if (options_
          .xla_tpu_alternate_memory_benefit_scaling_factor_for_large_buffers ==
      "NO_SCALE") {
    memory_boundedness = alternate_mem_benefit;
  } else {
    memory_boundedness = alternate_mem_benefit / std::sqrt(interval.size);
  }

  if (cache) {
    cache->memory_boundedness[interval.buffer->defining_position()] =
        memory_boundedness;
  }
  return memory_boundedness;
}

float CostAnalysis::GetAlternateMemoryBenefit(
    const HloPosition& position, CostAnalysis::Cache* cache) const {
  return GetAlternateMemoryBenefit(
      *position.instruction,
      GetInstructionElapsedDueToMemory(
          *position.instruction,
          /*operands_in_alternate_mem=*/{},
          /*outputs_in_alternate_mem=*/{position.index}),
      cache);
}

float CostAnalysis::GetAlternateMemoryBenefit(
    const HloUse& use, CostAnalysis::Cache* cache) const {
  return GetAlternateMemoryBenefit(
      *use.instruction,
      GetInstructionElapsedDueToMemory(
          *use.instruction,
          /*operands_in_alternate_mem=*/{std::make_pair(use.operand_number,
                                                        use.operand_index)}),
      cache);
}

int CostAnalysis::CalculateComputationNestLevel(
    const HloInstruction* instruction, bool while_only) const {
  int nest_level = 0;
  const HloComputation* computation = instruction->parent();
  while (!computation->IsEntryComputation()) {
    auto& node = call_graph_->GetNode(computation);
    auto callsites = node.caller_callsites();
    CHECK(node.computation()->IsAsyncComputation() || callsites.size() == 1)
        << "The module is not flattened!";
    auto& callsite = callsites[0];
    if (!while_only || callsite.instruction()->opcode() == HloOpcode::kWhile) {
      ++nest_level;
    }
    computation = callsite.instruction()->parent();
  }
  return nest_level;
}

// TODO(hanruobing): This function assumes all nested layers have the
// same hard-coded trip count for simplicity. I plan to replace it with the
// more accurate function (CalculateNestTripCount).
float CostAnalysis::GetWhileNestMultiplier(int while_nest_level) const {
  return IPow<float>(
      options_.xla_tpu_memory_space_assignment_while_execution_count,
      while_nest_level);
}

float CostAnalysis::CalculateNestTripCount(const HloInstruction* instruction,
                                           CostAnalysis::Cache* cache) const {
  float total_trip_count = 1.0;
  const HloComputation* computation = instruction->parent();
  while (!computation->IsEntryComputation()) {
    if (cache) {
      auto it = cache->computation_trip_count.find(computation);
      if (it != cache->computation_trip_count.end()) {
        if (computation == instruction->parent()) {
          return it->second;
        } else {
          total_trip_count *= it->second;
          break;
        }
      }
    }
    CallGraphNode& node = call_graph_->GetNode(computation);
    absl::Span<const CallSite> callsites = node.caller_callsites();
    const xla::CallSite& callsite = callsites[0];
    if (callsite.instruction()->opcode() == HloOpcode::kWhile) {
      HloInstruction* while_op = callsite.instruction();
      std::optional<float> trip_count;
      if (!trip_count.has_value()) {
        // TODO(hanruobing): Apply PrepareModuleForUnrolling on the module may
        // provide more accurate results for trip count analysis . However, it
        // may downgrade the performance of MSA. We need more evaluation to
        // decide whether to apply that pass before MSA.
        trip_count = ComputeWhileLoopTripCount(while_op);
      }
      total_trip_count *= trip_count.value_or(
          options_.xla_tpu_memory_space_assignment_while_execution_count);
    }
    computation = callsite.instruction()->parent();
  }
  if (cache) {
    cache->computation_trip_count[instruction->parent()] = total_trip_count;
  }
  return total_trip_count;
}

float CostAnalysis::GetDefaultMemoryAccessOverhead(
    const HloInstruction& instruction,
    absl::Span<const std::pair<int64_t, ShapeIndex>> operands_in_alternate_mem,
    absl::Span<const ShapeIndex> outputs_in_alternate_mem) const {
  // Calculate the pipeline overhead of accessing the default memory. We use the
  // maximum of the window size heuristic and the actual default memory bytes
  // accessed multiplied with the compute as the overhead. So, the math is:
  //
  // overhead = compute_per_iteration
  //          = compute_elapsed / num_iterations
  //          = compute_elapsed / (bytes_accessed / window_size)
  //          = (window_size / bytes_accessed) * compute_elapsed
  const float window_size_bytes =
      options_.pipeline_overhead_window_size_mib * 1024 * 1024;
  const float bytes_accessed = base_costs_.BytesAccessed(instruction);
  const float default_memory_bytes_accessed =
      bytes_accessed -
      GetBytesAccessedFromAlternateMemory(
          instruction, operands_in_alternate_mem, outputs_in_alternate_mem);
  const float compute_elapsed = GetInstructionElapsedDueToCompute(instruction);
  const float effective_window_size_bytes =
      std::min(window_size_bytes, default_memory_bytes_accessed);
  float overhead = 0;
  if (bytes_accessed > 0) {
    overhead = (effective_window_size_bytes / bytes_accessed) * compute_elapsed;
  }
  return overhead;
}

float CostAnalysis::GetDefaultMemoryBandwidthIdleTime(
    const HloInstruction& instruction,
    absl::Span<const std::pair<int64_t, ShapeIndex>> operands_in_alternate_mem,
    absl::Span<const ShapeIndex> outputs_in_alternate_mem) const {
  const float default_memory_bytes_accessed =
      base_costs_.BytesAccessed(instruction) -
      GetBytesAccessedFromAlternateMemory(
          instruction, operands_in_alternate_mem, outputs_in_alternate_mem);
  const float elapsed_due_to_default_mem =
      default_memory_bytes_accessed / base_costs_.BytesPerSecond();
  const float elapsed = GetInstructionElapsedInAlternateMemory(
      instruction, operands_in_alternate_mem, outputs_in_alternate_mem);
  return elapsed - elapsed_due_to_default_mem;
}

float CostAnalysis::GetBytesAccessedFromAlternateMemory(
    const HloInstruction& instruction,
    absl::Span<const std::pair<int64_t, ShapeIndex>> operands_in_alternate_mem,
    absl::Span<const ShapeIndex> outputs_in_alternate_mem) const {
  float bytes_accessed_from_alternate_mem = 0.0;
  for (auto& operand : operands_in_alternate_mem) {
    const float operand_bytes_accessed = base_costs_.OperandBytesAccessed(
        instruction, operand.first, operand.second);
    bytes_accessed_from_alternate_mem += operand_bytes_accessed;
  }

  for (auto& shape_idx : outputs_in_alternate_mem) {
    const float output_bytes_accessed =
        base_costs_.OutputBytesAccessed(instruction, shape_idx);
    bytes_accessed_from_alternate_mem += output_bytes_accessed;
  }
  return bytes_accessed_from_alternate_mem;
}

namespace {
// Returns true on async instructions since we assume they are already
// efficiently scheduled such that they are not in the critical path and appear
// to take no time.
bool ExcludeInstructionFromElapsed(const HloInstruction& instruction) {
  return instruction.opcode() == HloOpcode::kAllGatherStart ||
         instruction.opcode() == HloOpcode::kAllGatherDone ||
         instruction.opcode() == HloOpcode::kAllReduceStart ||
         instruction.opcode() == HloOpcode::kAllReduceDone ||
         instruction.opcode() == HloOpcode::kAsyncStart ||
         instruction.opcode() == HloOpcode::kAsyncDone ||
         instruction.opcode() == HloOpcode::kCollectivePermuteStart ||
         instruction.opcode() == HloOpcode::kCollectivePermuteDone ||
         instruction.opcode() == HloOpcode::kCopyStart ||
         instruction.opcode() == HloOpcode::kCopyDone;
}
}  // namespace

float CostAnalysis::GetInstructionElapsedDueToCompute(
    const HloInstruction& instruction) const {
  if (ExcludeInstructionFromElapsed(instruction)) {
    return 0.0f;
  }
  return base_costs_.ComputeSeconds(instruction);
}

float CostAnalysis::GetInstructionElapsedDueToMemory(
    const HloInstruction& instruction,
    absl::Span<const std::pair<int64_t, ShapeIndex>> operands_in_alternate_mem,
    absl::Span<const ShapeIndex> outputs_in_alternate_mem) const {
  if (ExcludeInstructionFromElapsed(instruction)) {
    return 0.0f;
  }
  float total_bytes_accessed = base_costs_.BytesAccessed(instruction);
  float bytes_accessed_from_alternate_mem = GetBytesAccessedFromAlternateMemory(
      instruction, operands_in_alternate_mem, outputs_in_alternate_mem);
  float elapsed_due_to_alternate_mem =
      bytes_accessed_from_alternate_mem /
      options_.alternate_mem_bandwidth_bytes_per_second;
  float elapsed_due_to_default_mem =
      (total_bytes_accessed - bytes_accessed_from_alternate_mem) /
      base_costs_.BytesPerSecond();
  return elapsed_due_to_alternate_mem + elapsed_due_to_default_mem;
}

float CostAnalysis::GetInstructionElapsedDueToMemory(
    const HloInstruction& instruction,
    IsInAlternateMemoryFun is_in_alternate_mem) const {
  if (ExcludeInstructionFromElapsed(instruction)) {
    return 0.0f;
  }
  float total_bytes_accessed = base_costs_.BytesAccessed(instruction);
  float bytes_accessed_from_alternate_mem = 0.0;
  for (int operand_num = 0; operand_num < instruction.operand_count();
       ++operand_num) {
    ShapeUtil::ForEachSubshape(
        instruction.operand(operand_num)->shape(),
        [&](const Shape& subshape, const ShapeIndex& index) {
          if (!subshape.IsArray()) {
            return;
          }
          if (is_in_alternate_mem(operand_num, index, subshape)) {
            bytes_accessed_from_alternate_mem +=
                base_costs_.OperandBytesAccessed(instruction, operand_num,
                                                 index);
          }
        });
  }
  ShapeUtil::ForEachSubshape(instruction.shape(), [&](const Shape& subshape,
                                                      const ShapeIndex& index) {
    if (!subshape.IsArray()) {
      return;
    }
    if (is_in_alternate_mem(/*operand_num=*/std::nullopt, index, subshape)) {
      bytes_accessed_from_alternate_mem +=
          base_costs_.OutputBytesAccessed(instruction, index);
    }
  });
  float elapsed_due_to_alternate_mem =
      bytes_accessed_from_alternate_mem /
      options_.alternate_mem_bandwidth_bytes_per_second;
  float elapsed_due_to_default_mem =
      (total_bytes_accessed - bytes_accessed_from_alternate_mem) /
      base_costs_.BytesPerSecond();
  return elapsed_due_to_alternate_mem + elapsed_due_to_default_mem;
}

float CostAnalysis::GetInstructionElapsed(
    const HloInstruction& instruction) const {
  if (ExcludeInstructionFromElapsed(instruction)) {
    return 0.0f;
  }
  float overhead = GetDefaultMemoryAccessOverhead(instruction);
  return std::max(GetInstructionElapsedDueToCompute(instruction),
                  GetInstructionElapsedDueToMemory(instruction) + overhead);
}

float CostAnalysis::GetInstructionElapsedInAlternateMemory(
    const HloInstruction& instruction,
    absl::Span<const std::pair<int64_t, ShapeIndex>> operands_in_alternate_mem,
    absl::Span<const ShapeIndex> outputs_in_alternate_mem) const {
  if (ExcludeInstructionFromElapsed(instruction)) {
    return 0.0f;
  }
  float overhead = GetDefaultMemoryAccessOverhead(
      instruction, operands_in_alternate_mem, outputs_in_alternate_mem);
  return std::max(
      GetInstructionElapsedDueToCompute(instruction),
      GetInstructionElapsedDueToMemory(instruction, operands_in_alternate_mem,
                                       outputs_in_alternate_mem) +
          overhead);
}

float CostAnalysis::GetInstructionElapsedInAlternateMemory(
    const HloInstruction& instruction,
    IsInAlternateMemoryFun is_in_alternate_mem) const {
  if (ExcludeInstructionFromElapsed(instruction)) {
    return 0.0f;
  }
  return std::max(
      GetInstructionElapsedDueToCompute(instruction),
      GetInstructionElapsedDueToMemory(instruction, is_in_alternate_mem));
}

float CostAnalysis::GetAsyncCopyElapsed(const Shape& shape) const {
  int64_t size_in_bytes = base_costs_.GetShapeSize(shape);
  return static_cast<float>(size_in_bytes) /
         (options_.async_copy_bandwidth_bytes_per_second *
          options_.async_copy_bandwidth_scaling_factor);
}

int64_t CostAnalysis::GetScheduleEndTime() const {
  return hlo_live_range_->schedule_end_time();
}

}  // namespace memory_space_assignment
}  // namespace xla

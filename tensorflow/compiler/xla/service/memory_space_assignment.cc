/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/memory_space_assignment.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <utility>

#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/memory_space_assignment_tuning_utils.h"
#include "tensorflow/compiler/xla/service/memory_space_assignment_utils.h"
#include "tensorflow/compiler/xla/service/tuple_util.h"
#include "tensorflow/core/lib/math/math_util.h"
namespace xla {

namespace memory_space_assignment {

namespace {
// Define a dummy chunk for chunks that will be allocated in the default memory
// space and for keeping track of number of asynchronous copies.
const HeapSimulator::Chunk kDummyChunk{-1, -1};
// For cross-program prefetched buffer, we only perform the freeing optimization
// if the buffer occupies less of the execution time ratio than this value.
const float kCrossProgramPrefetchOccupyFreeingLimit = 0.6;
// Each time we retry compilation, increase the preferred eviction end time by
// this amount multiplied by preferred overlap to async copy ratio.
const float kEvictionRetryMultiplier = 2.0;

bool LooksLikeAnActivation(const HloInstruction* inst) {
  for (HloInstruction* user : inst->users()) {
    switch (user->opcode()) {
      case HloOpcode::kConvolution:
      case HloOpcode::kDot:
        if (user->operand(0) == inst) {
          return true;
        }
        break;
      case HloOpcode::kGather:
        if (user->operand(1) == inst) {
          return true;
        }
        break;
      case HloOpcode::kFusion:
        for (int i = 0; i < user->operand_count(); ++i) {
          if (user->operand(i) == inst &&
              LooksLikeAnActivation(user->fused_parameter(i))) {
            return true;
          }
        }
        break;
      case HloOpcode::kBitcast:
        return LooksLikeAnActivation(user);
      default:
        return true;
    }
  }
  return false;
}

bool IsCrossProgramPrefetchCandidate(const HloValue& value,
                                     const Options& options) {
  return value.defining_instruction()->parent() ==
             value.defining_instruction()->GetModule()->entry_computation() &&
         value.defining_instruction()->opcode() == HloOpcode::kParameter &&
         (!value.shape().has_layout() ||
          value.shape().layout().memory_space() !=
              options.alternate_memory_space) &&
         value.index().size() <= 1 && value.shape().IsArray() &&
         !value.uses().empty() &&
         options.size_fn(value) <= options.max_size_in_bytes &&
         absl::c_all_of(value.uses(), [&](const HloUse& use) {
           const HloInstruction* inst =
               use.instruction->operand(use.operand_number);

           // Skip the LooksLikeAnActivation test since we're testing the
           // parent GTE/parameter and its children below.
           if (inst->opcode() == HloOpcode::kBitcast &&
               ((inst->operand(0)->opcode() == HloOpcode::kGetTupleElement &&
                 inst->operand(0)->operand(0)->opcode() ==
                     HloOpcode::kParameter) ||
                inst->operand(0)->opcode() == HloOpcode::kParameter)) {
             return true;
           }

           return (inst->opcode() == HloOpcode::kGetTupleElement ||
                   inst->opcode() == HloOpcode::kParameter) &&
                  !LooksLikeAnActivation(inst);
         });
}

absl::optional<MemorySpaceAssignment::BufferInterval>
FindCrossProgramPrefetchCandidate(const HloAliasAnalysis& alias_analysis,
                                  const HloLiveRange& hlo_live_range,
                                  const Options& options) {
  std::vector<MemorySpaceAssignment::BufferInterval> candidates;
  for (const HloBuffer& buffer : alias_analysis.buffers()) {
    CHECK_GE(buffer.values().size(), 1);
    const HloValue* value = buffer.values().at(0);
    if (IsCrossProgramPrefetchCandidate(*value, options)) {
      MemorySpaceAssignment::BufferInterval interval;
      interval.buffer = value;
      interval.size = options.size_fn(*value);
      interval.start = 0;
      interval.end = hlo_live_range.schedule_end_time();
      interval.need_allocation = true;
      interval.colocations = {++buffer.values().begin(), buffer.values().end()};
      candidates.emplace_back(interval);
    }
  }

  // The BufferIntervalCompare function used to sort buffers implements the
  // greater-than operator so that the most beneficial buffers are allocated
  // first. The size_compare function below hence uses the greater-than operator
  // to pick the largest buffer.
  auto size_compare = [](const auto& x, const auto& y) {
    if (x.size == y.size) {
      // When both buffers are of same size, we prefer the one that is used to
      // produce larger tensors in its consumer instructions.
      auto get_use_size =
          [](const MemorySpaceAssignment::BufferInterval& bi) -> int64_t {
        int64_t use_size = 0;
        for (const auto& use : bi.buffer->uses()) {
          use_size += ShapeUtil::ElementsInRecursive(use.instruction->shape());
        }
        return use_size;
      };
      return get_use_size(x) > get_use_size(y);
    }
    return x.size > y.size;
  };
  auto& compare = options.default_cross_program_prefetch_heuristic &&
                          options.buffer_interval_compare
                      ? *options.buffer_interval_compare
                      : size_compare;

  auto best_candidate = absl::c_min_element(candidates, compare);
  if (best_candidate == candidates.end()) {
    return absl::nullopt;
  }
  VLOG(3) << "Cross-program prefetch candidate picked: "
          << best_candidate->buffer->ToString();
  return *best_candidate;
}

Status InsertInstructionAndEnsureOperandsInserted(
    HloInstruction* new_instruction, HloInstructionSequence* new_sequence,
    absl::flat_hash_set<HloInstruction*>* inserted_instructions);

// Insert an instruction to the schedule, and make sure its dependencies
// (operands) are already in the schedule. If not, insert these operands
// before the instruction.
Status EnsureInstructionAndOperandsInserted(
    HloInstruction* new_instruction, HloInstructionSequence* new_sequence,
    absl::flat_hash_set<HloInstruction*>* inserted_instructions) {
  if (inserted_instructions->contains(new_instruction)) {
    return Status::OK();
  }
  return InsertInstructionAndEnsureOperandsInserted(
      new_instruction, new_sequence, inserted_instructions);
}

// Same as above, but does not check if instruction is already inserted. This is
// used when the caller already knows the instruction isn't inserted yet, to
// speed up compilation.
Status InsertInstructionAndEnsureOperandsInserted(
    HloInstruction* new_instruction, HloInstructionSequence* new_sequence,
    absl::flat_hash_set<HloInstruction*>* inserted_instructions) {
  for (HloInstruction* operand : new_instruction->operands()) {
    // CopyStart/CopyDone dependencies should always be already inserted; it is
    // a red flag when they haven't already been inserted.
    if (operand->opcode() == HloOpcode::kCopyStart ||
        operand->opcode() == HloOpcode::kCopyDone) {
      TF_RET_CHECK(inserted_instructions->contains(operand))
          << "Inserted instruction " << new_instruction->ToString()
          << " has un-inserted dependency: " << operand->ToString();
      continue;
    }
    TF_RETURN_IF_ERROR(EnsureInstructionAndOperandsInserted(
        operand, new_sequence, inserted_instructions));
  }
  VLOG(4) << "inserting: " << new_instruction->ToShortString();
  new_sequence->push_back(new_instruction);
  TF_RET_CHECK(inserted_instructions->insert(new_instruction).second);
  return Status::OK();
}

}  // namespace

/*static*/ StatusOr<std::unique_ptr<MemorySpaceAssignmentCostAnalysis>>
MemorySpaceAssignmentCostAnalysis::Create(const HloCostAnalysis& cost_analysis,
                                          const Options& options,
                                          const HloModule& module) {
  TF_ASSIGN_OR_RETURN(auto alias_analysis, HloAliasAnalysis::Run(&module));
  TF_ASSIGN_OR_RETURN(auto hlo_live_range,
                      HloLiveRange::Run(module.schedule(), *alias_analysis,
                                        module.entry_computation()));
  auto call_graph = CallGraph::Build(&module);
  return absl::WrapUnique(new MemorySpaceAssignmentCostAnalysis(
      cost_analysis, options, std::move(alias_analysis),
      std::move(hlo_live_range), std::move(call_graph)));
}

float MemorySpaceAssignmentCostAnalysis::GetAlternateMemoryBenefit(
    const HloInstruction& instruction, float elapsed_time_due_to_alternate_mem,
    MemorySpaceAssignmentCostAnalysis::Cache* cache) const {
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
        while_nest_multiplier = tensorflow::MathUtil::IPow<float>(
            options_.xla_tpu_memory_space_assignment_while_execution_count,
            CalculateComputationNestLevel(&instruction,
                                          /*while_only=*/true));
        cache->while_nest_multiplier[&instruction] = while_nest_multiplier;
      }
    } else {
      while_nest_multiplier = tensorflow::MathUtil::IPow<float>(
          options_.xla_tpu_memory_space_assignment_while_execution_count,
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

float MemorySpaceAssignmentCostAnalysis::GetMemoryBoundedness(
    const GlobalDecreasingSizeBestFitHeap<HloValue>::BufferInterval& interval,
    MemorySpaceAssignmentCostAnalysis::Cache* cache) const {
  float alternate_mem_benefit =
      GetAlternateMemoryBenefit(interval.buffer->defining_position(), cache);

  for (const HloBuffer* buffer : alias_analysis_->ComputeBuffersAt(
           interval.buffer->defining_position().instruction,
           interval.buffer->defining_position().index)) {
    for (const HloValue* value : buffer->values()) {
      for (const HloUse& use : value->uses()) {
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

  // Penalize larger buffers by dividing the benefit by the square root of the
  // size. Empirically, we observed this resulted in better performance compared
  // to dividing by the size.
  return alternate_mem_benefit / std::sqrt(interval.size);
}

float MemorySpaceAssignmentCostAnalysis::GetAlternateMemoryBenefit(
    const HloPosition& position,
    MemorySpaceAssignmentCostAnalysis::Cache* cache) const {
  return GetAlternateMemoryBenefit(
      *position.instruction,
      GetInstructionElapsedDueToMemory(
          *position.instruction,
          /*operands_in_alternate_mem=*/{},
          /*outputs_in_alternate_mem=*/{position.index}),
      cache);
}

float MemorySpaceAssignmentCostAnalysis::GetAlternateMemoryBenefit(
    const HloUse& use, MemorySpaceAssignmentCostAnalysis::Cache* cache) const {
  return GetAlternateMemoryBenefit(
      *use.instruction,
      GetInstructionElapsedDueToMemory(
          *use.instruction,
          /*operands_in_alternate_mem=*/{std::make_pair(use.operand_number,
                                                        use.operand_index)}),
      cache);
}

int MemorySpaceAssignmentCostAnalysis::CalculateComputationNestLevel(
    const HloInstruction* instruction, bool while_only) const {
  int nest_level = 0;
  const HloComputation* computation = instruction->parent();
  while (!computation->IsEntryComputation()) {
    auto node = call_graph_->GetNode(computation);
    auto callsites = node.caller_callsites();
    CHECK_EQ(callsites.size(), 1) << "The module is not flattened!";
    auto callsite = callsites[0];
    if (!while_only || callsite.instruction()->opcode() == HloOpcode::kWhile) {
      ++nest_level;
    }
    computation = callsite.instruction()->parent();
  }
  return nest_level;
}

float MemorySpaceAssignmentCostAnalysis::GetInstructionElapsedDueToCompute(
    const HloInstruction& instruction) const {
  return std::max(
      cost_analysis_.flop_count(instruction) /
          cost_analysis_.per_second_rate(HloCostAnalysis::kFlopsKey),
      cost_analysis_.transcendental_count(instruction) /
          cost_analysis_.per_second_rate(HloCostAnalysis::kTranscendentalsKey));
}

float MemorySpaceAssignmentCostAnalysis::GetInstructionElapsedDueToMemory(
    const HloInstruction& instruction,
    absl::Span<const std::pair<int64_t, ShapeIndex>> operands_in_alternate_mem,
    absl::Span<const ShapeIndex> outputs_in_alternate_mem) const {
  float total_bytes_accessed = cost_analysis_.bytes_accessed(instruction);
  float bytes_accessed_from_alternate_mem = 0.0;
  for (auto& operand : operands_in_alternate_mem) {
    float operand_bytes_accessed = cost_analysis_.operand_bytes_accessed(
        instruction, operand.first, operand.second);
    bytes_accessed_from_alternate_mem += operand_bytes_accessed;
  }

  for (auto& shape_idx : outputs_in_alternate_mem) {
    float output_bytes_accessed =
        cost_analysis_.output_bytes_accessed(instruction, shape_idx);
    bytes_accessed_from_alternate_mem += output_bytes_accessed;
  }
  float elapsed_due_to_alternate_mem =
      bytes_accessed_from_alternate_mem /
      options().alternate_mem_bandwidth_bytes_per_second;
  float elapsed_due_to_default_mem =
      (total_bytes_accessed - bytes_accessed_from_alternate_mem) /
      cost_analysis_.per_second_rate(HloCostAnalysis::kBytesAccessedKey);
  return elapsed_due_to_alternate_mem + elapsed_due_to_default_mem;
}

float MemorySpaceAssignmentCostAnalysis::GetInstructionElapsedDueToMemory(
    const HloInstruction& instruction,
    IsInAlternateMemoryFun is_in_alternate_mem) const {
  float total_bytes_accessed = cost_analysis_.bytes_accessed(instruction);
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
                cost_analysis_.operand_bytes_accessed(instruction, operand_num,
                                                      index);
          }
        });
  }
  ShapeUtil::ForEachSubshape(instruction.shape(), [&](const Shape& subshape,
                                                      const ShapeIndex& index) {
    if (!subshape.IsArray()) {
      return;
    }
    if (is_in_alternate_mem(/*operand_num=*/absl::nullopt, index, subshape)) {
      bytes_accessed_from_alternate_mem +=
          cost_analysis_.output_bytes_accessed(instruction, index);
    }
  });
  float elapsed_due_to_alternate_mem =
      bytes_accessed_from_alternate_mem /
      options().alternate_mem_bandwidth_bytes_per_second;
  float elapsed_due_to_default_mem =
      (total_bytes_accessed - bytes_accessed_from_alternate_mem) /
      cost_analysis_.per_second_rate(HloCostAnalysis::kBytesAccessedKey);
  return elapsed_due_to_alternate_mem + elapsed_due_to_default_mem;
}

float MemorySpaceAssignmentCostAnalysis::GetInstructionElapsed(
    const HloInstruction& instruction) const {
  return std::max(GetInstructionElapsedDueToCompute(instruction),
                  GetInstructionElapsedDueToMemory(instruction));
}

float MemorySpaceAssignmentCostAnalysis::GetInstructionElapsedInAlternateMemory(
    const HloInstruction& instruction,
    absl::Span<const std::pair<int64_t, ShapeIndex>> operands_in_alternate_mem,
    absl::Span<const ShapeIndex> outputs_in_alternate_mem) const {
  return std::max(
      GetInstructionElapsedDueToCompute(instruction),
      GetInstructionElapsedDueToMemory(instruction, operands_in_alternate_mem,
                                       outputs_in_alternate_mem));
}

float MemorySpaceAssignmentCostAnalysis::GetInstructionElapsedInAlternateMemory(
    const HloInstruction& instruction,
    IsInAlternateMemoryFun is_in_alternate_mem) const {
  return std::max(
      GetInstructionElapsedDueToCompute(instruction),
      GetInstructionElapsedDueToMemory(instruction, is_in_alternate_mem));
}

float MemorySpaceAssignmentCostAnalysis::GetAsyncCopyElapsed(
    const Shape& shape) const {
  int64_t size_in_bytes = cost_analysis_.GetShapeSize(shape);
  return static_cast<float>(size_in_bytes) /
         options().async_copy_bandwidth_bytes_per_second;
}

int64_t MemorySpaceAssignmentCostAnalysis::GetScheduleEndTime() const {
  return hlo_live_range_->schedule_end_time();
}

bool InstructionCountPrefetchIntervalPicker::CanAllocateInAlternateMemoryNoCopy(
    const Shape& shape, int64_t start_time, int64_t end_time) const {
  return end_time - start_time <= max_overlap_count_;
}

int64_t InstructionCountPrefetchIntervalPicker::PreferredEvictionEndTime(
    const Shape& shape, int64_t start_time, int64_t latest_end_time) const {
  return std::min(start_time + min_overlap_count_, latest_end_time);
}

int64_t InstructionCountPrefetchIntervalPicker::LatestPrefetchStartTime(
    const Shape& shape, int64_t start_time, int64_t end_time,
    const HloUse* use) const {
  return end_time - min_overlap_count_;
}

int64_t InstructionCountPrefetchIntervalPicker::PreferredPrefetchStartTime(
    const Shape& shape, int64_t earliest_prefetch_start_time,
    int64_t latest_prefetch_start_time, int64_t prefetch_end_time) const {
  return std::max(earliest_prefetch_start_time,
                  prefetch_end_time - max_overlap_count_);
}

int64_t InstructionCountPrefetchIntervalPicker::EstimatedPrefetchEndTime(
    const Shape& shape, int64_t start_time, int64_t end_time) const {
  // For testing, assume the end time is the estimated prefetch end time.
  return end_time;
}

float InstructionCountPrefetchIntervalPicker::GetLogicalIntervalElapsed(
    int64_t start_time, int64_t end_time) const {
  // For testing, just assume every HLO takes 1 second.
  return static_cast<float>(end_time - start_time - 1);
}

void InstructionCountPrefetchIntervalPicker::Begin(const HloUse& use,
                                                   int64_t start_time,
                                                   int64_t end_time) {
  end_time_ = end_time;
  const Shape& shape = ShapeUtil::GetSubshape(
      use.instruction->operand(use.operand_number)->shape(), use.operand_index);
  current_prefetch_time_ =
      PreferredPrefetchStartTime(shape, start_time, end_time, end_time);
}

int64_t InstructionCountPrefetchIntervalPicker::Next() {
  CHECK(!Done()) << "Prefetch interval picker's Next() is called even though "
                    "Done() is false";
  return current_prefetch_time_++;
}

bool InstructionCountPrefetchIntervalPicker::Done() const {
  return end_time_ - current_prefetch_time_ <= min_overlap_count_;
}

std::string InstructionCountPrefetchIntervalPicker::ToDebugString() const {
  return absl::StrCat("Overlapped HLOs = ", end_time_ - current_prefetch_time_);
}

std::string InstructionCountPrefetchIntervalPicker::ToNoCopyDebugString(
    const Shape& shape, int64_t start_time, int64_t end_time) const {
  return absl::StrCat("Overlapped HLOs = ", end_time - start_time);
}

CostAnalysisPrefetchIntervalPicker::CostAnalysisPrefetchIntervalPicker(
    const MemorySpaceAssignmentCostAnalysis& cost_analysis,
    float min_overlap_to_async_copy_ratio,
    float preferred_overlap_to_async_copy_ratio,
    float max_overlap_to_mem_size_async_copy_ratio, int64_t mem_size_bytes)
    : while_nest_level_(
          cost_analysis.hlo_live_range().instruction_schedule().size() + 1, 0),
      computation_nest_level_(
          cost_analysis.hlo_live_range().instruction_schedule().size() + 1, 0),
      cost_analysis_(cost_analysis),
      min_overlap_to_async_copy_ratio_(min_overlap_to_async_copy_ratio),
      preferred_overlap_to_async_copy_ratio_(
          preferred_overlap_to_async_copy_ratio),
      max_async_copy_elapsed_(
          cost_analysis_.GetAsyncCopyElapsed(
              ShapeUtil::MakeShape(S32, {mem_size_bytes / 4})) *
          max_overlap_to_mem_size_async_copy_ratio) {
  instruction_schedule_ =
      &cost_analysis_.hlo_live_range().instruction_schedule();

  // Create a vector of elapsed times and while nesting levels of HLO
  // instructions. The elapsed times are multiplied by
  // pow(while_execution_count, nest_level) to account for executing the HLOs
  // multiple times in while loops.
  std::vector<float> instructions_elapsed_time(
      instruction_schedule_->size() + 1, 0.0);
  int max_while_nest_level = 0;
  for (const auto& instruction_and_logical_time : *instruction_schedule_) {
    // To avoid double counting, don't include the elapsed time of while and
    // conditional HLOs.
    const HloInstruction* instruction = instruction_and_logical_time.first;
    int64_t logical_time = instruction_and_logical_time.second;
    if (logical_time >= instructions_elapsed_time.size()) {
      instructions_elapsed_time.resize(logical_time + 1, 0.0);
      while_nest_level_.resize(logical_time + 1, 0);
    }
    int while_nest_level = cost_analysis_.CalculateComputationNestLevel(
        instruction_and_logical_time.first, /*while_only=*/true);
    while_nest_level_[logical_time] = while_nest_level;
    max_while_nest_level = std::max(max_while_nest_level, while_nest_level);
    int computation_nest_level = cost_analysis_.CalculateComputationNestLevel(
        instruction_and_logical_time.first, /*while_only=*/false);
    computation_nest_level_[logical_time] = computation_nest_level;
    if (instruction->opcode() == HloOpcode::kWhile ||
        instruction->opcode() == HloOpcode::kConditional) {
      continue;
    }
    float elapsed_time = cost_analysis_.GetInstructionElapsed(
        *instruction_and_logical_time.first);
    instructions_elapsed_time[logical_time] =
        elapsed_time *
        tensorflow::MathUtil::IPow<float>(
            cost_analysis_.options()
                .xla_tpu_memory_space_assignment_while_execution_count,
            while_nest_level);
  }
  // As an optimization, create a cumulative sum vector of elapsed time.
  float cumsum = 0.0;
  elapsed_time_cumsum_.reserve(instructions_elapsed_time.size());
  for (float elapsed_time : instructions_elapsed_time) {
    cumsum += elapsed_time;
    elapsed_time_cumsum_.push_back(cumsum);
  }
  // To be able to accurately determine the minimum nest level between a start
  // time and an end time efficiently, populate a data structure that stores the
  // closest 'smaller' nest level change index.
  const int64_t size = instructions_elapsed_time.size();
  CHECK_EQ(size, while_nest_level_.size());
  std::vector<int> most_recent_by_level(while_nest_level_.size(), -1);
  int prev_nest_level = 0;
  int change_idx = -1;
  while_nest_level_change_.reserve(size);
  for (int i = 0; i < size; ++i) {
    int nest_level = while_nest_level_[i];
    if (nest_level != prev_nest_level) {
      prev_nest_level = nest_level;
      // Compute last change index by choosing the most recent instruction index
      // with smaller nesting level. Note that it may happen that even though
      // there were few different regions with other nest levels before, all of
      // then are same or bigger than this one, in which case we'll end up with
      // -1, e.g. if you got nest level 0 no need checking anything else.
      change_idx = -1;
      for (int smaller_level = 0; smaller_level < nest_level; smaller_level++) {
        change_idx = std::max(change_idx, most_recent_by_level[smaller_level]);
      }
    }
    most_recent_by_level[nest_level] = i;
    while_nest_level_change_.push_back(change_idx);
  }
  for (int i = 0; i <= max_while_nest_level; ++i) {
    while_execution_counts_.push_back(tensorflow::MathUtil::IPow<float>(
        cost_analysis_.options()
            .xla_tpu_memory_space_assignment_while_execution_count,
        i));
  }
}

float CostAnalysisPrefetchIntervalPicker::GetMaxElapsedInAlternateMemory(
    float async_copy_elapsed) const {
  return max_async_copy_elapsed_;
}

bool CostAnalysisPrefetchIntervalPicker::CanAllocateInAlternateMemoryNoCopy(
    const Shape& shape, int64_t start_time, int64_t end_time) const {
  // Even though this method returns if we allow the buffer in alternate memory
  // _without_ asynchronous copies, calculate how long it would have taken to
  // copy it and compare it to the elapsed time in the logical interval.
  float async_copy_elapsed = cost_analysis_.GetAsyncCopyElapsed(shape);
  float logical_interval_elapsed =
      GetLogicalIntervalElapsed(start_time, end_time);
  return GetMaxElapsedInAlternateMemory(async_copy_elapsed) >
         logical_interval_elapsed;
}

int64_t CostAnalysisPrefetchIntervalPicker::PreferredEvictionEndTime(
    const Shape& shape, int64_t start_time, int64_t latest_end_time) const {
  float async_copy_elapsed = cost_analysis_.GetAsyncCopyElapsed(shape);
  int64_t end_time;
  for (end_time = start_time + 1; end_time <= latest_end_time; ++end_time) {
    float logical_interval_elapsed =
        GetLogicalIntervalElapsed(start_time, end_time);
    if (logical_interval_elapsed >=
        (1 + kEvictionRetryMultiplier * retry_number_) *
            preferred_overlap_to_async_copy_ratio_ * async_copy_elapsed) {
      break;
    }
  }
  return end_time;
}

int64_t CostAnalysisPrefetchIntervalPicker::LatestPrefetchStartTime(
    const Shape& shape, int64_t start_time, int64_t end_time,
    const HloUse* use) const {
  // Find the earliest time that satisfies max_overlap_to_async_copy_ratio_.
  float async_copy_elapsed = cost_analysis_.GetAsyncCopyElapsed(shape);
  // If there is a use, estimate the time we would save by having this op in
  // alternate memory.
  float inst_elapsed_reduction = 0.0f;
  if (use) {
    float elapsed_time =
        cost_analysis_.GetInstructionElapsed(*use->instruction);
    float elapsed_time_in_alternate_mem =
        cost_analysis_.GetInstructionElapsedInAlternateMemory(
            *use->instruction,
            /*operands_in_alternate_mem=*/
            {std::make_pair(use->operand_number, use->operand_index)},
            /*outputs_in_alternate_mem=*/{});
    inst_elapsed_reduction = elapsed_time - elapsed_time_in_alternate_mem;
  }
  int end_nest_level = computation_nest_level_[end_time];

  // Find the latest time we're allowed to start prefetching.
  float min_interval = min_overlap_to_async_copy_ratio_ * async_copy_elapsed;
  int latest_prefetch_time;
  for (latest_prefetch_time = end_time - 1;
       latest_prefetch_time >= start_time &&
       (computation_nest_level_[latest_prefetch_time] != end_nest_level ||
        min_interval >
            GetLogicalIntervalElapsed(latest_prefetch_time, end_time) +
                inst_elapsed_reduction);
       --latest_prefetch_time) {
  }

  return latest_prefetch_time;
}

int64_t CostAnalysisPrefetchIntervalPicker::PreferredPrefetchStartTime(
    const Shape& shape, int64_t earliest_prefetch_start_time,
    int64_t latest_prefetch_start_time, int64_t prefetch_end_time) const {
  // Between the earliest and latest prefetch interval, find the interval
  // closest to the preferred interval and start iterating from there.
  float async_copy_elapsed = cost_analysis_.GetAsyncCopyElapsed(shape);
  int64_t preferred_prefetch_start_time = earliest_prefetch_start_time;
  float preferred_interval =
      preferred_overlap_to_async_copy_ratio_ * async_copy_elapsed;
  float best_interval = GetLogicalIntervalElapsed(earliest_prefetch_start_time,
                                                  prefetch_end_time);
  int end_nest_level = computation_nest_level_[prefetch_end_time];
  for (int64_t prefetch_start_time = earliest_prefetch_start_time + 1;
       prefetch_start_time <= latest_prefetch_start_time;
       ++prefetch_start_time) {
    float interval =
        GetLogicalIntervalElapsed(prefetch_start_time, prefetch_end_time);
    if (computation_nest_level_[prefetch_start_time] == end_nest_level &&
        std::abs(preferred_interval - interval) <
            std::abs(preferred_interval - best_interval)) {
      best_interval = interval;
      preferred_prefetch_start_time = prefetch_start_time;
    }
  }
  return preferred_prefetch_start_time;
}

int64_t CostAnalysisPrefetchIntervalPicker::LatestPrefetchEndTime(
    int64_t original_prefetch_end_time,
    int64_t proposed_prefetch_end_time) const {
  // Iterate towards the beginning until we find a suitable end time that is the
  // same while nest level as the original prefetch end time.
  int64_t original_nest_level =
      computation_nest_level_[original_prefetch_end_time];
  int64_t new_prefetch_end_time;
  for (new_prefetch_end_time = proposed_prefetch_end_time;
       computation_nest_level_[new_prefetch_end_time] != original_nest_level;
       --new_prefetch_end_time) {
  }
  return new_prefetch_end_time;
}

int64_t CostAnalysisPrefetchIntervalPicker::EstimatedPrefetchEndTime(
    const Shape& shape, int64_t start_time, int64_t end_time) const {
  float async_copy_elapsed = cost_analysis_.GetAsyncCopyElapsed(shape);
  int64_t estimated_end_time;
  for (estimated_end_time = start_time + 1; estimated_end_time < end_time;
       ++estimated_end_time) {
    float interval = GetLogicalIntervalElapsed(start_time, estimated_end_time);
    if (interval >= async_copy_elapsed) {
      break;
    }
  }
  return estimated_end_time;
}

void CostAnalysisPrefetchIntervalPicker::Begin(const HloUse& use,
                                               int64_t start_time,
                                               int64_t end_time) {
  const Shape& shape = ShapeUtil::GetSubshape(
      use.instruction->operand(use.operand_number)->shape(), use.operand_index);
  // Find the earliest time that satisfies max_overlap_to_async_copy_ratio_.
  async_copy_elapsed_ = cost_analysis_.GetAsyncCopyElapsed(shape);
  // Estimate the time we would save by having this op in alternate memory.
  float elapsed_time = cost_analysis_.GetInstructionElapsed(*use.instruction);
  float elapsed_time_in_alternate_mem =
      cost_analysis_.GetInstructionElapsedInAlternateMemory(
          *use.instruction, /*operands_in_alternate_mem=*/
          {std::make_pair(use.operand_number, use.operand_index)},
          /*outputs_in_alternate_mem=*/{});
  inst_elapsed_reduction_ = elapsed_time - elapsed_time_in_alternate_mem;
  end_logical_time_ = end_time;
  int end_nest_level = computation_nest_level_[end_logical_time_];

  // Find the latest time we're allowed to start prefetching.
  float min_interval = min_overlap_to_async_copy_ratio_ * async_copy_elapsed_;
  latest_prefetch_time_ =
      LatestPrefetchStartTime(shape, start_time, end_time, &use);

  // Find the earliest time we're allowed to start prefetching.
  float max_interval = GetMaxElapsedInAlternateMemory(async_copy_elapsed_);
  for (earliest_prefetch_time_ = start_time;
       earliest_prefetch_time_ < latest_prefetch_time_ &&
       (computation_nest_level_[earliest_prefetch_time_] != end_nest_level ||
        max_interval < GetLogicalIntervalElapsed(earliest_prefetch_time_,
                                                 end_logical_time_));
       ++earliest_prefetch_time_) {
  }
  if (earliest_prefetch_time_ > latest_prefetch_time_) {
    // There is no available prefetch interval for the given start and end
    // times. Set the iterators accordingly to ensure Done() returns true.
    increasing_prefetch_time_iterator_ = earliest_prefetch_time_;
    decreasing_prefetch_time_iterator_ = latest_prefetch_time_;
    CHECK(Done());
    return;
  }

  int64_t starting_prefetch_time = PreferredPrefetchStartTime(
      shape, earliest_prefetch_time_, latest_prefetch_time_, end_logical_time_);
  float preferred_interval =
      preferred_overlap_to_async_copy_ratio_ * async_copy_elapsed_;
  VLOG(4) << "Interval min/max/preferred = " << min_interval << " "
          << max_interval << " " << preferred_interval
          << " prefetch time earliest/latest/starting = "
          << earliest_prefetch_time_ << " " << latest_prefetch_time_ << " "
          << starting_prefetch_time;

  increasing_prefetch_time_iterator_ = starting_prefetch_time;
  decreasing_prefetch_time_iterator_ = starting_prefetch_time;
  using_increasing_prefetch_time_iterator_ = true;
  // Since both iterators start at the same position, call Next() once to
  // advance one of the iterators.
  Next();
}

int64_t CostAnalysisPrefetchIntervalPicker::Next() {
  CHECK(!Done()) << "Prefetch interval picker's Next() is called even though "
                    "Done() is false";
  if (using_increasing_prefetch_time_iterator_) {
    int64_t prefetch_time = increasing_prefetch_time_iterator_++;
    while (increasing_prefetch_time_iterator_ <= latest_prefetch_time_ &&
           computation_nest_level_[increasing_prefetch_time_iterator_] !=
               computation_nest_level_[end_logical_time_]) {
      ++increasing_prefetch_time_iterator_;
    }
    if (decreasing_prefetch_time_iterator_ >= earliest_prefetch_time_) {
      using_increasing_prefetch_time_iterator_ = false;
    }
    return prefetch_time;
  } else {
    int64_t prefetch_time = decreasing_prefetch_time_iterator_--;
    while (decreasing_prefetch_time_iterator_ >= earliest_prefetch_time_ &&
           computation_nest_level_[decreasing_prefetch_time_iterator_] !=
               computation_nest_level_[end_logical_time_]) {
      --decreasing_prefetch_time_iterator_;
    }
    if (increasing_prefetch_time_iterator_ <= latest_prefetch_time_) {
      using_increasing_prefetch_time_iterator_ = true;
    }
    return prefetch_time;
  }
}

bool CostAnalysisPrefetchIntervalPicker::Done() const {
  return increasing_prefetch_time_iterator_ > latest_prefetch_time_ &&
         decreasing_prefetch_time_iterator_ < earliest_prefetch_time_;
}

void CostAnalysisPrefetchIntervalPicker::SetRetryNumber(int retry_number) {
  retry_number_ = retry_number;
}

int CostAnalysisPrefetchIntervalPicker::GetMinWhileNestLevel(
    int64_t start_time, int64_t end_time) const {
  int min_nest_level =
      std::min(while_nest_level_[start_time], while_nest_level_[end_time]);
  int change_idx = while_nest_level_change_[end_time];
  while (change_idx >= start_time) {
    min_nest_level = std::min(min_nest_level, while_nest_level_[change_idx]);
    change_idx = while_nest_level_change_[change_idx];
  }
  return min_nest_level;
}

float CostAnalysisPrefetchIntervalPicker::GetLogicalIntervalElapsed(
    int64_t start_time, int64_t end_time) const {
  CHECK_LE(start_time, end_time);
  if (start_time == end_time) {
    return 0.0;
  }
  if (start_time < 0) {
    start_time = 0;
  }
  // Since elapsed_time_cumsum_ is already weighed by the while loop nesting
  // level, normalize the elapsed time by dividing with the nesting factor of
  // the interval (start and end times).
  int interval_while_nest_level = GetMinWhileNestLevel(start_time, end_time);
  return (elapsed_time_cumsum_[end_time - 1] -
          elapsed_time_cumsum_[start_time]) /
         while_execution_counts_[interval_while_nest_level];
}

std::string CostAnalysisPrefetchIntervalPicker::ToDebugString() const {
  int current_logical_prefetch_time = using_increasing_prefetch_time_iterator_
                                          ? increasing_prefetch_time_iterator_
                                          : decreasing_prefetch_time_iterator_;
  float logical_interval_elapsed = GetLogicalIntervalElapsed(
      current_logical_prefetch_time, end_logical_time_);
  return absl::StrCat(
      "Async copy elapsed (s) = ", async_copy_elapsed_,
      ", inst elapsed reduction (s) = ", inst_elapsed_reduction_,
      ", logical interval elapsed (s) = ", logical_interval_elapsed,
      ", interval = (", current_logical_prefetch_time, ", ", end_logical_time_,
      ")");
}

std::string CostAnalysisPrefetchIntervalPicker::ToNoCopyDebugString(
    const Shape& shape, int64_t start_time, int64_t end_time) const {
  float async_copy_elapsed = cost_analysis_.GetAsyncCopyElapsed(shape);
  float logical_interval_elapsed =
      GetLogicalIntervalElapsed(start_time, end_time);
  return absl::StrCat(
      "Async copy elapsed (s) = ", async_copy_elapsed,
      ", logical interval elapsed (s) = ", logical_interval_elapsed);
}

absl::optional<float>
CostAnalysisPrefetchIntervalPicker::BufferIntervalAlternateMemoryBenefit(
    const GlobalDecreasingSizeBestFitHeap<HloValue>::BufferInterval& interval)
    const {
  return cost_analysis_.GetMemoryBoundedness(interval);
}

bool MemorySpaceAssignment::Allocation::operator==(
    const MemorySpaceAssignment::Allocation& other) const {
  return defining_position() == other.defining_position() &&
         uses() == other.uses() && memory_space() == other.memory_space() &&
         chunk() == other.chunk() && start_time() == other.start_time() &&
         end_time() == other.end_time() &&
         earliest_available_time() == other.earliest_available_time() &&
         is_copy_allocation() == other.is_copy_allocation() &&
         is_scoped_allocation() == other.is_scoped_allocation();
}

bool MemorySpaceAssignment::CopyAllocation::operator==(
    const MemorySpaceAssignment::CopyAllocation& other) const {
  return static_cast<const Allocation&>(*this) ==
             static_cast<const Allocation&>(other) &&
         copy_done_schedule_before() == other.copy_done_schedule_before() &&
         copy_start_schedule_after() == other.copy_start_schedule_after() &&
         copy_start() == other.copy_start() && copy_done() == other.copy_done();
}

std::string MemorySpaceAssignment::AllocationValue::ToString() const {
  std::string out = absl::StrCat("computation = ", computation()->name());
  absl::StrAppend(&out,
                  (requires_contiguous_allocation_ ? " (cont alloc)" : ""));
  absl::StrAppend(&out, "\n position:\n");
  absl::StrAppend(&out, "  ", defining_position_.ToString(), "\n");
  absl::StrAppend(&out, " uses:\n");
  for (const Use& use : uses_) {
    absl::StrAppend(&out, "  ", use.hlo_use.ToString(), "\n");
  }
  return out;
}

std::string MemorySpaceAssignment::AllocationValue::ToShortString() const {
  return absl::StrCat("computation = ", computation()->name(),
                      ", position = ", defining_position_.ToString(),
                      ", value = ", value_->ToShortString(),
                      (requires_contiguous_allocation_ ? " (cont alloc)" : ""));
}

AlternateMemoryBestFitHeap::AlternateMemoryBestFitHeap(
    MemorySpaceAssignment::AllocationSequence* allocations,
    const Options& options, const HloAliasAnalysis& alias_analysis,
    const HloLiveRange& hlo_live_range)
    : GlobalDecreasingSizeBestFitHeap(options.alignment_in_bytes),
      allocations_(allocations),
      options_(options),
      alias_analysis_(alias_analysis),
      hlo_live_range_(hlo_live_range) {
  // Override buffer interval compare if provided.
  if (options.buffer_interval_compare) {
    buffer_interval_compare_ = *options.buffer_interval_compare;
  }

  std::vector<float> initial_resources(hlo_live_range.schedule_end_time(), 1.0);
  if (options.cost_analysis) {
    const std::vector<HloInstruction*>& flattened_instructions =
        hlo_live_range.flattened_instruction_sequence().instructions();
    for (int i = 0; i < flattened_instructions.size(); ++i) {
      const HloInstruction* inst = flattened_instructions[i];
      if (inst->opcode() == HloOpcode::kWhile ||
          inst->opcode() == HloOpcode::kConditional) {
        initial_resources[i] = 0;
      } else {
        initial_resources[i] =
            options.cost_analysis->GetInstructionElapsed(*inst);
      }
      VLOG(2) << "Initial resource[" << i << "] = " << initial_resources[i]
              << " (" << inst->name() << ")";
    }
  }
  prefetch_async_copy_resource_ = AsynchronousCopyResource(initial_resources);
  eviction_async_copy_resource_ = AsynchronousCopyResource(initial_resources);
}

void AlternateMemoryBestFitHeap::CreateAllocationValues(
    const AlternateMemoryBestFitHeap::BufferInterval& buffer_interval,
    std::vector<AllocationValue>& allocation_values) const {
  const HloValue* value = buffer_interval.buffer;
  VLOG(3) << "Creating AllocationValues for: " << value->ToString();

  // Find and sort all non-trivial (excluding GTE, Tuple, and bitcast)
  // positions. We create an AllocationValue object for each non-trivial
  // position. And for each AllocationValue object, we create an
  // AllocationSequence consisting of one or more Allocation objects.The reason
  // why we exclude the trivial positions from AllocationValue is because
  // Allocation objects have special support for tuples and bitcasts.
  const absl::flat_hash_map<const HloInstruction*, int64_t>&
      instruction_schedule = hlo_live_range_.instruction_schedule();
  std::vector<HloPosition> positions;
  for (const HloPosition& position : value->positions()) {
    const HloInstruction* instruction = position.instruction;
    if (instruction->opcode() != HloOpcode::kGetTupleElement &&
        instruction->opcode() != HloOpcode::kTuple &&
        instruction->opcode() != HloOpcode::kBitcast) {
      positions.push_back(position);
    }
  }
  absl::c_stable_sort(positions,
                      [&](const HloPosition& pos1, const HloPosition& pos2) {
                        return instruction_schedule.at(pos1.instruction) <
                               instruction_schedule.at(pos2.instruction);
                      });

  // Create an AllocationValue for each non-trivial position.
  absl::flat_hash_set<const HloComputation*> computations;
  int beginning_idx = allocation_values.size();
  for (int i = 0; i < positions.size(); ++i) {
    const HloPosition& position = positions.at(i);
    allocation_values.emplace_back(value, position, buffer_interval.size);
  }

  std::vector<HloUse> uses(value->uses());
  absl::c_stable_sort(uses, [&](const HloUse& use1, const HloUse& use2) {
    return instruction_schedule.at(use1.instruction) <
           instruction_schedule.at(use2.instruction);
  });

  // Associate each use with an AllocationValue. Each AllocationValue contains a
  // position and uses in the same computation. Furthermore, if the original
  // HloValue had multiple non-trivial positions in the same computation, those
  // will get their own AllocationValue as well. We split these HloValues so
  // that when we insert CopyStart/CopyDone in CopyAllocation::Process, they
  // point to the latest position. We then replace the operand of the use with
  // CopyStart/CopyDone with an operand of the latest position.
  for (const HloUse& use : uses) {
    int64_t use_time = instruction_schedule.at(use.instruction);
    HloComputation* use_computation = use.instruction->parent();

    AllocationValue* last_allocation_value = nullptr;
    for (int i = beginning_idx; i < allocation_values.size(); ++i) {
      AllocationValue* allocation_value = &allocation_values.at(i);
      if (HloDataflowAnalysis::IsAsynchronousOperationDone(
              use.instruction->opcode())) {
        if (allocation_value->defining_instruction() ==
            use.instruction->operand(0)) {
          last_allocation_value = allocation_value;
        }
      } else if (!HloDataflowAnalysis::IsAsynchronousOperationStart(
                     allocation_value->defining_instruction()->opcode()) &&
                 allocation_value->computation() == use_computation &&
                 instruction_schedule.at(
                     allocation_value->defining_position().instruction) <
                     use_time) {
        last_allocation_value = allocation_value;
      }
    }
    CHECK(last_allocation_value != nullptr);
    last_allocation_value->AddUse(use, use_time);
  }

  for (int i = beginning_idx; i < allocation_values.size(); ++i) {
    AllocationValue& allocation_value = allocation_values.at(i);
    if (HloDataflowAnalysis::IsAsynchronousOperationStart(
            allocation_value.defining_instruction()->opcode())) {
      CHECK_EQ(allocation_value.uses().size(), 1);
      CHECK(HloDataflowAnalysis::IsAsynchronousOperationDone(
          allocation_value.uses().at(0).hlo_use.instruction->opcode()));
      VLOG(3) << "Mark " << allocation_value.ToShortString()
              << " to require contiguous allocation.";
      allocation_value.set_requires_contiguous_allocation(true);
    }
    VLOG(3) << "Created allocation value: "
            << allocation_values.at(i).ToString();
  }
}

void AlternateMemoryBestFitHeap::FindAliases(
    std::vector<AllocationValue>* allocation_values) const {
  absl::flat_hash_map<const HloInstruction*,
                      std::vector<const AllocationValue*>>
      values_by_defining_inst;
  for (AllocationValue& value : *allocation_values) {
    values_by_defining_inst[value.defining_instruction()].push_back(&value);
  }
  auto maybe_add_alias_with_instruction = [&](const HloInstruction* instruction,
                                              AllocationValue::Use* use) {
    auto aliased_values_it = values_by_defining_inst.find(instruction);
    if (aliased_values_it != values_by_defining_inst.end()) {
      for (const AllocationValue* aliased_value : aliased_values_it->second) {
        VLOG(3) << "Adding aliasing for use " << use->hlo_use.ToString()
                << " to " << aliased_value->ToShortString();
        use->aliases.push_back(aliased_value->defining_position());
      }
    }
  };

  for (AllocationValue& value : *allocation_values) {
    for (AllocationValue::Use& use : value.uses()) {
      // Find any aliases with the instruction itself (operand and output must
      // alias).
      maybe_add_alias_with_instruction(use.hlo_use.instruction, &use);

      // Find any aliases with the parameters of called computations.
      for (const HloComputation* called_computation :
           use.hlo_use.instruction->called_computations()) {
        for (const HloInstruction* parameter_instruction :
             called_computation->parameter_instructions()) {
          maybe_add_alias_with_instruction(parameter_instruction, &use);
        }
      }

      // Special case for kWhile: the root of the body computation must alias as
      // well.
      if (use.hlo_use.instruction->opcode() == HloOpcode::kWhile) {
        HloPosition root_alias{
            use.hlo_use.instruction->while_body()->root_instruction(),
            use.hlo_use.operand_index};
        VLOG(3) << "Adding while body root aliasing for use "
                << use.hlo_use.ToString() << " to " << root_alias;
        use.aliases.push_back(root_alias);
      }
    }
  }
}

std::vector<const AlternateMemoryBestFitHeap::BufferInterval*>
AlternateMemoryBestFitHeap::GetSortedColocatedIntervals(
    const AlternateMemoryBestFitHeap::BufferInterval& interval) const {
  std::vector<const BufferInterval*> colocated_intervals;
  std::vector<const BufferInterval*> worklist = {&interval};
  while (!worklist.empty()) {
    const BufferInterval* item = worklist.back();
    worklist.pop_back();
    colocated_intervals.push_back(item);
    for (const HloValue* buffer_colocated : item->colocations) {
      worklist.push_back(&buffer_intervals_.at(buffer_colocated));
    }
  }

  absl::c_stable_sort(colocated_intervals, [&](const BufferInterval* x,
                                               const BufferInterval* y) {
    return std::make_pair(x->start, x->end) < std::make_pair(y->start, y->end);
  });
  return colocated_intervals;
}

bool AlternateMemoryBestFitHeap::IsUseAllowedInAlternateMemory(
    const AllocationValue& value, const HloUse& use) const {
  const auto& instruction_schedule = hlo_live_range_.instruction_schedule();
  if (!options_.is_use_allowed_in_alternate_mem_fn(use)) {
    return false;
  }
  if (use.instruction->opcode() == HloOpcode::kWhile) {
    HloComputation* while_body = use.instruction->while_body();

    // We don't want to allocate this buffer in alternate memory if it will be
    // evicted anyway. Find out if it has an early use or a late definition that
    // would make sense to keep it in the alternate memory.
    HloValue* parameter_value =
        &alias_analysis_.dataflow_analysis().GetUniqueValueAt(
            while_body->parameter_instruction(0), use.operand_index);
    int64_t parameter_time =
        instruction_schedule.at(while_body->parameter_instruction(0));
    int64_t root_time = instruction_schedule.at(while_body->root_instruction());
    int64_t min_use_time = root_time;
    for (const HloUse& parameter_use : parameter_value->uses()) {
      int64_t use_time = instruction_schedule.at(parameter_use.instruction);
      if (parameter_use.instruction->opcode() != HloOpcode::kGetTupleElement &&
          parameter_use.instruction->opcode() != HloOpcode::kTuple &&
          parameter_use.instruction->opcode() != HloOpcode::kBitcast &&
          use_time > parameter_time) {
        min_use_time = std::min(min_use_time, use_time);
      }
    }
    // If there is no use of this buffer inside the while loop, there is no need
    // to allocate it in the loop.
    if (min_use_time == root_time) {
      VLOG(4) << "While allocation not allowed in alternate memory. "
              << "use time = " << min_use_time << ", root time = " << root_time;
      return false;
    }
    const Shape& shape = parameter_value->shape();
    // Allow the buffer in alternate memory if the buffer has a short live range
    // either at the beginning or end of the while loop body.
    if (!options_.prefetch_interval_picker->CanAllocateInAlternateMemoryNoCopy(
            shape, parameter_time, min_use_time)) {
      VLOG(4) << "While allocation not allowed in alternate memory. "
              << "use time = " << min_use_time << ", root time = " << root_time;
      return false;
    }
    // Check if there is a required assignment for the while loop output.
    HloValue* while_value =
        &alias_analysis_.dataflow_analysis().GetUniqueValueAt(
            use.instruction, use.operand_index);
    int64_t while_time = instruction_schedule.at(use.instruction);
    auto existing_required_assignment =
        RequiredMemoryAssignmentAt(while_value, while_time);
    if (existing_required_assignment &&
        existing_required_assignment->memory_space == MemorySpace::kDefault) {
      VLOG(4) << "While allocation not allowed in alternate memory because "
                 "there is a required default memory assignment.";
      return false;
    }
  } else if (use.instruction->opcode() == HloOpcode::kConditional) {
    // For any use of this conditional (the same value might be passed into
    // multiple called computations), determine if the parameter->first use
    // dependency is short.
    int64_t conditional_time = instruction_schedule.at(use.instruction);
    for (const AllocationValue::Use& other_use : value.uses()) {
      if (other_use.hlo_use.instruction != use.instruction) {
        continue;
      }
      HloComputation* called_computation =
          use.instruction->called_computations().at(
              other_use.hlo_use.operand_number - 1);
      const HloInstruction* parameter_instruction =
          called_computation->parameter_instruction(0);
      HloValue* parameter_value =
          &alias_analysis_.dataflow_analysis().GetUniqueValueAt(
              parameter_instruction, other_use.hlo_use.operand_index);
      int64_t parameter_time = instruction_schedule.at(parameter_instruction);
      int64_t min_use_time = conditional_time;
      for (const HloUse& parameter_use : parameter_value->uses()) {
        if (parameter_use.instruction->parent() == called_computation &&
            parameter_use.instruction->opcode() !=
                HloOpcode::kGetTupleElement &&
            parameter_use.instruction->opcode() != HloOpcode::kTuple &&
            parameter_use.instruction->opcode() != HloOpcode::kBitcast) {
          min_use_time = std::min(
              min_use_time, instruction_schedule.at(parameter_use.instruction));
        }
      }
      if (options_.prefetch_interval_picker->CanAllocateInAlternateMemoryNoCopy(
              parameter_value->shape(), parameter_time, min_use_time)) {
        VLOG(4) << "Conditional allocation allowed in alternate memory for "
                   "computation = "
                << called_computation->name()
                << ", parameter time = " << parameter_time
                << ", min use time = " << min_use_time;
        return true;
      } else {
        VLOG(4) << "Conditional allocation not allowed in alternate memory for "
                   "computation = "
                << called_computation->name()
                << ", parameter time = " << parameter_time
                << ", min use time = " << min_use_time;
      }
    }
    return false;
  }

  return true;
}

void AlternateMemoryBestFitHeap::AppendBufferInfoDebugString(
    const AlternateMemoryBestFitHeap::BufferInterval& interval,
    std::string* debug_str) const {
  // Columns in buffer information:
  // buffer_id: int. This value can be used to match the allocation in
  // allocation information.
  // buffer_name: string.
  // alt_mem_benefit: float. Roughly corresponds to how much the cost analysis
  // thought it would be beneficial to put this in the alternate memory. The
  // higher the value, the more it is memory bound.
  // size: int. In bytes.
  // definition_time: int. Logical time this value was defined in the schedule.
  // use_times: string. This is a semicolon-separated list of integers for all
  // the use times.
  // use_names: string. This is a semicolon-separated list of string
  // representation of uses.
  if (debug_str->empty()) {
    // Append the column names.
    absl::StrAppend(debug_str,
                    "buffer_id,buffer_name,alt_mem_benefit,size,"
                    "definition_time,use_times,use_names\n");
  }
  const HloBuffer& buffer =
      alias_analysis_.GetBufferContainingValue(*interval.buffer);
  const auto& instruction_schedule = hlo_live_range_.instruction_schedule();
  int64_t definition_time =
      instruction_schedule.at(interval.buffer->defining_position().instruction);
  std::vector<std::pair<int64_t, std::string>> uses;
  for (const HloValue* value : buffer.values()) {
    for (const HloUse& use : value->uses()) {
      uses.push_back(
          {instruction_schedule.at(use.instruction), use.ToString()});
    }
  }
  absl::c_sort(uses);
  std::vector<int64_t> use_times;
  std::vector<std::string> use_names;
  use_times.reserve(uses.size());
  use_names.reserve(uses.size());
  for (const auto& use : uses) {
    use_times.push_back(use.first);
    use_names.push_back(use.second);
  }

  absl::StrAppend(debug_str, buffer.id(), ",");
  absl::StrAppend(debug_str, "\"", interval.buffer->ToShortString(), "\",");
  auto alternate_memory_benefit =
      options_.prefetch_interval_picker->BufferIntervalAlternateMemoryBenefit(
          interval);
  absl::StrAppend(
      debug_str, alternate_memory_benefit ? *alternate_memory_benefit : 0, ",");
  absl::StrAppend(debug_str, interval.size, ",");
  absl::StrAppend(debug_str, definition_time, ",");
  absl::StrAppend(debug_str, "\"", absl::StrJoin(use_times, ";"), "\",");
  absl::StrAppend(debug_str, "\"", absl::StrJoin(use_names, ";"), "\"");
  absl::StrAppend(debug_str, "\n");
}

void AlternateMemoryBestFitHeap::AppendAllocationInfoDebugString(
    const AllocationValue& value,
    const MemorySpaceAssignment::Allocation& allocation,
    std::string& debug_str) const {
  // Columns in allocation information:
  // buffer_id: int. This value can be used the match with buffer info.
  // size: int. In bytes.
  // offset: int. In bytes.
  // start_time: int. Logical start time of the allocation.
  // end_time: int. Logical end time of the allocation.
  if (debug_str.empty()) {
    // Append the column names.
    absl::StrAppend(&debug_str, "buffer_id,size,offset,start_time,end_time\n");
  }
  if (allocation.memory_space() == MemorySpace::kAlternate) {
    const HloBuffer& buffer =
        alias_analysis_.GetBufferContainingValue(*value.value());
    absl::StrAppend(&debug_str, buffer.id(), ",");
    absl::StrAppend(&debug_str, value.size(), ",");
    absl::StrAppend(&debug_str, allocation.chunk().offset, ",");
    absl::StrAppend(&debug_str, allocation.start_time(), ",");
    absl::StrAppend(&debug_str, allocation.end_time(), "\n");
  }
}

void AlternateMemoryBestFitHeap::DumpDebugStringsIfEnabled() const {
  if (!options_.dump_fn) {
    return;
  }
  options_.dump_fn("bufferinfo", buffer_info_str_);
  options_.dump_fn("allocinfo", allocation_info_str_);
}

HeapSimulator::Result<HloValue> AlternateMemoryBestFitHeap::Finish() {
  if (options_.autotuning_config.has_value()) {
    CHECK_EQ((*options_.autotuning_config).size(), buffer_intervals_.size());
  }

  AllocateReservedScopedAllocations();
  std::vector<BufferInterval> sorted_buffer_intervals =
      GetSortedBufferIntervals();
  memory_space_assignment::CustomizeSortedBufferInterval(
      options_.autotuning_config, sorted_buffer_intervals);

  // Calculate the memory pressure for the buffers that can be assigned in the
  // alternate memory.
  memory_pressure_ = 0;
  for (auto& interval : sorted_buffer_intervals) {
    if (!interval.need_allocation ||
        !MemorySpaceAssignmentUtils::IsIntervalAllowedInAlternateMemory(
            interval) ||
        interval.size > available_heap_size()) {
      continue;
    }
    memory_pressure_ += interval.size;
  }
  VLOG(1) << "Memory pressure = " << memory_pressure_;

  if (options_.enable_cross_program_prefetch) {
    absl::optional<AlternateMemoryBestFitHeap::BufferInterval>
        prefetch_candidate = FindCrossProgramPrefetchCandidate(
            alias_analysis_, hlo_live_range_, options_);
    if (prefetch_candidate) {
      HloModule* module =
          prefetch_candidate->buffer->instruction()->GetModule();
      AllocateCrossProgramPrefetchBuffer(module, prefetch_candidate);
    }
  }


  VLOG(1) << "Assigning buffers to alternate memory. Max heap size = "
          << options_.max_size_in_bytes;

  AddInputAndOutputRequiredAssignments();

  if (VLOG_IS_ON(3)) {
    VLOG(3) << "Flattened instruction sequence:";
    const auto& instruction_sequence =
        hlo_live_range_.flattened_instruction_sequence().instructions();
    for (int i = 0; i < instruction_sequence.size(); ++i) {
      VLOG(3) << " " << i << ": " << instruction_sequence[i]->parent()->name()
              << " " << instruction_sequence[i]->name();
    }
  }

  for (const auto& interval : sorted_buffer_intervals) {
    auto colocated_intervals = GetSortedColocatedIntervals(interval);
    if (AreIntervalsReservedInAlternateMemory(colocated_intervals)) {
      // Increment the reserved part of alternate memory so that it is not
      // available for other buffers.
      reserved_in_bytes_ += options_.size_fn(*interval.buffer);
    }
  }
  VLOG(2) << "Total reserved bytes = " << reserved_in_bytes_;

  for (auto& interval : sorted_buffer_intervals) {
    if (!interval.need_allocation) {
      continue;
    }

    if (!MemorySpaceAssignmentUtils::IsIntervalAllowedInAlternateMemory(
            interval)) {
      continue;
    }

    HloInstruction* inst = interval.buffer->instruction();
    HloModule* module = inst->GetModule();

    // Don't intra-program prefetch a cross program prefetch
    if (inst->opcode() == HloOpcode::kParameter &&
        absl::c_count(module->CrossProgramPrefetches(),
                      std::make_pair(inst->parameter_number(),
                                     interval.buffer->index())) > 0) {
      VLOG(3) << "Skip " << interval.buffer->ToShortString()
              << " because it is cross-program prefetched.";
      continue;
    }

    if (interval.size > available_heap_size()) {
      VLOG(3) << "Skip " << interval.buffer->ToShortString()
              << " because the buffer is larger than the heap size.";
      continue;
    }

    auto colocated_intervals = GetSortedColocatedIntervals(interval);

    if (AreIntervalsReservedInAlternateMemory(colocated_intervals)) {
      VLOG(3) << "Interval " << interval.buffer->ToShortString()
              << " is reserved in the alternate memory.";
      for (const BufferInterval* colocated_interval : colocated_intervals) {
        const HloValue* value = colocated_interval->buffer;
        // Color all of the aliased reserved buffers here because reserved
        // alternate memory allocations will not have an entry in preset
        // allocations that is normally used for coloring.
        for (auto& position : value->positions()) {
          VLOG(4) << "Coloring " << position.ToString();
          Shape* shape = ShapeUtil::GetMutableSubshape(
              position.instruction->mutable_shape(), position.index);
          CHECK(shape->IsArray()) << "Coloring a shape that is not an array: "
                                  << position.ToString();
          shape->mutable_layout()->set_memory_space(
              options_.alternate_memory_space);
        }
      }
      continue;
    }

    if (colocated_intervals.size() > 1 &&
        !options_.allocate_across_sequential_calls) {
      VLOG(4) << "Not allocating " << interval.buffer->ToShortString()
              << " because it aliases with another interval and "
              << " allocate_across_sequential_calls is false.";
      continue;
    }

    if (!ConsumeFuel("memory_space_assignment", [&] {
          return absl::StrCat("Ran out of fuel at buffer: ",
                              colocated_intervals[0]->buffer->ToShortString());
        })) {
      continue;
    }

    if (options_.dump_fn != nullptr || VLOG_IS_ON(3)) {
      // Only fill buffer_info_str_ if needed.
      AppendBufferInfoDebugString(interval, &buffer_info_str_);
    }

    std::vector<AllocationValue> allocation_values;
    CreateAllocationValuesFromColocatedIntervals(colocated_intervals,
                                                 allocation_values);

    // Retry allocating this value with larger limits if allocation fails.
    bool repacked = false;
    for (int retry_number = 0; retry_number < options_.max_retries;
         retry_number++) {
      AddRequiredAssignmentsForColocatedIntervals(colocated_intervals);
      options_.prefetch_interval_picker->SetRetryNumber(retry_number);
      Result result =
          AllocateAllocationValues(absl::MakeSpan(allocation_values));
      VLOG(2) << "Allocation result = "
              << absl::StrFormat("%x", static_cast<int>(result));
      if (result_requires_uncommit(result)) {
        UncommitPendingChunks(absl::MakeSpan(allocation_values));
        VLOG(2) << "Couldn't allocate. Retry number " << retry_number;
      } else if ((result_is(result, Result::kFailOutOfMemory) ||
                  options_.repack_after_every_allocation) &&
                 num_repacks_ < options_.max_repacks && !repacked) {
        UncommitPendingChunks(absl::MakeSpan(allocation_values));
        ++num_repacks_;
        repacked = true;
        CHECK_NE(options_.repacker, nullptr);
        std::vector<MemorySpaceAssignmentRepacker::AllocationBlock*>
            repack_allocation_blocks;
        ExportAllocationsForRepacking(repack_allocation_blocks);
        VLOG(2) << "Repacking.";
        auto repack_status =
            options_.repacker->Repack(absl::MakeSpan(repack_allocation_blocks));
        CHECK_EQ(repack_status.status(), Status::OK());
        VLOG(2) << "Repack complete. Modified = " << *repack_status;
        if (*repack_status) {
          ImportRepackedAllocations();
          --retry_number;
        }
      } else {
        FinalizeAllocations(absl::MakeSpan(allocation_values));
        break;
      }
    }
  }

  VLOG(3) << "Debug buffer info: ";
  XLA_VLOG_LINES(3, buffer_info_str_);
  VLOG(3) << "Debug allocation info: ";
  XLA_VLOG_LINES(3, allocation_info_str_);
  DumpDebugStringsIfEnabled();

  HeapSimulator::Result<HloValue> result;
  result.heap_size = result_.heap_size;
  result.heap_results.emplace_back(std::move(result_));
  return result;
}

void AlternateMemoryBestFitHeap::AddRequiredAssignmentsForColocatedIntervals(
    absl::Span<const AlternateMemoryBestFitHeap::BufferInterval* const>
        colocated_intervals) {
  // TODO(berkin): For now, place the phi values due to conditionals in
  // default memory.
  for (const BufferInterval* colocated_interval : colocated_intervals) {
    const HloValue* value = colocated_interval->buffer;
    for (const auto& position : value->positions()) {
      if (position.instruction->opcode() == HloOpcode::kConditional) {
        VLOG(3) << "Adding required assignment for condition output: "
                << value->ToShortString();
        AddRequiredAssignment(position.instruction, position.index,
                              MemorySpace::kDefault);
        for (const HloComputation* called_computation :
             position.instruction->called_computations()) {
          AddRequiredAssignment(called_computation->root_instruction(),
                                position.index, MemorySpace::kDefault);
        }
      }
    }
  }
}

void AlternateMemoryBestFitHeap::CreateAllocationValuesFromColocatedIntervals(
    absl::Span<const AlternateMemoryBestFitHeap::BufferInterval* const>
        colocated_intervals,
    std::vector<MemorySpaceAssignment::AllocationValue>& allocation_values) {
  // Create AllocationValues for all the colocated intervals.
  for (const auto& colocated_interval : colocated_intervals) {
    CreateAllocationValues(*colocated_interval, allocation_values);
  }
  // Go through the AllocationValues and delete the ones that have the identical
  // defining instruction and use instructions. This is useful for async
  // operations that can read and write to the same buffer, e.g., in-place
  // asynchronous collective permute. The AllocationValues that corresponds to
  // collective-permute-start{0} (the input) and collective-permute-start{1}
  // (the output) refer to the same buffer by definition (since they are created
  // from colocated intervals). If we don't delete one of these buffers, then
  // when we try to allocate the AllocationValue, we would think they overlap.
  auto create_instruction_vector = [](const AllocationValue& allocation_value) {
    std::vector<const HloInstruction*> instruction_vector;
    instruction_vector.push_back(allocation_value.defining_instruction());
    for (const AllocationValue::Use& use : allocation_value.uses()) {
      instruction_vector.push_back(use.hlo_use.instruction);
    }
    return instruction_vector;
  };
  for (int i = 0; i < allocation_values.size() - 1; ++i) {
    for (int j = i + 1; j < allocation_values.size(); ++j) {
      const AllocationValue& allocation_value_1 = allocation_values[i];
      const AllocationValue& allocation_value_2 = allocation_values[j];
      if (create_instruction_vector(allocation_value_1) ==
          create_instruction_vector(allocation_value_2)) {
        VLOG(3) << "Allocation values " << allocation_value_1.ToShortString()
                << " and " << allocation_value_2.ToShortString()
                << " are equivalent, deleting the second one.";
        allocation_values.erase(allocation_values.begin() + j);
        --j;
      }
    }
  }

  FindAliases(&allocation_values);
}

AlternateMemoryBestFitHeap::Result
AlternateMemoryBestFitHeap::AllocateAllocationValues(
    absl::Span<MemorySpaceAssignment::AllocationValue> allocation_values) {
  const auto& instruction_schedule = hlo_live_range_.instruction_schedule();

  // Find the use times across all of the related AllocationValues and sort
  // them. We use these to find allocations that are available throughout the
  // entire live range of all the AllocationValues.
  std::vector<int64_t> all_use_times;
  for (const AllocationValue& allocation_value : allocation_values) {
    absl::c_transform(allocation_value.uses(),
                      std::back_inserter(all_use_times),
                      [](const AllocationValue::Use& use) { return use.time; });
  }
  absl::c_sort(all_use_times);

  // Data structure to contain the preferred offset for a given computation.
  // We ensure that the same offset will be allocated outside the while loop
  // as well as inside the while loop.
  absl::flat_hash_map<const HloComputation*, AliasedOffset*>
      preferred_offset_for_computation;

  Result result = Result::kSuccess;
  for (AllocationValue& allocation_value : allocation_values) {
    int64_t definition_time =
        instruction_schedule.at(allocation_value.defining_instruction());

    AliasedOffset* preferred_offset = nullptr;
    auto preferred_offset_it =
        preferred_offset_for_computation.find(allocation_value.computation());
    if (preferred_offset_it != preferred_offset_for_computation.end()) {
      preferred_offset = preferred_offset_it->second;
    }

    // Iterate over the uses.
    for (int use_idx = 0; use_idx < allocation_value.uses().size(); ++use_idx) {
      const AllocationValue::Use& use = allocation_value.uses().at(use_idx);
      const HloUse hlo_use = use.hlo_use;
      int64_t use_time = instruction_schedule.at(hlo_use.instruction);
      int64_t latest_prefetch_time = use_time;
      bool allow_no_copy_alternate_mem_allocation = true;
      absl::optional<int64_t> earliest_prefetch_time = absl::nullopt;

      // Control flow  calls include kWhile, kCall, and kConditional opcodes.
      bool is_sequential_call =
          (GetInstructionCallContext(hlo_use.instruction->opcode()) ==
           CallContext::kControlFlow);
      if (is_sequential_call) {
        for (const HloComputation* called_computation :
             hlo_use.instruction->called_computations()) {
          const HloLiveRange::TimeBound& computation_span =
              hlo_live_range_.computation_span_times().at(called_computation);
          latest_prefetch_time =
              std::min(computation_span.start - 1, latest_prefetch_time);
        }
        if (hlo_use.instruction->opcode() == HloOpcode::kWhile) {
          // Given an example while loop and flattened schedule (logical times
          // shown on the left):
          //
          // 0:  a = ...
          // 1:  ...
          //     cond {
          // 2:   p = param(0)
          // 3:   ...
          //     }
          //     body {
          // 4:   p = param(0)
          // 5:   ...
          // 6:   ROOT ...
          //     }
          // 7:  w = while(a), body=body, cond=cond
          //
          // When processing "a" (time 0) and its while use (time 7), we update
          // the interval to time 0-4. This is so that the remaining interval
          // (5-6) can be allocated separately and this buffer doesn't waste
          // alternate memory space within the while loop body.
          HloComputation* while_body = hlo_use.instruction->while_body();
          // We require while body ROOTs to be the last in the schedule.
          CHECK_EQ(instruction_schedule.at(while_body->root_instruction()) + 1,
                   instruction_schedule.at(hlo_use.instruction))
              << "While body ROOTs need to be the last in the schedule!  "
                 "Please run RootInstructionSinker.";
          // Replace the use time with the parameter time so that we can decide
          // on alternate memory allocations within the while loop body when we
          // look at uses within the while loop body.
          use_time =
              instruction_schedule.at(while_body->parameter_instruction(0));
        } else if (hlo_use.instruction->opcode() == HloOpcode::kConditional) {
          // Replace the use time with the earliest parameter of called
          // computations.
          for (const HloComputation* called_computation :
               hlo_use.instruction->called_computations()) {
            use_time = std::min(
                use_time, instruction_schedule.at(
                              called_computation->parameter_instruction(0)));
          }
        }
      }

      // Add a required assignment in default memory if the use not allowed in
      // alternate memory.
      if (!IsUseAllowedInAlternateMemory(allocation_value, hlo_use)) {
        AddRequiredAssignment(allocation_value.value(), hlo_use.instruction,
                              MemorySpace::kDefault, use_time);
      } else if (use_idx > 0) {
        // We allow buffers in alternate memory that are passed into
        // conditionals to give up their alternate memory allocation inside the
        // called computation. This means that if a conditional operator has an
        // alternate memory allocation, subsequent uses cannot use the same
        // alternate memory allocation in order not to clobber data. So we force
        // default memory allocation for these subsequent uses.
        const AllocationValue::Use& previous_use =
            allocation_value.uses().at(use_idx - 1);
        if (previous_use.hlo_use.instruction->opcode() ==
                HloOpcode::kConditional &&
            previous_use.hlo_use.instruction != hlo_use.instruction) {
          allow_no_copy_alternate_mem_allocation = false;
          earliest_prefetch_time =
              instruction_schedule.at(previous_use.hlo_use.instruction);
          VLOG(3) << "Previous use (" << previous_use.hlo_use.ToString()
                  << ") of use (" << hlo_use.ToString()
                  << ") is a conditional, so this use will need to evict. "
                  << "Earliest prefetch time = " << *earliest_prefetch_time;
        }
      }

      // Bitcasts don't define buffers and don't directly consume buffers. Skip
      // allocating buffers for bitcast uses (unless they are the root
      // instruction). The uses that feed from bitcasts will be handled
      // specially.
      if (hlo_use.instruction->opcode() != HloOpcode::kBitcast ||
          hlo_use.instruction ==
              hlo_use.instruction->parent()->root_instruction()) {
        AllocationRequest request;
        // Rarely, (e.g., when conditional true and false parameters are the
        // same), definition time can be the time of the conditional and use
        // time is the parameter use, which is less.
        request.start_time = std::min(definition_time, use_time);
        request.end_time = use_time;
        request.latest_prefetch_time = latest_prefetch_time;
        request.size = allocation_value.size();
        request.allow_no_copy_alternate_mem_allocation =
            allow_no_copy_alternate_mem_allocation;
        request.earliest_prefetch_time = earliest_prefetch_time;
        request.preferred_offset = preferred_offset;
        request.use = &use;
        request.allocation_value = &allocation_value;
        request.all_use_times = all_use_times;
        result_mark(AllocateSegment(request), result);
        if (result_requires_uncommit(result)) {
          // If the allocation finding failed (e.g., due to running out of
          // asynchronous copies), then fall back to allocating the buffer
          // entirely in the default memory.
          return result;
        }

        // If there are multiple uses, they can try using the memory allocation
        // already at the alternate memory.
        definition_time = instruction_schedule.at(hlo_use.instruction);
      }

      // Propagate the allocation to any aliases this use might have had.
      MemorySpaceAssignment::Allocation* aliased_allocation =
          GetLiveAllocationAt(*allocation_value.allocation_sequence(),
                              use_time);
      for (const HloPosition& aliased_position : use.aliases) {
        AddAliasedRequiredAssignment(aliased_position.instruction,
                                     aliased_position.index,
                                     aliased_allocation);
      }

      if (hlo_use.instruction->opcode() == HloOpcode::kWhile &&
          aliased_allocation->memory_space() == MemorySpace::kAlternate) {
        // For while uses that are allocated in the alternate memory space, if
        // they also have an allocation in the default memory space in their
        // allocation sequence, create a "parent" allocation that mirrors this
        // default memory space allocation. When we process the parent
        // allocation, we add an additional parameter to the while that is a
        // reference to the buffer in the default memory space. With parent
        // allocations, we don't need to unnecessarily evict buffers since they
        // already have a copy in the default memory space. We search backwards
        // (latest to earliest in execution time) for a suitable allocation in
        // order to find the most recent one.
        if (absl::c_find_if(allocation_value.value()->positions(),
                            [&hlo_use](const HloPosition& position) {
                              return position.instruction ==
                                         hlo_use.instruction &&
                                     position.index == hlo_use.operand_index;
                            }) != allocation_value.value()->positions().end()) {
          auto allocation_sequence = allocation_value.allocation_sequence();
          auto prev_allocation_in_default_mem_it = std::find_if(
              allocation_sequence->rbegin(), allocation_sequence->rend(),
              [&](const auto& allocation) {
                return allocation->memory_space() == MemorySpace::kDefault &&
                       allocation->defining_position() ==
                           allocation_value.defining_position();
              });
          if (prev_allocation_in_default_mem_it !=
              allocation_sequence->rend()) {
            VLOG(3) << "Found a prev allocation in default mem for while use: "
                    << (*prev_allocation_in_default_mem_it)->ToString();
            auto body_allocation_value_it = absl::c_find_if(
                allocation_values, [&](const AllocationValue& value) {
                  return value.computation() ==
                             hlo_use.instruction->while_body() &&
                         value.defining_instruction()->opcode() ==
                             HloOpcode::kParameter;
                });
            CHECK_NE(body_allocation_value_it, allocation_values.end());
            VLOG(3) << "Body allocation value: "
                    << body_allocation_value_it->ToShortString();
            int64_t body_parameter_time = instruction_schedule.at(
                body_allocation_value_it->defining_instruction());
            body_allocation_value_it->allocation_sequence()->push_back(
                absl::make_unique<MemorySpaceAssignment::ParentAllocation>(
                    **prev_allocation_in_default_mem_it, hlo_use.instruction,
                    body_allocation_value_it->defining_position(),
                    body_parameter_time));
            VLOG(3) << "Created: "
                    << body_allocation_value_it->allocation_sequence()
                           ->back()
                           ->ToString();

            auto after_while_allocation_value_it = absl::c_find_if(
                allocation_values, [&](const AllocationValue& value) {
                  return value.defining_instruction() == hlo_use.instruction;
                });
            CHECK_NE(after_while_allocation_value_it, allocation_values.end());
            VLOG(3) << "After while allocation value: "
                    << after_while_allocation_value_it->ToShortString();
            int64_t while_time = instruction_schedule.at(hlo_use.instruction);
            after_while_allocation_value_it->allocation_sequence()->push_back(
                absl::make_unique<MemorySpaceAssignment::MirroredAllocation>(
                    **prev_allocation_in_default_mem_it, while_time));
            VLOG(3) << "Created: "
                    << after_while_allocation_value_it->allocation_sequence()
                           ->back()
                           ->ToString();
          }
        }
        // Special case for while loops since the root offset must agree with
        // other offsets: remember the preferred offset for the while loop body.
        preferred_offset_for_computation[hlo_use.instruction->while_body()] =
            GetAliasedOffset(*aliased_allocation);
      }
    }
  }
  return result;
}

bool operator<(const AsynchronousCopy& a, const AsynchronousCopy& b) {
  return a.AsTuple() < b.AsTuple();
}

bool operator==(const AsynchronousCopy& a, const AsynchronousCopy& b) {
  return a.AsTuple() == b.AsTuple();
}

bool operator!=(const AsynchronousCopy& a, const AsynchronousCopy& b) {
  return a.AsTuple() != b.AsTuple();
}

bool AsynchronousCopyResource::ConsumeResource(
    int64_t start_time, int64_t end_time, float resource,
    bool update_current_resource,
    const std::list<AsynchronousCopy>::iterator* current_copy,
    float resource_to_free) {
  VLOG(3) << "Consume resource: " << start_time << ", " << end_time << ", "
          << resource << ", delay: " << delay_[start_time + 1]
          << ", free: " << resource_to_free;

  // Nothing to do if we're not adding or removing any resources.
  if (resource == 0.0 && resource_to_free == 0.0) {
    return true;
  }

  // For the async copy we're adding, check the delay_ array to see how much
  // this copy would have to be delayed because of an earlier copy that wasn't
  // finished when this copy starts.
  if (current_copy == nullptr) {
    resource += delay_[start_time + 1];
  }

  // Find the copy that is right after this one. If there are leftover resources
  // by the time the next copy starts, the next copy will be pushed further
  // later in time.
  auto next_copy = async_copies_.end();
  if (current_copy != nullptr) {
    next_copy = std::next(*current_copy);
  } else {
    auto async_copy_time_it = async_copy_time_map_.upper_bound(start_time);
    if (async_copy_time_it != async_copy_time_map_.end()) {
      next_copy = async_copy_time_it->second;
    }
  }

  // Check if this copy will push the next copy later in time (or if removing
  // the resource, check if the removal of this copy move the next copy earlier
  // in time).
  absl::optional<float> delay_for_next_copy = absl::nullopt;
  float resource_freed = 0.0;
  for (int64_t time = start_time + 1; time < end_time && resource != 0;
       ++time) {
    // Iterate over the logical times that this copy spans. Note that the start
    // and end time ranges are exclusive.
    float used_resource = std::min(resource, initial_resources_[time]);
    if (next_copy != async_copies_.end() && next_copy->start_time == time - 1) {
      // This is the time where the next copy begins. If the resource is
      // non-zero at this point, the copy didn't finish by the time the next
      // copy started, so the next copy would need to be pushed later in time.
      delay_for_next_copy = resource;
      resource_to_free -= resource_freed;
    }
    if (update_current_resource && !delay_for_next_copy.has_value()) {
      // Update the delay_ vector and resource_freed variable with the amount
      // that was freed when removing the copy.
      float old_resource =
          std::max(0.0f, initial_resources_[time] - delay_[time]);
      delay_[time] = std::max(0.0f, resource - resource_to_free);
      float new_resource =
          std::max(0.0f, initial_resources_[time] - delay_[time]);
      resource_freed += std::max(0.0f, new_resource - old_resource);
    }
    // Update the resource with the used amount in this logical time.
    resource -= used_resource;
  }

  // If resource isn't satisfied by the end, we didn't have enough resources.
  if (resource > 0) {
    VLOG(3) << "Doesn't have enough resource; leftover resource = " << resource;
    return false;
  }

  // If this copy overlapped with another one, we recursively call
  // ConsumeResource with the amount of resource that needs to be added or
  // removed.
  if (delay_for_next_copy.has_value()) {
    return ConsumeResource(next_copy->start_time, next_copy->end_time,
                           *delay_for_next_copy + next_copy->resource,
                           update_current_resource, &next_copy,
                           resource_to_free);
  }
  return true;
}

void AsynchronousCopyResource::AddCopy(const AsynchronousCopy& copy) {
  CHECK(ConsumeResource(copy.start_time, copy.end_time, copy.resource,
                        /*update_current_resource=*/true));
  // Find the iterator for the copy that would be right after this copy and put
  // this copy right before it in async_copies_.
  auto async_copy_time_it = async_copy_time_map_.upper_bound(copy.start_time);
  auto insertion_it = (async_copy_time_it == async_copy_time_map_.end())
                          ? async_copies_.end()
                          : async_copy_time_it->second;
  auto inserted_it = async_copies_.insert(insertion_it, copy);
  // If this copy is the first copy we have seen with the start time, add the
  // inserted iterator into async_copy_time_map_ for fast lookups. Note that
  // async_copy_time_map_ always points to the very first copy with the same
  // start index. If there are multiple asynchronous copies that have the same
  // start time, the memory space assignment algorithm schedules them in the
  // same order that AddCopy was called.
  if (async_copy_time_map_.find(copy.start_time) ==
      async_copy_time_map_.end()) {
    async_copy_time_map_[copy.start_time] = inserted_it;
  }
}

void AsynchronousCopyResource::RemoveCopy(const AsynchronousCopy& copy) {
  CHECK(ConsumeResource(copy.start_time, copy.end_time, /*resource=*/0,
                        /*update_current_resource=*/true,
                        /*current_copy=*/nullptr,
                        /*resource_to_free=*/copy.resource));
  // Using async_copy_time_map_, find this copy to be removed. Note that the
  // iterator in async_copy_time_map_ points to the first-seen copy with the
  // given start time, so the copy to be removed might be later than the first
  // one.
  auto async_copy_time_it = async_copy_time_map_.find(copy.start_time);
  CHECK(async_copy_time_it != async_copy_time_map_.end());
  auto it = async_copy_time_it->second;
  for (; it != async_copies_.end() && *it != copy; ++it) {
  }
  CHECK(it != async_copies_.end());
  // If the copy to be removed is the value pointed by async_copy_time_map_, we
  // make the next copy with the same start time to be pointed by
  // async_copy_time_map_. If there are no such copies, we remove the key for
  // this copy start time.
  if (it == async_copy_time_it->second) {
    if (std::next(it) != async_copies_.end() &&
        std::next(it)->start_time == copy.start_time) {
      async_copy_time_it->second = std::next(it);
    } else {
      async_copy_time_map_.erase(async_copy_time_it);
    }
  }
  async_copies_.erase(it);
}

bool AsynchronousCopyResource::HasEnoughResource(int64_t start_time,
                                                 int64_t end_time,
                                                 float resource) {
  return ConsumeResource(start_time, end_time, resource,
                         /*update_current_resource=*/false);
}

AlternateMemoryBestFitHeap::AliasedOffset*
AlternateMemoryBestFitHeap::GetAliasedOffset(
    const MemorySpaceAssignment::Allocation& allocation) {
  auto aliased_offset_it = aliased_offset_map_.find(&allocation);
  CHECK(aliased_offset_it != aliased_offset_map_.end());
  return aliased_offset_it->second;
}

void AlternateMemoryBestFitHeap::CreateOrAddToAliasedOffset(
    const MemorySpaceAssignment::Allocation& allocation,
    AlternateMemoryBestFitHeap::AliasedOffset* aliased_offset) {
  CHECK(allocation.memory_space() == MemorySpace::kAlternate);
  CHECK(!aliased_offset_map_.contains(&allocation));
  if (!aliased_offset) {
    aliased_offsets_.push_back({allocation.chunk().offset});
    aliased_offset = &aliased_offsets_.back();
  }
  CHECK_EQ(allocation.chunk().offset, aliased_offset->offset);
  CHECK(aliased_offset->allocations.insert(&allocation).second);
  aliased_offset_map_[&allocation] = aliased_offset;
}

/*static*/ MemorySpaceAssignment::Allocation*
AlternateMemoryBestFitHeap::GetLiveAllocationAt(
    const MemorySpaceAssignment::AllocationSequence& allocations,
    int64_t time) {
  for (auto allocation_it = allocations.rbegin();
       allocation_it != allocations.rend(); ++allocation_it) {
    if ((*allocation_it)->start_time() <= time &&
        (*allocation_it)->end_time() >= time) {
      return allocation_it->get();
    }
  }
  return nullptr;
}

void AlternateMemoryBestFitHeap::AllocateCrossProgramPrefetchBuffer(
    HloModule* module, absl::optional<BufferInterval> prefetch_candidate) {
  if (!prefetch_candidate) {
    return;
  }

  ChunkCandidate chunk_candidate = FindChunkCandidate(*prefetch_candidate);
  if (chunk_candidate.chunk.offset != 0 ||
      chunk_candidate.heap_size > available_heap_size()) {
    LOG(WARNING)
        << "Could not allocate preferred memory for cross program prefetch";
    return;
  }

  const HloValue* buffer = prefetch_candidate->buffer;
  int64_t parameter = buffer->instruction()->parameter_number();
  module->AddCrossProgramPrefetch(parameter, buffer->index());

  MemorySpaceAssignment::AllocationSequence allocations;
  allocations.push_back(absl::make_unique<MemorySpaceAssignment::Allocation>(
      buffer->defining_position(), MemorySpace::kDefault, kDummyChunk,
      prefetch_candidate->start, prefetch_candidate->end,
      /*is_scoped_allocation=*/false));

  // Find the earliest use.
  const auto& instruction_schedule = hlo_live_range_.instruction_schedule();
  auto uses = buffer->uses();
  auto use_schedule_compare = [&](const HloUse& lhs, const HloUse& rhs) {
    return instruction_schedule.at(lhs.instruction) <
           instruction_schedule.at(rhs.instruction);
  };
  auto first_use = absl::c_min_element(uses, use_schedule_compare);
  int64_t latest_prefetch_time =
      instruction_schedule.at(first_use->instruction);

  // Find the latest use time.
  int64_t last_use_time = instruction_schedule.at(
      absl::c_max_element(uses, use_schedule_compare)->instruction);
  for (const HloValue* colocation : prefetch_candidate->colocations) {
    last_use_time = std::max(
        last_use_time,
        instruction_schedule.at(
            absl::c_max_element(colocation->uses(), use_schedule_compare)
                ->instruction));
  }

  int64_t end_of_program_prefetch_end_time = instruction_schedule.size();
  int64_t end_of_program_prefetch_latest_start_time =
      options_.prefetch_interval_picker->LatestPrefetchStartTime(
          buffer->defining_position().shape(), last_use_time,
          end_of_program_prefetch_end_time, nullptr);
  int64_t end_of_program_prefetch_start_time =
      options_.prefetch_interval_picker->PreferredPrefetchStartTime(
          buffer->defining_position().shape(), last_use_time,
          end_of_program_prefetch_latest_start_time,
          end_of_program_prefetch_end_time);
  VLOG(2) << "last use time = " << last_use_time
          << ", end-of-program prefetch start time = "
          << end_of_program_prefetch_start_time;
  float total_execution_time =
      options_.prefetch_interval_picker->GetLogicalIntervalElapsed(
          0, instruction_schedule.size());
  float buffer_occupied_time =
      options_.prefetch_interval_picker->GetLogicalIntervalElapsed(
          0, last_use_time) +
      options_.prefetch_interval_picker->GetLogicalIntervalElapsed(
          end_of_program_prefetch_start_time, end_of_program_prefetch_end_time);
  float buffer_occupied_ratio = buffer_occupied_time / total_execution_time;
  VLOG(2) << "Total execution time = " << total_execution_time
          << ", buffer occupied time = " << buffer_occupied_time
          << ", buffer occupied ratio = " << buffer_occupied_ratio;
  // Freeing buffer only makes sense if the buffer will be free for a
  // substantial time. Only perform this optimization if the ratio is below the
  // limit, and if the memory pressure is above the alternate memory size.
  bool free_buffer =
      (options_.enable_cross_program_prefetch_freeing &&
       memory_pressure_ > options_.max_size_in_bytes &&
       buffer_occupied_ratio < kCrossProgramPrefetchOccupyFreeingLimit &&
       end_of_program_prefetch_start_time > last_use_time &&
       end_of_program_prefetch_start_time < end_of_program_prefetch_end_time);
  int64_t cross_program_prefetch_end_time =
      free_buffer ? last_use_time : prefetch_candidate->end;

  AddAsyncCopy(*allocations.back(), MemorySpace::kAlternate,
               chunk_candidate.chunk, prefetch_candidate->start,
               cross_program_prefetch_end_time, latest_prefetch_time,
               &allocations, /*aliased_offset=*/nullptr,
               /*resource=*/0.0,
               /*is_cross_program_prefetch=*/true);
  absl::c_for_each(uses, [&](auto& use) { allocations.back()->AddUse(use); });
  AliasedOffset* cross_program_prefetch_offset =
      GetAliasedOffset(*allocations.back());

  if (free_buffer) {
    VLOG(2) << "Adding an end-of-program prefetch for freed "
               "cross-program-prefetched buffer.";
    AddAsyncCopy(*allocations.front(), MemorySpace::kAlternate,
                 chunk_candidate.chunk, end_of_program_prefetch_start_time,
                 end_of_program_prefetch_end_time,
                 end_of_program_prefetch_end_time, &allocations,
                 cross_program_prefetch_offset,
                 /*resource=*/0.0);
    CHECK_EQ(cross_program_prefetch_offset->offset,
             allocations.back()->chunk().offset);
  }

  const int allocations_initial_size = allocations_->size();
  for (auto& allocation : allocations) {
    if (allocation->memory_space() == MemorySpace::kAlternate) {
      BufferInterval buffer_interval;
      buffer_interval.start = allocation->start_time();
      buffer_interval.end = allocation->end_time();
      buffer_interval.size = allocation->chunk().size;
      buffer_interval.buffer = prefetch_candidate->buffer;
      AddToPendingChunks(buffer_interval, chunk_candidate);
    }
    allocations_->push_back(std::move(allocation));
  }

  // Add a repack allocation block for the Allocation objects in alternate
  // memory.
  for (int i = allocations_initial_size; i < allocations_->size(); ++i) {
    const auto& allocation = allocations_->at(i);
    if (allocation->memory_space() == MemorySpace::kAlternate) {
      repack_allocation_blocks_.push_back(MakeRepackAllocationBlock(
          allocation->start_time(), allocation->end_time(),
          allocation->chunk().size, allocation->chunk().offset,
          static_cast<int64_t>(repack_allocation_blocks_.size()),
          allocation.get()));
      RepackAllocationBlock* inserted = &repack_allocation_blocks_.back();
      for (RepackAllocationBlock& colocation : repack_allocation_blocks_) {
        colocation.colocations.push_back(inserted);
        if (&colocation != inserted) {
          inserted->colocations.push_back(&colocation);
        }
      }
    }
  }

  ClearPendingChunks();
}

void AlternateMemoryBestFitHeap::AllocateReservedScopedAllocations() {
  const auto& instruction_sequence =
      hlo_live_range_.flattened_instruction_sequence().instructions();
  std::vector<MemorySpaceAssignmentRepacker::AllocationBlock*> colocations;
  for (int i = 0; i < instruction_sequence.size(); ++i) {
    int64_t reserved_scoped_memory =
        options_.reserved_scoped_memory_fn(instruction_sequence[i]);
    if (reserved_scoped_memory != 0) {
      VLOG(1) << "Allocate reserved scoped memory at " << i << " ("
              << instruction_sequence[i]->name()
              << "): " << reserved_scoped_memory;
      MemorySpaceAssignment::BufferInterval interval;
      interval.buffer = nullptr;
      interval.size = reserved_scoped_memory;
      interval.start = i;
      interval.end = i;
      interval.need_allocation = true;
      interval.colocations = {};
      ChunkCandidate chunk_candidate =
          FindChunkCandidate(interval, /*preferred_offset=*/0);
      CHECK_EQ(chunk_candidate.chunk.offset, 0);
      AddToPendingChunks(interval, chunk_candidate);

      allocations_->push_back(
          absl::make_unique<MemorySpaceAssignment::Allocation>(
              HloPosition{instruction_sequence[i], {}}, MemorySpace::kAlternate,
              chunk_candidate.chunk, i, i, /*is_scoped_allocation=*/true));

      repack_allocation_blocks_.push_back(MakeRepackAllocationBlock(
          i, i, reserved_scoped_memory,
          /*initial_offset=*/0,
          static_cast<int64_t>(repack_allocation_blocks_.size()),
          allocations_->back().get()));
      colocations.push_back(&repack_allocation_blocks_.back());
    }
  }
  // If requested, make all scoped allocations to colocate with each other so
  // that when we repack, all scoped allocations get the same offsets. Since
  // they will all have the same scoped memory addresses, this increases the
  // opportunity to deduplicate different ops.  However, this may hurt the
  // memory packing efficiency.
  if (options_.allocate_reserved_scoped_memory_at_same_offset) {
    for (MemorySpaceAssignmentRepacker::AllocationBlock* repack_block :
         colocations) {
      repack_block->colocations = colocations;
    }
  }
}

absl::optional<AlternateMemoryBestFitHeap::RequiredMemoryAssignment>
AlternateMemoryBestFitHeap::RequiredMemoryAssignmentAt(const HloValue* buffer,
                                                       int64_t time) const {
  auto required_assignment_it = required_assignments_.find(buffer);
  absl::optional<RequiredMemoryAssignment> required_assignment_at_time;
  if (required_assignment_it != required_assignments_.end()) {
    for (const RequiredMemoryAssignment& required_assignment :
         required_assignment_it->second) {
      if (required_assignment.time == time) {
        // Sanity check that there is only one required at time.
        CHECK(!required_assignment_at_time);
        required_assignment_at_time = required_assignment;
      }
    }
  }
  return required_assignment_at_time;
}

absl::optional<AlternateMemoryBestFitHeap::RequiredMemoryAssignment>
AlternateMemoryBestFitHeap::AliasedRequiredAssignmentForUse(
    const AllocationValue::Use& use) const {
  absl::optional<RequiredMemoryAssignment> required_assignment;
  for (const HloPosition& position : use.aliases) {
    const HloValue* value =
        &alias_analysis_.dataflow_analysis().GetUniqueValueAt(
            position.instruction, position.index);
    int64_t time =
        hlo_live_range_.instruction_schedule().at(position.instruction);
    absl::optional<RequiredMemoryAssignment> required_assignment_for_alias =
        RequiredMemoryAssignmentAt(value, time);
    if (required_assignment == absl::nullopt) {
      required_assignment = required_assignment_for_alias;
    } else {
      CHECK(required_assignment_for_alias == absl::nullopt ||
            required_assignment->equals_ignoring_time(
                *required_assignment_for_alias));
    }
  }
  return required_assignment;
}

void AlternateMemoryBestFitHeap::AddAliasedRequiredAssignment(
    const HloInstruction* instruction, ShapeIndex index,
    const MemorySpaceAssignment::Allocation* aliased_allocation) {
  AliasedOffset* offset = nullptr;
  if (aliased_allocation->memory_space() == MemorySpace::kAlternate) {
    offset = GetAliasedOffset(*aliased_allocation);
  }
  AddRequiredAssignment(instruction, index, aliased_allocation->memory_space(),
                        offset);
}

void AlternateMemoryBestFitHeap::AddRequiredAssignment(
    const HloValue* value, const HloInstruction* instruction,
    MemorySpaceAssignment::MemorySpace memory_space, int64_t time,
    AliasedOffset* offset) {
  // Check for existing required assignment at this time and make sure it is the
  // same as this if there is one.
  auto existing_required_assignment = RequiredMemoryAssignmentAt(value, time);
  if (existing_required_assignment) {
    CHECK(memory_space == existing_required_assignment->memory_space)
        << "inst = " << instruction->ToString() << " at " << time;
    CHECK((!offset && !existing_required_assignment->offset) ||
          offset == existing_required_assignment->offset);
    VLOG(3) << "Not adding required assignment because there is one already: "
            << value->ToShortString() << " at " << time << " at "
            << (memory_space == MemorySpace::kDefault ? "def" : "alt");
  } else {
    VLOG(3) << "Adding required assignment: " << value->ToShortString()
            << " at " << time << " at "
            << (memory_space == MemorySpace::kDefault ? "def" : "alt");
    RequiredMemoryAssignment required_assignment{memory_space, time, offset};
    required_assignments_[value].push_back(required_assignment);
    pending_required_assignments_.push_back({value, required_assignment});
  }
}

void AlternateMemoryBestFitHeap::AddRequiredAssignment(
    const HloInstruction* instruction, ShapeIndex index,
    MemorySpace memory_space, AliasedOffset* offset) {
  const HloValue* value =
      &alias_analysis_.dataflow_analysis().GetUniqueValueAt(instruction, index);
  int64_t instruction_time =
      hlo_live_range_.instruction_schedule().at(instruction);
  AddRequiredAssignment(value, instruction, memory_space, instruction_time,
                        offset);
}

void AlternateMemoryBestFitHeap::AddInputAndOutputRequiredAssignments() {
  // Go through the parameters, outputs, and constants and pin them to the
  // corresponding memory by adding a required assignment.
  const HloModule& module = alias_analysis_.dataflow_analysis().module();
  const auto& instruction_schedule = hlo_live_range_.instruction_schedule();
  HloComputation* entry_computation = module.entry_computation();
  for (HloInstruction* parameter_instruction :
       entry_computation->parameter_instructions()) {
    int64_t parameter_instruction_time =
        instruction_schedule.at(parameter_instruction);
    ShapeUtil::ForEachSubshape(
        parameter_instruction->shape(),
        [&](const Shape& subshape, const ShapeIndex& index) {
          MemorySpace memory_space = MemorySpace::kDefault;
          if (subshape.has_layout() && subshape.layout().memory_space() ==
                                           options_.alternate_memory_space) {
            memory_space = MemorySpace::kAlternate;
          }
          for (const HloBuffer* buffer :
               alias_analysis_.ComputeBuffersAt(parameter_instruction, index)) {
            for (const HloValue* value : buffer->values()) {
              VLOG(3) << "Adding required assignment for parameter value = "
                      << value->ToShortString()
                      << " time = " << parameter_instruction_time << " space = "
                      << (memory_space == MemorySpace::kDefault ? "def"
                                                                : "alt");
              required_assignments_[value].push_back(
                  {memory_space, /*time=*/parameter_instruction_time});
            }
          }
        });
  }
  HloInstruction* root_instruction = entry_computation->root_instruction();
  int64_t root_instruction_time = instruction_schedule.at(root_instruction);
  ShapeUtil::ForEachSubshape(
      root_instruction->shape(),
      [&](const Shape& subshape, const ShapeIndex& index) {
        MemorySpace memory_space = MemorySpace::kDefault;
        if (subshape.has_layout() && subshape.layout().memory_space() ==
                                         options_.alternate_memory_space) {
          memory_space = MemorySpace::kAlternate;
        }
        for (const HloBuffer* buffer :
             alias_analysis_.ComputeBuffersAt(root_instruction, index)) {
          for (const HloValue* value : buffer->values()) {
            VLOG(3) << "Adding required assignment for output value = "
                    << value->ToShortString()
                    << " time = " << root_instruction_time << " space = "
                    << (memory_space == MemorySpace::kDefault ? "def" : "alt");
            required_assignments_[value].push_back(
                {memory_space, /*time=*/root_instruction_time});
          }
        }
      });

  for (const HloComputation* computation : module.MakeNonfusionComputations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kConstant) {
        auto constant_instruction_it = instruction_schedule.find(instruction);
        if (constant_instruction_it == instruction_schedule.end()) {
          continue;
        }
        int64_t constant_instruction_time = constant_instruction_it->second;
        for (const auto& indexed_shape :
             ShapeUtil::GetLeafShapes(instruction->shape())) {
          const ShapeIndex& index = indexed_shape.index;
          for (const HloBuffer* buffer :
               alias_analysis_.ComputeBuffersAt(instruction, index)) {
            for (const HloValue* value : buffer->values()) {
              VLOG(3) << "Adding required assignment for constant value = "
                      << value->ToShortString()
                      << " time = " << constant_instruction_time
                      << " space = def";
              required_assignments_[value].push_back(
                  {MemorySpace::kDefault, /*time=*/constant_instruction_time});
            }
          }
        }
      }
    }
  }
}

bool AlternateMemoryBestFitHeap::AreIntervalsReservedInAlternateMemory(
    absl::Span<const BufferInterval* const> colocated_intervals) const {
  auto is_position_in_alternate_memory = [&](const HloPosition& position) {
    const Shape& shape = position.shape();
    return shape.has_layout() &&
           shape.layout().memory_space() == options_.alternate_memory_space;
  };

  const HloModule& module = alias_analysis_.dataflow_analysis().module();
  const HloComputation* entry_computation = module.entry_computation();
  const HloInstruction* root_instruction =
      entry_computation->root_instruction();
  for (const BufferInterval* colocated_interval : colocated_intervals) {
    const HloValue* value = colocated_interval->buffer;
    if (value->defining_instruction()->opcode() == HloOpcode::kParameter &&
        value->defining_instruction()->parent() == entry_computation &&
        is_position_in_alternate_memory(value->defining_position())) {
      return true;
    }

    for (const HloPosition& position : value->positions()) {
      if (position.instruction == root_instruction &&
          is_position_in_alternate_memory(position)) {
        return true;
      }
    }
  }
  return false;
}

void AlternateMemoryBestFitHeap::ExportAllocationsForRepacking(
    std::vector<MemorySpaceAssignmentRepacker::AllocationBlock*>& allocations) {
  for (RepackAllocationBlock& allocation_block : repack_allocation_blocks_) {
    allocations.push_back(&allocation_block);
  }
}

void AlternateMemoryBestFitHeap::ImportRepackedAllocations() {
  interval_tree_ = {};
  for (RepackAllocationBlock& allocation_block : repack_allocation_blocks_) {
    MemorySpaceAssignment::Allocation* allocation = allocation_block.allocation;
    VLOG(3) << "Moved " << allocation->ToString() << ", size "
            << allocation->chunk().size << ", (" << allocation_block.start_time
            << ", " << allocation_block.end_time << ") from "
            << allocation_block.initial_offset << " to "
            << allocation_block.offset;
    allocation_block.allocation->mutable_chunk()->offset =
        allocation_block.offset;
    interval_tree_.Add(allocation_block.start_time, allocation_block.end_time,
                       {allocation_block.offset, allocation_block.size});
    allocation_block.initial_offset = allocation_block.offset;
    allocation_block.offset = -1;
  }
}

void AlternateMemoryBestFitHeap::UncommitPendingChunks(
    absl::Span<AllocationValue> allocation_values) {
  // Clear the allocation sequence of the allocation values so that in case we
  // retry allocation after uncommitting.
  for (AllocationValue& allocation_value : allocation_values) {
    allocation_value.allocation_sequence()->clear();
  }
  for (const auto& interval_and_chunk : pending_chunks_) {
    const BufferInterval& interval = interval_and_chunk.first;
    const Chunk& chunk = interval_and_chunk.second.chunk;
    VLOG(3) << "Uncommitting: (" << interval.start << ", " << interval.end
            << ") off = " << chunk.offset << " size = " << chunk.size;
    interval_tree_.Remove(interval.start, interval.end, chunk);
  }
  for (const auto& interval : pending_async_copies_) {
    if (interval.destination == MemorySpace::kAlternate) {
      prefetch_interval_tree_.Remove(interval.start_time, interval.end_time,
                                     kDummyChunk);
      prefetch_async_copy_resource_.RemoveCopy(interval);
    } else {
      eviction_interval_tree_.Remove(interval.start_time, interval.end_time,
                                     kDummyChunk);
      eviction_async_copy_resource_.RemoveCopy(interval);
    }
  }
  for (const auto& value_and_required_assignment :
       pending_required_assignments_) {
    auto& required_assignment_vector =
        required_assignments_[value_and_required_assignment.first];
    const RequiredMemoryAssignment& required_assignment =
        value_and_required_assignment.second;
    VLOG(3) << "Removing required assignment: "
            << (required_assignment.memory_space == MemorySpace::kDefault
                    ? "def"
                    : "alt")
            << " time = " << required_assignment.time << " off = "
            << (required_assignment.offset ? required_assignment.offset->offset
                                           : -1);
    for (auto it = required_assignment_vector.begin();
         it != required_assignment_vector.end(); ++it) {
      if (*it == value_and_required_assignment.second) {
        required_assignment_vector.erase(it);
        break;
      }
    }
  }
  ClearPendingChunks();
}

void AlternateMemoryBestFitHeap::FinalizeAllocations(
    absl::Span<AllocationValue> allocation_values) {
  absl::flat_hash_map<const AliasedOffset*,
                      std::vector<MemorySpaceAssignment::Allocation*>>
      colocation_map;
  for (AllocationValue& allocation_value : allocation_values) {
    for (auto& allocation : *allocation_value.allocation_sequence()) {
      if (options_.dump_fn != nullptr || VLOG_IS_ON(3)) {
        // Only fill buffer_info_str_ if needed.
        AppendAllocationInfoDebugString(allocation_value, *allocation,
                                        allocation_info_str_);
      }
      allocations_->push_back(std::move(allocation));
      MemorySpaceAssignment::Allocation* inserted_allocation =
          allocations_->back().get();
      if (inserted_allocation->memory_space() == MemorySpace::kAlternate) {
        colocation_map[GetAliasedOffset(*inserted_allocation)].push_back(
            inserted_allocation);
      }
    }
  }
  // The allocations that have the same AliasedOffset need to be colocated.
  // Export these to repack_allocation_blocks_ so that we can repack them to
  // reduce fragmentation.
  for (auto& colocation : colocation_map) {
    std::vector<MemorySpaceAssignmentRepacker::AllocationBlock*> colocations;
    for (MemorySpaceAssignment::Allocation* colocated_allocation :
         colocation.second) {
      repack_allocation_blocks_.push_back(MakeRepackAllocationBlock(
          colocated_allocation->start_time(), colocated_allocation->end_time(),
          colocated_allocation->chunk().size,
          colocated_allocation->chunk().offset,
          static_cast<int64_t>(repack_allocation_blocks_.size()),
          colocated_allocation));
      colocations.push_back(&repack_allocation_blocks_.back());
    }
    for (MemorySpaceAssignmentRepacker::AllocationBlock* repack_block :
         colocations) {
      repack_block->colocations = colocations;
    }
  }
  ClearPendingChunks();
}

void AlternateMemoryBestFitHeap::ClearPendingChunks() {
  pending_chunks_.clear();
  pending_async_copies_.clear();
  pending_required_assignments_.clear();
  aliased_offset_map_.clear();
  aliased_offsets_.clear();
}

void AlternateMemoryBestFitHeap::AddToPendingChunks(
    const BufferInterval& buffer_interval,
    const ChunkCandidate& chunk_candidate) {
  VLOG(3) << "Committing chunk: " << buffer_interval.start << "-"
          << buffer_interval.end << " : [" << chunk_candidate.chunk.offset
          << ", " << chunk_candidate.chunk.size << "]";
  pending_chunks_.emplace_back(buffer_interval, chunk_candidate);
  CommitChunk(buffer_interval, chunk_candidate);
}

AlternateMemoryBestFitHeap::Result AlternateMemoryBestFitHeap::AllocateSegment(
    const AllocationRequest& request) {
  auto allocation_sequence = request.allocation_value->allocation_sequence();
  // start_time == end_time is a special case where the value is consumed
  // multiple times by the same instruction. We can just find the previous
  // allocation and use that allocation.
  if (request.start_time == request.end_time) {
    MemorySpaceAssignment::Allocation* allocation =
        GetLiveAllocationAt(*allocation_sequence, request.end_time);
    CHECK_NE(allocation, nullptr);
    allocation->AddUse(request.use->hlo_use);
    return Result::kSuccess;
  }

  const HloPosition& defining_position =
      request.allocation_value->defining_position();
  VLOG(2) << "Finding allocation for "
          << request.allocation_value->ToShortString() << " ("
          << request.start_time << ", " << request.end_time
          << ") latest prefetch = " << request.latest_prefetch_time
          << " last use = " << request.allocation_value->uses().back().time
          << " use = " << request.use->hlo_use.ToString()
          << ". Size = " << request.size
          << ", def pos = " << defining_position.ToString();
  CHECK_LE(request.start_time, request.end_time);
  if (VLOG_IS_ON(3) && options_.cost_analysis) {
    VLOG(3) << "Definition benefit = "
            << options_.cost_analysis->GetAlternateMemoryBenefit(
                   request.allocation_value->defining_position())
            << " use benefit = "
            << options_.cost_analysis->GetAlternateMemoryBenefit(
                   request.use->hlo_use);
  }

  // There could be a requirement to pin this buffer to default memory either
  // because it is a parameter or an output.  If the buffer is a parameter, then
  // we're allowed to prefetch. If the use expects the output to be in default
  // memory, we cannot prefetch it because if we did, it would be in alternate
  // memory instead.
  auto required_assignment_at_start = RequiredMemoryAssignmentAt(
      request.allocation_value->value(), request.start_time);
  absl::optional<MemorySpace> required_memory_space_at_start;
  if (required_assignment_at_start) {
    required_memory_space_at_start = required_assignment_at_start->memory_space;
  }
  // Find required assignment both for the use and its aliases. If they are both
  // non-nullopt, then make sure they require the same assignment.
  auto required_assignment_at_end = RequiredMemoryAssignmentAt(
      request.allocation_value->value(), request.end_time);
  auto aliased_required_assignment_at_end =
      AliasedRequiredAssignmentForUse(*request.use);
  if (required_assignment_at_end != aliased_required_assignment_at_end) {
    if (required_assignment_at_end == absl::nullopt) {
      required_assignment_at_end = aliased_required_assignment_at_end;
    } else {
      CHECK(aliased_required_assignment_at_end == absl::nullopt ||
            aliased_required_assignment_at_end->equals_ignoring_time(
                *required_assignment_at_end));
    }
  }
  absl::optional<MemorySpace> required_memory_space_at_end;
  if (required_assignment_at_end) {
    required_memory_space_at_end = required_assignment_at_end->memory_space;
  }

  if (required_assignment_at_start) {
    bool needs_required_allocation = true;
    if (!allocation_sequence->empty()) {
      auto prev_allocation_it = std::find_if(
          allocation_sequence->rbegin(), allocation_sequence->rend(),
          [&](const auto& allocation) {
            return allocation->memory_space() ==
                       required_memory_space_at_start &&
                   allocation->defining_position() == defining_position;
          });
      if (prev_allocation_it != allocation_sequence->rend()) {
        (*prev_allocation_it)->Extend(request.start_time);
        needs_required_allocation = false;
      }
    }
    if (needs_required_allocation) {
      absl::optional<Chunk> aliased_chunk = absl::nullopt;
      if (required_assignment_at_start->memory_space ==
          MemorySpace::kAlternate) {
        aliased_chunk =
            Chunk{required_assignment_at_start->offset->offset, request.size};
      }
      allocation_sequence->push_back(
          absl::make_unique<MemorySpaceAssignment::Allocation>(
              defining_position, required_assignment_at_start->memory_space,
              aliased_chunk, request.start_time, request.start_time,
              /*is_scoped_allocation=*/false));
      if (required_assignment_at_start->memory_space ==
          MemorySpace::kAlternate) {
        CreateOrAddToAliasedOffset(*allocation_sequence->back(),
                                   required_assignment_at_start->offset);
      }
    }
  }

  Result allocation_result = Result::kSuccess;
  // First try keeping the allocation entirely in the alternate memory.
  if (required_memory_space_at_start != MemorySpace::kDefault &&
      required_memory_space_at_end != MemorySpace::kDefault &&
      request.allow_no_copy_alternate_mem_allocation) {
    allocation_result = AllocateInAlternateMemoryNoCopy(request);
    if (allocation_result == Result::kSuccess) {
      return Result::kSuccess;
    }
  }

  auto prev_allocation_it = allocation_sequence->rbegin();
  // Find a previous allocation that is in the default memory space (not
  // necessarily the very last allocation).
  auto prev_allocation_in_default_mem_it =
      std::find_if(allocation_sequence->rbegin(), allocation_sequence->rend(),
                   [&](const auto& allocation) {
                     return allocation->memory_space() == MemorySpace::kDefault;
                   });

  if (prev_allocation_in_default_mem_it == allocation_sequence->rend() &&
      prev_allocation_it != allocation_sequence->rend() &&
      (*prev_allocation_it)->memory_space() == MemorySpace::kAlternate &&
      (*prev_allocation_it)->defining_position() == defining_position &&
      !request.allocation_value->requires_contiguous_allocation()) {
    // If there was an allocation for this HloValue that was in the alternate
    // memory space, we also need to perform an eviction.
    Result eviction_result = Evict(request);
    if (eviction_result != Result::kSuccess) {
      // A non-success eviction requires us to uncommit previous allocations.
      return result_mark(Result::kFailRequiresUncommit, eviction_result);
    }
    prev_allocation_in_default_mem_it = allocation_sequence->rbegin();
  } else if (prev_allocation_in_default_mem_it == allocation_sequence->rend()) {
    allocation_sequence->push_back(
        absl::make_unique<MemorySpaceAssignment::Allocation>(
            defining_position, MemorySpace::kDefault, /*chunk=*/absl::nullopt,
            request.start_time, request.end_time,
            /*is_scoped_allocation=*/false));
    prev_allocation_in_default_mem_it = allocation_sequence->rbegin();
  }

  CHECK(prev_allocation_in_default_mem_it != allocation_sequence->rend());
  CHECK((*prev_allocation_in_default_mem_it)->memory_space() ==
        MemorySpace::kDefault);

  // If the buffer must be in default memory at the end_time, don't prefetch.
  if (required_memory_space_at_end == MemorySpace::kDefault) {
    VLOG(3)
        << "Not trying to prefetch because use requires buffer in default mem.";
    (*prev_allocation_in_default_mem_it)->Extend(request.end_time);
    (*prev_allocation_in_default_mem_it)->AddUse(request.use->hlo_use);
    return Result::kSuccess;
  }

  // Finally, try to prefetch the buffer into alternate memory.
  if (!request.allocation_value->requires_contiguous_allocation()) {
    Result prefetch_result =
        Prefetch(request, **prev_allocation_in_default_mem_it);
    if (prefetch_result == Result::kSuccess) {
      return Result::kSuccess;
    }
    result_mark(prefetch_result, allocation_result);
  }

  // If the end assignment was required to be in alternate memory but that
  // wasn't possible, then this allocation is invalid.
  if (required_memory_space_at_end == MemorySpace::kAlternate) {
    return result_mark(Result::kFailRequiresUncommit, allocation_result);
  }

  // If the start assignment was required to be in alternate memory and the
  // buffer needs a contiguous assignment, we couldn't satisfy this requirement
  // and must abort.
  if (required_memory_space_at_start == MemorySpace::kAlternate &&
      request.allocation_value->requires_contiguous_allocation()) {
    return result_mark(Result::kFailRequiresUncommit, allocation_result);
  }

  // If a copy wasn't inserted, then add this use to the latest allocation in
  // default memory.
  (*prev_allocation_in_default_mem_it)->Extend(request.end_time);
  (*prev_allocation_in_default_mem_it)->AddUse(request.use->hlo_use);
  return allocation_result;
}

void AlternateMemoryBestFitHeap::AddAsyncCopy(
    const MemorySpaceAssignment::Allocation& prev_allocation,
    MemorySpace memory_space, absl::optional<Chunk> chunk, int64_t start_time,
    int64_t end_time, int64_t copy_done_schedule_before_time,
    MemorySpaceAssignment::AllocationSequence* allocations,
    AliasedOffset* aliased_offset, float resource,
    bool is_cross_program_prefetch) {
  VLOG(3) << "Copy to "
          << (memory_space == MemorySpaceAssignment::MemorySpace::kDefault
                  ? "default"
                  : "alternate")
          << " memory between " << start_time << " and "
          << copy_done_schedule_before_time << " keeping until " << end_time
          << ", estimated copy resource is " << resource;
  CHECK_LT(start_time, copy_done_schedule_before_time);

  allocations->push_back(
      absl::make_unique<MemorySpaceAssignment::CopyAllocation>(
          prev_allocation, memory_space, chunk, start_time, end_time,
          copy_done_schedule_before_time, is_cross_program_prefetch));

  // Register the additional async copy with the interval tree to keep track of
  // the limit at any given time.
  pending_async_copies_.push_back({start_time, copy_done_schedule_before_time,
                                   resource, memory_space,
                                   next_async_copy_id_++});
  if (memory_space == MemorySpaceAssignment::MemorySpace::kAlternate) {
    prefetch_interval_tree_.Add(start_time, copy_done_schedule_before_time,
                                kDummyChunk);
    prefetch_async_copy_resource_.AddCopy(pending_async_copies_.back());
    CreateOrAddToAliasedOffset(*allocations->back(), aliased_offset);
  } else {
    eviction_interval_tree_.Add(start_time, copy_done_schedule_before_time,
                                kDummyChunk);
    eviction_async_copy_resource_.AddCopy(pending_async_copies_.back());
  }
}

bool AlternateMemoryBestFitHeap::ViolatesMaximumOutstandingAsyncCopies(
    int64_t start_time, int64_t end_time, bool is_prefetch,
    int64_t extra_async_copy_limit) const {
  if (options_.max_outstanding_prefetches < 0 && is_prefetch) {
    return false;
  }
  if (options_.max_outstanding_evictions < 0 && !is_prefetch) {
    return false;
  }

  // Count the prefetches/evictions in the interval tree for the given interval.
  if (is_prefetch) {
    int64_t num_prefetches =
        prefetch_interval_tree_.ChunksOverlappingInTime(start_time, end_time)
            .size();
    return num_prefetches >=
           options_.max_outstanding_prefetches + extra_async_copy_limit;
  } else {
    int64_t num_evictions =
        eviction_interval_tree_.ChunksOverlappingInTime(start_time, end_time)
            .size();
    return num_evictions >=
           options_.max_outstanding_evictions + extra_async_copy_limit;
  }
}

AlternateMemoryBestFitHeap::Result
AlternateMemoryBestFitHeap::AllocateInAlternateMemoryNoCopy(
    const AllocationRequest& request) {
  MemorySpaceAssignment::Allocation* prev_allocation = nullptr;
  bool can_eliminate_copy = false;
  if (request.allocation_value->allocation_sequence()->empty()) {
    // There hasn't been any allocations for this interval so far. We can
    // eliminate copy if the value can be placed in the alternate memory.
    can_eliminate_copy = options_.is_allowed_in_alternate_mem_fn(
        *request.allocation_value->value());
  } else {
    // If there has been a previous allocation, we can eliminate the copy if the
    // previous allocation was also in the alternate memory.
    prev_allocation =
        request.allocation_value->allocation_sequence()->back().get();
    can_eliminate_copy =
        (prev_allocation->memory_space() == MemorySpace::kAlternate);
  }

  if (!can_eliminate_copy) {
    return Result::kFailPrevAllocationNotInAlternateMem;
  }

  const HloPosition& defining_position =
      request.allocation_value->defining_position();
  if (!options_.prefetch_interval_picker->CanAllocateInAlternateMemoryNoCopy(
          defining_position.shape(), request.start_time + 1,
          request.end_time)) {
    return Result::kFailLiveRangeTooLong;
  }

  BufferInterval alternate_mem_interval;
  alternate_mem_interval.buffer = request.allocation_value->value();
  alternate_mem_interval.size = request.size;
  alternate_mem_interval.end = request.end_time;
  alternate_mem_interval.start = request.start_time;

  // Prefer the offset that was previously used for the previous allocation.
  AliasedOffset* preferred_offset = nullptr;
  if (prev_allocation != nullptr) {
    preferred_offset = GetAliasedOffset(*prev_allocation);
    // If there is a previous allocation, set the start time one after the end
    // of the previous allocation's end.
    alternate_mem_interval.start = prev_allocation->end_time() + 1;
  }

  if (request.preferred_offset) {
    // Sanity check that if there is a preferred offset provided in the request,
    // it matches with the previous allocation.
    CHECK(!preferred_offset || request.preferred_offset == preferred_offset)
        << "preferred_offset = " << preferred_offset->offset
        << ", request.preferred_offset = " << request.preferred_offset->offset;
    preferred_offset = request.preferred_offset;
  }

  VLOG(3) << "We can eliminate copy to alternate memory. Preferred offset = "
          << (preferred_offset ? preferred_offset->offset : -1);
  // In case there are additional uses after this use, we rely on the last use
  // time to try to reserve a chunk in the heap simulator. This is to prevent
  // the following scenario:
  //
  //                            +-------+
  //                           /         \
  //                   Producer--->Use1   +-->Use2
  //                       +---------+---------+
  // New buffer:           |         |         |
  //                       +---------+---------+
  //
  //                                     +-----------+
  // Current heap:                       | offset: 0 |
  //           --------------------------+-----------+------
  //
  // Because we allocate buffers greedily, Producer to Use1 segment first, and
  // then Use1 to Use2 segment, it is possible to allocate the first segment at
  // an offset that is available for the first segment (e.g. offset 0) but not
  // for the entire live range. This can result in unnecessary copies. By using
  // the last use time, we try to find an allocation that is available for the
  // entire Producer to Use2 range.
  absl::optional<ChunkCandidate> chunk_candidate = FindBestChunkCandidate(
      request, preferred_offset, &alternate_mem_interval);
  // Check if the new heap size fits within limits. Also ensure if a
  // preferred offset was provided, that offset was used.
  if (chunk_candidate) {
    VLOG(3) << "Keep the buffer in alternate memory. Offset = "
            << chunk_candidate->chunk.offset
            << ", size = " << chunk_candidate->chunk.size
            << ", heap_size = " << chunk_candidate->heap_size
            << ", prefetch picker = "
            << options_.prefetch_interval_picker->ToNoCopyDebugString(
                   defining_position.shape(), request.start_time,
                   request.end_time);
    AddToPendingChunks(alternate_mem_interval, *chunk_candidate);

    // If there was a previous allocation, the buffer location is the
    // same as the previous. Otherwise, it is the operand.
    if (prev_allocation != nullptr &&
        (prev_allocation->is_copy_allocation() ||
         prev_allocation->defining_position() == defining_position)) {
      prev_allocation->Extend(request.end_time);
    } else {
      request.allocation_value->allocation_sequence()->push_back(
          absl::make_unique<MemorySpaceAssignment::Allocation>(
              defining_position, MemorySpace::kAlternate,
              chunk_candidate->chunk, request.start_time, request.end_time,
              /*is_scoped_allocation=*/false));
      CreateOrAddToAliasedOffset(
          *request.allocation_value->allocation_sequence()->back(),
          preferred_offset);
    }
    request.allocation_value->allocation_sequence()->back()->AddUse(
        request.use->hlo_use);
    return Result::kSuccess;
  }
  return Result::kFailOutOfMemory;
}

AlternateMemoryBestFitHeap::Result AlternateMemoryBestFitHeap::Evict(
    const AllocationRequest& request) {
  CHECK_GT(request.allocation_value->allocation_sequence()->size(), 0);
  MemorySpaceAssignment::Allocation* prev_allocation =
      request.allocation_value->allocation_sequence()->back().get();
  int64_t eviction_start_time = prev_allocation->start_time();
  int64_t eviction_end_time = prev_allocation->end_time();
  CHECK(eviction_start_time <= eviction_end_time);

  int64_t preferred_eviction_end_time =
      std::max(options_.prefetch_interval_picker->PreferredEvictionEndTime(
                   request.allocation_value->defining_position().shape(),
                   eviction_start_time, request.end_time),
               eviction_end_time);
  // Evictions must complete by the time of this use.
  preferred_eviction_end_time =
      std::min(preferred_eviction_end_time, request.latest_prefetch_time);

  BufferInterval eviction_mem_interval;
  eviction_mem_interval.buffer = request.allocation_value->value();
  eviction_mem_interval.size = request.size;
  // Try to reserve a buffer from the end of the previous allocation to the
  // preferred eviction end time.
  eviction_mem_interval.start = eviction_end_time + 1;
  eviction_mem_interval.end = preferred_eviction_end_time;
  int64_t preferred_offset = prev_allocation->chunk().offset;
  VLOG(3) << "Eviction (" << eviction_start_time << ", " << eviction_end_time
          << ") preferred end time = " << eviction_mem_interval.end;

  for (; eviction_mem_interval.end > eviction_end_time;
       --eviction_mem_interval.end) {
    ChunkCandidate chunk_candidate =
        FindChunkCandidate(eviction_mem_interval, preferred_offset);
    if (chunk_candidate.chunk.offset == preferred_offset) {
      AddToPendingChunks(eviction_mem_interval, chunk_candidate);
      break;
    }
  }
  eviction_end_time = eviction_mem_interval.end;

  VLOG(3) << "Evicting buffer at " << prev_allocation->chunk().offset << " ("
          << eviction_start_time << ", " << eviction_end_time << ")";

  float eviction_resource =
      options_.cost_analysis
          ? options_.cost_analysis->GetAsyncCopyElapsed(
                request.allocation_value->defining_position().shape())
          : 0.1;

  bool eviction_interval_too_short = (eviction_start_time == eviction_end_time);
  bool eviction_violates_resource =
      !eviction_async_copy_resource_.HasEnoughResource(
          eviction_start_time, eviction_end_time, eviction_resource);
  if (eviction_violates_resource) {
    // If we're in the last retry, set resource to 0.
    if (options_.prefetch_interval_picker->retry_number() ==
        options_.max_retries - 1) {
      VLOG(3) << "Violates resource in last retry, setting resource = 0";
      eviction_resource = 0;
    }
    eviction_violates_resource =
        !eviction_async_copy_resource_.HasEnoughResource(
            eviction_start_time, eviction_end_time, eviction_resource);
  }
  bool eviction_violates_outstanding_copies =
      ViolatesMaximumOutstandingAsyncCopies(eviction_start_time,
                                            eviction_end_time,
                                            /*is_prefetch=*/false);

  // See if this interval would violate the asynchronous copy limit.
  if (!eviction_interval_too_short && !eviction_violates_outstanding_copies &&
      !eviction_violates_resource) {
    prev_allocation->Extend(eviction_end_time);
    AddAsyncCopy(*prev_allocation, MemorySpace::kDefault,
                 /*chunk=*/absl::nullopt, eviction_start_time,
                 prev_allocation->end_time(), eviction_end_time,
                 request.allocation_value->allocation_sequence(),
                 /*aliased_offset=*/nullptr, eviction_resource);
  } else {
    if (eviction_violates_outstanding_copies) {
      VLOG(3) << "This violates the maximum async copies.";
    } else if (eviction_violates_resource) {
      VLOG(3) << "This violates resource.";
    } else {
      VLOG(3) << "Eviction interval is too short (" << eviction_start_time
              << ", " << eviction_end_time << ").";
    }
    // If the original interval violated the limit, try sub-intervals within
    // this interval.
    bool eviction_scheduled = false;

    if (!eviction_scheduled) {
      // If the eviction couldn't be scheduled, then fail. This buffer will be
      // kept in the default memory.
      VLOG(3) << "Bailing: Could not evict " << request.use->hlo_use.ToString()
              << " because we hit the limit of maximum asynchronous copies "
              << "between "
              << hlo_live_range_.flattened_instruction_sequence()
                     .instructions()[eviction_start_time]
              << " and "
              << hlo_live_range_.flattened_instruction_sequence()
                     .instructions()[eviction_end_time];
      // return false;
      return Result::kFailOutOfAsyncCopies;
    }
  }
  // return true;
  return Result::kSuccess;
}

int64_t AlternateMemoryBestFitHeap::FindPrefetchEndTime(
    const AllocationRequest& request, int64_t earliest_prefetch_time) const {
  return request.latest_prefetch_time;
}

AlternateMemoryBestFitHeap::Result AlternateMemoryBestFitHeap::Prefetch(
    const AllocationRequest& request,
    const MemorySpaceAssignment::Allocation& prev_allocation_in_default_mem) {
  // Try partially placing the buffer in the alternate space. The time that is
  // overlapped will be used to asynchronously copy the buffer from the
  // default memory to the alternate memory.
  //
  //                      start                 end
  //                      time                  time
  //                      X---------------------X
  // Alternate:                          +------+
  // Default:             +---------------------+
  //                                     ^      ^
  //                                   Copy    Copy
  //                                   Start   Done
  int64_t earliest_prefetch_time =
      prev_allocation_in_default_mem.earliest_available_time();
  if (request.earliest_prefetch_time) {
    earliest_prefetch_time =
        std::max(earliest_prefetch_time, *request.earliest_prefetch_time);
  }
  int64_t prefetch_end_time =
      FindPrefetchEndTime(request, earliest_prefetch_time);

  options_.prefetch_interval_picker->Begin(
      request.use->hlo_use, earliest_prefetch_time, prefetch_end_time);
  VLOG(3) << "Trying prefetch picker = "
          << options_.prefetch_interval_picker->ToDebugString();

  // Create an alternate memory interval that starts at the earliest
  // possible position, given by max_prefetch_interval.
  BufferInterval alternate_mem_interval;
  alternate_mem_interval.buffer = request.allocation_value->value();
  alternate_mem_interval.size = request.size;
  const HloUse& use = request.use->hlo_use;
  const Shape& shape = ShapeUtil::GetSubshape(
      use.instruction->operand(use.operand_number)->shape(), use.operand_index);
  // While uses might be allowed to have additional outstanding prefetches.
  int64_t extra_async_copy_limit =
      request.use->hlo_use.instruction->opcode() == HloOpcode::kWhile
          ? options_.while_use_extra_outstanding_prefetch_limit
          : 0;
  Result result = Result::kSuccess;
  while (!options_.prefetch_interval_picker->Done()) {
    alternate_mem_interval.start = options_.prefetch_interval_picker->Next();
    CHECK_LT(alternate_mem_interval.start, prefetch_end_time);
    int64_t estimated_prefetch_end_time =
        options_.prefetch_interval_picker->EstimatedPrefetchEndTime(
            shape, alternate_mem_interval.start, prefetch_end_time);
    VLOG(4) << "Trying alternate memory allocation ("
            << alternate_mem_interval.start << ", " << request.end_time
            << "), estimated prefetch end time = "
            << estimated_prefetch_end_time;
    float prefetch_resource =
        options_.cost_analysis
            ? options_.cost_analysis->GetAsyncCopyElapsed(shape)
            : 0.1;
    if (!prefetch_async_copy_resource_.HasEnoughResource(
            alternate_mem_interval.start, prefetch_end_time,
            prefetch_resource)) {
      VLOG(2) << "This would violate asynchronous copy resource = "
              << prefetch_resource;
      result_mark(Result::kFailViolatesAsyncCopyResource, result);
      continue;
    }
    if (ViolatesMaximumOutstandingAsyncCopies(
            alternate_mem_interval.start, prefetch_end_time,
            /*is_prefetch=*/true, extra_async_copy_limit)) {
      VLOG(4) << "This would violate the outstanding async copy limit.";
      result_mark(Result::kFailOutOfAsyncCopies, result);
      continue;
    }

    auto chunk_candidate = FindBestChunkCandidate(
        request, request.preferred_offset, &alternate_mem_interval);
    // Check if we could find a suitable chunk.
    if (chunk_candidate) {
      VLOG(3) << "Move the buffer to alternate memory at "
              << alternate_mem_interval.start
              << ". Offset = " << chunk_candidate->chunk.offset
              << ", size = " << chunk_candidate->chunk.size
              << ", heap_size = " << chunk_candidate->heap_size
              << ", prefetch picker = "
              << options_.prefetch_interval_picker->ToDebugString();
      AddToPendingChunks(alternate_mem_interval, *chunk_candidate);

      AddAsyncCopy(prev_allocation_in_default_mem, MemorySpace::kAlternate,
                   chunk_candidate->chunk, alternate_mem_interval.start,
                   request.end_time, prefetch_end_time,
                   request.allocation_value->allocation_sequence(),
                   request.preferred_offset, prefetch_resource);

      request.allocation_value->allocation_sequence()->back()->AddUse(
          request.use->hlo_use);
      return Result::kSuccess;
    }
    result_mark(Result::kFailOutOfMemory, result);
  }
  // If we didn't consider any prefetch intervals, then the live range was too
  // short.
  if (result == Result::kSuccess) {
    return Result::kFailLiveRangeTooShort;
  } else {
    return result;
  }
}

absl::optional<AlternateMemoryBestFitHeap::ChunkCandidate>
AlternateMemoryBestFitHeap::FindBestChunkCandidate(
    const AllocationRequest& request, const AliasedOffset* preferred_offset,
    BufferInterval* alternate_mem_interval) const {
  int64_t end_time = request.end_time;
  if (!preferred_offset) {
    // First find the earliest use that is the same or later than the end time.
    const auto& use_times = request.all_use_times;
    auto use_time_it = use_times.begin();
    for (; *use_time_it < end_time; ++use_time_it) {
    }
    CHECK(use_time_it != use_times.end());
    int64_t earliest_use = *use_time_it;

    // Then find the latest use that can be allocated contiguously without
    // copies.
    const Shape& shape = request.allocation_value->defining_position().shape();
    for (;
         (use_time_it + 1) != use_times.end() &&
         options_.prefetch_interval_picker->CanAllocateInAlternateMemoryNoCopy(
             shape, *use_time_it, *(use_time_it + 1));
         ++use_time_it) {
    }
    CHECK(use_time_it != use_times.end());
    int64_t latest_contiguous_use_time = *use_time_it;

    // Find a chunk that's as long living as possible iterating in reverse over
    // the use times.
    for (; use_time_it >= use_times.begin() && *use_time_it >= end_time;
         --use_time_it) {
      alternate_mem_interval->end = *use_time_it;
      ChunkCandidate chunk_candidate =
          FindChunkCandidate(*alternate_mem_interval);
      if (chunk_candidate.heap_size <= available_heap_size()) {
        alternate_mem_interval->end = end_time;
        VLOG(3) << "FindBestChunkCandidate earliest use = " << earliest_use
                << ", latest contiguous use = " << latest_contiguous_use_time
                << ", use with available mem = " << *use_time_it
                << ", offset = " << chunk_candidate.chunk.offset;
        return chunk_candidate;
      }
    }
    alternate_mem_interval->end = end_time;
    return absl::nullopt;
  }
  // If a preferred offset is given, try to find an allocation at that offset
  // only.
  alternate_mem_interval->end = end_time;
  ChunkCandidate chunk_candidate =
      FindChunkCandidate(*alternate_mem_interval, preferred_offset->offset);
  if (chunk_candidate.chunk.offset == preferred_offset->offset) {
    return chunk_candidate;
  }
  return absl::nullopt;
}

StatusOr<MemorySpaceAssignment::AsyncCopyStats>
MemorySpaceAssignment::CalculateAsyncCopyStats() const {
  AsyncCopyStats stats;
  stats.max_outstanding_async_copies = 0;
  stats.num_prefetches = 0;
  stats.prefetch_bytes = 0;
  stats.num_evictions = 0;
  stats.eviction_bytes = 0;
  int64_t current_copies = 0;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloDataflowAnalysis> dataflow_analysis,
                      HloDataflowAnalysis::Run(*module_));
  for (const HloComputation* computation :
       module_->MakeNonfusionComputations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kCopyStart) {
        current_copies++;
      } else if (instruction->opcode() == HloOpcode::kCopyDone) {
        current_copies--;
        int64_t size =
            options_.size_fn(dataflow_analysis->GetUniqueValueAt(instruction));
        if (instruction->shape().layout().memory_space() ==
            options_.alternate_memory_space) {
          ++stats.num_prefetches;
          stats.prefetch_bytes += size;
        } else {
          ++stats.num_evictions;
          stats.eviction_bytes += size;
        }
      }
      stats.max_outstanding_async_copies =
          std::max(stats.max_outstanding_async_copies, current_copies);
    }
  }
  return stats;
}

/*static*/ MemorySpaceAssignment::BufferIntervalCompare
MemorySpaceAssignment::GetMemoryBoundednessBufferIntervalCompare(
    const MemorySpaceAssignmentCostAnalysis& cost_analysis,
    MemorySpaceAssignmentCostAnalysis::Cache* cache) {
  return [&cost_analysis, cache](const BufferInterval& x,
                                 const BufferInterval& y) {
    float x_memory_boundedness = cost_analysis.GetMemoryBoundedness(x, cache);
    float y_memory_boundedness = cost_analysis.GetMemoryBoundedness(y, cache);
    if (x_memory_boundedness != y_memory_boundedness) {
      return x_memory_boundedness > y_memory_boundedness;
    }
    // Tie-break if the memory boundedness is the same.
    return GlobalDecreasingSizeBestFitHeap<
        HloValue>::GetSpatialBufferIntervalCompare()(x, y);
  };
}

/*static*/ StatusOr<std::unique_ptr<PresetAssignments>>
MemorySpaceAssignment::Run(HloModule* module,
                           const HloLiveRange& hlo_live_range,
                           const HloAliasAnalysis& alias_analysis,
                           const Options& options) {
  CHECK(module->has_schedule());
  VLOG(3) << "Module before memory space assignment: ";
  XLA_VLOG_LINES(3, module->ToString());
  VLOG(3) << "Schedule: " << module->schedule().ToString();
  MemorySpaceAssignment memory_space_assignment(module, options,
                                                hlo_live_range);

  return memory_space_assignment.RunMemorySpaceAssignment(hlo_live_range,
                                                          alias_analysis);
}

StatusOr<std::unique_ptr<PresetAssignments>>
MemorySpaceAssignment::RunMemorySpaceAssignment(
    const HloLiveRange& hlo_live_range,
    const HloAliasAnalysis& alias_analysis) {
  TF_RETURN_IF_ERROR(FindAllocationSequence(hlo_live_range, alias_analysis));

  if (options_.cost_analysis) {
    float estimated_time =
        ComputeEstimatedElapsedTime(hlo_live_range, allocations_);
    VLOG(1) << "Estimated elapsed time (sec): " << estimated_time;
  }

  TF_RETURN_IF_ERROR(Process());
  ScheduleAsynchronousCopies();
  TF_RETURN_IF_ERROR(SimplifyGraph());
  TF_RETURN_IF_ERROR(FixSchedule());
  TF_RETURN_IF_ERROR(ExportAndColorBuffers());

  VLOG(3) << "Module after memory space assignment: ";
  XLA_VLOG_LINES(3, module_->ToString());
  TF_CHECK_OK(module_->schedule().Verify());
  TF_ASSIGN_OR_RETURN(AsyncCopyStats stats, CalculateAsyncCopyStats());
  VLOG(1) << "Maximum number of outstanding async copies: "
          << stats.max_outstanding_async_copies;
  VLOG(1) << "Number of prefetches: " << stats.num_prefetches
          << ", in bytes: " << stats.prefetch_bytes;
  VLOG(1) << "Number of evictions: " << stats.num_evictions
          << ", in bytes: " << stats.eviction_bytes;

  TF_RETURN_IF_ERROR(VerifyAndExportHeapSimulatorTrace());

  return std::move(preset_assignments_);
}

Status MemorySpaceAssignment::FindAllocationSequence(
    const HloLiveRange& hlo_live_range,
    const HloAliasAnalysis& alias_analysis) {
  auto algorithm = absl::make_unique<AlternateMemoryBestFitHeap>(
      &allocations_, options_, alias_analysis, hlo_live_range);

  HeapSimulator::Options heap_simulator_options;
  heap_simulator_options.may_reuse_operand_buffers = false;
  heap_simulator_options.alloc_constants = true;
  TF_RETURN_IF_ERROR(HeapSimulator::Run(std::move(algorithm), *module_,
                                        module_->schedule(), alias_analysis,
                                        options_.size_fn,
                                        heap_simulator_options)
                         .status());
  return Status::OK();
}

void MemorySpaceAssignment::Allocation::AddUse(HloUse use) {
  HloInstruction* operand =
      use.instruction->mutable_operand(use.operand_number);
  // If the use is a tuple, look inside the tuple to find the actual use.
  for (int64_t index : use.operand_index) {
    if (operand->opcode() != HloOpcode::kTuple) {
      break;
    }
    operand = operand->mutable_operand(index);
  }

  // Look beyond GetTupleElement(Tuple()) pattern for any bitcasts.
  std::function<HloInstruction*(HloInstruction*)> get_simplified_operand;
  get_simplified_operand = [&](HloInstruction* instruction) {
    while (instruction->opcode() == HloOpcode::kGetTupleElement) {
      HloInstruction* operand =
          get_simplified_operand(instruction->mutable_operand(0));
      if (operand->opcode() == HloOpcode::kTuple) {
        instruction = operand->mutable_operand(instruction->tuple_index());
      } else {
        return instruction;
      }
    }
    return instruction;
  };
  operand = get_simplified_operand(operand);

  uses_.push_back(use);
}

float MemorySpaceAssignment::ComputeEstimatedElapsedTime(
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
    float instruction_elapsed =
        options_.cost_analysis->GetInstructionElapsedInAlternateMemory(
            *instruction, operands_in_alternate_memory,
            outputs_in_alternate_memory);
    float while_nest_multiplier = tensorflow::MathUtil::IPow<float>(
        options_.xla_tpu_memory_space_assignment_while_execution_count,
        options_.cost_analysis->CalculateComputationNestLevel(
            instruction,
            /*while_only=*/true));
    total_elapsed += while_nest_multiplier * instruction_elapsed;
  }
  return total_elapsed;
}

Status MemorySpaceAssignment::Allocation::Process() {
  if (is_scoped_allocation()) {
    // Nothing to do here for scoped allocations.
    return Status::OK();
  }
  HloInstruction* producing_instruction = AddGetTupleElements();
  HloComputation* computation = producing_instruction->parent();
  for (const HloUse& use : uses_) {
    Shape operand_shape = use.instruction->operand(use.operand_number)->shape();
    HloInstruction* replacement_instruction = producing_instruction;
    if (operand_shape.IsTuple()) {
      TF_ASSIGN_OR_RETURN(
          replacement_instruction,
          ReplaceTupleWith(producing_instruction,
                           use.instruction->mutable_operand(use.operand_number),
                           use.operand_index));
    } else if (operand_shape != producing_instruction->shape()) {
      VLOG(4) << "Old shape = " << operand_shape.ToString()
              << ", new shape = " << producing_instruction->shape().ToString()
              << "; inserting a bitcast.";
      replacement_instruction = computation->AddInstruction(
          HloInstruction::CreateBitcast(operand_shape, producing_instruction));
    }
    TF_RETURN_IF_ERROR(use.instruction->ReplaceOperandWith(
        use.operand_number, replacement_instruction));
  }
  return Status::OK();
}

StatusOr<HloInstruction*> MemorySpaceAssignment::Allocation::ReplaceTupleWith(
    HloInstruction* new_instruction, HloInstruction* tuple,
    ShapeIndex shape_index) {
  const Shape& tuple_shape = tuple->shape();
  CHECK(tuple->shape().IsTuple())
      << "ReplaceTupleWith was called for a non-tuple. Tuple = "
      << tuple->ToString()
      << ", new_instruction = " << new_instruction->ToString()
      << ", shape_index = " << shape_index.ToString();
  // Check if the new instruction is a get-tuple-element of the correct index of
  // the tuple, and if so, simply return tuple.
  const HloInstruction* instruction = new_instruction;
  bool equivalent = true;
  for (int i = shape_index.size() - 1; i >= 0; --i) {
    int index = shape_index[i];
    if (instruction->opcode() != HloOpcode::kGetTupleElement ||
        instruction->tuple_index() != index) {
      equivalent = false;
      break;
    }
    instruction = instruction->operand(0);
  }
  if (equivalent && instruction == tuple) {
    VLOG(4) << "Instruction " << new_instruction->ToShortString()
            << " already exists at index " << shape_index.ToString() << " of "
            << tuple->ToShortString();
    return tuple;
  }

  HloComputation* computation = new_instruction->parent();
  std::vector<HloInstruction*> tuple_args(tuple_shape.tuple_shapes_size());
  CHECK_GE(tuple_shape.tuple_shapes_size(), shape_index[0]);
  for (int i = 0; i < tuple_shape.tuple_shapes_size(); ++i) {
    const Shape& subshape = tuple_shape.tuple_shapes(i);
    // If tuple is a tuple instruction, we can get the tuple instruction's
    // operand to construct the new tuple to improve compilation time
    // performance.
    auto get_operand = [&]() {
      if (tuple->opcode() == HloOpcode::kTuple) {
        return tuple->mutable_operand(i);
      } else {
        return computation->AddInstruction(
            HloInstruction::CreateGetTupleElement(subshape, tuple, i));
      }
    };
    if (i == shape_index[0]) {
      // If the subshape is still a tuple, recurse and pass a new shape index
      // for the one level deeper.
      if (subshape.IsTuple()) {
        TF_ASSIGN_OR_RETURN(tuple_args[i],
                            ReplaceTupleWith(new_instruction, get_operand(),
                                             ShapeIndex(shape_index.begin() + 1,
                                                        shape_index.end())));
      } else {
        if (subshape != new_instruction->shape()) {
          VLOG(4) << "Old shape = " << subshape.ToString()
                  << ", new shape = " << new_instruction->shape().ToString()
                  << "; inserting a bitcast.";
          new_instruction = computation->AddInstruction(
              HloInstruction::CreateBitcast(subshape, new_instruction));
        } else if (tuple->opcode() == HloOpcode::kTuple &&
                   tuple->operand(i) == new_instruction) {
          // If the tuple element is the same as the new instruction, we
          // actually don't have to create a new tuple, just return the original
          // tuple.
          VLOG(4) << "Tuple already contains the new instruction = "
                  << new_instruction->ToShortString()
                  << " tuple = " << tuple->ToShortString();
          return tuple;
        }
        tuple_args[i] = new_instruction;
      }
    } else {
      tuple_args[i] = get_operand();
    }
  }
  if (shape_index[0] == tuple_shape.tuple_shapes_size()) {
    // If shape_index[0] is equal to the tuple shape size, add the new
    // instruction as an additional argument.
    tuple_args.push_back(new_instruction);
  }
  return computation->AddInstruction(HloInstruction::CreateTuple(tuple_args));
}

HloInstruction* MemorySpaceAssignment::Allocation::AddGetTupleElements() const {
  HloInstruction* producing_instruction = defining_position().instruction;
  CHECK_NE(producing_instruction, nullptr);

  Shape shape = defining_position().shape();
  CHECK(shape.IsArray()) << "Allocation shape is not an array. Shape = "
                         << shape.ToString()
                         << " position = " << defining_position().shape();
  HloComputation* computation = producing_instruction->parent();

  // If the instruction we're processing is a tuple, we (recursively) search or
  // create kGetTupleElement instructions and copy that value. Asynchronous
  // copies only support array types.
  for (int64_t index : defining_position().index) {
    // We first search if there already is a get-tuple-element with the correct
    // index. If there is no such get-tuple-element, we create one.
    auto gte_it = absl::c_find_if(
        producing_instruction->users(), [index](const HloInstruction* use) {
          return use != use->parent()->root_instruction() &&
                 use->opcode() == HloOpcode::kGetTupleElement &&
                 use->tuple_index() == index;
        });
    if (gte_it != producing_instruction->users().end()) {
      producing_instruction = *gte_it;
    } else {
      producing_instruction =
          computation->AddInstruction(HloInstruction::CreateGetTupleElement(
              producing_instruction->shape().tuple_shapes(index),
              producing_instruction, index));
    }
  }
  return producing_instruction;
}

std::string MemorySpaceAssignment::Allocation::ToString() const {
  std::string memory_space_str = "def";
  if (memory_space_ == MemorySpace::kAlternate) {
    memory_space_str = absl::StrCat("alt (off: ", chunk_->offset, ")");
  }
  return absl::StrCat((is_scoped_allocation() ? "Scoped " : ""),
                      "Allocation in ", memory_space_str, " defined at ",
                      defining_position_.ToString());
}

std::string MemorySpaceAssignment::CopyAllocation::ToString() const {
  std::string memory_space_str = "def";
  if (memory_space_ == MemorySpace::kAlternate) {
    memory_space_str = absl::StrCat("alt (off: ", chunk_->offset, ")");
  }
  return absl::StrCat("Copy Allocation in ", memory_space_str, " from ",
                      prev_allocation_.ToString());
}

std::string MemorySpaceAssignment::MirroredAllocation::ToString() const {
  return absl::StrCat("Mirrored Allocation for ",
                      original_allocation_.ToString());
}

std::string MemorySpaceAssignment::ParentAllocation::ToString() const {
  return absl::StrCat("Parent Allocation mirrored at ",
                      defining_position_.ToString(), ", originally ",
                      original_allocation_.ToString());
}

Status MemorySpaceAssignment::CopyAllocation::Process() {
  // Copy allocations need to insert asynchronous copy nodes.
  Shape shape = defining_position().shape();
  HloInstruction* producing_instruction = AddGetTupleElements();
  HloComputation* computation = producing_instruction->parent();
  copy_start_ = computation->AddInstruction(HloInstruction::CreateCopyStart(
      ShapeUtil::MakeTupleShape({shape, shape, ShapeUtil::MakeShape(U32, {})}),
      producing_instruction, is_cross_program_prefetch_));
  copy_done_ = computation->AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCopyDone, copy_start_));
  VLOG(4) << "Created " << copy_start_->name()
          << " for position: " << defining_position().ToString();
  // Update the allocation position with the copy done instruction so that if
  // there are further copies from it, it can find the correct position.
  defining_position_ = HloPosition{copy_done_, {}};

  // Replace all the uses with the new copy instruction.
  for (HloUse use : uses_) {
    // If the operand is a tuple, we need to descend to the actual instruction
    // we want to replace.
    HloInstruction* replacement_instruction;
    Shape operand_shape = use.instruction->operand(use.operand_number)->shape();
    if (operand_shape.IsTuple()) {
      TF_ASSIGN_OR_RETURN(
          replacement_instruction,
          ReplaceTupleWith(copy_done_,
                           use.instruction->mutable_operand(use.operand_number),
                           use.operand_index));
    } else if (operand_shape != copy_done_->shape()) {
      VLOG(4) << "Old shape = " << operand_shape.ToString()
              << ", new shape = " << copy_done_->shape().ToString()
              << "; inserting a bitcast.";
      replacement_instruction = computation->AddInstruction(
          HloInstruction::CreateBitcast(operand_shape, copy_done_));
    } else {
      replacement_instruction = copy_done_;
    }
    TF_RETURN_IF_ERROR(use.instruction->ReplaceOperandWith(
        use.operand_number, replacement_instruction));
  }

  return Status::OK();
}

Status MemorySpaceAssignment::MirroredAllocation::Process() {
  defining_position_ = original_allocation_.defining_position();
  return Allocation::Process();
}

Status MemorySpaceAssignment::ParentAllocation::Process() {
  // Add an additional parameter to the while HLO with a reference to the buffer
  // in the default memory space.
  HloInstruction* producing_instruction =
      original_allocation_.AddGetTupleElements();
  int new_tuple_index = calling_instruction_->shape().tuple_shapes_size();

  TF_ASSIGN_OR_RETURN(HloInstruction * new_while_operand,
                      ReplaceTupleWith(producing_instruction,
                                       calling_instruction_->mutable_operand(0),
                                       {new_tuple_index}));
  TF_RETURN_IF_ERROR(calling_instruction_->ReplaceOperandWithDifferentShape(
      0, new_while_operand));
  *calling_instruction_->mutable_shape() = new_while_operand->shape();
  *calling_instruction_->while_condition()
       ->parameter_instruction(0)
       ->mutable_shape() = new_while_operand->shape();
  *calling_instruction_->while_body()
       ->parameter_instruction(0)
       ->mutable_shape() = new_while_operand->shape();
  defining_position_.index = {new_tuple_index};
  // Also replace the while op with a tuple that has the old shape. Note that we
  // need to first take a snapshot of the users before calling ExtractPrefix
  // since ExtractPrefix introduces additional gte users.
  std::vector<HloInstruction*> while_users = calling_instruction_->users();
  HloInstruction* tuple_with_old_shape =
      TupleUtil::ExtractPrefix(calling_instruction_, new_tuple_index);
  TF_RETURN_IF_ERROR(calling_instruction_->ReplaceAllUsesWithDifferentShape(
      while_users, tuple_with_old_shape));
  return Allocation::Process();
}

Status MemorySpaceAssignment::ParentAllocation::PostProcess() {
  // Update the root of the while body with the new parameter. The reason why we
  // need a separate post-process for this is because other allocations may have
  // while body root as a use, so they would update the old root instead of the
  // new root. Doing the post-process step later ensures the root has been
  // updated with other changes, and we can safely add the additional parameter.
  HloComputation* while_body = calling_instruction_->while_body();
  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_while_body_root,
      ReplaceTupleWith(AddGetTupleElements(), while_body->root_instruction(),
                       defining_position_.index));
  while_body->set_root_instruction(new_while_body_root,
                                   /*accept_different_shape=*/true);
  return Status::OK();
}

void MemorySpaceAssignment::Allocation::MarkIfNeeded(
    absl::flat_hash_set<const Allocation*>& needed_allocations) const {
  MarkNeeded(needed_allocations);
}

void MemorySpaceAssignment::Allocation::MarkNeeded(
    absl::flat_hash_set<const Allocation*>& needed_allocations) const {
  needed_allocations.insert(this);
}

void MemorySpaceAssignment::CopyAllocation::MarkNeeded(
    absl::flat_hash_set<const Allocation*>& needed_allocations) const {
  needed_allocations.insert(this);
  prev_allocation_.MarkNeeded(needed_allocations);
}

void MemorySpaceAssignment::ParentAllocation::MarkIfNeeded(
    absl::flat_hash_set<const Allocation*>& needed_allocations) const {
  // Parent allocations are only needed if they have any uses or if there is a
  // copy allocation that copies this value (in that case, the copy allocation
  // will call this allocation's MarkNeeded function).
  if (!uses_.empty()) {
    MarkNeeded(needed_allocations);
  }
}

void MemorySpaceAssignment::ParentAllocation::MarkNeeded(
    absl::flat_hash_set<const Allocation*>& needed_allocations) const {
  needed_allocations.insert(this);
  original_allocation_.MarkNeeded(needed_allocations);
}

void MemorySpaceAssignment::MirroredAllocation::MarkNeeded(
    absl::flat_hash_set<const Allocation*>& needed_allocations) const {
  needed_allocations.insert(this);
  original_allocation_.MarkNeeded(needed_allocations);
}

Status MemorySpaceAssignment::Process() {
  VLOG(1) << "Processing assigned buffers...";
  // Since some parent allocations may not be needed (e.g. when they don't have
  // any uses and if there is no other (non-parent) allocation that depends on
  // it, before we process the allocations, mark all allocations that are
  // needed.
  absl::flat_hash_set<const Allocation*> needed_allocations;
  for (auto& allocation : allocations_) {
    allocation->MarkIfNeeded(needed_allocations);
  }
  // Insert CopyStart/CopyDone pairs.
  for (auto& allocation : allocations_) {
    VLOG(3) << "Processing: " << allocation->ToString();
    if (!needed_allocations.contains(allocation.get())) {
      VLOG(3) << "Allocation not needed.";
      continue;
    }
    TF_RETURN_IF_ERROR(allocation->Process());
    // Add the offset and size of the allocation in the alternate memory to
    // the output map.
    if (allocation->is_scoped_allocation()) {
      CHECK(allocation->memory_space() == MemorySpace::kAlternate);
      scoped_memory_assignments_.emplace_back(
          allocation->defining_position().instruction, allocation->chunk());
      alternate_memory_size_ =
          std::max(alternate_memory_size_, allocation->chunk().chunk_end());
    } else if (allocation->memory_space() == MemorySpace::kAlternate) {
      alternate_memory_assignments_.emplace_back(
          allocation->defining_position(), allocation->chunk());
      alternate_memory_size_ =
          std::max(alternate_memory_size_, allocation->chunk().chunk_end());
    }
  }
  // Post-process allocations. This is only used for parent allocations where we
  // update the body root with a reference to the buffer in default memory
  // space.
  for (auto& allocation : allocations_) {
    if (needed_allocations.contains(allocation.get())) {
      VLOG(3) << "Post-Processing: " << allocation->ToString();
      TF_RETURN_IF_ERROR(allocation->PostProcess());
    }
  }
  return Status::OK();
}

Status MemorySpaceAssignment::ExportAndColorBuffers() {
  VLOG(1) << "Exporting buffers...";
  TF_ASSIGN_OR_RETURN(auto alias_analysis, HloAliasAnalysis::Run(module_));
  absl::flat_hash_map<int64_t, int64_t> seen_buffer_offsets;
  VLOG(3) << "Exported alternate memory allocations:";
  for (const auto& position_and_chunk : alternate_memory_assignments_) {
    const HloPosition& defining_position = position_and_chunk.first;
    const Chunk& chunk = position_and_chunk.second;
    const HloBuffer& buffer = alias_analysis->GetUniqueBufferAt(
        defining_position.instruction, defining_position.index);
    auto seen_buffer_offset_it = seen_buffer_offsets.find(buffer.id());
    if (seen_buffer_offset_it != seen_buffer_offsets.end()) {
      CHECK_EQ(chunk.offset, seen_buffer_offset_it->second)
          << "Mismatch in offset for positions that map to the same value: "
          << buffer.ToString() << ", pos: " << defining_position.ToString();
    } else {
      VLOG(3) << " [" << chunk.offset << ", " << chunk.size
              << "] : " << defining_position.ToString() << " ("
              << buffer.ToString() << ")";
      preset_assignments_->add_chunk(defining_position, chunk);
      seen_buffer_offsets[buffer.id()] = chunk.offset;
    }
  }

  VLOG(3) << "Exported scoped allocations in alternate memory:";
  for (const auto& instruction_and_chunk : scoped_memory_assignments_) {
    HloInstruction* instruction = instruction_and_chunk.first;
    const Chunk& chunk = instruction_and_chunk.second;
    VLOG(3) << " [" << chunk.offset << ", " << chunk.size
            << "] : " << instruction->name();
    preset_assignments_->add_scoped_allocation_chunk(instruction, chunk);
  }

  if (!preset_assignments_->chunks().empty() ||
      !preset_assignments_->scoped_allocation_chunks().empty()) {
    preset_assignments_
        ->assignment_information_for_space(options_.alternate_memory_space)
        ->size = alternate_memory_size_;
  }

  VLOG(3) << "Exported alternate memory sizes:";
  for (auto& pair : preset_assignments_->assignment_informations()) {
    VLOG(3) << "  space: " << pair.first << ", size: " << pair.second.size;
  }

  VLOG(1) << "Coloring buffers...";
  // Color the pending positions and all of their aliased buffers.
  for (const auto& defining_position_and_chunk :
       preset_assignments_->chunks()) {
    const HloPosition& defining_position = defining_position_and_chunk.first;
    for (auto& buffer : alias_analysis->ComputeBuffersAt(
             defining_position.instruction, defining_position.index)) {
      for (auto& value : buffer->values()) {
        for (auto& position : value->positions()) {
          VLOG(4) << "Coloring " << position.ToString();
          Shape* shape = ShapeUtil::GetMutableSubshape(
              position.instruction->mutable_shape(), position.index);
          CHECK(shape->IsArray()) << "Coloring a shape that is not an array: "
                                  << position.ToString();
          shape->mutable_layout()->set_memory_space(
              options_.alternate_memory_space);
        }
      }
    }
  }
  return Status::OK();
}

void MemorySpaceAssignment::RemoveAssignmentForInstruction(
    const HloInstruction* instruction) {
  for (auto& position_and_chunk : alternate_memory_assignments_) {
    const HloPosition& position = position_and_chunk.first;
    if (position.instruction == instruction) {
      VLOG(3) << "Removing instruction from alternate memory assignments.";
      // Swap the removed position and chunk with the back and pop back.
      position_and_chunk = alternate_memory_assignments_.back();
      alternate_memory_assignments_.pop_back();
      break;
    }
  }
}

Status MemorySpaceAssignment::SimplifyGraph() {
  VLOG(1) << "Simplifying graph...";
  for (HloComputation* computation : module_->MakeNonfusionComputations()) {
    // Parallel computations aren't in the schedule and don't need to be
    // modified.
    if (!computations_in_schedule_.contains(computation)) {
      VLOG(4) << "Not simplifying " << computation->name()
              << " because it's not in the schedule.";
      continue;
    }
    // Drop control dependencies. Since the computation is already scheduled, we
    // don't need control dependencies anymore, and having control
    // predecessors/successors prevents us from removing instructions without
    // users (HloComputation::IsSafelyRemovable returns false if there are
    // control dependencies).
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      TF_RETURN_IF_ERROR(instruction->DropAllControlDeps());
    }
    // We perform limited DCE and forward the tuple operand in patterns like
    // GetTupleElement(Tuple(a, b), 0). This is mostly because memory space
    // assignment is ran late in compilation (after DCE and arithmetic
    // simplification passes) and we don't want to generate redundant code.  Run
    // to fixed point.
    bool computation_modified = true;
    while (computation_modified) {
      computation_modified = false;
      VLOG(4) << "Running simplify graph loop over " << computation->name();
      for (HloInstruction* instruction :
           computation->MakeInstructionPostOrder()) {
        if (computation->IsSafelyRemovable(instruction) &&
            instruction->user_count() == 0 && !instruction->HasSideEffect() &&
            instruction != computation->root_instruction() &&
            instruction->opcode() != HloOpcode::kCopyStart &&
            instruction->opcode() != HloOpcode::kCopyDone) {
          VLOG(4) << "Instruction removed: " << instruction->ToString();
          // Ensure the alternate memory assignments don't contain a reference
          // to the removed instruction.
          RemoveAssignmentForInstruction(instruction);
          // Instead of deleting the instruction from the schedule, replace it
          // with a nullptr. This is needed because FixSchedule relies on the
          // logical time that is the index into flattened_instructions_ for
          // scheduling asynchronous copies.
          auto instruction_it =
              absl::c_find(flattened_instructions_, instruction);
          if (instruction_it != flattened_instructions_.end()) {
            *instruction_it = nullptr;
          }
          TF_RETURN_IF_ERROR(computation->RemoveInstruction(instruction));
          computation_modified = true;
        } else if (instruction->opcode() == HloOpcode::kGetTupleElement) {
          HloInstruction* operand = instruction->mutable_operand(0);
          if (operand->opcode() == HloOpcode::kTuple) {
            HloInstruction* forwarded_instruction =
                operand->mutable_operand(instruction->tuple_index());
            VLOG(4) << "Replacing uses of " << instruction->ToString()
                    << " with " << forwarded_instruction->ToString();
            TF_RETURN_IF_ERROR(
                instruction->ReplaceAllUsesWith(forwarded_instruction));
            computation_modified = true;
          }
        } else if (instruction->opcode() == HloOpcode::kTuple) {
          // Replace Tuple(GetTupleElement(x), ..., GetTupleElement(x)) pattern
          // with x.
          bool can_replace =
              instruction->operand_count() > 0 &&
              instruction->operand(0)->opcode() ==
                  HloOpcode::kGetTupleElement &&
              instruction->operand(0)
                      ->operand(0)
                      ->shape()
                      .tuple_shapes_size() == instruction->operand_count();
          for (int operand_number = 0;
               operand_number < instruction->operand_count();
               ++operand_number) {
            const HloInstruction* operand =
                instruction->operand(operand_number);
            if (operand->opcode() != HloOpcode::kGetTupleElement ||
                operand->tuple_index() != operand_number ||
                operand->operand(0) != instruction->operand(0)->operand(0)) {
              can_replace = false;
              break;
            }
          }
          if (can_replace) {
            HloInstruction* forwarded_instruction =
                instruction->mutable_operand(0)->mutable_operand(0);
            VLOG(4) << "Replacing uses of " << instruction->ToString()
                    << " with " << forwarded_instruction->ToString();
            TF_RETURN_IF_ERROR(
                instruction->ReplaceAllUsesWith(forwarded_instruction));
            computation_modified = true;
          }
        }
      }
    }
  }

  return Status::OK();
}

void MemorySpaceAssignment::ScheduleAsynchronousCopies() {
  VLOG(1) << "Scheduling asynchronous copies...";
  for (MemorySpace memory_space :
       {MemorySpace::kDefault, MemorySpace::kAlternate}) {
    std::vector<CopyAllocation*> copy_allocations;
    for (auto& allocation : allocations_) {
      if (allocation->is_copy_allocation()) {
        auto copy_allocation = static_cast<CopyAllocation*>(allocation.get());
        if (copy_allocation->memory_space() == memory_space) {
          copy_allocations.push_back(copy_allocation);
        }
      }
    }

    absl::c_stable_sort(
        copy_allocations, [](CopyAllocation* first, CopyAllocation* second) {
          return std::forward_as_tuple(first->copy_done_schedule_before(),
                                       first->copy_start_schedule_after()) <
                 std::forward_as_tuple(second->copy_done_schedule_before(),
                                       second->copy_start_schedule_after());
        });
    for (CopyAllocation* copy_allocation : copy_allocations) {
      // If the copy start doesn't happen to be scheduled at the correct
      // computation, delay it until the correct computation starts.
      int64_t copy_start_schedule_after =
          copy_allocation->copy_start_schedule_after();
      // Accessing flattened_instructions_ here without checking if it is
      // nullptr is safe because this method is called before SimplifyGraph.
      while (copy_allocation->defining_position().instruction->parent() !=
             flattened_instructions_[copy_start_schedule_after]->parent()) {
        VLOG(4) << "Delaying CopyStart (" << copy_start_schedule_after << " to "
                << (copy_start_schedule_after + 1) << ") for "
                << copy_allocation->copy_start()->ToString()
                << " because it is not in the correct computation.";
        copy_allocation->set_copy_start_schedule_after(
            ++copy_start_schedule_after);
      }

      schedule_after_[copy_allocation->copy_start_schedule_after()].push_back(
          copy_allocation->copy_start());
      schedule_before_[copy_allocation->copy_done_schedule_before()].push_back(
          copy_allocation->copy_done());
    }
  }
}

Status MemorySpaceAssignment::FixSchedule() {
  VLOG(1) << "Fixing schedule...";
  TF_RET_CHECK(module_->has_schedule());
  HloSchedule& schedule = module_->schedule();
  for (const HloComputation* computation :
       module_->MakeNonfusionComputations()) {
    // Parallel computations aren't in the schedule and don't need to be
    // modified.
    if (!computations_in_schedule_.contains(computation)) {
      VLOG(4) << "Not scheduling " << computation->name()
              << " because it's not in the schedule.";
      continue;
    }
    TF_RET_CHECK(schedule.is_computation_scheduled(computation));
    HloInstructionSequence new_sequence;

    absl::flat_hash_set<HloInstruction*> inserted_instructions;

    VLOG(4) << "Scheduling: " << computation->ToString();

    for (int64_t instruction_index = 0;; ++instruction_index) {
      auto insts_before_iter = schedule_before_.find(instruction_index);
      if (insts_before_iter != schedule_before_.end()) {
        for (HloInstruction* new_instruction : insts_before_iter->second) {
          if (new_instruction->parent() == computation) {
            VLOG(4) << "before " << instruction_index << ": "
                    << new_instruction->name();
            TF_RETURN_IF_ERROR(InsertInstructionAndEnsureOperandsInserted(
                new_instruction, &new_sequence, &inserted_instructions));
          }
        }
      }
      // We allow scheduling copy dones past the root instruction (for
      // end-of-program cross-program prefetch). So the loop exit condition is
      // actually here.
      if (instruction_index >= flattened_instructions_.size()) {
        break;
      }
      HloInstruction* instruction = flattened_instructions_[instruction_index];
      // Insert only if it is not deleted (SimplifyGraph sets it to nullptr if
      // it was deleted) and not previously inserted. Also bitcasts and tuples
      // are treated specially and only inserted as a result of operand
      // dependencies.
      if (instruction != nullptr && instruction->parent() == computation &&
          instruction->opcode() != HloOpcode::kBitcast &&
          instruction->opcode() != HloOpcode::kTuple &&
          !inserted_instructions.contains(instruction)) {
        VLOG(4) << "inst " << instruction_index << ": " << instruction->name();
        TF_RETURN_IF_ERROR(InsertInstructionAndEnsureOperandsInserted(
            instruction, &new_sequence, &inserted_instructions));
      }
      auto insts_after_iter = schedule_after_.find(instruction_index);
      if (insts_after_iter != schedule_after_.end()) {
        for (HloInstruction* new_instruction : insts_after_iter->second) {
          if (new_instruction->parent() == computation) {
            VLOG(4) << "after " << instruction_index << ": "
                    << new_instruction->name();
            TF_RETURN_IF_ERROR(InsertInstructionAndEnsureOperandsInserted(
                new_instruction, &new_sequence, &inserted_instructions));
          }
        }
      }
    }
    // For rare cases where the original sequence is empty, ensure the root
    // instruction and its dependencies are scheduled.
    TF_RETURN_IF_ERROR(EnsureInstructionAndOperandsInserted(
        computation->root_instruction(), &new_sequence,
        &inserted_instructions));
    CHECK_EQ(new_sequence.size(), computation->instruction_count())
        << "New sequence for computation " << computation->name() << " has "
        << new_sequence.size() << " instructions, expects "
        << computation->instruction_count() << ".";
    schedule.set_sequence(computation, new_sequence);
  }

  return Status::OK();
}

Status MemorySpaceAssignment::VerifyAndExportHeapSimulatorTrace() {
  VLOG(1) << "Verifying...";
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                      HloAliasAnalysis::Run(module_));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloLiveRange> hlo_live_range,
                      HloLiveRange::Run(module_->schedule(), *alias_analysis,
                                        module_->entry_computation()));

  BufferIntervalTree interval_tree;
  absl::flat_hash_set<int64_t> seen_buffers;
  // The key for events is: time, is_free, value_id. This is so that the events
  // are sorted first by time, then within the same time, allocations are sorted
  // earlier than frees, and finally the value id as a tie breaker.
  std::map<std::tuple<int64_t, bool, int64_t>,
           std::tuple<const HloValue*, Chunk, HeapSimulatorTrace::Event::Kind>>
      events;

  auto add_allocation_and_verify = [&](int64_t start_time, int64_t end_time,
                                       const Chunk& chunk,
                                       const HloValue* value) {
    events[std::make_tuple(start_time, /*is_free=*/false, value->id())] =
        std::make_tuple(value, chunk, HeapSimulatorTrace::Event::ALLOC);
    events[std::make_tuple(end_time, /*is_free=*/true, value->id())] =
        std::make_tuple(value, chunk, HeapSimulatorTrace::Event::FREE);

    // Get the chunks overlapping in time and search if they overlap in space
    // as well.
    // TODO(berkin): For now checking against end_time - 1 (exclusive), but we
    // really should check against end_time (inclusive) for cases where the
    // operand can't share buffer with user (see
    // HloDataflowAnalysis::CanShareOperandBufferWithUser).
    for (const Chunk& overlapping_chunk :
         interval_tree.ChunksOverlappingInTime(start_time, end_time - 1)) {
      if (chunk.OverlapsWith(overlapping_chunk)) {
        return InternalError(
            ("Value %s (%d, %d) off: %d size: %d overlaps with another chunk"
             " off: %d size: %d"),
            value->ToShortString(), start_time, end_time, chunk.offset,
            chunk.size, overlapping_chunk.offset, overlapping_chunk.size);
      }
    }
    interval_tree.Add(start_time, end_time - 1, chunk);
    return Status::OK();
  };

  // Go through all instructions in the module to ensure CopyStart/CopyDone
  // instructions copy between alternate memory and default memory.
  for (const HloComputation* computation :
       module_->MakeNonfusionComputations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kCopyStart) {
        int64_t from_memory_space =
            ShapeUtil::GetSubshape(instruction->shape(), {1})
                .layout()
                .memory_space();
        int64_t to_memory_space =
            ShapeUtil::GetSubshape(instruction->shape(), {0})
                .layout()
                .memory_space();
        CHECK_NE(from_memory_space, to_memory_space)
            << "Asynchronous copy to the same memory space: "
            << instruction->ToString();
      }
    }
  }

  for (const auto& position_and_chunk : preset_assignments_->chunks()) {
    const HloPosition& position = position_and_chunk.first;
    const Chunk& chunk = position_and_chunk.second;
    const HloBuffer& buffer =
        alias_analysis->GetUniqueBufferAt(position.instruction, position.index);
    CHECK(!seen_buffers.contains(buffer.id()))
        << "Multiple preset assignments for the same buffer: "
        << buffer.ToString() << ", pos: " << position.ToString()
        << ", off: " << chunk.offset << ", size: " << chunk.size;
    seen_buffers.insert(buffer.id());

    for (const HloValue* value : buffer.values()) {
      const HloLiveRange::TimeBound& time_bound =
          hlo_live_range->buffer_live_ranges().at(value);
      const HloInstruction* last_use_instruction = nullptr;
      int64_t last_use_time = time_bound.start;
      for (const HloUse& use : value->uses()) {
        int64_t use_time =
            hlo_live_range->instruction_schedule().at(use.instruction);
        if (use_time > last_use_time) {
          last_use_time = use_time;
          last_use_instruction = use.instruction;
        }
      }

      std::function<Status(const HloInstruction*, int64_t, int64_t,
                           absl::string_view)>
          split_conditional_buffer;
      split_conditional_buffer = [&](const HloInstruction* use_instruction,
                                     int64_t start_time, int64_t end_time,
                                     absl::string_view indent_string) {
        // Special case when verifying conditional: we internally split the use
        // of alternate memory in conditionals, so fish them out from the
        // conditionals.
        VLOG(3) << indent_string
                << "Splitting conditional buffer: " << buffer.ToString()
                << " value: " << value->ToShortString() << ": (" << start_time
                << ", " << end_time << ") off: " << chunk.offset
                << ", size: " << chunk.size;
        int64_t earliest_computation_start_time = end_time;
        for (const HloComputation* called_computation :
             use_instruction->called_computations()) {
          earliest_computation_start_time =
              std::min(earliest_computation_start_time,
                       hlo_live_range->computation_span_times()
                           .at(called_computation)
                           .start);
          int64_t parameter_time = -1;
          int64_t last_use_time = -1;
          const HloInstruction* last_use_instruction = nullptr;
          for (const HloPosition& position : value->positions()) {
            if (position.instruction->opcode() == HloOpcode::kParameter &&
                position.instruction->parent() == called_computation) {
              parameter_time = hlo_live_range->instruction_schedule().at(
                  position.instruction);
              break;
            }
          }
          for (const HloUse& use : value->uses()) {
            int64_t use_time =
                hlo_live_range->instruction_schedule().at(use.instruction);
            if (use.instruction->parent() == called_computation &&
                use_time > last_use_time) {
              last_use_time = use_time;
              last_use_instruction = use.instruction;
            }
          }
          if (last_use_time != -1) {
            CHECK_NE(parameter_time, -1);
            VLOG(3) << indent_string
                    << " computation: " << called_computation->name() << ": ("
                    << parameter_time << ", " << last_use_time << ")";
            CHECK(last_use_instruction);
            if (last_use_instruction->opcode() == HloOpcode::kConditional) {
              // The last use is another (nested) conditional. Call this
              // function recursively.
              TF_RETURN_IF_ERROR(split_conditional_buffer(
                  last_use_instruction, parameter_time, last_use_time,
                  absl::StrCat(indent_string, "  ")));
            } else {
              last_use_time = std::min(last_use_time, end_time);
              TF_RETURN_IF_ERROR(add_allocation_and_verify(
                  parameter_time, last_use_time, chunk, value));
            }
          }
        }
        VLOG(3) << indent_string << " from beginning until first computation: ("
                << start_time << ", " << (earliest_computation_start_time - 1)
                << ")";
        TF_RETURN_IF_ERROR(add_allocation_and_verify(
            start_time, earliest_computation_start_time - 1, chunk, value));
        return Status::OK();
      };

      if (last_use_instruction &&
          last_use_instruction->opcode() == HloOpcode::kConditional) {
        TF_RETURN_IF_ERROR(split_conditional_buffer(
            last_use_instruction, time_bound.start, time_bound.end, " "));
      } else if (!value->uses().empty()) {
        last_use_time = std::min(last_use_time, time_bound.end);
        VLOG(3) << " buffer: " << buffer.ToString()
                << " value: " << value->ToShortString() << ": ("
                << time_bound.start << ", " << last_use_time
                << ") off: " << chunk.offset << ", size: " << chunk.size;
        TF_RETURN_IF_ERROR(add_allocation_and_verify(
            time_bound.start, last_use_time, chunk, value));
      }
    }
  }

  HeapSimulatorTrace* heap_trace =
      &preset_assignments_
           ->assignment_information_for_space(options_.alternate_memory_space)
           ->heap_simulator_trace;
  int64_t memory_usage = 0;
  int64_t max_memory_usage = 0;
  for (const auto& event : events) {
    int64_t time;
    bool is_free;
    int64_t buffer_id;
    std::tie(time, is_free, buffer_id) = event.first;
    const HloValue* value;
    Chunk chunk;
    HeapSimulatorTrace::Event::Kind kind;
    std::tie(value, chunk, kind) = event.second;
    HeapSimulatorTrace::Event* heap_trace_event = heap_trace->add_events();
    heap_trace_event->set_kind(kind);
    heap_trace_event->set_buffer_id(buffer_id);
    heap_trace_event->set_instruction_name(value->instruction()->name());
    heap_trace_event->set_computation_name(
        value->instruction()->parent()->name());

    if (kind == HeapSimulatorTrace::Event::ALLOC) {
      memory_usage += chunk.size;
    } else {
      CHECK_EQ(kind, HeapSimulatorTrace::Event::FREE);
      memory_usage -= chunk.size;
    }
    max_memory_usage = std::max(max_memory_usage, memory_usage);
    VLOG(4) << "Memory usage: " << memory_usage << " at time: " << time;
  }
  VLOG(1) << "Max memory usage ignoring fragmentation: " << max_memory_usage;

  return Status::OK();
}
}  // namespace memory_space_assignment
}  // namespace xla

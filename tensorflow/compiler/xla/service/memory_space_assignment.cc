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

namespace xla {

namespace {
// Define a dummy chunk for chunks that will be allocated in the default memory
// space and for keeping track of number of asynchronous copies.
const HeapSimulator::Chunk kDummyChunk{-1, -1};
}  // namespace

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
    absl::optional<int64> operand_in_alternate_mem,
    bool output_in_alternate_mem) const {
  float bytes_accessed = cost_analysis_.bytes_accessed(instruction);
  float elapsed_due_to_bytes =
      bytes_accessed /
      cost_analysis_.per_second_rate(HloCostAnalysis::kBytesAccessedKey);
  if (operand_in_alternate_mem) {
    // Estimate the elapsed time due to the operand being in the alternate
    // memory space.
    float operand_bytes_accessed = cost_analysis_.operand_bytes_accessed(
        instruction, *operand_in_alternate_mem);
    float elapsed_due_to_operand_bytes =
        operand_bytes_accessed / alternate_mem_bandwidth_bytes_per_second_;
    bytes_accessed -= operand_bytes_accessed;
    elapsed_due_to_bytes =
        elapsed_due_to_operand_bytes +
        bytes_accessed /
            cost_analysis_.per_second_rate(HloCostAnalysis::kBytesAccessedKey);
  }
  if (output_in_alternate_mem) {
    // Estimate the elapsed time due to the output being in the alternate memory
    // space.
    float output_bytes_accessed =
        cost_analysis_.output_bytes_accessed(instruction);
    float elapsed_due_to_output_bytes =
        output_bytes_accessed / alternate_mem_bandwidth_bytes_per_second_;
    bytes_accessed -= output_bytes_accessed;
    elapsed_due_to_bytes =
        elapsed_due_to_output_bytes +
        bytes_accessed /
            cost_analysis_.per_second_rate(HloCostAnalysis::kBytesAccessedKey);
  }
  return elapsed_due_to_bytes;
}

float MemorySpaceAssignmentCostAnalysis::GetInstructionElapsed(
    const HloInstruction& instruction,
    absl::optional<int64> operand_in_alternate_mem,
    bool output_in_alternate_mem) const {
  return std::max(
      GetInstructionElapsedDueToCompute(instruction),
      GetInstructionElapsedDueToMemory(instruction, operand_in_alternate_mem,
                                       output_in_alternate_mem));
}

float MemorySpaceAssignmentCostAnalysis::GetAsyncCopyElapsed(
    const Shape& shape) const {
  int64 size_in_bytes = cost_analysis_.GetShapeSize(shape);
  return static_cast<float>(size_in_bytes) /
         async_copy_bandwidth_bytes_per_second_;
}

bool InstructionCountPrefetchIntervalPicker::CanAllocateInAlternateMemoryNoCopy(
    const Shape& shape, int64 start_time, int64 end_time) const {
  return end_time - start_time <= max_overlap_count_;
}

int64 InstructionCountPrefetchIntervalPicker::PreferredEvictionEndTime(
    const Shape& shape, int64 start_time, int64 latest_end_time) const {
  return std::min(start_time + min_overlap_count_, latest_end_time);
}

void InstructionCountPrefetchIntervalPicker::Begin(const HloUse& use,
                                                   int64 start_time,
                                                   int64 end_time) {
  end_time_ = end_time;
  current_prefetch_time_ = std::max(start_time, end_time_ - max_overlap_count_);
}

int64 InstructionCountPrefetchIntervalPicker::Next() {
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
    const Shape& shape, int64 start_time, int64 end_time) const {
  return absl::StrCat("Overlapped HLOs = ", end_time - start_time);
}

void CostAnalysisPrefetchIntervalPicker::SetInstructionSchedule(
    const absl::flat_hash_map<const HloInstruction*, int64>&
        instruction_schedule) {
  // First create a vector of elapsed times of HLO instructions.
  std::vector<float> instructions_elapsed_time(instruction_schedule.size(),
                                               0.0);

  for (const auto& instruction_and_logical_time : instruction_schedule) {
    float elapsed_time = cost_analysis_.cost_analysis().optimal_seconds(
        *instruction_and_logical_time.first);
    int64 logical_time = instruction_and_logical_time.second;
    if (logical_time >= instructions_elapsed_time.size()) {
      instructions_elapsed_time.resize(logical_time + 1, 0.0);
    }
    instructions_elapsed_time[logical_time] = elapsed_time;
  }
  // As an optimization, create a cumulative sum vector of elapsed time.
  float cumsum = 0.0;
  for (float elapsed_time : instructions_elapsed_time) {
    cumsum += elapsed_time;
    elapsed_time_cumsum_.push_back(cumsum);
  }
}

bool CostAnalysisPrefetchIntervalPicker::CanAllocateInAlternateMemoryNoCopy(
    const Shape& shape, int64 start_time, int64 end_time) const {
  // Even though this method returns if we allow the buffer in alternate memory
  // _without_ asynchronous copies, calculate how long it would have taken to
  // copy it and compare it to the elapsed time in the logical interval.
  float async_copy_elapsed = cost_analysis_.GetAsyncCopyElapsed(shape);
  float logical_interval_elapsed =
      GetLogicalIntervalElapsed(start_time, end_time);
  return max_async_copy_to_overlap_ratio_ * async_copy_elapsed >
         logical_interval_elapsed;
}

int64 CostAnalysisPrefetchIntervalPicker::PreferredEvictionEndTime(
    const Shape& shape, int64 start_time, int64 latest_end_time) const {
  float async_copy_elapsed = cost_analysis_.GetAsyncCopyElapsed(shape);
  int64 end_time;
  for (end_time = start_time + 1; end_time <= latest_end_time; ++end_time) {
    float logical_interval_elapsed =
        GetLogicalIntervalElapsed(start_time, end_time);
    if (logical_interval_elapsed >=
        min_async_copy_to_overlap_ratio_ * async_copy_elapsed) {
      break;
    }
  }
  return end_time;
}

void CostAnalysisPrefetchIntervalPicker::Begin(const HloUse& use,
                                               int64 start_time,
                                               int64 end_time) {
  const Shape& shape = use.instruction->operand(use.operand_number)->shape();
  // Find the earliest time that satisfies max_async_copy_to_overlap_ratio_.
  async_copy_elapsed_ = cost_analysis_.GetAsyncCopyElapsed(shape);
  // Estimate the time we would save by having this op in alternate memory.
  float elapsed_time = cost_analysis_.GetInstructionElapsed(*use.instruction);
  float elapsed_time_in_alternate_mem = cost_analysis_.GetInstructionElapsed(
      *use.instruction, use.operand_number);
  inst_elapsed_reduction_ = elapsed_time - elapsed_time_in_alternate_mem;
  end_logical_time_ = end_time;
  // Find the earliest time we're allowed to start prefetching.
  for (current_logical_prefetch_time_ = start_time;
       current_logical_prefetch_time_ <= end_logical_time_ &&
       max_async_copy_to_overlap_ratio_ * async_copy_elapsed_ <
           GetLogicalIntervalElapsed(current_logical_prefetch_time_,
                                     end_logical_time_);
       ++current_logical_prefetch_time_) {
  }
}

int64 CostAnalysisPrefetchIntervalPicker::Next() {
  CHECK(!Done()) << "Prefetch interval picker's Next() is called even though "
                    "Done() is false";
  return current_logical_prefetch_time_++;
}

bool CostAnalysisPrefetchIntervalPicker::Done() const {
  // The end time is inclusive, so we're done if the prefetch time is greater
  // than that.
  if (current_logical_prefetch_time_ > end_logical_time_) {
    return true;
  }
  float logical_interval_elapsed = GetLogicalIntervalElapsed(
      current_logical_prefetch_time_, end_logical_time_);
  return async_copy_elapsed_ * min_async_copy_to_overlap_ratio_ >
         logical_interval_elapsed + inst_elapsed_reduction_;
}

float CostAnalysisPrefetchIntervalPicker::GetLogicalIntervalElapsed(
    int64 start_time, int64 end_time) const {
  return elapsed_time_cumsum_[end_time - 1] - elapsed_time_cumsum_[start_time];
}

std::string CostAnalysisPrefetchIntervalPicker::ToDebugString() const {
  float logical_interval_elapsed = GetLogicalIntervalElapsed(
      current_logical_prefetch_time_, end_logical_time_);
  return absl::StrCat(
      "Async copy elapsed (s) = ", async_copy_elapsed_,
      ", inst elapsed reduction (s) = ", inst_elapsed_reduction_,
      ", logical interval elapsed (s) = ", logical_interval_elapsed);
}

std::string CostAnalysisPrefetchIntervalPicker::ToNoCopyDebugString(
    const Shape& shape, int64 start_time, int64 end_time) const {
  float async_copy_elapsed = cost_analysis_.GetAsyncCopyElapsed(shape);
  float logical_interval_elapsed =
      GetLogicalIntervalElapsed(start_time, end_time);
  return absl::StrCat(
      "Async copy elapsed (s) = ", async_copy_elapsed,
      ", logical interval elapsed (s) = ", logical_interval_elapsed);
}

std::vector<const GlobalDecreasingSizeBestFitHeap::BufferInterval*>
AlternateMemoryBestFitHeap::GetSortedColocatedIntervals(
    const GlobalDecreasingSizeBestFitHeap::BufferInterval& interval) const {
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

  absl::c_sort(colocated_intervals, [&](const BufferInterval* x,
                                        const BufferInterval* y) {
    return std::make_pair(x->start, x->end) < std::make_pair(y->start, y->end);
  });
  return colocated_intervals;
}

HeapSimulator::Result AlternateMemoryBestFitHeap::Finish() {
  std::vector<BufferInterval> sorted_buffer_intervals =
      GetSortedBufferIntervals();

  VLOG(1) << "Assigning buffers to alternate memory. Max heap size = "
          << options_.max_size_in_bytes;

  AddInputAndOutputRequiredAssignments();
  options_.prefetch_interval_picker->SetInstructionSchedule(
      hlo_live_range_.instruction_schedule());

  for (auto& interval : sorted_buffer_intervals) {
    if (!interval.need_allocation) {
      continue;
    }

    // Skip if we have already allocated for this buffer.
    if (allocation_map_->contains(interval.buffer)) {
      continue;
    }

    // If the buffer is a tuple, don't use this algorithm for now. The buffers
    // that are pointed to by the tuple will still use this algorithm.  Because
    // tuples are cheap to place in the alternate memory (they are just
    // pointers) we don't need to use prefetch/evict logic.
    if (interval.buffer->shape().IsTuple()) {
      VLOG(4) << "Keeping value " << interval.buffer->ToShortString()
              << " in default mem because it is a tuple.";
      continue;
    }

    auto colocated_intervals = GetSortedColocatedIntervals(interval);

    if (AreIntervalsReservedInAlternateMemory(colocated_intervals)) {
      VLOG(4) << "Interval " << interval.buffer->ToShortString()
              << " is reserved in the alternate memory. Total reserved bytes = "
              << reserved_in_bytes_;
      for (const BufferInterval* colocated_interval : colocated_intervals) {
        const HloValue* value = colocated_interval->buffer;
        // Color all of the aliased reserved buffers here because reserved
        // alternate memory allocations will not have an entry in preset
        // allocations that is normally used for coloring.
        for (auto& position : value->positions()) {
          VLOG(3) << "Coloring " << position.ToString();
          Shape* shape = ShapeUtil::GetMutableSubshape(
              position.instruction->mutable_shape(), position.index);
          CHECK(shape->IsArray()) << "Coloring a shape that is not an array: "
                                  << position.ToString();
          shape->mutable_layout()->set_memory_space(
              options_.alternate_memory_space);
        }
      }
      // Increment the reserved part of alternate memory so that it is not
      // available for other buffers. Since all colocated intervals should have
      // the same size, just use the first one.
      reserved_in_bytes_ += options_.size_fn(*colocated_intervals[0]->buffer);
      continue;
    }

    if (colocated_intervals.size() > 1 &&
        !options_.allocate_across_sequential_calls) {
      VLOG(4) << "Not allocating " << interval.buffer->ToShortString()
              << " because it aliases with another interval and "
              << " allocate_across_sequential_calls is false.";
      continue;
    }

    const HloComputation* defining_computation =
        colocated_intervals[0]->buffer->defining_instruction()->parent();
    MemorySpaceAssignment::Allocation* aliased_allocation = nullptr;
    for (const BufferInterval* colocated_interval : colocated_intervals) {
      const HloValue* value = colocated_interval->buffer;
      const auto& instruction_schedule = hlo_live_range_.instruction_schedule();
      MemorySpaceAssignment::AllocationSequence* allocation_sequence =
          &(*allocation_map_)[value];
      int64 definition_time =
          instruction_schedule.at(value->defining_instruction());
      // Sort the uses by the use time.
      std::vector<HloUse> uses = value->uses();
      absl::c_sort(uses, [&](HloUse use1, HloUse use2) {
        return instruction_schedule.at(use1.instruction) <
               instruction_schedule.at(use2.instruction);
      });

      // If there was an aliased allocation for this buffer, propagate that for
      // this HloValue.
      if (aliased_allocation != nullptr) {
        VLOG(3) << "Adding an aliased allocation: ("
                << aliased_allocation->start_time() << ", "
                << aliased_allocation->end_time()
                << ") pos: " << aliased_allocation->defining_position()
                << " mem space: "
                << (aliased_allocation->memory_space() == MemorySpace::kDefault
                        ? "default"
                        : "alt");
        allocation_sequence->push_back(
            absl::make_unique<MemorySpaceAssignment::Allocation>(
                value->defining_instruction(), value->defining_position(),
                aliased_allocation->memory_space(), aliased_allocation->chunk(),
                definition_time, definition_time));
      }

      // Iterate over the uses.
      for (HloUse use : uses) {
        int64 use_time = instruction_schedule.at(use.instruction);
        int64 last_use_time = instruction_schedule.at(uses.back().instruction);
        int64 latest_prefetch_time = use_time;

        if (use.instruction->parent() != defining_computation) {
          VLOG(3) << "skip use " << use.ToString()
                  << " because it's in a different computation.";
          continue;
        }

        // Sequential calls include kWhile, kCall, and kConditional opcodes.
        bool is_sequential_call =
            (GetInstructionCallContext(use.instruction->opcode()) ==
             CallContext::kSequential);
        if (is_sequential_call) {
          for (const HloComputation* called_computation :
               use.instruction->called_computations()) {
            const HloLiveRange::TimeBound& computation_span =
                hlo_live_range_.computation_span_times().at(called_computation);
            latest_prefetch_time =
                std::min(computation_span.start, latest_prefetch_time);
          }
        }

        // Bitcasts don't define buffers and don't directly consume buffers.
        // Skip allocating buffers for bitcast uses. The uses that feed from
        // bitcasts will be handled specially.
        if (use.instruction->opcode() != HloOpcode::kBitcast) {
          if (!FindAllocation(definition_time, use_time, last_use_time,
                              latest_prefetch_time, value->defining_position(),
                              use, value, colocated_interval->size,
                              allocation_sequence)) {
            // If the allocation finding failed (e.g., due to running out of
            // asynchronous copies), then fall back to allocating the buffer
            // entirely in the default memory.
            pending_chunks_.clear();
            pending_async_copies_.clear();
            allocation_sequence->clear();
            break;
          }

          // If there are multiple uses, they can try using the memory
          // allocation already at the alternate memory.
          definition_time = use_time;
        }

        // If the use has been a sequential call (e.g. a while loop), the other
        // colocated intervals must alias with this allocation.
        if (is_sequential_call && !allocation_sequence->empty()) {
          aliased_allocation = allocation_sequence->back().get();
        }
      }
    }

    CommitPendingChunks();
  }

  if (VLOG_IS_ON(3)) {
    for (const auto& alloc_pair : *allocation_map_) {
      VLOG(3) << "Allocation for " << alloc_pair.first->ToShortString();
      for (const auto& alloc : alloc_pair.second) {
        std::string addr_str = ": default";
        if (alloc->memory_space() == MemorySpace::kAlternate) {
          addr_str = absl::StrCat(": alt ", alloc->chunk().offset);
        }

        VLOG(3) << "  " << alloc->start_time() << "-" << alloc->end_time()
                << addr_str << ", " << alloc->uses().size() << " uses";
      }
    }
  }

  return result_;
}

bool operator<(const AsynchronousCopy& a, const AsynchronousCopy& b) {
  return (a.start_time < b.start_time && a.end_time <= b.end_time) ||
         (a.start_time <= b.start_time && a.end_time < b.end_time);
}

void AsynchronousCopyOrdering::AddCopy(const AsynchronousCopy& copy) {
  auto it_and_inserted = ranges_.insert(copy);
  CHECK(it_and_inserted.second ||
        it_and_inserted.first->start_time == copy.start_time);
}

bool AsynchronousCopyOrdering::ViolatesOrdering(int64 start_time,
                                                int64 end_time) const {
  // We allow identical start and end times. It is enough to check for just the
  // start time in case we find a match in ranges_ because the found value will
  // either be identical to {start_time, end_time} (and this doesn't violate) or
  // its start_time will be smaller and end_time will be larger (this violates).
  auto copy_it = ranges_.find(
      {start_time, end_time, MemorySpaceAssignment::MemorySpace::kAlternate});
  return copy_it != ranges_.end() && copy_it->start_time != start_time;
}

void AlternateMemoryBestFitHeap::AddInputAndOutputRequiredAssignments() {
  // Go through the parameters and outputs and pin them to the corresponding
  // memory by adding a required assignment.
  const HloModule& module = alias_analysis_.dataflow_analysis().module();
  const auto& instruction_schedule = hlo_live_range_.instruction_schedule();
  HloComputation* entry_computation = module.entry_computation();
  for (HloInstruction* parameter_instruction :
       entry_computation->parameter_instructions()) {
    int64 parameter_instruction_time =
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
  int64 root_instruction_time = instruction_schedule.at(root_instruction);
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

void AlternateMemoryBestFitHeap::CommitPendingChunks() {
  for (auto interval_and_chunk : pending_chunks_) {
    VLOG(3) << "Committing chunk: " << interval_and_chunk.first.start << "-"
            << interval_and_chunk.first.end << " : ["
            << interval_and_chunk.second.chunk.offset << ", "
            << interval_and_chunk.second.chunk.size << "]";
    CommitChunk(interval_and_chunk.first, interval_and_chunk.second);
  }
  pending_chunks_.clear();
  // Also add the pending async copies to the interval tree.
  for (const auto& interval : pending_async_copies_) {
    if (options_.max_outstanding_async_copies >= 0) {
      async_copy_interval_tree_.Add(interval.start_time, interval.end_time,
                                    kDummyChunk);
    }
    if (interval.destination == MemorySpace::kAlternate) {
      async_copy_ordering_.AddCopy(interval);
    }
  }
  pending_async_copies_.clear();
}

void AlternateMemoryBestFitHeap::AddToPendingChunks(
    const BufferInterval& buffer_interval,
    const ChunkCandidate& chunk_candidate) {
  pending_chunks_.emplace_back(buffer_interval, chunk_candidate);
}

bool AlternateMemoryBestFitHeap::FindAllocation(
    int64 start_time, int64 end_time, int64 last_use_time,
    int64 latest_prefetch_time, HloPosition defining_position, HloUse use,
    const HloValue* buffer, int64 size,
    MemorySpaceAssignment::AllocationSequence* allocations) {
  HloInstruction* operand =
      use.instruction->mutable_operand(use.operand_number);
  // If the operand is a bitcast, we look at bitcast's operand until we find a
  // non-bitcast operand.
  HloInstruction* non_bitcast_operand = operand;
  while (non_bitcast_operand->opcode() == HloOpcode::kBitcast) {
    non_bitcast_operand = non_bitcast_operand->mutable_operand(0);
  }
  // Create an alternate memory interval that starts at the earliest
  // possible position, given by max_prefetch_interval.
  BufferInterval alternate_mem_interval;
  alternate_mem_interval.buffer = buffer;
  alternate_mem_interval.size = size;
  alternate_mem_interval.end = end_time;

  VLOG(2) << "Finding allocation for " << buffer->ToShortString() << " ("
          << start_time << ", " << end_time
          << ") latest prefetch = " << latest_prefetch_time
          << " last use = " << last_use_time << " use = " << use.ToString()
          << ". Size = " << size
          << ", def pos = " << defining_position.ToString()
          << ", operand = " << operand->ToShortString()
          << (non_bitcast_operand != operand
                  ? ", non_bitcast_operand = " +
                        non_bitcast_operand->ToShortString()
                  : "");
  CHECK_LE(start_time, end_time);

  // There could be a requirement to pin this buffer to default memory either at
  // the definition site (e.g., parameters) or at the use site (e.g., outputs).
  // If there is a definition requirement, then we're allowed to prefetch, but
  // if it's a use requirement, we cannot prefetch the buffer. If the use
  // expects the buffer to be in default memory, we cannot prefetch it because
  // if we did, it would be in alternate memory instead.
  bool definition_requires_buffer_in_default_mem = false;
  bool use_requires_buffer_in_default_mem = false;
  auto required_assignment_it = required_assignments_.find(buffer);
  if (required_assignment_it != required_assignments_.end()) {
    for (const RequiredMemoryAssignment& required_assignment :
         required_assignment_it->second) {
      VLOG(3) << "Required assignment at time = " << required_assignment.time
              << " space = "
              << (required_assignment.memory_space == MemorySpace::kDefault
                      ? "def"
                      : "alt");
      if (required_assignment.memory_space == MemorySpace::kDefault) {
        if (required_assignment.time == start_time) {
          definition_requires_buffer_in_default_mem = true;
          VLOG(3) << "Definition requires buffer in default memory.";
        }
        if (required_assignment.time == end_time) {
          use_requires_buffer_in_default_mem = true;
          VLOG(3) << "Use requires buffer in default memory.";
        }
      }
    }
  }

  // First try keeping the allocation entirely in the alternate memory.
  if (!definition_requires_buffer_in_default_mem &&
      !use_requires_buffer_in_default_mem &&
      TryAllocatingInAlternateMemoryNoCopy(
          start_time, end_time, last_use_time, defining_position, use,
          alternate_mem_interval, non_bitcast_operand, allocations)) {
    return true;
  }

  MemorySpaceAssignment::Allocation* prev_allocation = nullptr;
  if (!allocations->empty()) {
    prev_allocation = allocations->back().get();
  }
  // Find a previous allocation that is in the default memory space (not
  // necessarily the very last allocation).
  MemorySpaceAssignment::Allocation* prev_allocation_in_default_mem = nullptr;
  for (auto allocation_it = allocations->rbegin();
       allocation_it != allocations->rend(); ++allocation_it) {
    if ((*allocation_it)->memory_space() == MemorySpace::kDefault &&
        (*allocation_it)->defining_position() == defining_position) {
      prev_allocation_in_default_mem = allocation_it->get();
      break;
    }
  }

  // Since copies couldn't be removed, create an allocation in the default
  // memory space.
  if (prev_allocation_in_default_mem != nullptr) {
    if (prev_allocation == prev_allocation_in_default_mem) {
      // The latest allocation is also in the default memory, simply extend
      // that.
      prev_allocation->Extend(end_time);
    } else {
      // The latest allocation is different. Create a new allocation in default
      // memory.
      allocations->push_back(
          absl::make_unique<MemorySpaceAssignment::Allocation>(
              non_bitcast_operand, defining_position, MemorySpace::kDefault,
              kDummyChunk, prev_allocation_in_default_mem->end_time(),
              end_time));
    }
  } else if (prev_allocation != nullptr &&
             prev_allocation->memory_space() == MemorySpace::kAlternate &&
             prev_allocation->defining_position() == defining_position) {
    // If there was an allocation for this HloValue that was in the alternate
    // memory space, we also need to perform an eviction.
    int64 eviction_start_time = prev_allocation->start_time();
    int64 eviction_end_time = prev_allocation->end_time();
    CHECK(eviction_start_time <= eviction_end_time);

    int64 preferred_eviction_end_time = std::max(
        options_.prefetch_interval_picker->PreferredEvictionEndTime(
            non_bitcast_operand->shape(), eviction_start_time, end_time),
        eviction_end_time);

    BufferInterval eviction_mem_interval;
    eviction_mem_interval.buffer = buffer;
    eviction_mem_interval.size = size;
    // Try to reserve a buffer from the end of the previous allocation to the
    // preferred eviction end time.
    eviction_mem_interval.start = prev_allocation->end_time() + 1;
    eviction_mem_interval.end = preferred_eviction_end_time;
    int64 preferred_offset = prev_allocation->chunk().offset;
    VLOG(4) << "Eviction (" << eviction_start_time << ", " << eviction_end_time
            << ") preferred end time = " << preferred_eviction_end_time;

    while (preferred_eviction_end_time > eviction_end_time) {
      ChunkCandidate chunk_candidate =
          FindChunkCandidate(eviction_mem_interval, preferred_offset);
      if (chunk_candidate.chunk.offset == preferred_offset) {
        eviction_end_time = preferred_eviction_end_time;
        AddToPendingChunks(eviction_mem_interval, chunk_candidate);
        break;
      }
      eviction_mem_interval.end = --preferred_eviction_end_time;
    }

    VLOG(3) << "Evicting buffer at " << prev_allocation->chunk().offset << " ("
            << eviction_start_time << ", " << eviction_end_time << ")";

    bool eviction_interval_too_short =
        (eviction_start_time == eviction_end_time);
    bool eviction_violates_outstanding_copies =
        ViolatesMaximumOutstandingAsyncCopies(eviction_start_time,
                                              eviction_end_time);

    // See if this interval would violate the asynchronous copy limit.
    if (!eviction_interval_too_short && !eviction_violates_outstanding_copies) {
      prev_allocation->Extend(eviction_end_time);
      AddAsyncCopy(*prev_allocation, MemorySpace::kDefault, kDummyChunk,
                   eviction_start_time, prev_allocation->end_time(),
                   eviction_end_time, allocations);
    } else {
      if (eviction_violates_outstanding_copies) {
        VLOG(3) << "This violates the maximum async copies.";
      } else {
        VLOG(3) << "Eviction interval is too short (" << eviction_start_time
                << ", " << eviction_end_time << ").";
      }
      // If the original interval violated the limit, try sub-intervals within
      // this interval.
      bool eviction_scheduled = false;
      for (int64 time = eviction_start_time; time < eviction_end_time; ++time) {
        VLOG(3) << "Try evicting (" << time << ", " << time << ")";
        if (!ViolatesMaximumOutstandingAsyncCopies(time, time)) {
          VLOG(3) << "Eviction successful.";
          AddAsyncCopy(*prev_allocation, MemorySpace::kDefault, kDummyChunk,
                       time, time, time, allocations);
          eviction_scheduled = true;
          break;
        }
      }

      if (!eviction_scheduled) {
        // If the eviction couldn't be scheduled, then fail. This buffer will be
        // kept in the default memory.
        VLOG(3) << "Bailing: Could not evict " << use.ToString()
                << " because we hit the limit of maximum asynchronous copies "
                << "between "
                << hlo_live_range_.flattened_instruction_sequence()
                       .instructions()[eviction_start_time]
                << " and "
                << hlo_live_range_.flattened_instruction_sequence()
                       .instructions()[eviction_end_time];
        return false;
      }
    }
  } else {
    allocations->push_back(absl::make_unique<MemorySpaceAssignment::Allocation>(
        non_bitcast_operand, defining_position, MemorySpace::kDefault,
        kDummyChunk, start_time, end_time));
  }

  // If the use requires the buffer to be in default memory, don't try to
  // prefetch.
  if (use_requires_buffer_in_default_mem) {
    VLOG(4)
        << "Not trying to prefetch because use requires buffer in default mem.";
    allocations->back()->AddUse(use);
    return true;
  }

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
  options_.prefetch_interval_picker->Begin(use, start_time,
                                           latest_prefetch_time);
  VLOG(4) << "Trying prefetch picker = "
          << options_.prefetch_interval_picker->ToDebugString();
  while (!options_.prefetch_interval_picker->Done()) {
    alternate_mem_interval.start = options_.prefetch_interval_picker->Next();
    VLOG(4) << "Trying alternate memory allocation ("
            << alternate_mem_interval.start << ", "
            << alternate_mem_interval.end << ")";
    // If this additional asynchronous copy would violate the limit, try a
    // different interval.
    if (ViolatesMaximumOutstandingAsyncCopies(alternate_mem_interval.start,
                                              alternate_mem_interval.end)) {
      VLOG(4) << "This would violate the outstanding async copy limit.";
      continue;
    }
    if (async_copy_ordering_.ViolatesOrdering(alternate_mem_interval.start,
                                              alternate_mem_interval.end)) {
      VLOG(4) << "This would violate asynchronous copy ordering.";
      continue;
    }

    ChunkCandidate chunk_candidate = FindChunkCandidate(alternate_mem_interval);
    // Check if the new heap size fits within limits.
    if (chunk_candidate.heap_size < available_heap_size()) {
      VLOG(3) << "Move the buffer to alternate memory at "
              << alternate_mem_interval.start
              << ". Offset = " << chunk_candidate.chunk.offset
              << ", size = " << chunk_candidate.chunk.size
              << ", heap_size = " << chunk_candidate.heap_size
              << ", prefetch picker = "
              << options_.prefetch_interval_picker->ToDebugString();
      AddToPendingChunks(alternate_mem_interval, chunk_candidate);

      AddAsyncCopy(*allocations->back().get(), MemorySpace::kAlternate,
                   chunk_candidate.chunk, alternate_mem_interval.start,
                   end_time, latest_prefetch_time, allocations);

      allocations->back()->AddUse(use);
      return true;
    }
  }

  // If a copy wasn't inserted, then add this use to the latest allocation.
  allocations->back()->AddUse(use);
  return true;
}

void AlternateMemoryBestFitHeap::AddAsyncCopy(
    const MemorySpaceAssignment::Allocation& prev_allocation,
    MemorySpace memory_space, Chunk chunk, int64 start_time, int64 end_time,
    int64 copy_done_schedule_before_time,
    MemorySpaceAssignment::AllocationSequence* allocations) {
  VLOG(3) << "Copy to "
          << (memory_space == MemorySpaceAssignment::MemorySpace::kDefault
                  ? "default"
                  : "alternate")
          << " memory between " << start_time << " and "
          << copy_done_schedule_before_time << " keeping until " << end_time;

  allocations->push_back(
      absl::make_unique<MemorySpaceAssignment::CopyAllocation>(
          prev_allocation, memory_space, chunk, start_time, end_time,
          copy_done_schedule_before_time));

  // Register the additional async copy with the interval tree to keep track of
  // the limit at any given time.
  pending_async_copies_.push_back({start_time, end_time, memory_space});
}

bool AlternateMemoryBestFitHeap::ViolatesMaximumOutstandingAsyncCopies(
    int64 start_time, int64 end_time) const {
  if (options_.max_outstanding_async_copies < 0) {
    return false;
  }

  // Count both the asynchronous copies in the interval tree as well as the
  // pending asynchronous copies belonging to this buffer.
  int64 num_async_copies =
      async_copy_interval_tree_.ChunksOverlappingInTime(start_time, end_time)
          .size();

  for (const auto& interval : pending_async_copies_) {
    if (interval.start_time > start_time && interval.end_time < end_time) {
      num_async_copies++;
    }
  }
  // Add one because we are checking if adding an additional asynchronous copy
  // would violate the limit.
  return num_async_copies + 1 > options_.max_outstanding_async_copies;
}

bool AlternateMemoryBestFitHeap::TryAllocatingInAlternateMemoryNoCopy(
    int64 start_time, int64 end_time, int64 last_use_time,
    HloPosition defining_position, HloUse use,
    BufferInterval alternate_mem_interval, HloInstruction* non_bitcast_operand,
    MemorySpaceAssignment::AllocationSequence* allocations) {
  MemorySpaceAssignment::Allocation* prev_allocation = nullptr;
  bool can_eliminate_copy = false;
  if (allocations->empty()) {
    // There hasn't been any allocations for this interval so far. We can
    // eliminate copy if the value can be placed in the alternate memory.
    can_eliminate_copy =
        options_.is_allowed_in_alternate_mem_fn(*alternate_mem_interval.buffer);
  } else {
    // If there has been a previous allocation, we can eliminate the copy if the
    // previous allocation was also in the alternate memory.
    prev_allocation = allocations->back().get();
    can_eliminate_copy =
        (prev_allocation->memory_space() == MemorySpace::kAlternate);
  }

  if (!can_eliminate_copy) {
    return false;
  }

  if (!options_.prefetch_interval_picker->CanAllocateInAlternateMemoryNoCopy(
          non_bitcast_operand->shape(), start_time, end_time)) {
    return false;
  }

  alternate_mem_interval.start = start_time;

  // Prefer the offset that was previously used for the previous allocation.
  int64 preferred_offset = -1;
  if (prev_allocation != nullptr) {
    preferred_offset = prev_allocation->chunk().offset;
    // If there is a previous allocation, set the start time one after the end
    // of the previous allocation's end.
    alternate_mem_interval.start = prev_allocation->end_time() + 1;
  }

  VLOG(4) << "We can eliminate copy to alternate memory. Preferred offset = "
          << preferred_offset;
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
  alternate_mem_interval.end = last_use_time;
  ChunkCandidate chunk_candidate =
      FindChunkCandidate(alternate_mem_interval, preferred_offset);
  alternate_mem_interval.end = end_time;
  // Check if the new heap size fits within limits. Also ensure if a
  // preferred offset was provided, that offset was used.
  if (chunk_candidate.heap_size <= available_heap_size() &&
      (preferred_offset == -1 ||
       preferred_offset == chunk_candidate.chunk.offset)) {
    VLOG(3) << "Keep the buffer in alternate memory. Offset = "
            << chunk_candidate.chunk.offset
            << ", size = " << chunk_candidate.chunk.size
            << ", heap_size = " << chunk_candidate.heap_size
            << ", prefetch picker = "
            << options_.prefetch_interval_picker->ToNoCopyDebugString(
                   non_bitcast_operand->shape(), start_time, end_time);
    AddToPendingChunks(alternate_mem_interval, chunk_candidate);

    // If there was a previous allocation, the buffer location is the
    // same as the previous. Otherwise, it is the operand.
    if (prev_allocation != nullptr &&
        (prev_allocation->is_copy_allocation() ||
         prev_allocation->defining_position() == defining_position)) {
      prev_allocation->Extend(end_time);
    } else {
      allocations->push_back(
          absl::make_unique<MemorySpaceAssignment::Allocation>(
              non_bitcast_operand, defining_position, MemorySpace::kAlternate,
              chunk_candidate.chunk, start_time, end_time));
    }
    allocations->back()->AddUse(use);
    return true;
  }
  return false;
}

/*static*/ int64 MemorySpaceAssignment::CountMaximumOutstandingAsyncCopies(
    const HloModule& module) {
  int64 max_copies = 0;
  int64 current_copies = 0;
  for (HloInstruction* instruction :
       module.schedule().sequence(module.entry_computation()).instructions()) {
    if (instruction->opcode() == HloOpcode::kCopyStart) {
      current_copies++;
    } else if (instruction->opcode() == HloOpcode::kCopyDone) {
      current_copies--;
    }
    max_copies = std::max(max_copies, current_copies);
  }
  return max_copies;
}

/*static*/ MemorySpaceAssignment::BufferIntervalCompare
MemorySpaceAssignment::GetMemoryBoundednessBufferIntervalCompare(
    const MemorySpaceAssignmentCostAnalysis& cost_analysis) {
  return [&](const BufferInterval& x, const BufferInterval& y) {
    // Returns a heuristic value that captures how much putting this tensor to
    // the alternate memory would help if the op is memory bound, or otherwise
    // how far off is the op to memory boundedness. The larger this number, the
    // higher priority it will be placed in the alternate memory.
    auto get_alternate_mem_benefit =
        [&](const HloInstruction& instruction,
            float elapsed_time_due_to_alternate_mem) {
          float elapsed_time_due_to_compute =
              cost_analysis.GetInstructionElapsedDueToCompute(instruction);
          float elapsed_time_due_to_memory =
              cost_analysis.GetInstructionElapsedDueToMemory(instruction);
          if (elapsed_time_due_to_memory > elapsed_time_due_to_compute) {
            // Memory bound, return how much alternate memory is better.
            return elapsed_time_due_to_memory -
                   elapsed_time_due_to_alternate_mem;
          } else {
            // Compute bound, return how far off are we to memory boundedness.
            return elapsed_time_due_to_memory - elapsed_time_due_to_compute;
          }
        };

    auto get_memory_boundedness = [&](const BufferInterval& interval) {
      const HloInstruction& defining_instruction =
          *interval.buffer->defining_instruction();
      float alternate_mem_benefit = get_alternate_mem_benefit(
          defining_instruction, cost_analysis.GetInstructionElapsedDueToMemory(
                                    defining_instruction,
                                    /*operand_in_alternate_mem=*/{},
                                    /*output_in_alternate_mem=*/true));
      for (const HloUse& use : interval.buffer->uses()) {
        float use_alternate_mem_benefit = get_alternate_mem_benefit(
            *use.instruction, cost_analysis.GetInstructionElapsedDueToMemory(
                                  *use.instruction, use.operand_number));
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
      return alternate_mem_benefit;
    };

    float x_memory_boundedness = get_memory_boundedness(x);
    float y_memory_boundedness = get_memory_boundedness(y);
    if (x_memory_boundedness != y_memory_boundedness) {
      return x_memory_boundedness > y_memory_boundedness;
    }
    // Tie-break if the memory boundedness is the same.
    return GlobalDecreasingSizeBestFitHeap::GetSpatialBufferIntervalCompare()(
        x, y);
  };
}

/*static*/ StatusOr<std::unique_ptr<PresetAssignments>>
MemorySpaceAssignment::Run(HloModule* module, const Options& options) {
  CHECK(module->has_schedule());
  VLOG(4) << "Module before memory space assignment: ";
  XLA_VLOG_LINES(4, module->ToString());
  VLOG(4) << "Schedule: " << module->schedule().ToString();
  TF_ASSIGN_OR_RETURN(auto alias_analysis, HloAliasAnalysis::Run(module));

  const HloComputation* entry_computation = module->entry_computation();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloLiveRange> hlo_live_range,
                      HloLiveRange::Run(module->schedule(), *alias_analysis,
                                        entry_computation));
  MemorySpaceAssignment memory_space_assignment(
      module, options.alternate_memory_space, *hlo_live_range);
  auto algorithm = absl::make_unique<AlternateMemoryBestFitHeap>(
      &memory_space_assignment.allocation_map_, options, *alias_analysis,
      *hlo_live_range);

  HeapSimulator::Options heap_simulator_options;
  heap_simulator_options.may_reuse_operand_buffers = false;
  TF_RETURN_IF_ERROR(HeapSimulator::Run(std::move(algorithm), *module,
                                        module->schedule(),
                                        *alias_analysis.get(), options.size_fn,
                                        heap_simulator_options)
                         .status());

  TF_RETURN_IF_ERROR(memory_space_assignment.Process());
  memory_space_assignment.ScheduleAsynchronousCopies();
  TF_RETURN_IF_ERROR(memory_space_assignment.SimplifyGraph());
  TF_RETURN_IF_ERROR(memory_space_assignment.FixSchedule());

  VLOG(4) << "Module after memory space assignment: ";
  XLA_VLOG_LINES(4, module->ToString());
  TF_CHECK_OK(module->schedule().Verify());
  VLOG(1) << "Maximum number of outstanding async copies: "
          << CountMaximumOutstandingAsyncCopies(*module);

  if (options.verify || VLOG_IS_ON(1)) {
    TF_RETURN_IF_ERROR(memory_space_assignment.Verify());
  }

  return std::move(memory_space_assignment.preset_assignments_);
}

void MemorySpaceAssignment::Allocation::AddUse(HloUse use) {
  HloInstruction* operand =
      use.instruction->mutable_operand(use.operand_number);
  // If the use is a tuple, look inside the tuple to find the actual use.
  for (int64 index : use.operand_index) {
    if (operand->opcode() != HloOpcode::kTuple) {
      break;
    }
    operand = operand->mutable_operand(index);
  }
  // When the operand of a use is a bitcast, we place the bitcast in a separate
  // data structure.
  if (operand->opcode() == HloOpcode::kBitcast) {
    bitcasts_.push_back(operand);
  } else {
    uses_.push_back(use);
  }
}

Status MemorySpaceAssignment::Allocation::Process(
    MemorySpaceAssignment* memory_space_assignment) {
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

  HloComputation* computation = new_instruction->parent();
  std::vector<HloInstruction*> tuple_args(tuple_shape.tuple_shapes_size());
  for (int64 i = 0; i < tuple_shape.tuple_shapes_size(); ++i) {
    const Shape& subshape = tuple_shape.tuple_shapes(i);
    if (i == shape_index[0]) {
      // If the subshape is still a tuple, recurse and pass a new shape index
      // for the one level deeper.
      if (subshape.IsTuple()) {
        HloInstruction* get_tuple_element = computation->AddInstruction(
            HloInstruction::CreateGetTupleElement(subshape, tuple, i));
        TF_ASSIGN_OR_RETURN(tuple_args[i],
                            ReplaceTupleWith(new_instruction, get_tuple_element,
                                             ShapeIndex(shape_index.begin() + 1,
                                                        shape_index.end())));
      } else {
        tuple_args[i] = new_instruction;
      }
    } else {
      HloInstruction* get_tuple_element = computation->AddInstruction(
          HloInstruction::CreateGetTupleElement(subshape, tuple, i));
      tuple_args[i] = get_tuple_element;
    }
  }
  return computation->AddInstruction(HloInstruction::CreateTuple(tuple_args));
}

Status MemorySpaceAssignment::CopyAllocation::Process(
    MemorySpaceAssignment* memory_space_assignment) {
  // Copy allocations need to insert asynchronous copy nodes.
  HloInstruction* producing_instruction = defining_position().instruction;
  CHECK_NE(producing_instruction, nullptr);

  Shape shape = defining_position().shape();
  CHECK(shape.IsArray()) << "CopyAllocation shape is not an array. Shape = "
                         << shape.ToString()
                         << " position = " << defining_position().shape();
  HloComputation* computation = producing_instruction->parent();

  // If the instruction we're copying from is a tuple, we (recursively) create
  // kGetTupleElement instructions and copy that value. Asynchronous copies only
  // support array types.
  if (!producing_instruction->shape().IsArray()) {
    producing_instruction = defining_position().instruction;
    for (int64 index : defining_position().index) {
      producing_instruction =
          computation->AddInstruction(HloInstruction::CreateGetTupleElement(
              producing_instruction->shape().tuple_shapes(index),
              producing_instruction, index));
    }
  }
  copy_start_ = computation->AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeTupleShape({shape, ShapeUtil::MakeShape(U32, {})}),
      HloOpcode::kCopyStart, producing_instruction));
  copy_done_ = computation->AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCopyDone, copy_start_));
  // Update the allocation with the copy done instruction so that if there
  // are further copies from it, it can find the correct instruction.
  instruction_ = copy_done_;

  // Also update the defining position.
  defining_position_ = HloPosition{copy_done_, {}};

  // Replace all the uses with the new copy instruction.
  for (HloUse use : uses_) {
    // If the operand is a tuple, we need to descend to the actual instruction
    // we want to replace.
    HloInstruction* replacement_instruction;
    if (use.instruction->operand(use.operand_number)->shape().IsTuple()) {
      TF_ASSIGN_OR_RETURN(
          replacement_instruction,
          ReplaceTupleWith(copy_done_,
                           use.instruction->mutable_operand(use.operand_number),
                           use.operand_index));
    } else {
      replacement_instruction = copy_done_;
    }
    TF_RETURN_IF_ERROR(use.instruction->ReplaceOperandWith(
        use.operand_number, replacement_instruction));
  }

  // Replace all the bitcasts with the new copy instruction. Note that if there
  // is a chain of bitcasts, their operands will be replaced with copy done.
  // For example:
  //
  // a = Foo()
  // b = Bitcast(a)
  // c = Bitcast(b)
  //
  // If a is moved to the alternate memory asynchronously, the graph will be
  // changed into:
  //
  // a = Foo()
  // cs = CopyStart(a)
  // cd = CopyDone(cs)
  // b = Bitcast(cd)
  // c = Bitcast(cd)
  //
  // Because of the potential shape change in the operand (b -> cd), we use
  // ReplaceOperandWithDifferentShape.
  for (HloInstruction* bitcast : bitcasts_) {
    TF_RETURN_IF_ERROR(bitcast->ReplaceOperandWithDifferentShape(
        /*operand_num=*/0, copy_done_));
  }

  return Status::OK();
}

Status MemorySpaceAssignment::Process() {
  // Insert CopyStart/CopyDone pairs.
  int64 alternate_memory_size = 0;
  for (auto& buffer_and_sequence : allocation_map_) {
    for (auto& allocation : buffer_and_sequence.second) {
      TF_RETURN_IF_ERROR(allocation->Process(this));
      // Add the offset and size of the allocation in the alternate memory to
      // the output map. Special case for bitcast: since bitcast doesn't define
      // its own buffer, that shouldn't be exported as a preset chunk.
      if (allocation->memory_space() == MemorySpace::kAlternate &&
          allocation->instruction()->opcode() != HloOpcode::kBitcast) {
        preset_assignments_->add_chunk(allocation->defining_position(),
                                       allocation->chunk());
        alternate_memory_size =
            std::max(alternate_memory_size, allocation->chunk().chunk_end());
      }
    }
  }

  if (!preset_assignments_->chunks().empty()) {
    preset_assignments_->add_size(alternate_memory_space_,
                                  alternate_memory_size);
  }

  if (VLOG_IS_ON(3)) {
    VLOG(3) << "Exported alternate memory allocations:";
    for (auto& pair : preset_assignments_->chunks()) {
      VLOG(3) << " [" << pair.second.offset << ", " << pair.second.size
              << "] : " << pair.first.ToString();
    }
    VLOG(3) << "Exported alternate memory sizes:";
    for (auto& pair : preset_assignments_->sizes()) {
      VLOG(3) << "  space: " << pair.first << ", size: " << pair.second;
    }
  }

  // Color the pending positions and all of their aliased buffers.
  TF_ASSIGN_OR_RETURN(auto alias_analysis, HloAliasAnalysis::Run(module_));
  for (const auto& defining_position_and_chunk :
       preset_assignments_->chunks()) {
    const HloPosition& defining_position = defining_position_and_chunk.first;
    for (auto& buffer : alias_analysis->ComputeBuffersAt(
             defining_position.instruction, defining_position.index)) {
      for (auto& value : buffer->values()) {
        for (auto& position : value->positions()) {
          VLOG(3) << "Coloring " << position.ToString();
          Shape* shape = ShapeUtil::GetMutableSubshape(
              position.instruction->mutable_shape(), position.index);
          CHECK(shape->IsArray()) << "Coloring a shape that is not an array: "
                                  << position.ToString();
          shape->mutable_layout()->set_memory_space(alternate_memory_space_);
        }
      }
    }
  }

  return Status::OK();
}

void PresetAssignments::RemoveAssignmentForInstruction(
    const HloInstruction* instruction) {
  for (auto& position_and_chunk : chunks_) {
    const HloPosition& position = position_and_chunk.first;
    if (position.instruction == instruction) {
      VLOG(3) << "Removing instruction from preset assignments.";
      // Swap the removed position and chunk with the back and pop back.
      position_and_chunk = chunks_.back();
      chunks_.pop_back();
      break;
    }
  }
}

Status MemorySpaceAssignment::SimplifyGraph() {
  for (HloComputation* computation : module_->MakeNonfusionComputations()) {
    // Parallel computations aren't in the schedule and don't need to be
    // modified.
    if (!computations_in_schedule_.contains(computation)) {
      VLOG(4) << "Not simplifying " << computation->name()
              << " because it's not in the schedule.";
      continue;
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
            instruction != computation->root_instruction()) {
          VLOG(4) << "Instruction removed: " << instruction->ToString();
          // Ensure the exported preset assignments don't contain a reference to
          // the removed instruction.
          preset_assignments_->RemoveAssignmentForInstruction(instruction);
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
        }
      }
    }
  }

  return Status::OK();
}

void MemorySpaceAssignment::EnsureInstructionAndOperandsInserted(
    HloInstruction* new_instruction, HloInstructionSequence* new_sequence,
    absl::flat_hash_set<HloInstruction*>* inserted_instructions) const {
  if (inserted_instructions->contains(new_instruction)) {
    return;
  }
  for (HloInstruction* operand : new_instruction->operands()) {
    // CopyStart/CopyDone dependencies should always be already inserted; it is
    // a red flag when they haven't already been inserted.
    CHECK((operand->opcode() != HloOpcode::kCopyStart &&
           operand->opcode() != HloOpcode::kCopyDone) ||
          inserted_instructions->contains(operand))
        << "Inserted instruction " << new_instruction->ToString()
        << " has un-inserted dependency: " << operand->ToString();
    EnsureInstructionAndOperandsInserted(operand, new_sequence,
                                         inserted_instructions);
  }
  VLOG(4) << "inserting: " << new_instruction->ToShortString();
  new_sequence->push_back(new_instruction);
  inserted_instructions->insert(new_instruction);
}

void MemorySpaceAssignment::ScheduleAsynchronousCopies() {
  for (MemorySpace memory_space :
       {MemorySpace::kDefault, MemorySpace::kAlternate}) {
    std::vector<CopyAllocation*> copy_allocations;
    for (auto& buffer_and_sequence : allocation_map_) {
      for (auto& allocation : buffer_and_sequence.second) {
        if (allocation->is_copy_allocation()) {
          auto copy_allocation = static_cast<CopyAllocation*>(allocation.get());
          if (copy_allocation->memory_space() == memory_space) {
            copy_allocations.push_back(copy_allocation);
          }
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

    CopyAllocation* prev_copy_allocation = nullptr;
    for (CopyAllocation* copy_allocation : copy_allocations) {
      // If the copy start doesn't happen to be scheduled at the correct
      // computation, delay it until the correct computation starts.
      int64 copy_start_schedule_after =
          copy_allocation->copy_start_schedule_after();
      // Accessing flattened_instructions_ here without checking if it is
      // nullptr is safe because this method is called before SimplifyGraph.
      while (copy_allocation->instruction()->parent() !=
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
      prev_copy_allocation = copy_allocation;
    }
  }
}

Status MemorySpaceAssignment::FixSchedule() {
  CHECK(module_->has_schedule());
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
    CHECK(schedule.is_computation_scheduled(computation));
    HloInstructionSequence new_sequence;

    absl::flat_hash_set<HloInstruction*> inserted_instructions;

    VLOG(4) << "Scheduling: " << computation->ToString();

    for (int64 instruction_index = 0;
         instruction_index < flattened_instructions_.size();
         ++instruction_index) {
      auto insts_before_iter = schedule_before_.find(instruction_index);
      if (insts_before_iter != schedule_before_.end()) {
        for (HloInstruction* new_instruction : insts_before_iter->second) {
          if (new_instruction->parent() == computation) {
            EnsureInstructionAndOperandsInserted(new_instruction, &new_sequence,
                                                 &inserted_instructions);
          }
        }
      }
      HloInstruction* instruction = flattened_instructions_[instruction_index];
      // Insert only if it is not deleted (SimplifyGraph sets it to nullptr if
      // it was deleted) and not previously inserted. Also bitcasts and tuples
      // are treated specially and only inserted as a result of operand
      // dependencies.
      if (instruction != nullptr &&
          !inserted_instructions.contains(instruction) &&
          instruction->parent() == computation &&
          instruction->opcode() != HloOpcode::kBitcast &&
          instruction->opcode() != HloOpcode::kTuple) {
        EnsureInstructionAndOperandsInserted(instruction, &new_sequence,
                                             &inserted_instructions);
      }
      auto insts_after_iter = schedule_after_.find(instruction_index);
      if (insts_after_iter != schedule_after_.end()) {
        for (HloInstruction* new_instruction : insts_after_iter->second) {
          if (new_instruction->parent() == computation) {
            EnsureInstructionAndOperandsInserted(new_instruction, &new_sequence,
                                                 &inserted_instructions);
          }
        }
      }
    }
    // For rare cases where the original sequence is empty, ensure the root
    // instruction and its dependencies are scheduled.
    EnsureInstructionAndOperandsInserted(computation->root_instruction(),
                                         &new_sequence, &inserted_instructions);
    CHECK_EQ(new_sequence.size(), computation->instruction_count())
        << "New sequence for computation " << computation->name() << " has "
        << new_sequence.size() << " instructions, expects "
        << computation->instruction_count() << ".";
    schedule.set_sequence(computation, new_sequence);
  }

  return Status::OK();
}

Status MemorySpaceAssignment::Verify() const {
  VLOG(3) << "Verifying:";
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                      HloAliasAnalysis::Run(module_));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloLiveRange> hlo_live_range,
                      HloLiveRange::Run(module_->schedule(), *alias_analysis,
                                        module_->entry_computation()));

  BufferIntervalTree interval_tree;
  absl::flat_hash_set<int64> seen_buffers;

  for (const auto& position_and_chunk : preset_assignments_->chunks()) {
    const HloPosition& position = position_and_chunk.first;
    const Chunk& chunk = position_and_chunk.second;
    const HloBuffer& buffer =
        alias_analysis->GetUniqueBufferAt(position.instruction, position.index);
    if (seen_buffers.contains(buffer.id())) {
      continue;
    }
    seen_buffers.insert(buffer.id());

    int64 start_time = INT64_MAX;
    int64 end_time = -1;
    for (const HloValue* value : buffer.values()) {
      const HloLiveRange::TimeBound& time_bound =
          hlo_live_range->buffer_live_ranges().at(value);
      start_time = std::min(start_time, time_bound.start);
      end_time = std::max(end_time, time_bound.end);
    }
    CHECK_GE(start_time, 0);
    CHECK_GT(end_time, 0);
    // Get the chunks overlapping in time and search if they overlap in space as
    // well.
    // TODO(berkin): For now checking against end_time - 1 (exclusive), but we
    // really should check against end_time (inclusive) for cases where the
    // operand can't share buffer with user (see
    // HloDataflowAnalysis::CanShareOperandBufferWithUser).
    for (const Chunk& overlapping_chunk :
         interval_tree.ChunksOverlappingInTime(start_time, end_time - 1)) {
      if (chunk.OverlapsWith(overlapping_chunk)) {
        return InternalError(
            ("Buffer %s (%d, %d) off: %d size: %d overlaps with another chunk"
             " off: %d size: %d"),
            buffer.ToString(), start_time, end_time, chunk.offset, chunk.size,
            overlapping_chunk.offset, overlapping_chunk.size);
      }
    }
    interval_tree.Add(start_time, end_time - 1, chunk);
    VLOG(3) << " buffer: " << buffer.ToString() << ": (" << start_time << ", "
            << end_time << ") off: " << position_and_chunk.second.offset
            << ", size: " << position_and_chunk.second.size;
  }

  return Status::OK();
}

}  // namespace xla

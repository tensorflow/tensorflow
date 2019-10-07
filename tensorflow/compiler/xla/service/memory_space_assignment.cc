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
          << max_size_in_bytes_
          << ", min prefetch interval = " << min_prefetch_interval_
          << ", max prefetch interval = " << max_prefetch_interval_;

  AddInputAndOutputRequiredAssignments();

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
    bool keep_in_default_memory = false;
    for (const BufferInterval* colocated_interval : colocated_intervals) {
      const HloValue* value = colocated_interval->buffer;
      // If any of the colocated values are phi buffers, we keep them in the
      // default memory for now.
      if (value->is_phi()) {
        keep_in_default_memory = true;
        VLOG(4) << "Keeping value " << value->ToShortString()
                << " because it contains a phi node.";
        break;
      }
    }

    // At this point, none of the colocated buffers contain any phi buffers.
    for (const BufferInterval* colocated_interval : colocated_intervals) {
      if (keep_in_default_memory) {
        break;
      }
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
      // Iterate over the uses.
      for (HloUse use : uses) {
        int64 use_time = instruction_schedule.at(use.instruction);
        int64 last_use_time = instruction_schedule.at(uses.back().instruction);

        // Bitcasts don't define buffers and don't directly consume buffers.
        // Skip allocating buffers for bitcast uses. The uses that feed from
        // bitcasts will be handled specially.
        if (use.instruction->opcode() != HloOpcode::kBitcast) {
          if (!FindAllocation(definition_time, use_time, last_use_time,
                              value->defining_position(), use, value,
                              colocated_interval->size, allocation_sequence)) {
            // If the allocation finding failed (e.g., due to running out of
            // asynchronous copies), then fall back to allocating the buffer
            // entirely in the default memory.
            pending_chunks_.clear();
            pending_async_copies_.clear();
            allocation_sequence->clear();
            keep_in_default_memory = true;
            break;
          }

          // If there are multiple uses, they can try using the memory
          // allocation already at the alternate memory.
          definition_time = use_time;
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

void AlternateMemoryBestFitHeap::AddInputAndOutputRequiredAssignments() {
  // Go through the parameters and outputs and pin them to default memory by
  // adding a required assignment.
  // TODO(berkin): If these values are already marked alternate memory, use
  // those instead.
  const HloDataflowAnalysis& dataflow_analysis =
      alias_analysis_.dataflow_analysis();
  const HloModule& module = dataflow_analysis.module();
  const auto& instruction_schedule = hlo_live_range_.instruction_schedule();
  HloComputation* entry_computation = module.entry_computation();
  for (HloInstruction* parameter_instruction :
       entry_computation->parameter_instructions()) {
    int64 parameter_instruction_time =
        instruction_schedule.at(parameter_instruction);
    ShapeUtil::ForEachSubshape(
        parameter_instruction->shape(),
        [&](const Shape& /*subshape*/, const ShapeIndex& index) {
          for (const HloValue* value :
               dataflow_analysis.GetValueSet(parameter_instruction, index)
                   .values()) {
            VLOG(3) << "Adding required assignment for parameter value = "
                    << value->ToShortString()
                    << " time = " << parameter_instruction_time;
            required_assignments_[value].push_back(
                {/*memory_space=*/MemorySpace::kDefault,
                 /*time=*/parameter_instruction_time});
          }
        });
  }
  HloInstruction* root_instruction = entry_computation->root_instruction();
  int64 root_instruction_time = instruction_schedule.at(root_instruction);
  ShapeUtil::ForEachSubshape(
      root_instruction->shape(),
      [&](const Shape& /*subshape*/, const ShapeIndex& index) {
        for (const HloValue* value :
             dataflow_analysis.GetValueSet(root_instruction, index).values()) {
          VLOG(3) << "Adding required assignment for output value = "
                  << value->ToShortString()
                  << " time = " << root_instruction_time;
          required_assignments_[value].push_back(
              {/*memory_space=*/MemorySpace::kDefault,
               /*time=*/root_instruction_time});
        }
      });
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
  if (max_outstanding_async_copies_ >= 0) {
    for (auto interval : pending_async_copies_) {
      async_copy_interval_tree_.Add(interval.first, interval.second,
                                    kDummyChunk);
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
    HloPosition defining_position, HloUse use, const HloValue* buffer,
    int64 size, MemorySpaceAssignment::AllocationSequence* allocations) {
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
  alternate_mem_interval.start =
      std::max(start_time, end_time - max_prefetch_interval_);
  alternate_mem_interval.end = end_time;

  VLOG(2) << "Finding allocation for " << buffer->ToShortString() << " ("
          << start_time << ", " << end_time << ") last use = " << last_use_time
          << " use = " << use.ToString() << ". Size = " << size
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
      VLOG(3) << "Required assignment at time = " << required_assignment.time;
      // TODO(berkin): Handle memory requirements for alternate memory space.
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

  if (defining_position.instruction->parent() != use.instruction->parent() ||
      (!use.instruction->called_computations().empty() &&
       use.instruction->opcode() != HloOpcode::kFusion)) {
    VLOG(3) << "Use is in a different computation or calls a computation.";
    // Fail because we do not allow asynchronous copies while in the bodies of
    // other computation.
    return false;
  }

  MemorySpaceAssignment::Allocation* prev_allocation = nullptr;
  if (!allocations->empty()) {
    prev_allocation = allocations->back().get();
  }

  // Since copies couldn't be removed, create an allocation in the default
  // memory space.
  if (prev_allocation != nullptr &&
      prev_allocation->memory_space() == MemorySpace::kAlternate &&
      prev_allocation->defining_position() == defining_position) {
    // If there was an allocation for this HloValue that was in the alternate
    // memory space, we also need to perform an eviction.
    // TODO(berkin): For now evictions happen relative to the most recent
    // allocation in the alternate memory. We can potentially start evictions
    // earlier and end later.
    VLOG(3) << "Evicting buffer at " << prev_allocation->chunk().offset << " ("
            << prev_allocation->start_time() << ", "
            << prev_allocation->end_time() << ")";

    // See if this interval would violate the asynchronous copy limit.
    if (!ViolatesMaximumOutstandingAsyncCopies(prev_allocation->start_time(),
                                               prev_allocation->end_time())) {
      AddAsyncCopy(*prev_allocation, MemorySpace::kDefault, kDummyChunk,
                   prev_allocation->start_time(), prev_allocation->end_time(),
                   allocations);

    } else {
      VLOG(3) << "This violates the maximum async copies.";
      // If the original interval violated the limit, try sub-intervals within
      // this interval.
      bool eviction_scheduled = false;
      for (int64 time = prev_allocation->start_time();
           time <= prev_allocation->end_time(); ++time) {
        VLOG(3) << "Try evicting (" << time << ", " << time << ")";
        if (!ViolatesMaximumOutstandingAsyncCopies(time, time)) {
          VLOG(3) << "Eviction successful.";
          AddAsyncCopy(*prev_allocation, MemorySpace::kDefault, kDummyChunk,
                       time, time, allocations);
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
                       .instructions()[prev_allocation->start_time()]
                << " and "
                << hlo_live_range_.flattened_instruction_sequence()
                       .instructions()[prev_allocation->end_time()];
        return false;
      }
    }
  } else if (prev_allocation != nullptr &&
             prev_allocation->memory_space() == MemorySpace::kDefault &&
             prev_allocation->defining_position() == defining_position) {
    // If the previous allocation was in the default memory space and was
    // defined by the same instruction, extend that.  Otherwise, create a new
    // allocation.
    prev_allocation->Extend(end_time);
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
  for (alternate_mem_interval.start =
           std::max(start_time, end_time - max_prefetch_interval_);
       alternate_mem_interval.end - alternate_mem_interval.start >
       min_prefetch_interval_;
       ++alternate_mem_interval.start) {
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
    ChunkCandidate chunk_candidate = FindChunkCandidate(alternate_mem_interval);
    // Check if the new heap size fits within limits.
    if (chunk_candidate.heap_size < max_size_in_bytes_) {
      VLOG(3) << "Move the buffer to alternate memory at "
              << alternate_mem_interval.start
              << ". Offset = " << chunk_candidate.chunk.offset
              << ", size = " << chunk_candidate.chunk.size
              << ", heap_size = " << chunk_candidate.heap_size;
      AddToPendingChunks(alternate_mem_interval, chunk_candidate);

      AddAsyncCopy(*allocations->back().get(), MemorySpace::kAlternate,
                   chunk_candidate.chunk, alternate_mem_interval.start,
                   end_time, allocations);

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
    MemorySpaceAssignment::AllocationSequence* allocations) {
  VLOG(3) << "Copy to "
          << (memory_space == MemorySpaceAssignment::MemorySpace::kDefault
                  ? "default"
                  : "alternate")
          << " memory between " << start_time << " and " << end_time;

  allocations->push_back(
      absl::make_unique<MemorySpaceAssignment::CopyAllocation>(
          prev_allocation, memory_space, chunk, start_time, end_time));

  // Register the additional async copy with the interval tree to keep track of
  // the limit at any given time.
  pending_async_copies_.emplace_back(start_time, end_time);
}

bool AlternateMemoryBestFitHeap::ViolatesMaximumOutstandingAsyncCopies(
    int64 start_time, int64 end_time) const {
  if (max_outstanding_async_copies_ < 0) {
    return false;
  }

  // Count both the asynchronous copies in the interval tree as well as the
  // pending asynchronous copies belonging to this buffer.
  int64 num_async_copies =
      async_copy_interval_tree_.ChunksOverlappingInTime(start_time, end_time)
          .size();

  for (auto interval : pending_async_copies_) {
    if (interval.second > start_time && interval.first < end_time) {
      num_async_copies++;
    }
  }
  // Add one because we are checking if adding an additional asynchronous copy
  // would violate the limit.
  return num_async_copies + 1 > max_outstanding_async_copies_;
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
        is_allowed_in_alternate_mem_(*alternate_mem_interval.buffer);
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

  if (alternate_mem_interval.start != start_time) {
    return false;
  }

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
  if (chunk_candidate.heap_size < max_size_in_bytes_ &&
      (preferred_offset == -1 ||
       preferred_offset == chunk_candidate.chunk.offset)) {
    VLOG(3) << "Keep the buffer in alternate memory. Offset = "
            << chunk_candidate.chunk.offset
            << ", size = " << chunk_candidate.chunk.size
            << ", heap_size = " << chunk_candidate.heap_size;
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

/*static*/ StatusOr<std::unique_ptr<PresetAssignments>>
MemorySpaceAssignment::Run(
    HloModule* module, int64 alternate_memory_space, int64 max_size_in_bytes,
    int64 min_prefetch_interval, int64 max_prefetch_interval,
    int64 alternate_memory_space_alignment_in_bytes,
    BufferValue::SizeFunction size_fn,
    AlternateMemoryBestFitHeap::IsAllowedInAlternateMemoryFunction
        is_allowed_in_alternate_mem,
    int64 max_outstanding_async_copies) {
  CHECK(module->has_schedule());
  VLOG(4) << "Module before memory space assignment: ";
  XLA_VLOG_LINES(4, module->ToString());
  VLOG(4) << "Schedule: " << module->schedule().ToString();
  TF_ASSIGN_OR_RETURN(auto alias_analysis, HloAliasAnalysis::Run(module));

  MemorySpaceAssignment memory_space_assignment(module, alternate_memory_space);
  const HloComputation* entry_computation = module->entry_computation();
  TF_ASSIGN_OR_RETURN(memory_space_assignment.hlo_live_range_,
                      HloLiveRange::Run(module->schedule(), *alias_analysis,
                                        entry_computation));
  // TODO(berkin): Explore heap algorithms other than kSpatial.
  auto algorithm = absl::make_unique<AlternateMemoryBestFitHeap>(
      &memory_space_assignment.allocation_map_, max_size_in_bytes,
      min_prefetch_interval, max_prefetch_interval, *alias_analysis,
      *memory_space_assignment.hlo_live_range_,
      alternate_memory_space_alignment_in_bytes,
      GlobalDecreasingSizeBestFitHeap::Type::kSpatial,
      is_allowed_in_alternate_mem, max_outstanding_async_copies);

  TF_RETURN_IF_ERROR(HeapSimulator::Run(std::move(algorithm), *module,
                                        module->schedule(),
                                        *alias_analysis.get(), size_fn)
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

  return std::move(memory_space_assignment.preset_assignments_);
}

void MemorySpaceAssignment::Allocation::AddUse(HloUse use) {
  HloInstruction* operand =
      use.instruction->mutable_operand(use.operand_number);
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
  // For non-copy allocations, all we need to do is to update the output memory
  // space if placed in the alternate memory.
  if (memory_space_ == MemorySpace::kAlternate) {
    memory_space_assignment->AddPositionInAlternateMemorySpace(
        defining_position());
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

  if (memory_space_ == MemorySpace::kAlternate) {
    memory_space_assignment->AddPositionInAlternateMemorySpace({copy_done_});
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
  for (HloPosition defining_position : pending_positions_in_alternate_mem_) {
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
          // Ensure the exported preset assignments don't contain a refence to
          // the removed instruction.
          preset_assignments_->RemoveAssignmentForInstruction(instruction);
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
    EnsureInstructionAndOperandsInserted(operand, new_sequence,
                                         inserted_instructions);
  }
  VLOG(4) << "inserting: " << new_instruction->ToShortString();
  new_sequence->push_back(new_instruction);
  inserted_instructions->insert(new_instruction);
}

void MemorySpaceAssignment::AddPositionInAlternateMemorySpace(
    HloPosition position) {
  pending_positions_in_alternate_mem_.push_back(position);
}

void MemorySpaceAssignment::ScheduleAsynchronousCopies() {
  // For asynchronous copies of both directions (default to alternate and vice
  // versa), sort them by their completion time. Then, if in the sorted order we
  // see that the start time is earlier than the start time of an asynchronous
  // copy that ends earlier, we delay the start of this. As a result, given
  // asynchronous copies that might look like:
  //
  //   CS          CD
  // a +-----------+
  // b    +-----------+
  // c  +---------+
  //
  // We'll first sort by completion time:
  //
  // c  +---------+
  // a +-----------+
  // b    +-----------+
  //
  // Then, delay a because c starts later than a despite also ending earlier:
  //
  // c  +---------+
  // a   +---------+
  // b    +-----------+
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
      if (prev_copy_allocation &&
          prev_copy_allocation->copy_start_schedule_after() >
              copy_allocation->copy_start_schedule_after()) {
        VLOG(4) << "Delaying CopyStart ("
                << copy_allocation->copy_start_schedule_after() << " to "
                << prev_copy_allocation->copy_start_schedule_after() << ") for "
                << copy_allocation->copy_start()->ToString() << " because of "
                << prev_copy_allocation->copy_start()->ToString();
        copy_allocation->set_copy_start_schedule_after(
            prev_copy_allocation->copy_start_schedule_after());
      }

      // If the copy start doesn't happen to be scheduled at the correct
      // computation, delay it until the correct computation starts.
      const auto& flattened_instructions =
          hlo_live_range_->flattened_instruction_sequence().instructions();
      int64 copy_start_schedule_after =
          copy_allocation->copy_start_schedule_after();
      while (copy_allocation->instruction()->parent() !=
             flattened_instructions[copy_start_schedule_after]->parent()) {
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
    CHECK(schedule.is_computation_scheduled(computation));
    const HloInstructionSequence& sequence =
        hlo_live_range_->flattened_instruction_sequence();
    HloInstructionSequence new_sequence;

    absl::flat_hash_set<HloInstruction*> inserted_instructions;

    VLOG(4) << "Scheduling: " << computation->ToString();

    for (int64 instruction_index = 0;
         instruction_index < sequence.instructions().size();
         ++instruction_index) {
      HloInstruction* instruction = sequence.instructions()[instruction_index];
      if (!computation->ContainsInstruction(instruction)) {
        continue;
      }
      auto insts_before_iter = schedule_before_.find(instruction_index);
      if (insts_before_iter != schedule_before_.end()) {
        for (HloInstruction* new_instruction : insts_before_iter->second) {
          EnsureInstructionAndOperandsInserted(new_instruction, &new_sequence,
                                               &inserted_instructions);
        }
      }
      // Insert only if not previously inserted.
      if (!inserted_instructions.contains(instruction)) {
        EnsureInstructionAndOperandsInserted(instruction, &new_sequence,
                                             &inserted_instructions);
      }
      auto insts_after_iter = schedule_after_.find(instruction_index);
      if (insts_after_iter != schedule_after_.end()) {
        for (HloInstruction* new_instruction : insts_after_iter->second) {
          EnsureInstructionAndOperandsInserted(new_instruction, &new_sequence,
                                               &inserted_instructions);
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

}  // namespace xla

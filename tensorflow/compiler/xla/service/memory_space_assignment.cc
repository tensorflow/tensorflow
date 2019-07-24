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
// space.
const HeapSimulator::Chunk kDefaultMemorySpaceDummyChunk{-1, -1};
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

  for (auto& interval : sorted_buffer_intervals) {
    if (!interval.need_allocation) {
      continue;
    }

    // Skip if we have already allocated for this buffer.
    const HloBuffer& buffer =
        alias_analysis_.GetBufferContainingValue(*interval.buffer);
    if (allocation_map_->contains(&buffer)) {
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

    MemorySpaceAssignment::AllocationSequence* allocation_sequence =
        &(*allocation_map_)[&buffer];
    if (keep_in_default_memory) {
      continue;
    }

    // At this point, none of the colocated buffers contain any phi buffers.
    for (const BufferInterval* colocated_interval : colocated_intervals) {
      const HloValue* value = colocated_interval->buffer;
      int64 definition_time =
          instruction_schedule_->at(value->defining_instruction());
      // Iterate over the uses.
      for (HloUse use : value->uses()) {
        int64 use_time = instruction_schedule_->at(use.instruction);

        FindAllocation(definition_time, use_time, use, *colocated_interval,
                       allocation_sequence);
        // If there are multiple uses, they can try using the memory allocation
        // already at the alternate memory.
        definition_time = use_time;
      }
    }
  }

  if (VLOG_IS_ON(3)) {
    for (const auto& alloc_pair : *allocation_map_) {
      VLOG(3) << "Allocation for " << alloc_pair.first->ToString();
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

HloInstruction* AlternateMemoryBestFitHeap::GetInstructionAt(int64 time) const {
  return flattened_instruction_sequence_->instructions()[time];
}

void AlternateMemoryBestFitHeap::FindAllocation(
    int64 start_time, int64 end_time, HloUse use,
    const BufferInterval& interval,
    MemorySpaceAssignment::AllocationSequence* allocations) {
  HloInstruction* def_instruction =
      use.instruction->mutable_operand(use.operand_number);
  // Create an alternate memory interval that starts at the earliest
  // possible position, given by max_prefetch_interval.
  BufferInterval alternate_mem_interval;
  alternate_mem_interval.buffer = interval.buffer;
  alternate_mem_interval.size = interval.size;
  alternate_mem_interval.start =
      std::max(start_time, end_time - max_prefetch_interval_);
  alternate_mem_interval.end = end_time;

  VLOG(2) << "Finding allocation for " << interval.buffer->ToShortString()
          << " (" << start_time << ", " << end_time
          << "). Size = " << interval.size;

  MemorySpaceAssignment::Allocation* prev_allocation = nullptr;
  bool can_eliminate_copy = false;
  if (allocations->empty()) {
    // There hasn't been any allocations for this interval so far. We can
    // eliminate copy if the value can be placed in the alternate memory.
    can_eliminate_copy = is_allowed_in_alternate_mem_(*interval.buffer);
  } else {
    // If there has been a previous allocation, we can eliminate the copy if the
    // previous allocation was also in the alternate memory.
    prev_allocation = allocations->back().get();
    can_eliminate_copy =
        (prev_allocation->memory_space() == MemorySpace::kAlternate);
  }

  if (alternate_mem_interval.start == start_time && can_eliminate_copy) {
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
    ChunkCandidate chunk_candidate =
        FindChunkCandidate(alternate_mem_interval, preferred_offset);
    // Check if the new heap size fits within limits. Also ensure if a
    // preferred offset was provided, that offset was used.
    if (chunk_candidate.heap_size < max_size_in_bytes_ &&
        (preferred_offset == -1 ||
         preferred_offset == chunk_candidate.chunk.offset)) {
      VLOG(3) << "Keep the buffer in alternate memory. Offset = "
              << chunk_candidate.chunk.offset
              << ", size = " << chunk_candidate.chunk.size
              << ", heap_size = " << chunk_candidate.heap_size;
      CommitChunk(alternate_mem_interval, chunk_candidate);

      // If there was a previous allocation, the buffer location is the
      // same as the previous. Otherwise, it is the operand.
      if (prev_allocation != nullptr &&
          prev_allocation->defining_instruction() == def_instruction) {
        prev_allocation->Extend(end_time);
      } else {
        allocations->push_back(
            absl::make_unique<MemorySpaceAssignment::Allocation>(
                def_instruction, MemorySpace::kAlternate, chunk_candidate.chunk,
                start_time, end_time));
      }
      allocations->back()->AddUse(use);
      return;
    }
  }

  // Since copies couldn't be removed, create an allocation in the default
  // memory space.
  if (prev_allocation != nullptr &&
      prev_allocation->memory_space() == MemorySpace::kAlternate &&
      prev_allocation->defining_instruction() == def_instruction) {
    // If there was an allocation for this HloValue that was in the alternate
    // memory space, we also need to perform an eviction.
    // TODO(berkin): For now evictions happen relative to the most recent
    // allocation in the alternate memory. We can potentially start evictions
    // earlier and end later.
    HloInstruction* earliest_instruction =
        GetInstructionAt(prev_allocation->start_time());
    HloInstruction* latest_instruction =
        GetInstructionAt(prev_allocation->end_time());

    VLOG(3) << "Evicting buffer at " << prev_allocation->chunk().offset << " ("
            << prev_allocation->start_time() << ", "
            << prev_allocation->end_time() << ")";
    VLOG(3) << "Copy to default mem between instructions "
            << earliest_instruction->ToString() << " - "
            << latest_instruction->ToString();

    // The live range of this buffer is from the start time of the previous
    // buffer that was in the alternate memory so that a buffer is allocated
    // during the copy.
    allocations->push_back(
        absl::make_unique<MemorySpaceAssignment::CopyAllocation>(
            *prev_allocation, MemorySpace::kDefault,
            kDefaultMemorySpaceDummyChunk, prev_allocation->start_time(),
            end_time, earliest_instruction, latest_instruction));
  } else if (prev_allocation != nullptr &&
             prev_allocation->memory_space() == MemorySpace::kDefault &&
             prev_allocation->defining_instruction() == def_instruction) {
    // If the previous allocation was in the default memory space and was
    // defined by the same instruction, extend that.  Otherwise, create a new
    // allocation.
    prev_allocation->Extend(end_time);
  } else {
    allocations->push_back(absl::make_unique<MemorySpaceAssignment::Allocation>(
        def_instruction, MemorySpace::kDefault, kDefaultMemorySpaceDummyChunk,
        start_time, end_time));
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
    ChunkCandidate chunk_candidate = FindChunkCandidate(alternate_mem_interval);
    // Check if the new heap size fits within limits.
    if (chunk_candidate.heap_size < max_size_in_bytes_) {
      HloInstruction* earliest_instruction =
          GetInstructionAt(alternate_mem_interval.start);
      VLOG(3) << "Move the buffer to alternate memory at "
              << alternate_mem_interval.start
              << ". Offset = " << chunk_candidate.chunk.offset
              << ", size = " << chunk_candidate.chunk.size
              << ", heap_size = " << chunk_candidate.heap_size;
      VLOG(3) << "Copy to alternate mem between instructions "
              << earliest_instruction->ToString() << " - "
              << use.instruction->ToString();
      CommitChunk(alternate_mem_interval, chunk_candidate);

      // Since copies couldn't be removed, create an allocation in the
      // default memory space.
      allocations->push_back(
          absl::make_unique<MemorySpaceAssignment::CopyAllocation>(
              *allocations->back().get(), MemorySpace::kAlternate,
              chunk_candidate.chunk, alternate_mem_interval.start, end_time,
              earliest_instruction, use.instruction));
      allocations->back()->AddUse(use);
      return;
    }
  }

  // If a copy wasn't inserted, then add this use to the latest allocation.
  allocations->back()->AddUse(use);
}

/*static*/ StatusOr<bool> MemorySpaceAssignment::Run(
    HloModule* module, int64 alternate_memory_space, int64 max_size_in_bytes,
    int64 min_prefetch_interval, int64 max_prefetch_interval,
    int64 alternate_memory_space_alignment_in_bytes,
    BufferValue::SizeFunction size_fn,
    AlternateMemoryBestFitHeap::IsAllowedInAlternateMemoryFunction
        is_allowed_in_alternate_mem) {
  CHECK(module->has_schedule());
  VLOG(4) << "Module before memory space assignment: " << module->ToString();
  VLOG(4) << "Schedule: " << module->schedule().ToString();
  TF_ASSIGN_OR_RETURN(auto alias_analysis, HloAliasAnalysis::Run(module));

  MemorySpaceAssignment memory_space_assignment(module, alternate_memory_space);
  // TODO(berkin): Explore heap algorithms other than kSpatial.
  auto algorithm = absl::make_unique<AlternateMemoryBestFitHeap>(
      &memory_space_assignment.allocation_map_, max_size_in_bytes,
      min_prefetch_interval, max_prefetch_interval, *alias_analysis,
      alternate_memory_space_alignment_in_bytes,
      GlobalDecreasingSizeBestFitHeap::Type::kSpatial,
      is_allowed_in_alternate_mem);

  TF_RETURN_IF_ERROR(HeapSimulator::Run(std::move(algorithm), *module,
                                        module->schedule(),
                                        *alias_analysis.get(), size_fn)
                         .status());

  TF_RETURN_IF_ERROR(memory_space_assignment.Process());
  TF_RETURN_IF_ERROR(memory_space_assignment.FixSchedule());

  VLOG(4) << "Module after memory space assignment: " << module->ToString();
  VLOG(4) << "Schedule: " << module->schedule().ToString();
  TF_CHECK_OK(module->schedule().Verify());

  return true;
}

Status MemorySpaceAssignment::Allocation::Process(
    MemorySpaceAssignment* memory_space_assignment) {
  // For non-copy allocations, all we need to do is to update the output memory
  // space if placed in the alternate memory.
  if (memory_space_ == MemorySpace::kAlternate) {
    Layout* layout = defining_instruction_->mutable_shape()->mutable_layout();
    layout->set_memory_space(memory_space_assignment->alternate_memory_space_);
  }
  return Status::OK();
}

Status MemorySpaceAssignment::CopyAllocation::Process(
    MemorySpaceAssignment* memory_space_assignment) {
  // Copy allocations need to insert asynchronous copy nodes.
  HloInstruction* def_instruction = defining_instruction();
  CHECK_NE(def_instruction, nullptr);

  Shape shape = def_instruction->shape();
  HloComputation* computation = def_instruction->parent();

  // Set the layout to include the memory space.
  Layout* layout = shape.mutable_layout();
  if (memory_space_ == MemorySpace::kAlternate) {
    layout->set_memory_space(memory_space_assignment->alternate_memory_space_);
  } else {
    layout->set_memory_space(0);
  }

  HloInstruction* copy_start =
      computation->AddInstruction(HloInstruction::CreateUnary(
          ShapeUtil::MakeTupleShape({shape, ShapeUtil::MakeShape(U32, {})}),
          HloOpcode::kCopyStart, def_instruction));
  HloInstruction* copy_done = computation->AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCopyDone, copy_start));
  // Update the allocation with the defining instruction so that if there
  // are further copies from it, it can find the correct instruction.
  defining_instruction_ = copy_done;

  // Replace all the uses with the new copy instruction.
  for (HloUse use : uses_) {
    TF_RETURN_IF_ERROR(
        use.instruction->ReplaceOperandWith(use.operand_number, copy_done));
  }

  // Insert the new instructions at the appropriate places in the schedule.
  // FixSchedule will process the maps to actually insert them.
  memory_space_assignment->ScheduleAsynchronousCopy(
      copy_start, copy_start_schedule_after_, copy_done,
      copy_done_schedule_before_);
  return Status::OK();
}

Status MemorySpaceAssignment::Process() {
  // Insert CopyStart/CopyDone pairs.
  for (auto& buffer_and_sequence : allocation_map_) {
    for (auto& allocation : buffer_and_sequence.second) {
      TF_RETURN_IF_ERROR(allocation->Process(this));
    }
  }
  return Status::OK();
}

void MemorySpaceAssignment::ScheduleAsynchronousCopy(
    HloInstruction* copy_start, HloInstruction* copy_start_schedule_after,
    HloInstruction* copy_done, HloInstruction* copy_done_schedule_before) {
  schedule_after_[copy_start_schedule_after].push_back(copy_start);
  schedule_before_[copy_done_schedule_before].push_back(copy_done);
}

Status MemorySpaceAssignment::FixSchedule() {
  CHECK(module_->has_schedule());
  HloSchedule& schedule = module_->schedule();
  for (const HloComputation* computation : module_->computations()) {
    const HloInstructionSequence& sequence = schedule.sequence(computation);
    HloInstructionSequence new_sequence;

    for (HloInstruction* instruction : sequence.instructions()) {
      auto insts_before_iter = schedule_before_.find(instruction);
      if (insts_before_iter != schedule_before_.end()) {
        for (HloInstruction* new_instruction : insts_before_iter->second) {
          new_sequence.push_back(new_instruction);
          VLOG(4) << "before: " << new_instruction->ToString();
        }
      }
      new_sequence.push_back(instruction);
      VLOG(4) << instruction->ToString();
      auto insts_after_iter = schedule_after_.find(instruction);
      if (insts_after_iter != schedule_after_.end()) {
        for (HloInstruction* new_instruction : insts_after_iter->second) {
          new_sequence.push_back(new_instruction);
          VLOG(4) << "after: " << new_instruction->ToString();
        }
      }
    }
    schedule.set_sequence(computation, new_sequence);
  }

  return Status::OK();
}

}  // namespace xla

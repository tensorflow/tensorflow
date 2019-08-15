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

    // If the buffer is a tuple, don't use this algorithm for now. The buffers
    // that are pointed to by the tuple will still use this algorithm.
    // TODO(berkin): Because tuples are cheap to place in the alternate memory
    // (they are just pointers) we don't need to use prefetch/evict logic.
    if (buffer.values()[0]->shape().IsTuple()) {
      VLOG(4) << "Keeping buffer " << buffer.ToString()
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
      // Sort the uses by the use time.
      std::vector<HloUse> uses = value->uses();
      absl::c_sort(uses, [&](HloUse use1, HloUse use2) {
        return instruction_schedule_->at(use1.instruction) <
               instruction_schedule_->at(use2.instruction);
      });
      // Iterate over the uses.
      for (HloUse use : uses) {
        int64 use_time = instruction_schedule_->at(use.instruction);

        // Bitcasts don't define buffers and don't directly consume buffers.
        // Skip allocating buffers for bitcast uses. The uses that feed from
        // bitcasts will be handled specially.
        if (use.instruction->opcode() != HloOpcode::kBitcast) {
          FindAllocation(definition_time, use_time, value->defining_position(),
                         use, value, colocated_interval->size,
                         allocation_sequence);
          // If there are multiple uses, they can try using the memory
          // allocation already at the alternate memory.
          definition_time = use_time;
        }
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
    int64 start_time, int64 end_time, HloPosition defining_position, HloUse use,
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
  alternate_mem_interval.start =
      std::max(start_time, end_time - max_prefetch_interval_);
  alternate_mem_interval.end = end_time;

  VLOG(2) << "Finding allocation for " << buffer->ToShortString() << " ("
          << start_time << ", " << end_time << "). Size = " << size
          << ", def pos = " << defining_position.ToString()
          << ", operand = " << operand->ToString()
          << (non_bitcast_operand != operand
                  ? ", non_bitcast_operand = " + non_bitcast_operand->ToString()
                  : "");
  CHECK_LT(start_time, end_time);

  // First try keeping the allocation entirely in the alternate memory.
  if (TryAllocatingInAlternateMemoryNoCopy(
          start_time, end_time, defining_position, use, alternate_mem_interval,
          non_bitcast_operand, allocations)) {
    return;
  }

  MemorySpaceAssignment::Allocation* prev_allocation = nullptr;
  if (!allocations->empty()) {
    prev_allocation = allocations->back().get();
  }

  // Since copies couldn't be removed, create an allocation in the default
  // memory space.
  if (prev_allocation != nullptr &&
      prev_allocation->memory_space() == MemorySpace::kAlternate &&
      prev_allocation->instruction() == non_bitcast_operand) {
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
             prev_allocation->instruction() == non_bitcast_operand) {
    // If the previous allocation was in the default memory space and was
    // defined by the same instruction, extend that.  Otherwise, create a new
    // allocation.
    prev_allocation->Extend(end_time);
  } else {
    allocations->push_back(absl::make_unique<MemorySpaceAssignment::Allocation>(
        non_bitcast_operand, defining_position, MemorySpace::kDefault,
        kDefaultMemorySpaceDummyChunk, start_time, end_time));
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

bool AlternateMemoryBestFitHeap::TryAllocatingInAlternateMemoryNoCopy(
    int64 start_time, int64 end_time, HloPosition defining_position, HloUse use,
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
        prev_allocation->instruction() == non_bitcast_operand) {
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

/*static*/ StatusOr<std::unique_ptr<PresetAssignments>>
MemorySpaceAssignment::Run(
    HloModule* module, int64 alternate_memory_space, int64 max_size_in_bytes,
    int64 min_prefetch_interval, int64 max_prefetch_interval,
    int64 alternate_memory_space_alignment_in_bytes,
    BufferValue::SizeFunction size_fn,
    AlternateMemoryBestFitHeap::IsAllowedInAlternateMemoryFunction
        is_allowed_in_alternate_mem) {
  CHECK(module->has_schedule());
  VLOG(4) << "Module before memory space assignment: ";
  XLA_VLOG_LINES(4, module->ToString());
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

  VLOG(4) << "Module after memory space assignment: ";
  XLA_VLOG_LINES(4, module->ToString());
  TF_CHECK_OK(module->schedule().Verify());

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

Status MemorySpaceAssignment::Allocation::PropagateMemorySpaceToBitcasts(
    const MemorySpaceAssignment& memory_space_assignment) {
  for (HloInstruction* bitcast : bitcasts_) {
    if (memory_space_ == MemorySpace::kAlternate) {
      Layout* bitcast_layout = bitcast->mutable_shape()->mutable_layout();
      bitcast_layout->set_memory_space(
          memory_space_assignment.alternate_memory_space_);
    }
  }
  return Status::OK();
}

Status MemorySpaceAssignment::Allocation::Process(
    MemorySpaceAssignment* memory_space_assignment) {
  // For non-copy allocations, all we need to do is to update the output memory
  // space if placed in the alternate memory.
  if (memory_space_ == MemorySpace::kAlternate) {
    Layout* layout = instruction_->mutable_shape()->mutable_layout();
    layout->set_memory_space(memory_space_assignment->alternate_memory_space_);
  }
  TF_RETURN_IF_ERROR(PropagateMemorySpaceToBitcasts(*memory_space_assignment));
  return Status::OK();
}

Status MemorySpaceAssignment::CopyAllocation::Process(
    MemorySpaceAssignment* memory_space_assignment) {
  // Copy allocations need to insert asynchronous copy nodes.
  HloInstruction* producing_instruction = instruction();
  CHECK_NE(producing_instruction, nullptr);

  Shape shape = producing_instruction->shape();
  HloComputation* computation = producing_instruction->parent();

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
          HloOpcode::kCopyStart, producing_instruction));
  HloInstruction* copy_done = computation->AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCopyDone, copy_start));
  // Update the allocation with the copy done instruction so that if there
  // are further copies from it, it can find the correct instruction.
  instruction_ = copy_done;
  // Also update the defining position. Note that the output of CopyDone is
  // actually defined in the item {0} of CopyStart.
  defining_position_ = HloPosition{copy_start, {0}};

  // Replace all the uses with the new copy instruction.
  for (HloUse use : uses_) {
    TF_RETURN_IF_ERROR(
        use.instruction->ReplaceOperandWith(use.operand_number, copy_done));
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
        /*operand_num=*/0, instruction_));
  }

  // Propagate the memory space to all bitcasts.
  TF_RETURN_IF_ERROR(PropagateMemorySpaceToBitcasts(*memory_space_assignment));

  // Insert the new instructions at the appropriate places in the schedule.
  // FixSchedule will process the maps to actually insert them.
  memory_space_assignment->ScheduleAsynchronousCopy(
      copy_start, copy_start_schedule_after_, copy_done,
      copy_done_schedule_before_);
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
  return Status::OK();
}

void MemorySpaceAssignment::ScheduleAsynchronousCopy(
    HloInstruction* copy_start, HloInstruction* copy_start_schedule_after,
    HloInstruction* copy_done, HloInstruction* copy_done_schedule_before) {
  schedule_after_[copy_start_schedule_after].push_back(copy_start);
  schedule_before_[copy_done_schedule_before].push_back(copy_done);
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
  VLOG(4) << "inserting: " << new_instruction->ToString();
  new_sequence->push_back(new_instruction);
  inserted_instructions->insert(new_instruction);
}

Status MemorySpaceAssignment::FixSchedule() {
  CHECK(module_->has_schedule());
  HloSchedule& schedule = module_->schedule();
  for (const HloComputation* computation :
       module_->MakeNonfusionComputations()) {
    CHECK(schedule.is_computation_scheduled(computation));
    const HloInstructionSequence& sequence = schedule.sequence(computation);
    HloInstructionSequence new_sequence;

    absl::flat_hash_set<HloInstruction*> inserted_instructions;

    for (HloInstruction* instruction : sequence.instructions()) {
      auto insts_before_iter = schedule_before_.find(instruction);
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
      auto insts_after_iter = schedule_after_.find(instruction);
      if (insts_after_iter != schedule_after_.end()) {
        for (HloInstruction* new_instruction : insts_after_iter->second) {
          EnsureInstructionAndOperandsInserted(new_instruction, &new_sequence,
                                               &inserted_instructions);
        }
      }
    }
    schedule.set_sequence(computation, new_sequence);
  }

  return Status::OK();
}

}  // namespace xla

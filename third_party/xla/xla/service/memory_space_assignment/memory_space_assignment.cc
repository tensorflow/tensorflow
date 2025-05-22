/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/service/memory_space_assignment/memory_space_assignment.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/layout.h"
#include "xla/service/buffer_value.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_value.h"
#include "xla/service/memory_space_assignment/algorithm.h"
#include "xla/service/memory_space_assignment/allocation.h"
#include "xla/service/memory_space_assignment/memory_space_assignment.pb.h"
#include "xla/service/memory_space_assignment/options.h"
#include "xla/service/memory_space_assignment/simulator.h"
#include "xla/service/memory_space_assignment/slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace memory_space_assignment {
namespace {

// Insert an instruction to the schedule, and make sure its dependencies
// (operands) are already in the schedule. If not, insert these operands
// before the instruction.
void InsertInstructionAndEnsureOperandsInserted(
    HloInstruction* new_instruction, HloInstructionSequence* new_sequence,
    absl::flat_hash_set<HloInstruction*>* inserted_instructions) {
  if (!inserted_instructions->insert(new_instruction).second) {
    VLOG(1) << "Already inserted: " << new_instruction->ToString();
  } else {
    for (HloInstruction* operand : new_instruction->operands()) {
      InsertInstructionAndEnsureOperandsInserted(operand, new_sequence,
                                                 inserted_instructions);
    }
    VLOG(1) << "Inserting: " << new_instruction->ToShortString();
    new_sequence->push_back(new_instruction);
  }
}

std::string InstructionScheduleToString(const HloLiveRange& hlo_live_range) {
  const absl::flat_hash_map<const HloInstruction*, HloLiveRange::LogicalTime>&
      instruction_schedule = hlo_live_range.instruction_schedule();
  std::vector<std::pair<int64_t, const HloInstruction*>> instructions;
  instructions.reserve(instruction_schedule.size());
  for (const auto& instruction : instruction_schedule) {
    instructions.push_back({instruction.second, instruction.first});
  }
  std::string instruction_schedule_str = "\n";
  absl::c_sort(instructions);
  for (auto& instruction : instructions) {
    absl::StrAppend(&instruction_schedule_str,
                    "LogicalTime: ", instruction.first, " ",
                    instruction.second->ToString(), "\n");
  }
  return instruction_schedule_str;
}

void EnsureParentAllocationIsAvailableForCopy(CopyAllocation* copy_allocation) {
  Allocation& parent_allocation = copy_allocation->mutable_prev_allocation();
  parent_allocation.Extend(copy_allocation->copy_done_schedule_before());
  if (parent_allocation.is_copy_allocation()) {
    auto parent_copy_allocation =
        tensorflow::down_cast<CopyAllocation*>(&parent_allocation);
    parent_copy_allocation->set_copy_done_schedule_before(
        std::min(parent_copy_allocation->copy_done_schedule_before(),
                 copy_allocation->start_time()));
    parent_copy_allocation->set_copy_start_schedule_after(
        std::min(parent_copy_allocation->copy_start_schedule_after(),
                 parent_copy_allocation->copy_done_schedule_before() - 1));
  }
}

void MakeCopyAllocationJitForSingleUse(CopyAllocation* copy_allocation,
                                       int64_t use_time) {
  copy_allocation->set_start_time(use_time - 1);
  copy_allocation->set_copy_start_schedule_after(use_time - 1);
  copy_allocation->set_end_time(use_time);
  copy_allocation->set_copy_done_schedule_before(use_time);
  EnsureParentAllocationIsAvailableForCopy(copy_allocation);
}

int64_t GetUseTime(const HloUse& use, const HloLiveRange& hlo_live_range) {
  return hlo_live_range.instruction_schedule().at(use.instruction);
}

void ProcessPrefetchesToAlternateMemory(AllocationSequence& allocations,
                                        const HloLiveRange& hlo_live_range) {
  std::vector<Allocation*> allocations_in_raw_pointers =
      GetAllocationSequenceInRawPointers(allocations);
  for (auto allocation : allocations_in_raw_pointers) {
    if (allocation->is_copy_allocation() && allocation->is_in_alternate_mem() &&
        !allocation->uses().empty()) {
      CopyAllocation* prefetch =
          tensorflow::down_cast<CopyAllocation*>(allocation);
      std::vector<HloUse> uses = prefetch->uses();  // Create a copy of uses.
      prefetch->clear_uses();                       // Clear old uses.
      // For every prefetch, update prefetch to serve earliest use just in time.
      prefetch->AddUse(uses[0]);
      MakeCopyAllocationJitForSingleUse(prefetch,
                                        GetUseTime(uses[0], hlo_live_range));
      // For every use after the first use, create a new prefetch from the same
      // parent allocation.
      for (size_t use_index = 1; use_index < uses.size(); ++use_index) {
        const HloUse& use = uses[use_index];
        int64_t use_time = GetUseTime(use, hlo_live_range);
        auto jit_single_use_prefetch = std::make_unique<CopyAllocation>(
            prefetch->mutable_prev_allocation(), MemorySpace::kAlternate,
            prefetch->chunk(), use_time - 1, use_time, use_time);
        jit_single_use_prefetch->set_copy_start_schedule_after(use_time - 1);
        jit_single_use_prefetch->AddUse(use);
        EnsureParentAllocationIsAvailableForCopy(jit_single_use_prefetch.get());
        allocations.push_back(std::move(jit_single_use_prefetch));
      }
    }
  }
}

void MakeEvictionImmediate(CopyAllocation* eviction) {
  const Allocation& parent_allocation = eviction->prev_allocation();
  eviction->set_start_time(parent_allocation.start_time());
  eviction->set_copy_start_schedule_after(parent_allocation.start_time());
  eviction->set_copy_done_schedule_before(parent_allocation.start_time() + 1);
  eviction->Extend(parent_allocation.start_time() + 1);
}

absl::flat_hash_map<Allocation*, CopyAllocation*> GetEvictionsMap(
    std::vector<Allocation*>& allocations) {
  absl::flat_hash_map<Allocation*, CopyAllocation*> evictions_map;
  for (auto& allocation : allocations) {
    if (allocation->is_copy_allocation() && allocation->is_in_default_mem()) {
      auto eviction = tensorflow::down_cast<CopyAllocation*>(allocation);
      Allocation& parent_allocation = eviction->mutable_prev_allocation();
      if (!parent_allocation.is_copy_allocation()) {
        evictions_map[&parent_allocation] = eviction;
      }
    }
  }
  return evictions_map;
}

void ProcessBuffersProducedInAlternateMemory(
    AllocationSequence& allocations, const HloLiveRange& hlo_live_range) {
  std::vector<Allocation*> allocations_in_raw_pointers =
      GetAllocationSequenceInRawPointers(allocations);
  // For all parent allocations produced in alternate memory, create a map from
  // parent allocation -> eviction.
  absl::flat_hash_map<Allocation*, CopyAllocation*> evictions_map =
      GetEvictionsMap(allocations_in_raw_pointers);
  // Make all such evictions immediate.
  for (auto& [_, eviction] : evictions_map) {
    MakeEvictionImmediate(eviction);
  }
  if (VLOG_IS_ON(2)) {
    LOG(INFO) << "AllocationSequence after making spills immediate spills\n";
    XLA_LOG_LINES(INFO, AllocationSequenceToString(allocations, true));
  }
  // Process all buffers produced in the alternate memory:
  // 1. Make the buffer short lived.
  // 2. Service immediate use if any.
  // 3. If buffer is also used later get or create an immediate eviction.
  // 4. For every later use prefetch just in time from the eviction.
  for (auto allocation : allocations_in_raw_pointers) {
    if (!allocation->is_copy_allocation() &&
        allocation->is_in_alternate_mem()) {
      std::vector<HloUse> uses = allocation->uses();  // Create a copy of uses.
      allocation->clear_uses();                       // Clear old uses.
      // Make buffer short lived.
      allocation->set_end_time(allocation->start_time() + 1);
      for (const HloUse& use : uses) {
        int64_t use_time = GetUseTime(use, hlo_live_range);
        if (allocation->start_time() + 1 == use_time) {
          allocation->AddUse(use);
          continue;
        }
        if (!evictions_map.contains(allocation)) {
          auto eviction_unique_ptr = std::make_unique<CopyAllocation>(
              *allocation, MemorySpace::kDefault, std::nullopt,
              allocation->start_time(), allocation->start_time() + 1,
              allocation->start_time() + 1);
          eviction_unique_ptr->set_copy_start_schedule_after(
              allocation->start_time());
          evictions_map[allocation] = eviction_unique_ptr.get();
          allocations.push_back(std::move(eviction_unique_ptr));
        }
        CopyAllocation* eviction = evictions_map[allocation];
        auto jit_single_use_prefetch = std::make_unique<CopyAllocation>(
            *eviction, MemorySpace::kAlternate, allocation->chunk(),
            use_time - 1, use_time, use_time);
        jit_single_use_prefetch->set_copy_start_schedule_after(use_time - 1);
        jit_single_use_prefetch->AddUse(use);
        EnsureParentAllocationIsAvailableForCopy(jit_single_use_prefetch.get());
        allocations.push_back(std::move(jit_single_use_prefetch));
      }
    }
  }
}

void TransformAllocationSequenceToSpill(AllocationSequence& allocations,
                                        const HloLiveRange& hlo_live_range) {
  if (VLOG_IS_ON(2)) {
    LOG(INFO) << "InstructionSchedule before transform\n";
    XLA_LOG_LINES(INFO, InstructionScheduleToString(hlo_live_range));
    LOG(INFO) << "AllocationSequence before transform\n";
    XLA_LOG_LINES(INFO, AllocationSequenceToString(allocations, true));
  }
  ProcessPrefetchesToAlternateMemory(allocations, hlo_live_range);
  if (VLOG_IS_ON(2)) {
    LOG(INFO) << "AllocationSequence after processing prefetches\n";
    XLA_LOG_LINES(INFO, AllocationSequenceToString(allocations, true));
  }
  ProcessBuffersProducedInAlternateMemory(allocations, hlo_live_range);
  if (VLOG_IS_ON(2)) {
    VLOG(2) << "AllocationSequence after processing buffers produced in kAlt\n";
    XLA_LOG_LINES(INFO, AllocationSequenceToString(allocations, true));
  }
  SortAllocationSequence(allocations);
}

}  // namespace

absl::StatusOr<MemorySpaceAssignment::AsyncCopyStats>
MemorySpaceAssignment::CalculateAsyncCopyStats(
    const HloDataflowAnalysis& dataflow_analysis) const {
  AsyncCopyStats stats;
  int64_t current_copies = 0;
  for (const HloComputation* computation :
       module_->MakeNonfusionComputations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kCopyStart ||
          (instruction->opcode() == HloOpcode::kAsyncStart &&
           instruction->async_wrapped_instruction()->opcode() ==
               HloOpcode::kSlice)) {
        current_copies++;
      } else if (instruction->opcode() == HloOpcode::kCopyDone ||
                 (instruction->opcode() == HloOpcode::kAsyncDone &&
                  instruction->async_wrapped_instruction()->opcode() ==
                      HloOpcode::kSlice)) {
        current_copies--;
        int64_t size =
            options_.size_fn(dataflow_analysis.GetUniqueValueAt(instruction));
        if (instruction->shape().layout().memory_space() ==
            options_.alternate_memory_space) {
          ++stats.num_prefetches;
          stats.prefetch_bytes += size;
          if (instruction->opcode() == HloOpcode::kAsyncDone &&
              instruction->async_wrapped_instruction()->opcode() ==
                  HloOpcode::kSlice) {
            ++stats.num_sliced_prefetch_slices;
          }
        } else {
          ++stats.num_evictions;
          stats.eviction_bytes += size;
        }
      } else if (instruction->IsCustomCall(kConcatBitcastCustomCall)) {
        ++stats.num_sliced_prefetches;
      }
      stats.max_outstanding_async_copies =
          std::max(stats.max_outstanding_async_copies, current_copies);
    }
  }
  return stats;
}

/*static*/ absl::StatusOr<std::unique_ptr<PresetAssignments>>
MemorySpaceAssignment::Run(HloModule* module,
                           const HloLiveRange& hlo_live_range,
                           const HloAliasAnalysis& alias_analysis,
                           const Options& options) {
  CHECK(module->has_schedule());
  if (VLOG_IS_ON(3)) {
    LOG(INFO) << "Module before memory space assignment: ";
    XLA_LOG_LINES(INFO, module->ToString());
    LOG(INFO) << "Schedule: " << module->schedule().ToString();
  }
  MemorySpaceAssignment memory_space_assignment(module, options,
                                                hlo_live_range);

  return memory_space_assignment.RunMemorySpaceAssignment(hlo_live_range,
                                                          alias_analysis);
}

absl::Status MemorySpaceAssignment::VerifyAllocations() const {
  BufferIntervalTree interval_tree;
  // Checks the chunks that overlap with a given allocation in time do not
  // overlap with the allocation's chunk in the memory range. If they do, we
  // throw an error, otherwise we add the allocation's chunk to the interval
  // tree and return an OK status.
  auto add_allocation_and_verify =
      [&](const Allocation* allocation) -> absl::Status {
    for (const HeapSimulator::Chunk& overlapping_chunk :
         interval_tree.ChunksOverlappingInTime(allocation->start_time(),
                                               allocation->end_time() - 1)) {
      CHECK(!allocation->chunk().OverlapsWith(overlapping_chunk))
          << "Chunks are overlapping at Allocation level (before fixing the "
             "schedule): "
          << allocation->ToString()
          << " overlaps with allocated chunk: " << overlapping_chunk.ToString();
    }
    interval_tree.Add(allocation->start_time(), allocation->end_time() - 1,
                      allocation->chunk());
    return absl::OkStatus();
  };
  // Verify that all alternate memory allocations are free of overlapping
  // Allocations in time and space, and add them to interval_tree one by one.
  for (const auto& allocation : allocations_) {
    if (allocation->memory_space() == MemorySpace::kAlternate) {
      TF_RETURN_IF_ERROR(add_allocation_and_verify(allocation.get()));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<PresetAssignments>>
MemorySpaceAssignment::RunMemorySpaceAssignment(
    const HloLiveRange& hlo_live_range,
    const HloAliasAnalysis& alias_analysis) {
  bool splitting_enabled = options_.determine_split_dimension_fn != nullptr &&
                           options_.init_split_tree_fn != nullptr &&
                           options_.shape_size_fn != nullptr;
  if (splitting_enabled) {
    CHECK_EQ(options_.sliced_prefetch_options.max_slices(), 0)
        << "TODO(b/167392593): Support sliced prefetches for split shapes.";
    CHECK(!options_.enable_window_prefetch)
        << "TODO(b/167392593): Support split shapes for window "
           "prefetches.";
  }
  TF_RETURN_IF_ERROR(FindAllocationSequence(hlo_live_range, alias_analysis));

  std::optional<RuntimeSimulator> runtime_simulator = std::nullopt;
  if (options_.cost_analysis) {
    runtime_simulator.emplace(options_.cost_analysis,
                              options_.alternate_memory_space);
    if (VLOG_IS_ON(1)) {
      float estimated_time =
          runtime_simulator->SimulateElapsedTimeWithoutAsyncCopyLikes(
              hlo_live_range, allocations_);
      LOG(INFO) << "Estimated elapsed time without async copies (sec): "
                << estimated_time;
    }
  }

  TF_RETURN_IF_ERROR(Process(hlo_live_range));
  if (options_.verify) {
    TF_RETURN_IF_ERROR(VerifyAllocations());
  }

  // DEBUG_LOG_ALLOCATIONS_AT
  //
  // Uncomment the following to log the alternate memory allocations that MSA
  // made at a given schedule time.
  //
  // AllocationSequenceDebugging::LogAltMemAllocationsAt(
  //     allocations_, /*time*/1);
  ScheduleAsynchronousCopies();
  TF_RETURN_IF_ERROR(SimplifyGraph());
  TF_RETURN_IF_ERROR(FixSchedule());
  TF_ASSIGN_OR_RETURN(auto alias, HloAliasAnalysis::Run(module_));
  TF_RETURN_IF_ERROR(ExportAndColorBuffers(*alias));
  std::vector<int64_t> alt_mem_bytes_occupied;
  // alt_mem_bytes_occupied is used for logging in the RuntimeSimulator below.
  // We only populate it in VerifyAndExportHeapSimulatorTrace if the
  // RuntimeSimulator is present.
  TF_RETURN_IF_ERROR(VerifyAndExportHeapSimulatorTrace(
      *alias,
      runtime_simulator.has_value() ? &alt_mem_bytes_occupied : nullptr));
  if (VLOG_IS_ON(2) && runtime_simulator.has_value()) {
    float estimated_time = runtime_simulator->SimulateElapsedTime(
        module_, allocations_, &alt_mem_bytes_occupied);
    LOG(INFO) << "Estimated elapsed time with async copies (sec): "
              << estimated_time;
  }
  if (VLOG_IS_ON(3)) {
    LOG(INFO) << "Module after memory space assignment: ";
    XLA_LOG_LINES(INFO, module_->ToString());
  }
  TF_CHECK_OK(module_->schedule().Verify());
  if (VLOG_IS_ON(1)) {
    TF_ASSIGN_OR_RETURN(AsyncCopyStats stats,
                        CalculateAsyncCopyStats(alias->dataflow_analysis()));
    LOG(INFO) << "Maximum number of outstanding async copies/slices: "
              << stats.max_outstanding_async_copies;
    LOG(INFO) << "Number of prefetches: " << stats.num_prefetches
              << ", in bytes: " << stats.prefetch_bytes;
    LOG(INFO) << "Number of sliced prefetches: " << stats.num_sliced_prefetches
              << ", consuming number of slices: "
              << stats.num_sliced_prefetch_slices;
    LOG(INFO) << "Number of evictions: " << stats.num_evictions
              << ", in bytes: " << stats.eviction_bytes;
  }

  return std::move(preset_assignments_);
}

absl::Status MemorySpaceAssignment::FindAllocationSequence(
    const HloLiveRange& hlo_live_range,
    const HloAliasAnalysis& alias_analysis) {
  auto algorithm = std::make_unique<MsaAlgorithm>(
      module_, &allocations_, options_, alias_analysis, hlo_live_range);

  HeapSimulator::Options heap_simulator_options;
  heap_simulator_options.may_reuse_operand_buffers = false;
  heap_simulator_options.alloc_constants = true;
  TF_RETURN_IF_ERROR(HeapSimulator::Run(std::move(algorithm), *module_,
                                        module_->schedule(), alias_analysis,
                                        options_.size_fn,
                                        heap_simulator_options)
                         .status());
  return absl::OkStatus();
}

MemorySpaceAssignment::ScopedMemorySource
MemorySpaceAssignment::ScopedMemorySource::ForInstruction(
    HloInstruction* instruction) {
  return ScopedMemorySource{false, instruction};
}

MemorySpaceAssignment::ScopedMemorySource
MemorySpaceAssignment::ScopedMemorySource::ForPostModule() {
  return ScopedMemorySource{true, nullptr};
}

std::string MemorySpaceAssignment::ScopedMemorySource::ToString() const {
  if (is_post_module) {
    return "<post-module>";
  }
  return std::string(instruction->name());
}

absl::Status MemorySpaceAssignment::Process(
    const HloLiveRange& hlo_live_range) {
  VLOG(1) << "Processing assigned buffers...";
  // Since some parent allocations may not be needed (e.g. when they don't have
  // any uses and if there is no other (non-parent) allocation that depends on
  // it, before we process the allocations, mark all allocations that are
  // needed.
  absl::flat_hash_set<const Allocation*> needed_allocations;
  if (options_.always_spill_to_default_memory) {
    TransformAllocationSequenceToSpill(allocations_, hlo_live_range);
  }
  for (auto& allocation : allocations_) {
    allocation->MarkIfNeeded(needed_allocations);
  }
  // Insert CopyStart/CopyDone and SliceStart/SliceDone pairs.
  for (auto& allocation : allocations_) {
    VLOG(3) << "Processing: " << allocation->ToString();
    if (!needed_allocations.contains(allocation.get())) {
      VLOG(3) << "Allocation not needed.";
      continue;
    }
    TF_RETURN_IF_ERROR(allocation->Process(options_.bitcast_split_fn));
    // Add the offset and size of the allocation in the alternate memory to
    // the output map.
    if (allocation->is_scoped_allocation()) {
      CHECK(allocation->memory_space() == MemorySpace::kAlternate);
      ScopedAllocation* scoped_allocation =
          static_cast<ScopedAllocation*>(allocation.get());
      if (scoped_allocation->is_post_module()) {
        scoped_memory_assignments_.emplace_back(
            ScopedMemorySource::ForPostModule(), allocation->chunk());
      } else {
        scoped_memory_assignments_.emplace_back(
            ScopedMemorySource::ForInstruction(
                scoped_allocation->defining_position().instruction),
            allocation->chunk());
      }
      alternate_memory_size_ =
          std::max(alternate_memory_size_, allocation->chunk().chunk_end());
    } else if (allocation->memory_space() == MemorySpace::kAlternate) {
      if (allocation->is_sliced_copy_allocation()) {
        // Add slices
        const SlicedCopyAllocation& sliced_copy_allocation =
            *static_cast<const SlicedCopyAllocation*>(allocation.get());
        for (const SlicedCopyAllocation::SliceDetail& details :
             sliced_copy_allocation.slice_details_sorted_by_start_time()) {
          alternate_memory_assignments_.push_back(
              {{details.copy_done, {}}, details.slice_decision.chunk});
          alternate_memory_size_ = std::max(
              alternate_memory_size_, details.slice_decision.chunk.chunk_end());
        }
        CHECK(
            !sliced_copy_allocation.cross_program_prefetch_index().has_value());
      }
      alternate_memory_assignments_.emplace_back(
          allocation->defining_position(), allocation->chunk());
      alternate_memory_size_ =
          std::max(alternate_memory_size_, allocation->chunk().chunk_end());
      if (allocation->split_shape().has_value()) {
        auto result = split_map_.insert({allocation->defining_position(),
                                         &allocation->split_shape()->layout()});
        if (!result.second) {
          CHECK_EQ(*result.first->second,
                   allocation->split_shape().value().layout());
        }
      }
      if (allocation->cross_program_prefetch_index().has_value()) {
        TF_RETURN_IF_ERROR(module_->SetCrossProgramPrefetchOffset(
            *allocation->cross_program_prefetch_index(),
            allocation->chunk().offset));
      }
    }
  }

  // Post-process allocations. This is only used for parent allocations where we
  // update the body root with a reference to the buffer in default memory
  // space.
  absl::flat_hash_set<HloPosition> seen_pinned_positions;
  for (auto& allocation : allocations_) {
    if (needed_allocations.contains(allocation.get())) {
      VLOG(3) << "Post-Processing: " << allocation->ToString();
      TF_RETURN_IF_ERROR(allocation->PostProcess());
      if (allocation->is_pinned_allocation() &&
          !allocation->is_scoped_allocation()) {
        auto [it, inserted] =
            seen_pinned_positions.insert(allocation->defining_position());
        TF_RET_CHECK(inserted)
            << "Multiple pinned allocations defined for position "
            << allocation->defining_position().ToString();
      }
    }
  }
  return absl::OkStatus();
}

absl::Status MemorySpaceAssignment::ExportAndColorBuffers(
    const HloAliasAnalysis& alias_analysis) {
  VLOG(1) << "Exporting buffers...";
  absl::flat_hash_map<int64_t, int64_t> seen_buffer_offsets;
  VLOG(3) << "Exported alternate memory allocations:";
  for (const auto& position_and_chunk : alternate_memory_assignments_) {
    const HloPosition& defining_position = position_and_chunk.first;
    const HeapSimulator::Chunk& chunk = position_and_chunk.second;
    const HloBuffer& buffer = alias_analysis.GetUniqueBufferAt(
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
    ScopedMemorySource scoped_memory_source = instruction_and_chunk.first;
    const HeapSimulator::Chunk& chunk = instruction_and_chunk.second;
    if (scoped_memory_source.is_post_module) {
      preset_assignments_->set_post_module_scoped_alternate_memory_chunk(chunk);
    } else {
      preset_assignments_->add_scoped_allocation_chunk(
          scoped_memory_source.instruction, chunk);
    }
    VLOG(3) << " [" << chunk.offset << ", " << chunk.size
            << "] : " << scoped_memory_source.ToString();
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
    auto split_result = split_map_.find(defining_position);
    for (auto& buffer : alias_analysis.ComputeBuffersAt(
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
          if (split_result != split_map_.end()) {
            CHECK_EQ(shape->layout().split_configs().size(), 0);
            shape->mutable_layout()->add_split_configs(
                split_result->second->split_configs(0));
          }
        }
      }
    }
  }

  return absl::OkStatus();
}

void MemorySpaceAssignment::RemoveAssignmentForInstruction(
    const HloInstruction* instruction) {
  auto it = alternate_memory_assignments_.begin();
  auto end = alternate_memory_assignments_.end();
  while (it != end) {
    const HloPosition& position = it->first;
    if (position.instruction == instruction) {
      VLOG(3) << "Removing instruction from alternate memory assignments.";
      if (std::next(it) == end) {
        alternate_memory_assignments_.pop_back();
        break;
      } else {
        // Swap the removed position and chunk with the back and pop back.
        *it = alternate_memory_assignments_.back();
        alternate_memory_assignments_.pop_back();
        end = alternate_memory_assignments_.end();
      }
    } else {
      ++it;
    }
  }
}

absl::Status MemorySpaceAssignment::SimplifyGraph() {
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
            instruction->IsDead() && !instruction->HasSideEffect() &&
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
          bool can_replace = instruction->operand_count() > 0 &&
                             instruction->operand(0)->opcode() ==
                                 HloOpcode::kGetTupleElement &&
                             instruction->operand(0)
                                     ->operand(0)
                                     ->shape()
                                     .tuple_shapes()
                                     .size() == instruction->operand_count();
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

  return absl::OkStatus();
}

namespace {

// An interface that is used to wrap asynchronous copies, asynchronous slices,
// and asynchronous slice concat operations, for use in MSA's scheduling
// algorithm (ScheduleAsynchronousCopies).
//
// Each AsyncCopy step represents 1 copy, 1 slice, or 1 concat. Each step
// has an optional start phase (e.g., to start a copy or slice), and a required
// done phase (e.g., to finish a copy or slice, or to perform a concat).
class AsyncCopyStep {
 public:
  struct StartPhase {
    int64_t schedule_after_time;
    HloInstruction* instruction;
  };
  struct DonePhase {
    int64_t schedule_before_time;
    HloInstruction* instruction;
  };

  virtual ~AsyncCopyStep() = default;

  bool operator<(const AsyncCopyStep& rhs) const {
    std::optional<StartPhase> lhs_start_phase = start_phase();
    auto lhs_tuple = std::make_tuple(
        done_phase().schedule_before_time,
        (lhs_start_phase.has_value() ? lhs_start_phase->schedule_after_time
                                     : done_phase().schedule_before_time));
    std::optional<StartPhase> rhs_start_phase = rhs.start_phase();
    auto rhs_tuple = std::make_tuple(
        rhs.done_phase().schedule_before_time,
        (rhs_start_phase.has_value() ? rhs_start_phase->schedule_after_time
                                     : rhs.done_phase().schedule_before_time));

    return lhs_tuple < rhs_tuple;
  }

  virtual HloPosition defining_position() const = 0;

  virtual std::optional<StartPhase> start_phase() const = 0;
  virtual void set_start_phase_schedule_after_time(int64_t schedule_after) = 0;
  virtual DonePhase done_phase() const = 0;

 protected:
  AsyncCopyStep() = default;
};

class AsyncCopyStepForCopyAllocation : public AsyncCopyStep {
 public:
  explicit AsyncCopyStepForCopyAllocation(CopyAllocation* copy_allocation)
      : AsyncCopyStep(), copy_allocation_(copy_allocation) {}

  ~AsyncCopyStepForCopyAllocation() override = default;

  HloPosition defining_position() const override {
    return copy_allocation_->defining_position();
  }

  std::optional<StartPhase> start_phase() const override {
    StartPhase phase{copy_allocation_->copy_start_schedule_after(),
                     copy_allocation_->copy_start()};

    return phase;
  }

  void set_start_phase_schedule_after_time(int64_t schedule_after) override {
    copy_allocation_->set_copy_start_schedule_after(schedule_after);
  }

  DonePhase done_phase() const override {
    return {copy_allocation_->copy_done_schedule_before(),
            copy_allocation_->copy_done()};
  }

 private:
  CopyAllocation* copy_allocation_ = nullptr;
};

class AsyncCopyStepForSlice : public AsyncCopyStep {
 public:
  AsyncCopyStepForSlice(SlicedCopyAllocation* sliced_copy_allocation,
                        size_t slice_index)
      : AsyncCopyStep(),
        sliced_copy_allocation_(sliced_copy_allocation),
        slice_index_(slice_index) {}

  ~AsyncCopyStepForSlice() override = default;

  HloPosition defining_position() const override {
    return sliced_copy_allocation_->defining_position();
  }

  std::optional<StartPhase> start_phase() const override {
    const SlicedCopyAllocation::SliceDetail& slice_details =
        sliced_copy_allocation_
            ->slice_details_sorted_by_start_time()[slice_index_];
    StartPhase phase{slice_details.copy_start_after_time,
                     slice_details.copy_start};

    return phase;
  }

  void set_start_phase_schedule_after_time(int64_t schedule_after) override {
    sliced_copy_allocation_
        ->mutable_slice_details_sorted_by_start_time()[slice_index_]
        .copy_start_after_time = schedule_after;
  }

  DonePhase done_phase() const override {
    const SlicedCopyAllocation::SliceDetail& slice_details =
        sliced_copy_allocation_
            ->slice_details_sorted_by_start_time()[slice_index_];
    DonePhase phase{slice_details.copy_done_before_time,
                    slice_details.copy_done};

    return phase;
  }

 private:
  SlicedCopyAllocation* sliced_copy_allocation_ = nullptr;
  size_t slice_index_;
};

class AsyncCopyStepForSliceConcat : public AsyncCopyStep {
 public:
  explicit AsyncCopyStepForSliceConcat(
      SlicedCopyAllocation* sliced_copy_allocation)
      : AsyncCopyStep(), sliced_copy_allocation_(sliced_copy_allocation) {}

  ~AsyncCopyStepForSliceConcat() override = default;

  HloPosition defining_position() const override {
    return sliced_copy_allocation_->defining_position();
  }

  std::optional<StartPhase> start_phase() const override {
    return std::nullopt;
  }

  void set_start_phase_schedule_after_time(int64_t schedule_after) override {}

  DonePhase done_phase() const override {
    return {sliced_copy_allocation_->earliest_available_time(),
            sliced_copy_allocation_->concat()};
  }

 private:
  SlicedCopyAllocation* sliced_copy_allocation_ = nullptr;
};

}  // namespace

void MemorySpaceAssignment::ScheduleAsynchronousCopies() {
  VLOG(1) << "Scheduling asynchronous copies...";
  for (MemorySpace memory_space :
       {MemorySpace::kDefault, MemorySpace::kAlternate}) {
    std::vector<std::unique_ptr<AsyncCopyStep>> async_copy_steps;
    for (auto& allocation : allocations_) {
      if (allocation->memory_space() != memory_space) {
        continue;
      }

      if (allocation->is_copy_allocation()) {
        auto copy_allocation = static_cast<CopyAllocation*>(allocation.get());
        async_copy_steps.push_back(
            std::make_unique<AsyncCopyStepForCopyAllocation>(copy_allocation));
      } else if (allocation->is_sliced_copy_allocation()) {
        auto sliced_copy_allocation =
            static_cast<SlicedCopyAllocation*>(allocation.get());
        for (int i = 0; i < sliced_copy_allocation
                                ->mutable_slice_details_sorted_by_start_time()
                                .size();
             ++i) {
          async_copy_steps.push_back(std::make_unique<AsyncCopyStepForSlice>(
              sliced_copy_allocation, i));
        }
        async_copy_steps.push_back(
            std::make_unique<AsyncCopyStepForSliceConcat>(
                sliced_copy_allocation));
      }
    }

    absl::c_stable_sort(
        async_copy_steps,
        [](const std::unique_ptr<AsyncCopyStep>& lhs,
           const std::unique_ptr<AsyncCopyStep>& rhs) { return *lhs < *rhs; });
    for (std::unique_ptr<AsyncCopyStep>& async_copy_step : async_copy_steps) {
      std::optional<AsyncCopyStep::StartPhase> start_phase =
          async_copy_step->start_phase();
      if (start_phase.has_value()) {
        // If the copy start doesn't happen to be scheduled at the correct
        // computation, delay it until the correct computation starts.
        int64_t copy_start_schedule_after = start_phase->schedule_after_time;

        // Accessing flattened_instructions_ here without checking if it is
        // nullptr is safe because this method is called before SimplifyGraph.
        while (
            async_copy_step->defining_position().instruction->parent() !=
            flattened_instructions_[
                // We can't use -1 to index into flatten_instructions_. However,
                // if we want to place the copy as first instruction, i.e.,
                // after the -1 scheduling position, its parent will be the same
                // as the first instruction, i.e., the one at the 0th position.
                std::max<int64_t>(0, copy_start_schedule_after)]
                ->parent()) {
          VLOG(4) << "Delaying CopyStart (" << copy_start_schedule_after
                  << " to " << (copy_start_schedule_after + 1) << ") for "
                  << start_phase->instruction->ToString()
                  << " because it is not in the correct computation.";
          async_copy_step->set_start_phase_schedule_after_time(
              ++copy_start_schedule_after);
        }
        start_phase = async_copy_step->start_phase();
        schedule_after_[start_phase->schedule_after_time].push_back(
            start_phase->instruction);
      }

      AsyncCopyStep::DonePhase done_phase = async_copy_step->done_phase();
      schedule_before_[done_phase.schedule_before_time].push_back(
          done_phase.instruction);
    }
  }
}

absl::Status MemorySpaceAssignment::FixSchedule() {
  VLOG(1) << "Fixing schedule...";
  TF_RET_CHECK(module_->has_schedule());
  HloSchedule& schedule = module_->schedule();
  for (const HloComputation* computation :
       module_->MakeNonfusionComputations()) {
    // Parallel computations aren't in the schedule and don't need to be
    // modified.
    if (!computations_in_schedule_.contains(computation)) {
      if (computation->IsAsyncComputation()) {
        VLOG(4) << "Created a dummy schedule for async computation "
                << computation->name();
        schedule.GetOrCreateSequence(computation);
        continue;
      }
      VLOG(4) << "Not scheduling " << computation->name()
              << " because it's not in the schedule.";
      continue;
    }
    TF_RET_CHECK(schedule.is_computation_scheduled(computation));
    HloInstructionSequence new_sequence;

    absl::flat_hash_set<HloInstruction*> inserted_instructions;

    VLOG(4) << "Scheduling: " << computation->ToString();

    for (int64_t instruction_index = -1;; ++instruction_index) {
      auto insts_before_iter = schedule_before_.find(instruction_index);
      if (insts_before_iter != schedule_before_.end()) {
        for (HloInstruction* new_instruction : insts_before_iter->second) {
          if (new_instruction->parent() == computation) {
            VLOG(4) << "before " << instruction_index << ": "
                    << new_instruction->ToString();
            InsertInstructionAndEnsureOperandsInserted(
                new_instruction, &new_sequence, &inserted_instructions);
          }
        }
      }

      if (instruction_index != -1) {
        // We allow scheduling copy dones past the root instruction (for
        // end-of-program cross-program prefetch). So the loop exit condition is
        // actually here.
        if (instruction_index >= flattened_instructions_.size()) {
          break;
        }

        HloInstruction* instruction =
            flattened_instructions_[instruction_index];
        // Insert only if it is not deleted (SimplifyGraph sets it to nullptr if
        // it was deleted) and not previously inserted. Also bitcasts and tuples
        // are treated specially and only inserted as a result of operand
        // dependencies.
        if (instruction != nullptr && instruction->parent() == computation &&
            instruction->opcode() != HloOpcode::kBitcast &&
            instruction->opcode() != HloOpcode::kTuple &&
            !inserted_instructions.contains(instruction)) {
          VLOG(4) << "inst " << instruction_index << ": "
                  << instruction->ToString();
          InsertInstructionAndEnsureOperandsInserted(instruction, &new_sequence,
                                                     &inserted_instructions);
        }
      }

      auto insts_after_iter = schedule_after_.find(instruction_index);
      if (insts_after_iter != schedule_after_.end()) {
        for (HloInstruction* new_instruction : insts_after_iter->second) {
          if (new_instruction->parent() == computation) {
            VLOG(4) << "after " << instruction_index << ": "
                    << new_instruction->ToString();
            InsertInstructionAndEnsureOperandsInserted(
                new_instruction, &new_sequence, &inserted_instructions);
          }
        }
      }
    }

    // For rare cases where the original sequence is empty, ensure the root
    // instruction and its dependencies are scheduled.
    InsertInstructionAndEnsureOperandsInserted(
        computation->root_instruction(), &new_sequence, &inserted_instructions);

    CHECK_EQ(new_sequence.size(), computation->instruction_count())
        << "New sequence for computation " << computation->name() << " has "
        << new_sequence.size() << " instructions, expects "
        << computation->instruction_count() << ".";
    schedule.set_sequence(computation, new_sequence);
  }

  TF_RETURN_IF_ERROR(schedule.Update());

  return absl::OkStatus();
}

absl::Status MemorySpaceAssignment::VerifyAndExportHeapSimulatorTrace(
    const HloAliasAnalysis& alias_analysis,
    std::vector<int64_t>* alt_mem_bytes_occupied) {
  VLOG(1) << "Verifying...";
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloLiveRange> hlo_live_range,
                      HloLiveRange::Run(module_->schedule(), alias_analysis,
                                        module_->entry_computation()));

  BufferIntervalTree interval_tree;
  absl::flat_hash_set<int64_t> seen_buffers;
  // The key for events is: time, is_free, value_id. This is so that the events
  // are sorted first by time, then within the same time, allocations are sorted
  // earlier than frees, and finally the value id as a tie breaker.
  std::map<std::tuple<int64_t, bool, int64_t>,
           std::tuple<const HloValue*, HeapSimulator::Chunk,
                      HeapSimulatorTrace::Event::Kind>>
      events;

  auto add_allocation_and_verify = [&](int64_t start_time, int64_t end_time,
                                       const HeapSimulator::Chunk& chunk,
                                       const HloValue* value) -> absl::Status {
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
    for (const HeapSimulator::Chunk& overlapping_chunk :
         interval_tree.ChunksOverlappingInTime(start_time, end_time - 1)) {
      if (chunk.OverlapsWith(overlapping_chunk)) {
        return Internal(
            ("Value %s (%d, %d) off: %d size: %d overlaps with another chunk"
             " off: %d size: %d"),
            value->ToShortString(), start_time, end_time, chunk.offset,
            chunk.size, overlapping_chunk.offset, overlapping_chunk.size);
      }
    }
    interval_tree.Add(start_time, end_time - 1, chunk);
    return absl::OkStatus();
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
    const HeapSimulator::Chunk& chunk = position_and_chunk.second;
    const HloBuffer& buffer =
        alias_analysis.GetUniqueBufferAt(position.instruction, position.index);
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
      for (const HloUse& use : value->GetUses()) {
        int64_t use_time =
            hlo_live_range->instruction_schedule().at(use.instruction);
        if (use_time > last_use_time) {
          last_use_time = use_time;
          last_use_instruction = use.instruction;
        }
      }

      std::function<absl::Status(const HloInstruction*, int64_t, int64_t,
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
          int64_t computation_start_time =
              hlo_live_range->computation_span_times()
                  .at(called_computation)
                  .start;
          earliest_computation_start_time =
              std::min(earliest_computation_start_time, computation_start_time);
          int64_t last_use_time = -1;
          const HloInstruction* last_use_instruction = nullptr;
          for (const HloUse& use : value->GetUses()) {
            int64_t use_time =
                hlo_live_range->instruction_schedule().at(use.instruction);
            if (use.instruction->parent() == called_computation &&
                use_time > last_use_time) {
              last_use_time = use_time;
              last_use_instruction = use.instruction;
            }
          }
          if (last_use_time != -1) {
            VLOG(3) << indent_string
                    << " computation: " << called_computation->name() << ": ("
                    << computation_start_time << ", " << last_use_time << ")";
            CHECK(last_use_instruction);
            last_use_time = std::min(last_use_time, end_time);
            if (last_use_instruction->opcode() == HloOpcode::kConditional) {
              // The last use is another (nested) conditional. Call this
              // function recursively.
              TF_RETURN_IF_ERROR(split_conditional_buffer(
                  last_use_instruction, computation_start_time, last_use_time,
                  absl::StrCat(indent_string, "  ")));
            } else {
              TF_RETURN_IF_ERROR(add_allocation_and_verify(
                  computation_start_time, last_use_time, chunk, value));
            }
          }
        }
        VLOG(3) << indent_string << " from beginning until first computation: ("
                << start_time << ", " << (earliest_computation_start_time - 1)
                << ")";
        TF_RETURN_IF_ERROR(add_allocation_and_verify(
            start_time, earliest_computation_start_time - 1, chunk, value));
        return absl::OkStatus();
      };

      if (last_use_instruction &&
          last_use_instruction->opcode() == HloOpcode::kConditional) {
        TF_RETURN_IF_ERROR(split_conditional_buffer(
            last_use_instruction, time_bound.start, time_bound.end, " "));
      } else if (!value->GetUses().empty()) {
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
  int64_t prev_time = 0;
  int64_t prev_memory_usage = 0;
  if (alt_mem_bytes_occupied) {
    // Populate alt_mem_bytes_occupied with -1, for each instruction.
    alt_mem_bytes_occupied->resize(
        hlo_live_range->flattened_instruction_sequence().instructions().size(),
        -1);
  }
  for (const auto& event : events) {
    int64_t time;
    bool is_free;
    int64_t buffer_id;
    std::tie(time, is_free, buffer_id) = event.first;
    const HloValue* value;
    HeapSimulator::Chunk chunk;
    HeapSimulatorTrace::Event::Kind kind;
    std::tie(value, chunk, kind) = event.second;
    HeapSimulatorTrace::Event* heap_trace_event = heap_trace->add_events();
    heap_trace_event->set_kind(kind);
    heap_trace_event->set_buffer_id(buffer_id);
    *heap_trace_event->mutable_instruction_name() =
        std::string(value->instruction()->name());
    *heap_trace_event->mutable_computation_name() =
        std::string(value->instruction()->parent()->name());

    if (prev_time != time) {
      VLOG(2) << "Memory usage: " << std::max(memory_usage, prev_memory_usage)
              << " at time: " << prev_time << " ("
              << hlo_live_range->flattened_instruction_sequence()
                     .instructions()
                     .at(prev_time)
                     ->name()
              << ")";
      prev_time = time;
      prev_memory_usage = memory_usage;
    }
    if (kind == HeapSimulatorTrace::Event::ALLOC) {
      memory_usage += chunk.size;
    } else {
      CHECK_EQ(kind, HeapSimulatorTrace::Event::FREE);
      memory_usage -= chunk.size;
    }
    prev_memory_usage = std::max(prev_memory_usage, memory_usage);
    max_memory_usage = std::max(max_memory_usage, memory_usage);
    if (alt_mem_bytes_occupied) {
      // Update known alt mem usage.
      (*alt_mem_bytes_occupied)[time] = memory_usage;
    }
  }
  if (alt_mem_bytes_occupied) {
    // Replace the -1s in alt_mem_bytes_occupied with the previous alt memory
    // usage.
    int64_t prev_bytes = 0;
    for (int64_t i = 0; i < alt_mem_bytes_occupied->size(); ++i) {
      if ((*alt_mem_bytes_occupied)[i] == -1) {
        (*alt_mem_bytes_occupied)[i] = prev_bytes;
      }
      prev_bytes = (*alt_mem_bytes_occupied)[i];
    }
  }

  VLOG(1) << "Max memory usage ignoring fragmentation: " << max_memory_usage;

  return absl::OkStatus();
}

}  // namespace memory_space_assignment
}  // namespace xla

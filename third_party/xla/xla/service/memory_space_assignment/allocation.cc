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

#include "xla/service/memory_space_assignment/allocation.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/service/heap_simulator/allocation_block.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo_value.h"
#include "xla/service/memory_space_assignment/memory_space_assignment.pb.h"
#include "xla/service/memory_space_assignment/slice.h"
#include "xla/service/time_utils.h"
#include "xla/service/tuple_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla::memory_space_assignment {
namespace {

std::string UsesToString(const std::vector<HloUse>& uses) {
  if (uses.empty()) {
    return "none";
  }
  std::vector<std::string> uses_str;
  uses_str.reserve(uses.size());
  for (const auto& use : uses) {
    uses_str.push_back(use.ToString());
  }
  return absl::StrJoin(uses_str, ",");
}

// Helper function to compute the start time for a SlicedCopyAllocation.
int64_t GetSlicedCopyAllocationExclusiveStartTime(
    const std::vector<SliceDecision>&
        slice_decisions_sorted_by_exclusive_start_time) {
  if (slice_decisions_sorted_by_exclusive_start_time.empty()) {
    return -1;
  }

  return slice_decisions_sorted_by_exclusive_start_time.front()
      .exclusive_start_time;
}

// Helper function to compute the underlying Allocation chunk for a
// SlicedCopyAllocation.
std::optional<HeapSimulator::Chunk> GetSlicedCopyAllocationChunk(
    const std::vector<SliceDecision>& slice_decisions_sorted_by_start_time) {
  if (slice_decisions_sorted_by_start_time.empty()) {
    return std::nullopt;
  }
  auto offset_cmp = [](const SliceDecision& lhs, const SliceDecision& rhs) {
    return lhs.chunk.offset < rhs.chunk.offset;
  };
  auto end_cmp = [](const SliceDecision& lhs, const SliceDecision& rhs) {
    return lhs.chunk.chunk_end() < rhs.chunk.chunk_end();
  };
  return HeapSimulator::Chunk::FromOffsetEnd(
      std::min_element(slice_decisions_sorted_by_start_time.begin(),
                       slice_decisions_sorted_by_start_time.end(), offset_cmp)
          ->chunk.offset,
      std::max_element(slice_decisions_sorted_by_start_time.begin(),
                       slice_decisions_sorted_by_start_time.end(), end_cmp)
          ->chunk.chunk_end());
}

}  // namespace

std::string MemorySpaceToString(MemorySpace memory_space) {
  switch (memory_space) {
    case MemorySpace::kDefault:
      return "def";
    case MemorySpace::kAlternate:
      return "alt";
  }
}

std::optional<int64_t> Allocation::cross_program_prefetch_index() const {
  return cross_program_prefetch_index_;
}

HeapSimulator::Chunk Allocation::chunk() const {
  CHECK(chunk_.has_value());
  return *chunk_;
}

void Allocation::set_offset(int64_t offset) {
  CHECK(chunk_.has_value());
  *chunk_ = HeapSimulator::Chunk::FromOffsetSize(offset, chunk_->size);
}

bool Allocation::is_in_alternate_mem() const {
  return memory_space_ == MemorySpace::kAlternate;
}

bool Allocation::is_in_default_mem() const {
  return memory_space_ == MemorySpace::kDefault;
}

void Allocation::RemoveUse(HloUse use) {
  uses_.erase(std::remove_if(uses_.begin(), uses_.end(),
                             [=](const auto& u) { return u == use; }),
              uses_.end());
}

void Allocation::AddUse(HloUse use) {
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

absl::Status Allocation::UpdateUses(HloComputation* computation,
                                    HloInstruction* producing_instruction,
                                    const BitcastSplitFn& bitcast_split_fn) {
  for (const HloUse& use : uses()) {
    HloInstruction* replacement_instruction = producing_instruction;
    Shape operand_shape = use.instruction->operand(use.operand_number)->shape();
    if (operand_shape.IsTuple()) {
      TF_ASSIGN_OR_RETURN(
          replacement_instruction,
          TupleUtil::ReplaceTupleWith(
              producing_instruction,
              use.instruction->mutable_operand(use.operand_number),
              use.operand_index));
    } else if (!Shape::Equal().IgnoreSplitConfigInLayout()(
                   operand_shape, producing_instruction->shape())) {
      // When processing allocations, we treat bitcasts as trivial positions and
      // do not create allocations for them. We insert bitcasts after copies, to
      // account for the fact that we don't have an allocation for the bitcast.
      VLOG(4) << "Old shape = " << operand_shape.ToString()
              << ", new shape = " << producing_instruction->shape().ToString()
              << "; inserting a bitcast.";
      replacement_instruction = computation->AddInstruction(
          HloInstruction::CreateBitcast(operand_shape, producing_instruction));
      if (mutable_split_shape().has_value() &&
          producing_instruction->shape().layout().split_configs_size() != 0) {
        TF_ASSIGN_OR_RETURN(
            int64_t split_dim,
            bitcast_split_fn(
                replacement_instruction,
                mutable_split_shape()->layout().split_configs(0).dimension()));
        SplitConfig split_config(
            split_dim,
            {mutable_split_shape()->layout().split_configs(0).split_indices(
                0)});
        replacement_instruction->mutable_shape()
            ->mutable_layout()
            ->add_split_configs(split_config);
      }
    }
    TF_RETURN_IF_ERROR(use.instruction->ReplaceOperandWith(
        use.operand_number, replacement_instruction));
  }
  return absl::OkStatus();
}

bool Allocation::is_copy_like_allocation() const {
  return is_copy_allocation() || is_sliced_copy_allocation();
}

HloInstruction* Allocation::AddGetTupleElements() const {
  CHECK_NE(defining_position().instruction, nullptr);

  Shape shape = defining_position().shape();
  CHECK(shape.IsArray()) << "Allocation shape is not an array. Shape = "
                         << shape.ToString()
                         << " position = " << defining_position().shape();
  return TupleUtil::AddGetTupleElements(defining_position());
}

Allocation::Allocation(HloPosition defining_position, MemorySpace memory_space,
                       std::optional<HeapSimulator::Chunk> chunk,
                       int64_t start_time, int64_t end_time,
                       bool is_scoped_allocation,
                       std::optional<int64_t> cross_program_prefetch_index)
    : original_defining_position_(std::move(defining_position)),
      memory_space_(memory_space),
      chunk_(chunk),
      start_time_(start_time),
      end_time_(end_time),
      is_scoped_allocation_(is_scoped_allocation),
      cross_program_prefetch_index_(cross_program_prefetch_index),
      split_shape_(std::nullopt) {
  CHECK(!is_scoped_allocation ||
        original_defining_position_.index == ShapeIndex({}));
}

HloPosition Allocation::original_defining_position() const {
  return original_defining_position_;
}

void Allocation::set_original_defining_position(HloPosition defining_position) {
  original_defining_position_ = std::move(defining_position);
}

bool Allocation::base_is_equal(const Allocation& other) const {
  return defining_position() == other.defining_position() &&
         uses() == other.uses() && memory_space() == other.memory_space() &&
         chunk() == other.chunk() && start_time() == other.start_time() &&
         end_time() == other.end_time() &&
         earliest_available_time() == other.earliest_available_time() &&
         is_copy_allocation() == other.is_copy_allocation() &&
         is_scoped_allocation() == other.is_scoped_allocation();
}

PinnedAllocation::PinnedAllocation(HloPosition defining_position,
                                   MemorySpace memory_space,
                                   std::optional<HeapSimulator::Chunk> chunk,
                                   int64_t start_time, int64_t end_time,
                                   bool is_scoped_allocation)
    : Allocation(std::move(defining_position), memory_space, chunk, start_time,
                 end_time, is_scoped_allocation,
                 /*cross_program_prefetch_index=*/std::nullopt) {}

HloPosition PinnedAllocation::defining_position() const {
  return original_defining_position();
}

bool PinnedAllocation::operator==(const PinnedAllocation& other) const {
  return this->base_is_equal(static_cast<const Allocation&>(other));
}

bool MirroredAllocation::operator==(const MirroredAllocation& other) const {
  return this->base_is_equal(static_cast<const Allocation&>(other));
}

bool ParentAllocation::operator==(const ParentAllocation& other) const {
  return this->base_is_equal(static_cast<const Allocation&>(other));
}

bool PinnedAllocation::operator==(const Allocation& other) const {
  const PinnedAllocation* casted_other =
      dynamic_cast<const PinnedAllocation*>(&other);
  return casted_other != nullptr && (*this) == (*casted_other);
}

absl::Status PinnedAllocation::Process(const BitcastSplitFn& bitcast_split_fn) {
  if (is_scoped_allocation()) {
    // Nothing to do here for scoped allocations.
    return absl::OkStatus();
  }
  HloInstruction* producing_instruction = AddGetTupleElements();
  HloComputation* computation = producing_instruction->parent();

  if (memory_space() == MemorySpace::kAlternate &&
      mutable_split_shape().has_value()) {
    CHECK(Shape::Equal().IgnoreSplitConfigInLayout()(
        producing_instruction->shape(), mutable_split_shape().value()));
    *producing_instruction->mutable_shape() = mutable_split_shape().value();
  }
  return UpdateUses(computation, producing_instruction, bitcast_split_fn);
}

std::string PinnedAllocation::ToString() const {
  std::string memory_space_str = MemorySpaceToString(memory_space());
  std::optional<HeapSimulator::Chunk> chunk = maybe_chunk();
  if (chunk) {
    absl::StrAppend(&memory_space_str, " (off: ", chunk->offset,
                    ", size: ", chunk->size, ")");
  }
  return absl::StrCat((is_scoped_allocation() ? "Scoped " : ""),
                      "PinnedAllocation in ", memory_space_str, " defined at ",
                      original_defining_position().ToString(),
                      ", start_time:", start_time(), ", end_time:", end_time(),
                      ", uses: ", UsesToString(uses()));
}

void PinnedAllocation::MarkIfNeeded(
    absl::flat_hash_set<const Allocation*>& needed_allocations) const {
  MarkNeeded(needed_allocations);
}

void PinnedAllocation::MarkNeeded(
    absl::flat_hash_set<const Allocation*>& needed_allocations) const {
  needed_allocations.insert(this);
}

CopyAllocation::CopyAllocation(
    Allocation& prev_allocation, MemorySpace memory_space,
    std::optional<HeapSimulator::Chunk> chunk,
    int64_t copy_start_schedule_after_time,
    int64_t copy_done_schedule_before_time, int64_t end_time,
    std::optional<int64_t> cross_program_prefetch_index,
    HloInstruction* sync_mem_op)
    : Allocation(
          /*defining_position=*/{nullptr, {}}, memory_space, chunk,
          // Allocation uses an inclusive start time
          ExclusiveToInclusiveStartTime(copy_start_schedule_after_time),
          end_time,
          /*is_scoped_allocation=*/false, cross_program_prefetch_index),
      prev_allocation_(prev_allocation),
      copy_start_schedule_after_(copy_start_schedule_after_time),
      copy_done_schedule_before_(copy_done_schedule_before_time),
      sync_mem_op_(sync_mem_op) {}

int64_t CopyAllocation::earliest_available_time() const {
  return copy_done_schedule_before_;
}

absl::Status CopyAllocation::Process(const BitcastSplitFn& bitcast_split_fn) {
  // Copy allocations need to insert asynchronous copy nodes.
  Shape shape = defining_position().shape();
  if (memory_space() == MemorySpace::kAlternate && sync_mem_op_ != nullptr &&
      mutable_split_shape().has_value()) {
    *sync_mem_op_->mutable_shape() = mutable_split_shape().value();
  }
  HloInstruction* producing_instruction = AddGetTupleElements();
  HloComputation* computation = producing_instruction->parent();
  if (sync_mem_op_ != nullptr && sync_mem_op_->opcode() != HloOpcode::kCopy) {
    if (sync_mem_op_->opcode() == HloOpcode::kSlice ||
        sync_mem_op_->opcode() == HloOpcode::kDynamicSlice) {
      TF_ASSIGN_OR_RETURN(copy_done_,
                          computation->CreateAsyncInstructions(
                              sync_mem_op_, {ShapeUtil::MakeShape(S32, {})},
                              HloInstruction::kMainExecutionThread, false));
    } else {
      return Internal("Sync mem op is not a copy, slice, or dynamic slice.");
    }
    copy_start_ = copy_done_->mutable_operand(0);
    // If the shape of the copy start operand is not compatible with the
    // shape of the producing instruction, we insert a bitcast to make them
    // compatible.
    if (!ShapeUtil::CompatibleIgnoringFpPrecision(
            producing_instruction->shape(), copy_start_->operand(0)->shape())) {
      producing_instruction =
          computation->AddInstruction(HloInstruction::CreateBitcast(
              copy_start_->operand(0)->shape(), producing_instruction));
    }
    TF_RETURN_IF_ERROR(
        copy_start_->ReplaceOperandWith(0, producing_instruction));
  } else {
    Shape dest_shape;
    if (memory_space() == MemorySpace::kAlternate &&
        mutable_split_shape().has_value()) {
      dest_shape = mutable_split_shape().value();
    } else if (memory_space() == MemorySpace::kDefault && shape.has_layout() &&
               shape.layout().split_configs_size() > 0) {
      dest_shape = shape;
      dest_shape.mutable_layout()->clear_split_configs();
    } else {
      dest_shape = shape;
    }
    copy_start_ = computation->AddInstruction(HloInstruction::CreateCopyStart(
        ShapeUtil::MakeTupleShape(
            {dest_shape, shape, ShapeUtil::MakeShape(U32, {})}),
        producing_instruction, cross_program_prefetch_index()));
    copy_done_ = computation->AddInstruction(HloInstruction::CreateUnary(
        dest_shape, HloOpcode::kCopyDone, copy_start_));
  }
  VLOG(4) << "Created " << copy_start_->name()
          << " for copy allocation: " << ToString();

  // Update the allocation position with the copy complete instruction, so that
  // if there are further copies from it, they can find the correct position.
  set_original_defining_position(HloPosition{copy_done_, {}});
  return UpdateUses(computation, copy_done_, bitcast_split_fn);
}

void CopyAllocation::MarkIfNeeded(
    absl::flat_hash_set<const Allocation*>& needed_allocations) const {
  MarkNeeded(needed_allocations);
}

void CopyAllocation::MarkNeeded(
    absl::flat_hash_set<const Allocation*>& needed_allocations) const {
  needed_allocations.insert(this);
  prev_allocation_.MarkNeeded(needed_allocations);
}

std::string CopyAllocation::ToString() const {
  std::string memory_space_str = MemorySpaceToString(memory_space());
  std::optional<HeapSimulator::Chunk> chunk = maybe_chunk();
  if (chunk) {
    absl::StrAppend(&memory_space_str, " (off: ", chunk->offset,
                    ", size: ", chunk->size, ")");
  }
  return absl::StrCat("Copy Allocation in ", memory_space_str,
                      ", start_time:", start_time(), ", end_time:", end_time(),
                      ", copy_start_after_time: ", copy_start_schedule_after(),
                      ", copy_done_before_time: ", copy_done_schedule_before(),
                      ", uses: ", UsesToString(uses()), ", sync_mem_op: ",
                      sync_mem_op_ ? sync_mem_op_->name() : "none", ", from ",
                      prev_allocation_.ToString());
}

HloPosition CopyAllocation::defining_position() const {
  // Unless explicitly set, the defining position of a copy allocation is
  // retrieved from the previous allocation. This is because we don't create
  // new CopyStart/CopyDone instructions until later and the position should
  // point to the previous (copy or otherwise) allocation's position for the
  // original defining position.
  HloPosition defining_position = original_defining_position();
  if (defining_position.instruction == nullptr) {
    return prev_allocation_.defining_position();
  }
  return defining_position;
}

bool CopyAllocation::operator==(const CopyAllocation& other) const {
  return this->base_is_equal(static_cast<const Allocation&>(other)) &&
         copy_done_schedule_before() == other.copy_done_schedule_before() &&
         copy_start_schedule_after() == other.copy_start_schedule_after() &&
         copy_start() == other.copy_start() && copy_done() == other.copy_done();
}

bool CopyAllocation::operator==(const Allocation& other) const {
  const CopyAllocation* casted_other =
      dynamic_cast<const CopyAllocation*>(&other);
  return casted_other != nullptr && (*this) == (*casted_other);
}

void CopyAllocation::set_copy_start_schedule_after(
    int64_t copy_start_schedule_after) {
  copy_start_schedule_after_ = copy_start_schedule_after;
}

void CopyAllocation::set_copy_done_schedule_before(
    int64_t copy_done_schedule_before) {
  copy_done_schedule_before_ = copy_done_schedule_before;
}

int64_t CopyAllocation::copy_start_schedule_after() const {
  return copy_start_schedule_after_;
}

int64_t CopyAllocation::copy_done_schedule_before() const {
  return copy_done_schedule_before_;
}

SlicedCopyAllocation::SlicedCopyAllocation(
    const Allocation& prev_allocation, MemorySpace memory_space,
    std::vector<SliceDecision> slice_decisions_sorted_by_exclusive_start_time,
    int64_t copy_done_schedule_before_time, int64_t end_time,
    const SlicedPrefetchOptions& sliced_prefetch_options,
    absl::FunctionRef<Shape(const Shape&)> get_equivalent_s8_shape_fn,
    HloInstruction* sync_mem_op)
    : Allocation(
          /*defining_position=*/{nullptr, {}}, memory_space,
          GetSlicedCopyAllocationChunk(
              slice_decisions_sorted_by_exclusive_start_time),
          // Allocation uses an inclusive start time
          ExclusiveToInclusiveStartTime(
              GetSlicedCopyAllocationExclusiveStartTime(
                  slice_decisions_sorted_by_exclusive_start_time)),
          end_time,
          /*is_scoped_allocation=*/false,
          /*cross_program_prefetch_index=*/std::nullopt),
      original_shape_to_slice_(prev_allocation.defining_position().shape()),
      prev_allocation_(prev_allocation),
      sliced_prefetch_options_(sliced_prefetch_options),
      get_equivalent_s8_shape_fn_(get_equivalent_s8_shape_fn),
      sync_mem_op_(sync_mem_op) {
  CHECK_GE(slice_decisions_sorted_by_exclusive_start_time.size(), 2);
  slice_details_sorted_by_exclusive_start_time_.reserve(
      slice_decisions_sorted_by_exclusive_start_time.size());
  for (SliceDecision& decision :
       slice_decisions_sorted_by_exclusive_start_time) {
    int64_t copy_done_schedule_after_time = decision.exclusive_start_time;
    slice_details_sorted_by_exclusive_start_time_.push_back(SliceDetail{
        std::move(decision),
        copy_done_schedule_after_time,
        copy_done_schedule_before_time,
        /*copy_start=*/nullptr,
        /*copy_done=*/nullptr,
    });
  }
}

absl::Status SlicedCopyAllocation::Process(
    const BitcastSplitFn& bitcast_split_fn) {
  Shape shape = defining_position().shape();
  HloInstruction* producing_instruction = AddGetTupleElements();

  // Calling Process() over the previous allocation might
  // have modified the defining position, and hence the shape that was used when
  // we computed the slices. In cases where the shape has changed, we insert a
  // bitcast, so slice instructions operate on the originally sliced shape.
  //
  // Note, these bitcasts are being inserted in the same cases that
  // UpdateUses() is inserting bitcasts, except we are
  // inserting the bitcasts before the copy, instead of after the copy.
  if (!Shape::Equal().IgnoreMemorySpaceInLayout()(shape,
                                                  original_shape_to_slice_)) {
    int64_t new_memory_space = shape.layout().memory_space();
    shape = original_shape_to_slice_;
    shape.mutable_layout()->set_memory_space(new_memory_space);
    producing_instruction = producing_instruction->parent()->AddInstruction(
        HloInstruction::CreateBitcast(shape, producing_instruction));
  }

  HloComputation* computation = producing_instruction->parent();
  std::vector<HloInstruction*> slice_dones;
  slice_dones.reserve(slice_details_sorted_by_exclusive_start_time_.size());

  // If we are trying to make all slices a uniform size, we bitcast the
  // producing instruction to an array of bytes, so it is easy to slice into any
  // size.
  Shape slice_shape = shape;
  if (IsUniformSliceSizingEnabled(sliced_prefetch_options_)) {
    slice_shape = get_equivalent_s8_shape_fn_(shape);
    producing_instruction = producing_instruction->parent()->AddInstruction(
        HloInstruction::CreateBitcast(slice_shape, producing_instruction));
  }

  // Sliced copy allocations need to insert asynchronous copy nodes.
  for (SliceDetail& slice_detail :
       slice_details_sorted_by_exclusive_start_time_) {
    TF_RETURN_IF_ERROR(slice_detail.CreateAsyncSlice(
        slice_shape, *producing_instruction, *computation));
    VLOG(4) << "Created " << slice_detail.copy_start->name()
            << " for sliced copy allocation: " << ToString();
    slice_dones.push_back(slice_detail.copy_done);
  }

  TF_RETURN_IF_ERROR(CreateBitcastConcat(shape, slice_dones));

  // If we bitcast to an array of bytes above, the result of the concatenated
  // slices will also be an array of bytes. Thus, we need to cast the
  // concatenation back to the original shape.
  if (IsUniformSliceSizingEnabled(sliced_prefetch_options_)) {
    concat_ = concat_->parent()->AddInstruction(
        HloInstruction::CreateBitcast(shape, concat_));
  }

  // Update the allocation position with the copy complete instruction, so that
  // if there are further copies from it, they can find the correct position.
  set_original_defining_position(HloPosition{concat_, {}});
  return UpdateUses(computation, concat_, bitcast_split_fn);
}

void SlicedCopyAllocation::MarkIfNeeded(
    absl::flat_hash_set<const Allocation*>& needed_allocations) const {
  MarkNeeded(needed_allocations);
}

void SlicedCopyAllocation::MarkNeeded(
    absl::flat_hash_set<const Allocation*>& needed_allocations) const {
  needed_allocations.insert(this);
  prev_allocation_.MarkNeeded(needed_allocations);
}

HloPosition SlicedCopyAllocation::defining_position() const {
  // Unless explicitly set, the defining position of a sliced copy allocation is
  // retrieved from the previous allocation. This is because we don't create
  // new CopyStart/CopyDone instructions until later and the position should
  // point to the previous (copy or otherwise) allocation's position for the
  // original defining position.
  HloPosition defining_position = original_defining_position();
  if (defining_position.instruction == nullptr) {
    return prev_allocation_.defining_position();
  }
  return defining_position;
}

int64_t SlicedCopyAllocation::earliest_available_time() const {
  return slice_details_sorted_by_start_time().back().copy_done_before_time;
}

std::vector<int64_t> SlicedCopyAllocation::SliceOffsetsSortedByStartTime()
    const {
  std::vector<int64_t> offsets;
  offsets.reserve(slice_details_sorted_by_exclusive_start_time_.size());

  for (const SliceDetail& slice_detail :
       slice_details_sorted_by_exclusive_start_time_) {
    offsets.push_back(slice_detail.slice_decision.chunk.offset);
  }

  return offsets;
}

void SlicedCopyAllocation::AddDiffToAllSliceOffsets(int64_t diff) {
  for (SliceDetail& slice_detail :
       slice_details_sorted_by_exclusive_start_time_) {
    HeapSimulator::Chunk& chunk = slice_detail.slice_decision.chunk;
    chunk =
        HeapSimulator::Chunk::FromOffsetSize(chunk.offset + diff, chunk.size);
  }
}

void SlicedCopyAllocation::ImportRepackedSliceData(
    const SlicedAllocationData& data) {
  int num_slices = slice_details_sorted_by_exclusive_start_time_.size();
  CHECK_EQ(data.slices_sorted_by_offset.size(), num_slices);

  std::vector<SliceDetail*> slice_details_sorted_by_offset;
  slice_details_sorted_by_offset.reserve(num_slices);
  for (SliceDetail& slice_detail :
       slice_details_sorted_by_exclusive_start_time_) {
    slice_details_sorted_by_offset.push_back(&slice_detail);
  }
  absl::c_sort(slice_details_sorted_by_offset, [](const SliceDetail* lhs,
                                                  const SliceDetail* rhs) {
    return lhs->slice_decision.chunk.offset < rhs->slice_decision.chunk.offset;
  });

  for (int i = 0; i < num_slices; ++i) {
    SliceDetail* slice_detail = slice_details_sorted_by_offset[i];
    HeapSimulator::Chunk& chunk = slice_detail->slice_decision.chunk;
    const AllocatedSlice& repacked_slice_data = data.slices_sorted_by_offset[i];
    chunk = HeapSimulator::Chunk::FromOffsetSize(repacked_slice_data.offset,
                                                 chunk.size);
    slice_detail->copy_start_after_time =
        repacked_slice_data.inclusive_start_time - 1;
    slice_detail->slice_decision.exclusive_start_time =
        InclusiveToExclusiveStartTime(repacked_slice_data.inclusive_start_time);
  }

  absl::c_sort(slice_details_sorted_by_exclusive_start_time_,
               [](const SliceDetail& lhs, const SliceDetail& rhs) {
                 return std::make_tuple(lhs.copy_start_after_time,
                                        lhs.slice_decision.chunk.offset) <
                        std::make_tuple(rhs.copy_start_after_time,
                                        rhs.slice_decision.chunk.offset);
               });
}

const std::vector<SlicedCopyAllocation::SliceDetail>&
SlicedCopyAllocation::slice_details_sorted_by_start_time() const {
  return slice_details_sorted_by_exclusive_start_time_;
}

std::vector<SlicedCopyAllocation::SliceDetail>&
SlicedCopyAllocation::mutable_slice_details_sorted_by_start_time() {
  return slice_details_sorted_by_exclusive_start_time_;
}

bool SlicedCopyAllocation::operator==(const SlicedCopyAllocation& other) const {
  return this->base_is_equal(static_cast<const Allocation&>(other)) &&
         slice_details_sorted_by_exclusive_start_time_ ==
             other.slice_details_sorted_by_exclusive_start_time_ &&
         concat_ == other.concat_;
}

std::string SlicedCopyAllocation::ToString() const {
  std::string memory_space_str = MemorySpaceToString(memory_space());
  if (memory_space() == MemorySpace::kAlternate) {
    absl::StrAppend(&memory_space_str, " (off: ", maybe_chunk()->offset, ")");
  }
  return absl::StrCat(
      "Sliced Copy Allocation in ", memory_space_str,
      ", start_time:", start_time(), ", end_time:", end_time(),
      ", first_slice_copy_start_after_time: ",
      slice_details_sorted_by_start_time().front().copy_start_after_time,
      ", last_slice_copy_done_before_time: ",
      slice_details_sorted_by_start_time().back().copy_done_before_time,
      ", uses: ", UsesToString(uses()),
      ", sync_mem_op: ", sync_mem_op_ ? sync_mem_op_->name() : "none",
      ", from ", prev_allocation_.ToString());
}

absl::Status SlicedCopyAllocation::CreateBitcastConcat(
    const Shape& shape, absl::Span<HloInstruction* const> slices) {
  CHECK(!slices.empty());
  concat_ =
      slices.front()->parent()->AddInstruction(HloInstruction::CreateCustomCall(
          shape, slices,
          xla::memory_space_assignment::kConcatBitcastCustomCall));
  return absl::OkStatus();
}

std::string SlicedCopyAllocation::SliceDetail::ToString() const {
  return absl::StrCat("{ slice_decision: ", slice_decision.ToString(),
                      ", copy_start_after_time: ", copy_start_after_time,
                      ", copy_done_before_time: ", copy_done_before_time, " }");
}

std::tuple<const SliceDecision&, int64_t, int64_t, const HloInstruction*,
           const HloInstruction*>
SliceDetailToTuple(const SlicedCopyAllocation::SliceDetail& slice_detail) {
  return std::make_tuple(std::ref(slice_detail.slice_decision),
                         slice_detail.copy_start_after_time,
                         slice_detail.copy_done_before_time,
                         slice_detail.copy_start, slice_detail.copy_done);
}

bool SlicedCopyAllocation::SliceDetail::operator==(
    const SliceDetail& other) const {
  return SliceDetailToTuple(*this) == SliceDetailToTuple(other);
}

absl::Status SlicedCopyAllocation::SliceDetail::CreateAsyncSlice(
    const Shape& original_shape, HloInstruction& producer,
    HloComputation& parent) {
  if (original_shape.dimensions().size() !=
      slice_decision.sizing.slice_params.size()) {
    return FailedPrecondition(
        "%s", absl::StrCat("The number of SlicedCopyAllocation parameters ",
                           slice_decision.sizing.slice_params.size(),
                           " does not match the rank ",
                           original_shape.dimensions().size(),
                           " of the tensor we are slicing."));
  }

  std::vector<int64_t> start_indices;
  start_indices.reserve(slice_decision.sizing.slice_params.size());
  std::vector<int64_t> limit_indices;
  limit_indices.reserve(slice_decision.sizing.slice_params.size());
  std::vector<int64_t> strides;
  strides.reserve(slice_decision.sizing.slice_params.size());

  for (int i = 0; i < slice_decision.sizing.slice_params.size(); ++i) {
    const SliceParam& slice_param = slice_decision.sizing.slice_params[i];
    start_indices.push_back(slice_param.start_inclusive);
    limit_indices.push_back(slice_param.end_exclusive);
    strides.push_back(1);
    const int64_t new_dim =
        slice_param.end_exclusive - slice_param.start_inclusive;
    if (new_dim <= 0) {
      return FailedPrecondition(
          "%s", absl::StrCat("SlicedCopyAllocation new dimension size is ",
                             new_dim, ", expected something > 0."));
    }
    if (original_shape.dimensions(i) < new_dim) {
      return FailedPrecondition(
          "%s",
          absl::StrCat("SlicedCopyAllocation sliced dimension size ", new_dim,
                       " is bigger than its original dimension size of ",
                       original_shape.dimensions(i), "."));
    }
  }

  HloInstruction* slice = parent.AddInstruction(
      HloInstruction::CreateSlice(slice_decision.sizing.slice_shape, &producer,
                                  start_indices, limit_indices, strides));
  TF_ASSIGN_OR_RETURN(copy_done, parent.CreateAsyncInstructions(
                                     slice, {ShapeUtil::MakeShape(S32, {})}));
  copy_start = copy_done->mutable_operand(0);

  return absl::OkStatus();
}

bool SlicedCopyAllocation::operator==(const Allocation& other) const {
  const SlicedCopyAllocation* casted_other =
      dynamic_cast<const SlicedCopyAllocation*>(&other);
  return casted_other != nullptr && (*this) == (*casted_other);
}

HloPosition MirroredAllocation::defining_position() const {
  return original_defining_position();
}

std::string MirroredAllocation::ToString() const {
  return absl::StrCat("Mirrored Allocation for ",
                      original_allocation_.ToString());
}

std::string ParentAllocation::ToString() const {
  return absl::StrCat("Parent Allocation mirrored at ",
                      original_defining_position().ToString(), ", originally ",
                      original_allocation_.ToString());
}

MirroredAllocation::MirroredAllocation(const Allocation& original_allocation,
                                       int64_t time)
    : Allocation(original_allocation.defining_position(), MemorySpace::kDefault,
                 original_allocation.maybe_chunk(),
                 /*start_time=*/time,
                 /*end_time=*/time, /*is_scoped_allocation=*/false,
                 /*cross_program_prefetch_index=*/std::nullopt),
      original_allocation_(original_allocation) {}

absl::Status MirroredAllocation::Process(
    const BitcastSplitFn& bitcast_split_fn) {
  set_original_defining_position(original_allocation_.defining_position());
  if (is_scoped_allocation()) {
    // Nothing to do here for scoped allocations.
    return absl::OkStatus();
  }
  HloInstruction* producing_instruction = AddGetTupleElements();
  HloComputation* computation = producing_instruction->parent();
  return UpdateUses(computation, producing_instruction, bitcast_split_fn);
}

ParentAllocation::ParentAllocation(const Allocation& original_allocation,
                                   HloInstruction* calling_instruction,
                                   HloPosition position, int64_t time)
    : Allocation(std::move(position), MemorySpace::kDefault,
                 original_allocation.maybe_chunk(),
                 /*start_time=*/time,
                 /*end_time=*/time, /*is_scoped_allocation=*/false,
                 /*cross_program_prefetch_index=*/std::nullopt),
      original_allocation_(original_allocation),
      calling_instruction_(calling_instruction) {}

HloPosition ParentAllocation::defining_position() const {
  return original_defining_position();
}

absl::Status ParentAllocation::Process(const BitcastSplitFn& bitcast_split_fn) {
  // Add an additional parameter to the while HLO with a reference to the buffer
  // in the default memory space.
  HloInstruction* producing_instruction =
      original_allocation_.AddGetTupleElements();
  int new_tuple_index = calling_instruction_->shape().tuple_shapes_size();

  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_while_operand,
      TupleUtil::ReplaceTupleWith(producing_instruction,
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
  HloPosition defining_position = original_defining_position();
  defining_position.index = {new_tuple_index};
  set_original_defining_position(defining_position);
  // Also replace the while op with a tuple that has the old shape. Note that we
  // need to first take a snapshot of the users before calling ExtractPrefix
  // since ExtractPrefix introduces additional gte users.
  std::vector<HloInstruction*> while_users = calling_instruction_->users();
  HloInstruction* tuple_with_old_shape =
      TupleUtil::ExtractPrefix(calling_instruction_, new_tuple_index);
  TF_RETURN_IF_ERROR(calling_instruction_->ReplaceAllUsesWithDifferentShape(
      while_users, tuple_with_old_shape));

  if (is_scoped_allocation()) {
    // Nothing to do here for scoped allocations.
    return absl::OkStatus();
  }
  HloInstruction* final_instruction = AddGetTupleElements();
  HloComputation* computation = final_instruction->parent();
  return UpdateUses(computation, final_instruction, bitcast_split_fn);
}

absl::Status ParentAllocation::PostProcess() {
  // Update the root of the while body with the new parameter. The reason why we
  // need a separate post-process for this is because other allocations may have
  // while body root as a use, so they would update the old root instead of the
  // new root. Doing the post-process step later ensures the root has been
  // updated with other changes, and we can safely add the additional parameter.
  HloComputation* while_body = calling_instruction_->while_body();
  TF_ASSIGN_OR_RETURN(HloInstruction * new_while_body_root,
                      TupleUtil::ReplaceTupleWith(
                          AddGetTupleElements(), while_body->root_instruction(),
                          original_defining_position().index));
  while_body->set_root_instruction(new_while_body_root,
                                   /*accept_different_shape=*/true);
  return absl::OkStatus();
}

void ParentAllocation::MarkIfNeeded(
    absl::flat_hash_set<const Allocation*>& needed_allocations) const {
  // Parent allocations are only needed if they have any uses or if there is a
  // copy allocation that copies this value (in that case, the copy allocation
  // will call this allocation's MarkNeeded function).
  if (!has_no_uses()) {
    MarkNeeded(needed_allocations);
  }
}

void ParentAllocation::MarkNeeded(
    absl::flat_hash_set<const Allocation*>& needed_allocations) const {
  needed_allocations.insert(this);
  original_allocation_.MarkNeeded(needed_allocations);
}

bool ParentAllocation::operator==(const Allocation& other) const {
  const ParentAllocation* casted_other =
      dynamic_cast<const ParentAllocation*>(&other);
  return casted_other != nullptr && (*this) == (*casted_other);
}

void MirroredAllocation::MarkIfNeeded(
    absl::flat_hash_set<const Allocation*>& needed_allocations) const {
  MarkNeeded(needed_allocations);
}

void MirroredAllocation::MarkNeeded(
    absl::flat_hash_set<const Allocation*>& needed_allocations) const {
  needed_allocations.insert(this);
  original_allocation_.MarkNeeded(needed_allocations);
}

bool MirroredAllocation::operator==(const Allocation& other) const {
  const MirroredAllocation* casted_other =
      dynamic_cast<const MirroredAllocation*>(&other);
  return casted_other != nullptr && (*this) == (*casted_other);
}

WindowPrefetchedAllocation::WindowPrefetchedAllocation(
    Allocation& prev_allocation, HloUse use, const HeapSimulator::Chunk& chunk,
    int64_t prefetch_start_schedule_after_time,
    int64_t prefetch_done_schedule_before_time, const Options& options)
    : Allocation(
          {nullptr, {}}, MemorySpace::kAlternate, chunk,
          ExclusiveToInclusiveStartTime(prefetch_start_schedule_after_time),
          InclusiveToExclusiveEndTime(prefetch_done_schedule_before_time),
          /*is_scoped_allocation=*/false,
          /*cross_program_prefetch_index=*/std::nullopt),
      options_(options),
      prev_allocation_(prev_allocation),
      use_(use),
      prefetch_start_schedule_after_(prefetch_start_schedule_after_time),
      prefetch_done_schedule_before_(prefetch_done_schedule_before_time),
      bytes_(chunk.size) {}

HloPosition WindowPrefetchedAllocation::defining_position() const {
  HloPosition defining_position = original_defining_position();
  if (defining_position.instruction == nullptr) {
    return prev_allocation_.defining_position();
  }
  return defining_position;
}

int64_t WindowPrefetchedAllocation::earliest_available_time() const {
  return prefetch_done_schedule_before_;
}

absl::Status WindowPrefetchedAllocation::InsertWindowPrefetchInstruction(
    HloInstruction* producing_instruction, HloInstruction* use_instruction,
    HloComputation* computation) {
  // Derive the shape for window buffer.
  Shape buffer_shape = ShapeUtil::MakeShape(U8, {options_.bytes});
  Layout layout = LayoutUtil::MakeLayout({0});
  layout.set_memory_space(options_.alternate_memory_space);
  *buffer_shape.mutable_layout() = layout;
  // Sync flag shape
  Shape sflag_shape = ShapeUtil::MakeShape(S32, {});
  // Output shape of the WindowPrefetch op.
  Shape output_shape = ShapeUtil::MakeTupleShape({buffer_shape, sflag_shape});

  // Insert WindowPrefetch op.
  HloInstruction* custom_call =
      computation->AddInstruction(HloInstruction::CreateCustomCall(
          output_shape, {producing_instruction}, "WindowPrefetch"));
  HloInstruction* get_buffer = computation->AddInstruction(
      HloInstruction::CreateGetTupleElement(buffer_shape, custom_call, 0));
  HloInstruction* get_sflag = computation->AddInstruction(
      HloInstruction::CreateGetTupleElement(sflag_shape, custom_call, 1));
  use_instruction->AppendOperand(get_buffer);
  use_instruction->AppendOperand(get_sflag);

  // The buffer's defining position is the get_tuple_element instruction.
  prefetch_instruction_ = get_buffer;
  return absl::OkStatus();
}

absl::Status WindowPrefetchedAllocation::Process(
    const BitcastSplitFn& bitcast_split_fn) {
  HloInstruction* producing_instruction = AddGetTupleElements();
  HloComputation* computation = producing_instruction->parent();
  HloInstruction* use_instruction = use_.instruction;
  int64_t use_operand = use_instruction->operand_count();
  CHECK_EQ(use_instruction->opcode(), HloOpcode::kFusion);

  TF_RETURN_IF_ERROR(InsertWindowPrefetchInstruction(
      producing_instruction, use_instruction, computation));

  // Notify the backend that an operand has been appended as a window prefetch
  // buffer.
  options_.notify_operand_appended_fn(use_instruction, use_.operand_number,
                                      use_operand);

  // Set the original defining position to the window prefetch instruction.
  set_original_defining_position(HloPosition{prefetch_instruction_, {}});
  AddUse(HloUse{use_instruction, use_operand});
  return absl::OkStatus();
}

void WindowPrefetchedAllocation::MarkIfNeeded(
    absl::flat_hash_set<const Allocation*>& needed_allocations) const {
  MarkNeeded(needed_allocations);
}

void WindowPrefetchedAllocation::MarkNeeded(
    absl::flat_hash_set<const Allocation*>& needed_allocations) const {
  needed_allocations.insert(this);
  prev_allocation_.MarkNeeded(needed_allocations);
}

std::string WindowPrefetchedAllocation::ToString() const {
  return absl::StrCat("WindowPrefetched Allocation");
}

bool WindowPrefetchedAllocation::operator==(
    const WindowPrefetchedAllocation& other) const {
  return this->base_is_equal(static_cast<const Allocation&>(other)) &&
         prefetch_done_schedule_before() ==
             other.prefetch_done_schedule_before() &&
         prefetch_start_schedule_after() ==
             other.prefetch_start_schedule_after() &&
         prefetch() == other.prefetch() && bytes_ == other.bytes_;
}

bool WindowPrefetchedAllocation::operator==(const Allocation& other) const {
  const WindowPrefetchedAllocation* casted_other =
      dynamic_cast<const WindowPrefetchedAllocation*>(&other);
  return casted_other != nullptr && (*this) == (*casted_other);
}

std::tuple<int64_t, bool, int64_t> GetAllocationSortTuple(
    const std::unique_ptr<Allocation>& allocation) {
  int64_t scheduled_on_or_before = allocation->start_time();
  int64_t scheduled_on_or_after = allocation->start_time();
  if (allocation->is_copy_allocation()) {
    auto copy_allocation =
        tensorflow::down_cast<CopyAllocation*>(allocation.get());
    scheduled_on_or_before = copy_allocation->copy_done_schedule_before();
    scheduled_on_or_after = copy_allocation->copy_start_schedule_after();
  }
  return std::forward_as_tuple(scheduled_on_or_before,
                               !allocation->is_copy_allocation(),
                               scheduled_on_or_after);
}

void SortAllocationSequence(AllocationSequence& allocations) {
  absl::c_sort(allocations, [](const std::unique_ptr<Allocation>& lhs,
                               const std::unique_ptr<Allocation>& rhs) {
    return GetAllocationSortTuple(lhs) < GetAllocationSortTuple(rhs);
  });
}

std::string AllocationSequenceToString(AllocationSequence& allocations,
                                       bool sort_allocations) {
  if (sort_allocations) {
    SortAllocationSequence(allocations);
  }
  std::string allocations_str = "\n";
  for (const std::unique_ptr<Allocation>& allocation : allocations) {
    absl::StrAppend(&allocations_str, allocation->ToString(), "\n");
  }
  return allocations_str;
}

std::vector<Allocation*> GetAllocationSequenceInRawPointers(
    AllocationSequence& allocations) {
  std::vector<Allocation*> allocations_in_raw_pointers;
  for (const std::unique_ptr<Allocation>& allocation : allocations) {
    allocations_in_raw_pointers.push_back(allocation.get());
  }
  return allocations_in_raw_pointers;
}

namespace {

struct AllocationSummary {
  static void Add(const Allocation& allocation,
                  std::vector<AllocationSummary>& data) {
    if (!allocation.is_sliced_copy_allocation()) {
      data.push_back(
          AllocationSummary{allocation.chunk(), allocation.start_time(),
                            allocation.end_time(), GetName(allocation)});
      return;
    }
    const SlicedCopyAllocation& sliced_copy_allocation =
        dynamic_cast<const SlicedCopyAllocation&>(allocation);
    for (size_t i = 0;
         i < sliced_copy_allocation.slice_details_sorted_by_start_time().size();
         ++i) {
      const SlicedCopyAllocation::SliceDetail& slice_detail =
          sliced_copy_allocation.slice_details_sorted_by_start_time()[i];
      data.push_back(AllocationSummary{
          slice_detail.slice_decision.chunk,
          ExclusiveToInclusiveStartTime(
              slice_detail.slice_decision.exclusive_start_time),
          sliced_copy_allocation.end_time(), GetName(allocation, i)});
    }
  }

  static std::string GetName(const Allocation& allocation,
                             std::optional<size_t> slice_index = std::nullopt) {
    std::vector<std::string> attributes;
    if (slice_index.has_value()) {
      attributes.push_back(absl::StrCat("slice ", *slice_index));
    }
    if (allocation.is_scoped_allocation()) {
      attributes.push_back("scoped");
    }
    if (allocation.cross_program_prefetch_index().has_value()) {
      attributes.push_back("cross-program prefetch");
    }
    if (attributes.empty()) {
      return allocation.defining_position().ToString();
    }
    return absl::StrCat(allocation.defining_position().ToString(), " (",
                        absl::StrJoin(attributes, ", "), ")");
  }

  HeapSimulator::Chunk chunk;
  int64_t start_time_inclusive;
  int64_t end_time_inclusive;
  std::string name;
};

}  // namespace

void AllocationSequenceDebugging::LogAltMemAllocationsAt(
    const AllocationSequence& allocations, int64_t time) {
  std::vector<AllocationSummary> data_vector;
  for (const std::unique_ptr<Allocation>& allocation : allocations) {
    if (allocation->start_time() <= time && allocation->end_time() >= time &&
        allocation->is_in_alternate_mem()) {
      AllocationSummary::Add(*allocation, data_vector);
    }
  }
  absl::c_sort(data_vector,
               [](const AllocationSummary& a, const AllocationSummary& b) {
                 if (a.chunk.offset != b.chunk.offset) {
                   return a.chunk.offset < b.chunk.offset;
                 }
                 return a.start_time_inclusive < b.start_time_inclusive;
               });
  LOG(INFO) << "Live allocations in alternate mem at instruction time " << time
            << " (before MSA alters the graph):";
  for (const AllocationSummary& data : data_vector) {
    LOG(INFO) << "Alt mem allocation in chunk " << data.chunk.ToString()
              << " during [" << data.start_time_inclusive << ","
              << data.end_time_inclusive << "], holding " << data.name;
  }
}

}  // namespace xla::memory_space_assignment

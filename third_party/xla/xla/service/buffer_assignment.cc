/* Copyright 2017 The OpenXLA Authors.

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

// Defines the data returned by the XLA buffer assignment packages.

#include "xla/service/buffer_assignment.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iterator>
#include <memory>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_op_metadata.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/map_util.h"
#include "xla/service/buffer_assignment.pb.h"
#include "xla/service/buffer_value.h"
#include "xla/service/call_graph.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_value.h"
#include "xla/service/logical_buffer.h"
#include "xla/service/memory_space_assignment/memory_space_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/numbers.h"

namespace xla {
namespace {

using absl::flat_hash_map;
using absl::flat_hash_set;
using absl::StrAppend;
using absl::StrAppendFormat;
using memory_space_assignment::PresetAssignments;
using ::tsl::strings::HumanReadableNumBytes;

absl::flat_hash_map<int64_t, const HloInstruction*> BuildIdToHloInstructionMap(
    const HloModule* module) {
  // Build a map from a unique_id to corresponding HloInstruction in the module.
  absl::flat_hash_map<int64_t, const HloInstruction*> id_to_hlo_instruction;
  for (const HloComputation* computation : module->computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      id_to_hlo_instruction[instruction->unique_id()] = instruction;
    }
  }
  return id_to_hlo_instruction;
}

absl::StatusOr<absl::flat_hash_map<int64_t, const HloValue*>>
BuildIdToLogicalBufferMap(
    const BufferAssignmentProto& proto,
    const absl::flat_hash_map<int64_t, const HloInstruction*>&
        id_to_hlo_instruction,
    const std::unique_ptr<HloAliasAnalysis>& alias_analysis) {
  absl::flat_hash_map<int64_t, const HloValue*> id_to_logical_buffer;
  // Process each logical buffer in the proto.
  for (const LogicalBufferProto& logical_buffer_proto :
       proto.logical_buffers()) {
    TF_RET_CHECK(logical_buffer_proto.has_defined_at())
        << "Expected logical buffer to have location information in the proto.";
    TF_RET_CHECK(id_to_hlo_instruction.contains(
        logical_buffer_proto.defined_at().instruction_id()))
        << "Expected hlo instruction " << "with the id '"
        << logical_buffer_proto.defined_at().instruction_id()
        << "' in the proto to also exist in the "
           "HLO module.";
    // Assumption: An hlo module loaded from an hlo proto
    // preserves each instruction's unique_id. An instruction's name is a
    // deprecated field in the LocationProto.
    const HloInstruction* hlo_instruction = id_to_hlo_instruction.at(
        logical_buffer_proto.defined_at().instruction_id());

    std::vector<int64_t> shape_idx_vals;
    absl::c_copy(logical_buffer_proto.defined_at().shape_index(),
                 std::back_inserter(shape_idx_vals));
    ShapeIndex proto_shape_index(shape_idx_vals);

    // Look up logical buffer by hlo instruction and shape index.
    auto& logical_buffer = alias_analysis->dataflow_analysis().GetUniqueValueAt(
        hlo_instruction, proto_shape_index);

    // Assign color to a logical buffer from the proto.
    logical_buffer.set_color(logical_buffer_proto.color());

    id_to_logical_buffer[logical_buffer_proto.id()] = &logical_buffer;
  }
  return id_to_logical_buffer;
}

}  // namespace

absl::Status GatherComputationsByAllocationType(
    const HloModule* module,
    std::vector<const HloComputation*>* thread_local_computations,
    std::vector<const HloComputation*>* global_computations) {
  // Create a worklist of computations paired with whether the allocation must
  // be thread-local.
  std::deque<std::pair<const HloComputation*, bool>> worklist;
  worklist.push_back(std::make_pair(module->entry_computation(),
                                    /*is_thread_local=*/false));

  // Sets for quickly checking membership. Computations are returned in vectors
  // for stable iteration.
  flat_hash_set<const HloComputation*> thread_local_set;
  flat_hash_set<const HloComputation*> global_set;

  while (!worklist.empty()) {
    auto worklist_front = worklist.front();
    worklist.pop_front();
    const HloComputation* computation = worklist_front.first;
    bool is_thread_local = worklist_front.second;
    bool in_thread_local_set = thread_local_set.contains(computation);
    bool in_global_set = global_set.contains(computation);

    // If the computation has already been added to the respective set, then
    // nothing to do.
    if ((is_thread_local && in_thread_local_set) ||
        (!is_thread_local && in_global_set)) {
      continue;
    }

    // If the computation has already been added to the other set this is an
    // error condition because the global call to the computation (eg,
    // while/call) may return a reference to one of the thread-local buffers to
    // the calling computation which will become a dangling reference when the
    // thread-local is deallocated with the call return.
    if ((is_thread_local && in_global_set) ||
        (!is_thread_local && in_thread_local_set)) {
      return InvalidArgument(
          "computation %s has conflicting allocation requirements (global "
          "and thread-local)",
          computation->name());
    }

    if (is_thread_local) {
      thread_local_set.insert(computation);
    } else {
      global_set.insert(computation);
    }

    for (auto* instruction : computation->instructions()) {
      for (HloComputation* subcomputation :
           instruction->called_computations()) {
        switch (instruction->opcode()) {
          case HloOpcode::kCall:
          case HloOpcode::kConditional:
          case HloOpcode::kWhile:
          case HloOpcode::kAsyncStart:
          case HloOpcode::kAsyncUpdate:
          case HloOpcode::kAsyncDone:
            // Call, conditional, while, and async operations inherit their
            // thread-locality from their parent computation.
            worklist.push_back(std::make_pair(subcomputation, is_thread_local));
            break;
          case HloOpcode::kCustomCall:
          case HloOpcode::kAllReduce:
          case HloOpcode::kReduceScatter:
          case HloOpcode::kAllReduceStart:
          case HloOpcode::kMap:
          case HloOpcode::kReduce:
          case HloOpcode::kReduceWindow:
          case HloOpcode::kScatter:
          case HloOpcode::kSelectAndScatter:
          case HloOpcode::kSort:
          case HloOpcode::kFusion:
            // Map/reduce etc computations are always thread-local.
            worklist.push_back(std::make_pair(subcomputation, true));
            break;
          default:
            return Internal("Unexpected calling opcode: %s",
                            HloOpcodeString(instruction->opcode()));
        }
      }
    }
  }

  // Add the computations to the vectors in post order.
  for (auto* computation : module->MakeComputationPostOrder()) {
    if (thread_local_set.contains(computation)) {
      thread_local_computations->push_back(computation);
    } else if (global_set.contains(computation)) {
      global_computations->push_back(computation);
    }
    // If the computation is not reachable from the entry computation, then it
    // will not appear in either thread_local_set or global_set. We don't bother
    // assigning buffers for these.
  }
  return absl::OkStatus();
}

std::string BufferAllocation::Slice::ToString() const {
  return absl::StrCat("{index:", allocation_ == nullptr ? -1 : index(),
                      ", offset:", offset_, ", size:", size_, "}");
}

absl::StatusOr<xla::buffer_assignment::BufferAllocationSliceProto>
BufferAllocation::Slice::ToProto() const {
  xla::buffer_assignment::BufferAllocationSliceProto proto;
  proto.set_offset(offset());
  proto.set_size(size());
  proto.set_buffer_allocation_index(allocation() == nullptr ? -1 : index());
  return proto;
}

absl::StatusOr<BufferAllocation::Slice> BufferAllocation::Slice::FromProto(
    const xla::buffer_assignment::BufferAllocationSliceProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations) {
  if (proto.buffer_allocation_index() < 0 ||
      proto.buffer_allocation_index() >= buffer_allocations.size()) {
    return absl::OutOfRangeError(absl::StrCat("Buffer allocation index ",
                                              proto.buffer_allocation_index(),
                                              " is out of range."));
  }
  const BufferAllocation& allocation =
      buffer_allocations[proto.buffer_allocation_index()];
  return BufferAllocation::Slice(&allocation, proto.offset(), proto.size());
}

BufferAllocation::Slice BufferAllocation::GetSlice(
    const HloValue& buffer) const {
  const OffsetSize os = FindOrDie(assigned_buffers_, &buffer);
  return Slice(this, os.offset, os.size);
}

void BufferAllocation::AddAssignment(const HloValue& buffer, int64_t offset,
                                     int64_t size) {
  VLOG(4) << "Adding the following buffer to allocation #" << index()
          << absl::StrFormat(" (size=%d, offset=%d) %s", size, offset,
                             buffer.ToShortString());
  CHECK(!assigned_buffers_.contains(&buffer))
      << "LogicalBuffer " << buffer << " already assigned to allocation "
      << index_;
  CHECK_LE(offset, size_) << "LogicalBuffer " << buffer
                          << " offset out of range";
  CHECK_LE(offset + size, size_)
      << "LogicalBuffer " << buffer
      << " size out of range at offset: " << offset << " with size: " << size;
  if (!(IsPreallocatedTempBuffer() && color() != 0)) {
    CHECK_EQ(buffer.color(), color())
        << "Buffer color " << buffer.color() << " for buffer " << buffer
        << " does not match allocation color " << color() << ".";
  }
  OffsetSize offset_size;
  offset_size.offset = offset;
  offset_size.size = size;
  assigned_buffers_.emplace(&buffer, offset_size);
  // For debugging purposes, store the assigned memory space in the
  // instruction's layout.
  for (HloPosition position : buffer.positions()) {
    Shape* shape = ShapeUtil::GetMutableSubshape(
        position.instruction->mutable_shape(), position.index);
    if (shape->has_layout()) {
      shape->mutable_layout()->set_memory_space(buffer.color());
    }
  }
}

BufferAllocationProto BufferAllocation::ToProto() const {
  BufferAllocationProto proto;
  proto.set_index(index_);
  proto.set_size(size_);
  proto.set_is_thread_local(is_thread_local_);
  proto.set_is_tuple(is_tuple_);
  proto.set_color(color_);
  if (is_entry_computation_parameter_) {
    proto.set_is_entry_computation_parameter(true);
    for (int64_t idx : param_shape_index()) {
      proto.add_parameter_shape_index(idx);
    }
    proto.set_parameter_number(parameter_number_);
  }
  proto.set_is_constant(is_constant_);
  proto.set_maybe_live_out(maybe_live_out_);
  for (const auto& buffer_offset_size : assigned_buffers_) {
    BufferAllocationProto::Assigned* proto_assigned = proto.add_assigned();
    proto_assigned->set_logical_buffer_id(buffer_offset_size.first->id());
    proto_assigned->set_offset(buffer_offset_size.second.offset);
    proto_assigned->set_size(buffer_offset_size.second.size);
  }
  absl::c_sort(*proto.mutable_assigned(),
               [](const BufferAllocationProto::Assigned& assign1,
                  const BufferAllocationProto::Assigned& assign2) {
                 return assign1.logical_buffer_id() <
                        assign2.logical_buffer_id();
               });
  return proto;
}

static bool CompareHloValuesById(const HloValue* a, const HloValue* b) {
  return a->id() < b->id();
}

// Returns parameter instruction corresponding to the allocation or nullptr.
static const HloInstruction* GetEntryParameterInstruction(
    const BufferAllocation& alloc) {
  for (const auto& p : alloc.assigned_buffers()) {
    const HloValue* value = p.first;
    const HloInstruction* instr = value->instruction();
    if (instr->opcode() == HloOpcode::kParameter &&
        instr->parent() == instr->GetModule()->entry_computation()) {
      return instr;
    }
  }
  return nullptr;
}

// Returns root module output instruction corresponding to the allocation or
// nullptr.
static const HloInstruction* GetOutputInstruction(
    const BufferAllocation& alloc) {
  for (const auto& p : alloc.assigned_buffers()) {
    const HloValue* value = p.first;
    for (const HloPosition& position : value->positions()) {
      const HloInstruction* instr = position.instruction;
      if (position.index.empty() &&
          instr->parent()->root_instruction() == instr &&
          instr->parent()->IsEntryComputation()) {
        return instr;
      }
    }
  }
  return nullptr;
}

std::string BufferAllocation::ToShortString(bool human_readable_size) const {
  std::string output;
  if (human_readable_size) {
    StrAppendFormat(&output, "allocation %d: size %s", index_,
                    HumanReadableNumBytes(size()));
  } else {
    StrAppendFormat(&output, "allocation %d: size %d", index_, size());
  }
  if (color() != 0) {
    StrAppend(&output, ", color ", color());
  }
  if (is_entry_computation_parameter()) {
    const HloInstruction* param = GetEntryParameterInstruction(*this);
    StrAppend(&output, ", parameter ", parameter_number(), ", shape |",
              param ? param->shape().ToString(/*print_layout=*/false)
                    : "<unknown shape>",
              "| at ShapeIndex ", param_shape_index().ToString());
  }
  if (const HloInstruction* instr = GetOutputInstruction(*this)) {
    StrAppend(&output, ", output shape is |",
              instr->shape().ToString(/*print_layout=*/false), "|");
  }
  if (is_constant()) {
    StrAppend(&output, ", constant");
  }
  if (is_thread_local()) {
    StrAppend(&output, ", thread-local");
  }
  if (maybe_live_out()) {
    StrAppend(&output, ", maybe-live-out");
  }
  if (IsPreallocatedTempBuffer()) {
    StrAppend(&output, ", preallocated-temp");
  }
  StrAppend(&output, ":\n");
  return output;
}

std::string BufferAllocation::ToString() const {
  std::string output = ToShortString();
  // Dump the assigned buffers ordered by id.
  std::vector<const HloValue*> sorted_buffers;
  for (const auto& buffer_offset_size : assigned_buffers_) {
    sorted_buffers.push_back(buffer_offset_size.first);
  }
  absl::c_sort(sorted_buffers, &CompareHloValuesById);
  for (const HloValue* buffer : sorted_buffers) {
    const OffsetSize& offset_size = FindOrDie(assigned_buffers_, buffer);
    StrAppend(&output,
              absl::StrFormat(
                  " value: %s (size=%d,offset=%d): %s\n",
                  buffer->ToShortString(), offset_size.size, offset_size.offset,
                  ShapeUtil::HumanStringWithLayout(buffer->shape())));
  }
  return output;
}

std::ostream& operator<<(std::ostream& out, const BufferAllocation& buffer) {
  out << buffer.ToString();
  return out;
}

std::ostream& operator<<(std::ostream& out, const BufferAllocation::Slice& s) {
  out << s.ToString();
  return out;
}

bool BufferAssignment::HasAllocation(const HloValue& value) const {
  return allocation_index_for_value_.contains(&value);
}

bool BufferAssignment::HasAllocation(HloValue::Id value_id) const {
  return HasAllocation(dataflow_analysis().GetValue(value_id));
}

bool BufferAssignment::HasAllocation(const HloBuffer& buffer) const {
  return allocation_index_for_value_.contains(buffer.values()[0]);
}

const BufferAllocation& BufferAssignment::GetAssignedAllocation(
    const HloValue& value) const {
  CHECK(HasAllocation(value));
  return GetAllocation(allocation_index_for_value_.at(&value));
}

const BufferAllocation& BufferAssignment::GetAssignedAllocation(
    const HloBuffer& hlo_buffer) const {
  return GetAssignedAllocation(*hlo_buffer.values()[0]);
}

BufferAllocation* BufferAssignment::GetMutableAssignedAllocation(
    const HloBuffer& buffer) {
  return const_cast<BufferAllocation*>(&GetAssignedAllocation(buffer));
}

std::set<BufferAllocation::Slice> BufferAssignment::GetAllSlices(
    const HloInstruction* instruction, const ShapeIndex& index) const {
  std::set<BufferAllocation::Slice> result;
  for (const HloValue* value :
       dataflow_analysis().GetValueSet(instruction, index).values()) {
    if (HasAllocation(*value)) {
      result.insert(GetAssignedAllocation(*value).GetSlice(*value));
    }
  }
  return result;
}

const BufferAllocation& BufferAssignment::GetAllocation(
    BufferAllocation::Index index) const {
  CHECK_GE(index, 0);
  CHECK_LT(index, allocations_.size());
  return allocations_[index];
}

const BufferAllocation* BufferAssignment::GetInstructionAllocation(
    const HloInstruction* hlo, const ShapeIndex& shape_index) const {
  const HloValue* value =
      dataflow_analysis().GetValueSet(hlo, shape_index).values()[0];

  if (!HasAllocation(*value)) {
    return nullptr;
  }

  const BufferAllocation& instruction_allocation =
      GetAssignedAllocation(*value);
  return &instruction_allocation;
}

BufferAllocation* BufferAssignment::GetMutableAllocation(
    BufferAllocation::Index index) {
  return const_cast<BufferAllocation*>(&GetAllocation(index));
}

bool BufferAssignment::HasAllocationAt(const HloInstruction* instruction,
                                       const ShapeIndex& index) const {
  return absl::c_any_of(
      dataflow_analysis().GetValueSet(instruction, index).values(),
      IsKeyIn(allocation_index_for_value_));
}

bool BufferAssignment::HasTopLevelAllocation(
    const HloInstruction* instruction) const {
  return HasAllocationAt(instruction, /*index=*/{});
}

absl::StatusOr<BufferAllocation::Slice> BufferAssignment::GetUniqueSlice(
    const HloInstruction* instruction, const ShapeIndex& index) const {
  VLOG(3) << "Trying to find unique slice for " << instruction->name() << " ["
          << index << "]";
  BufferAllocation::Slice result;
  for (const HloValue* value :
       dataflow_analysis().GetValueSet(instruction, index).values()) {
    VLOG(3) << "Examining value " << *value;
    if (HasAllocation(*value)) {
      VLOG(3) << "Has allocation";
      const BufferAllocation::Slice slice =
          GetAssignedAllocation(*value).GetSlice(*value);
      if (result.allocation() == nullptr) {
        result = slice;
      } else if (result != slice) {
        return FailedPrecondition(
            "BufferAllocation::Slice for instruction %s at index %s cannot "
            "be determined at compile-time.",
            instruction->name(), index.ToString());
      }
    } else {
      VLOG(3) << "No allocation";
    }
  }
  if (result.allocation() == nullptr) {
    return FailedPrecondition(
        "BufferAllocation::Slice not assigned for instruction %s at index %s",
        instruction->name(), index.ToString());
  }
  return result;
}

absl::StatusOr<BufferAllocation::Slice>
BufferAssignment::GetUniqueTopLevelSlice(
    const HloInstruction* instruction) const {
  return GetUniqueSlice(instruction, /*index=*/{});
}

bool BufferAssignment::SharesSliceAtIndex(
    const HloInstruction* hlo_a, const ShapeIndex& shape_index_a,
    const HloInstruction* hlo_b, const ShapeIndex& shape_index_b) const {
  return GetUniqueSlice(hlo_a, shape_index_a).value() ==
         GetUniqueSlice(hlo_b, shape_index_b).value();
}

bool BufferAssignment::HaveDisjointSlices(const HloInstruction* hlo_a,
                                          const HloInstruction* hlo_b) const {
  using SliceSet = flat_hash_set<BufferAllocation::Slice>;
  // Gets the slices all of instr's subshapes.  If any subshape doesn't have an
  // assigned slice, returns the empty set.
  auto collect_slices = [&](const HloInstruction* instr) -> SliceSet {
    SliceSet slices;
    absl::Status status = ShapeUtil::ForEachSubshapeWithStatus(
        instr->shape(),
        [&](const Shape& /*subshape*/,
            const ShapeIndex& index) -> absl::Status {
          auto shape_slices = GetAllSlices(instr, index);
          if (shape_slices.empty()) {
            return InvalidArgument("No slices assigned to part of instr.");
          }
          slices.insert(shape_slices.begin(), shape_slices.end());
          return absl::OkStatus();
        });
    if (!status.ok()) {
      return {};
    }
    return slices;
  };

  SliceSet slices_a = collect_slices(hlo_a);
  SliceSet slices_b = collect_slices(hlo_b);
  // hlo_a and hlo_b have disjoint slices if collect_slices succeeded (i.e.
  // didn't return the empty set) for both HLOs, and the two resulting sets of
  // slices are disjoint.
  return !slices_a.empty() && !slices_b.empty() &&
         absl::c_none_of(slices_a, [&](const BufferAllocation::Slice& slice) {
           return slices_b.contains(slice);
         });
}

absl::StatusOr<BufferAllocation::Slice>
BufferAssignment::GetUniqueTopLevelOutputSlice() const {
  return GetUniqueTopLevelSlice(
      module_->entry_computation()->root_instruction());
}

BufferAllocation* BufferAssignment::NewEmptyAllocation(
    int64_t size, LogicalBuffer::Color color) {
  BufferAllocation::Index index = allocations_.size();
  allocations_.emplace_back(index, size, color);
  BufferAllocation* allocation = &allocations_.back();
  return allocation;
}

BufferAllocation* BufferAssignment::NewAllocation(const HloBuffer& buffer,
                                                  int64_t size) {
  BufferAllocation* allocation = NewEmptyAllocation(size, buffer.color());
  AddAssignment(allocation, buffer, /*offset=*/0, size);
  allocation->peak_buffers_.push_back(buffer.values()[0]);
  return allocation;
}

void BufferAssignment::AddAssignment(BufferAllocation* allocation,
                                     const HloBuffer& buffer, int64_t offset,
                                     int64_t size) {
  CHECK(allocation->is_reusable() || allocation->assigned_buffers().empty())
      << "Non-reusable allocation already assigned a buffer: "
      << allocation->ToString();

  for (const HloValue* buffer_value : buffer.values()) {
    CHECK(!allocation_index_for_value_.contains(buffer_value))
        << "BufferValue " << buffer_value << " already has an allocation.";
    allocation->AddAssignment(*buffer_value, offset, size);
    allocation_index_for_value_[buffer_value] = allocation->index();
  }

  if (alias_analysis().BufferLivesOut(buffer)) {
    VLOG(3) << "HloBuffer lives out: " << buffer.ToString();
    VLOG(3) << "Set maybe live out: " << allocation->ToString();
    allocation->set_maybe_live_out(true);
  }
}

void BufferAssignment::AddAssignment(BufferAllocation* allocation,
                                     const HloValue& value, int64_t offset,
                                     int64_t size) {
  allocation->AddAssignment(value, offset, size);
  allocation_index_for_value_[&value] = allocation->index();
  const HloValue& hlo_value =
      *CHECK_NOTNULL(dynamic_cast<const HloValue*>(&value));
  if (alias_analysis().ValueLivesOut(hlo_value)) {
    VLOG(3) << "HloValue lives out: " << hlo_value.ToString();
    VLOG(3) << "Set maybe live out: " << allocation->ToString();
    allocation->set_maybe_live_out(true);
  }
}

// Combines allocations of temporary buffers of the same color into one big
// BufferAllocation.
void BufferAssignment::CombineTempAllocations(
    const absl::flat_hash_set<BufferValue::Color>& private_stack_colors,
    std::optional<BufferValue::Color> temp_buffer_color) {
  VLOG(1) << "CombineTempAllocations()";
  // Stores the combined allocations.
  std::deque<BufferAllocation> combined_allocations;
  // Holds the pointer to a combined allocation of each color, if any.
  flat_hash_map<BufferValue::Color, BufferAllocation*> combined_allocation_map;

  // Move all temp allocations into a single run at the end of the allocations
  // vector.
  const auto first_temp_it =
      std::partition(allocations_.begin(), allocations_.end(),
                     [](const BufferAllocation& allocation) {
                       return !allocation.IsPreallocatedTempBuffer();
                     });

  // Walk over the run of temp allocations, collecting the allocations belonging
  // to the same color.
  if (first_temp_it != allocations_.end()) {
    for (auto it = first_temp_it; it != allocations_.end(); ++it) {
      BufferAllocation& temp_allocation = *it;
      BufferValue::Color color = temp_allocation.color();
      auto combined_it = combined_allocation_map.find(color);
      if (combined_it == combined_allocation_map.end()) {
        // We have found the first temp allocation of this color. Collect
        // the other temp allocations of the same color into it subject to the
        // size constraint.
        VLOG(1) << "Combined temp allocation for color " << color
                << " is: " << temp_allocation;
        combined_allocations.push_back(temp_allocation);
        combined_allocation_map.emplace(color, &combined_allocations.back());
        continue;
      }
      if (combined_it->second->size() + it->size() >=
          multiheap_size_constraint_per_heap_) {
        // We cannot put more into the current combined_it. So, appoint a new
        // combined_it.
        VLOG(1) << "Due to size constraint, reset temp allocation for color "
                << color << " to: " << temp_allocation;
        combined_allocations.push_back(temp_allocation);
        combined_allocation_map.emplace(color, &combined_allocations.back());
        continue;
      }

      BufferAllocation* combined_allocation = combined_it->second;
      VLOG(1) << "Combined allocation absorbing temp allocation: "
              << temp_allocation;

      // Each temp allocation is placed end-to-end, accounting for alignment.
      // The offset of each buffer in the combined allocation is computed from
      // the base offset of the allocation. For private stack color, we assume
      // each allocation object corresponds to one of the independent executions
      // of the private stack computations, so it is safe to reuse offsets in
      // that case.
      int64_t alignment = color_alignment_(color);
      int64_t base;
      bool is_private_stack = private_stack_colors.contains(color);
      if (is_private_stack) {
        base = 0;
        combined_allocation->set_size(std::max(base, temp_allocation.size()));
      } else {
        base = RoundUpTo(combined_allocation->size(), alignment);
        combined_allocation->set_size(base + temp_allocation.size());
      }
      for (const auto& buffer_offset_size : temp_allocation.assigned_buffers_) {
        const HloValue* value = buffer_offset_size.first;
        const int64_t offset = buffer_offset_size.second.offset;
        const int64_t size = buffer_offset_size.second.size;
        combined_allocation->AddAssignment(*value, base + offset, size);
      }
      if (!temp_allocation.HeapTraces().empty()) {
        CHECK_EQ(temp_allocation.HeapTraces().size(), 1);
        combined_allocation->AddHeapTrace(temp_allocation.HeapTraces().front());
      }

      if (is_private_stack) {
        if (temp_allocation.size() == combined_allocation->size()) {
          combined_allocation->peak_buffers_ = temp_allocation.peak_buffers_;
        }
      } else {
        combined_allocation->peak_buffers_.insert(
            combined_allocation->peak_buffers_.end(),
            temp_allocation.peak_buffers_.begin(),
            temp_allocation.peak_buffers_.end());
      }

      if (temp_buffer_color.has_value()) {
        if (combined_allocation->color() == 0) {
          combined_allocation->set_color(temp_buffer_color.value());
        }
      }
    }
    // Replace all existing temporary allocations with the new combined
    // allocations.
    allocations_.erase(first_temp_it, allocations_.end());
    for (BufferAllocation& combined : combined_allocations) {
      temp_allocation_total_size_ += combined.size();
      allocations_.push_back(std::move(combined));
    }
  }

  // Update allocation indices to their new positions.
  allocation_index_for_value_.erase(allocation_index_for_value_.begin(),
                                    allocation_index_for_value_.end());
  for (size_t index = 0; index < allocations_.size(); ++index) {
    BufferAllocation* allocation = &allocations_[index];
    allocation->set_index(index);
    std::vector<const HloValue*> sorted_values;
    sorted_values.reserve(allocation->assigned_buffers_.size());
    for (const auto& buffer_offset_size : allocation->assigned_buffers_) {
      const HloValue* value = buffer_offset_size.first;
      sorted_values.emplace(sorted_values.end(), value);
    }
    absl::c_sort(sorted_values, &CompareHloValuesById);
    for (const HloValue* value : sorted_values) {
      allocation_index_for_value_[value] = index;
    }
  }
}

void BufferAssignment::ComputeSummaryStats() {
  for (auto& allocation : Allocations()) {
    if (allocation.is_entry_computation_parameter()) {
      stats_.parameter_allocation_count++;
      stats_.parameter_allocation_bytes += allocation.size();
    }
    if (allocation.is_constant()) {
      stats_.constant_allocation_count++;
      stats_.constant_allocation_bytes += allocation.size();
    }
    if (allocation.maybe_live_out()) {
      stats_.maybe_live_out_allocation_count++;
      stats_.maybe_live_out_allocation_bytes += allocation.size();
    }
    if (allocation.IsPreallocatedTempBuffer()) {
      stats_.preallocated_temp_allocation_count++;
      stats_.preallocated_temp_allocation_bytes += allocation.size();
    }
    stats_.total_allocation_count++;
    stats_.total_allocation_bytes += allocation.size();
  }
}

absl::StatusOr<int64_t> BufferAssignment::ComputeTotalFragmentationBytes()
    const {
  // Only compute total fragmentation if all computations have schedules.
  HloSchedule schedule(module_);
  bool schedule_complete = true;
  for (const auto& computation : module_->computations()) {
    if (!computation->IsFusionComputation()) {
      const HloInstructionSequence* sequence =
          hlo_ordering().SequentialOrder(*computation);
      if (sequence == nullptr) {
        schedule_complete = false;
      } else {
        schedule.set_sequence(computation, *sequence);
      }
    }
  }
  if (schedule_complete) {
    TF_RETURN_IF_ERROR(schedule.Verify());
    TF_ASSIGN_OR_RETURN(
        const int64_t min_size,
        HeapSimulator::MinimumMemoryForModule(schedule, buffer_size_));
    return stats_.total_allocation_bytes - min_size;
  }
  return -1;
}

std::string BufferAssignment::StatsString(
    bool report_total_fragmentation) const {
  std::string s;
  StrAppendFormat(&s, "BufferAssignment stats:\n");
  StrAppendFormat(&s, "             parameter allocation: %10s\n",
                  HumanReadableNumBytes(stats_.parameter_allocation_bytes));
  StrAppendFormat(&s, "              constant allocation: %10s\n",
                  HumanReadableNumBytes(stats_.constant_allocation_bytes));
  StrAppendFormat(
      &s, "        maybe_live_out allocation: %10s\n",
      HumanReadableNumBytes(stats_.maybe_live_out_allocation_bytes));
  StrAppendFormat(
      &s, "     preallocated temp allocation: %10s\n",
      HumanReadableNumBytes(stats_.preallocated_temp_allocation_bytes));
  if (stats_.preallocated_temp_fragmentation_bytes >= 0) {
    const double percent = 100. * stats_.preallocated_temp_fragmentation_bytes /
                           stats_.preallocated_temp_allocation_bytes;
    StrAppendFormat(
        &s, "  preallocated temp fragmentation: %10s (%.2f%%)\n",
        HumanReadableNumBytes(stats_.preallocated_temp_fragmentation_bytes),
        percent);
  }
  StrAppendFormat(&s, "                 total allocation: %10s\n",
                  HumanReadableNumBytes(stats_.total_allocation_bytes));
  if (report_total_fragmentation) {
    auto total_fragmentation_bytes = ComputeTotalFragmentationBytes();
    if (total_fragmentation_bytes.ok() && *total_fragmentation_bytes >= 0) {
      const double percent =
          100. * *total_fragmentation_bytes / stats_.total_allocation_bytes;
      StrAppendFormat(&s, "              total fragmentation: %10s (%.2f%%)\n",
                      HumanReadableNumBytes(*total_fragmentation_bytes),
                      percent);
    }
  }
  return s;
}

std::string BufferAssignment::ToString() const {
  std::string output;
  absl::StrAppend(&output, "BufferAssignment:\n");
  std::vector<const HloValue*> used_values;
  int64_t total_size = 0;
  for (auto& allocation : allocations_) {
    total_size += allocation.size();
    absl::StrAppend(&output, allocation.ToString());
    for (const auto& p : allocation.assigned_buffers()) {
      used_values.push_back(p.first);
    }
  }
  absl::StrAppend(&output, "\nTotal bytes used: ", total_size, " (",
                  HumanReadableNumBytes(total_size), ")\n");
  absl::StrAppend(&output, "\nUsed values:\n");
  absl::c_sort(used_values, &CompareHloValuesById);
  for (const HloValue* value : used_values) {
    absl::StrAppend(&output, value->ToString());
  }
  return output;
}

std::string BufferAssignment::MemoryUsageReport(float percentile,
                                                int64_t more_than_k) const {
  std::string output;
  int64_t total_size = 0;
  for (auto& allocation : allocations_) {
    total_size += allocation.size();
  }
  absl::StrAppend(&output, "Total bytes used: ", total_size, " (",
                  HumanReadableNumBytes(total_size), ")\n");

  absl::StrAppend(&output, "\nAllocations sorted by size:\n\n");
  auto allocations = allocations_;
  std::sort(allocations.begin(), allocations.end(),
            [](const BufferAllocation& a, const BufferAllocation& b) {
              if (a.size() > b.size()) return true;
              if (a.size() < b.size()) return false;
              return a.index() < b.index();
            });

  int64_t cumulative_size = 0;
  absl::StrAppend(
      &output, "cumulative_size; total_size - cumulative_size; allocation\n");
  absl::StrAppend(&output,
                  "------------------------------------------------------------"
                  "------------------\n");
  int64_t index = 0;
  for (auto& allocation : allocations) {
    cumulative_size += allocation.size();
    absl::StrAppend(
        &output,
        absl::StrFormat("%10s(%3.0f%%); %10s; %s",
                        HumanReadableNumBytes(cumulative_size),
                        100. * cumulative_size / total_size,
                        HumanReadableNumBytes(total_size - cumulative_size),
                        allocation.ToShortString(true)));

    // Skip the rest of the allocations if they are less than percentile of the
    // total size and not more than k.
    if (++index > more_than_k &&
        total_size - cumulative_size < total_size * percentile) {
      absl::StrAppend(
          &output,
          absl::StrFormat(
              "The rest %d allocations are less than %d%% of the total "
              "size and not shown.\n",
              allocations.size() - index, static_cast<int>(percentile * 100)));
      break;
    }
  }

  absl::StrAppend(&output,
                  "\n\nAllocations sorted by size with their values:\n");
  for (auto& allocation : allocations) {
    if (allocation.assigned_buffers().size() == 1) {
      absl::StrAppend(&output, allocation.ToShortString(true));
    } else {
      StrAppendFormat(
          &output, "%s\n%s\n", allocation.ToShortString(true),
          allocation.MemoryUsageReport("\t", percentile, more_than_k));
    }
  }
  return output;
}

std::string BufferAllocation::MemoryUsageReport(const std::string& prefix,
                                                float percentile,
                                                int64_t more_than_k) const {
  std::string output;

  struct OffsetInfo {
    std::vector<const HloValue*> values;
    OffsetSize offset_size;
  };

  // Group the values by their offset in the allocation.
  absl::flat_hash_map<int64_t, OffsetInfo> offset_to_buffers;
  for (const auto& element : assigned_buffers_) {
    const HloValue* value = element.first;
    OffsetInfo& offset_info = offset_to_buffers[element.second.offset];
    offset_info.values.push_back(value);
    offset_info.offset_size.offset = element.second.offset;
    offset_info.offset_size.size =
        std::max(offset_info.offset_size.size, element.second.size);
  }

  // Sort the offset infos by the max size of the values in the group.
  std::vector<OffsetInfo> sorted_offset_infos;
  int64_t total_size = 0;
  for (auto& element : offset_to_buffers) {
    total_size += element.second.offset_size.size;
    sorted_offset_infos.push_back(std::move(element.second));
  }
  absl::c_sort(sorted_offset_infos,
               [](const OffsetInfo& a, const OffsetInfo& b) {
                 return a.offset_size.size > b.offset_size.size;
               });

  StrAppend(&output, prefix,
            "cumulative_size;       size;       offset; used_by_n_values; "
            "shapes_list\n");
  StrAppend(&output, prefix,
            "------------------------------------------------------------\n");
  int64_t cumulative_size = 0;
  int64_t index = 0;
  for (const auto& offset_info : sorted_offset_infos) {
    cumulative_size += offset_info.offset_size.size;
    StrAppendFormat(&output, "%s%9s(%3.0f%%); %10s; %12d; %16d; ", prefix,
                    xla::HumanReadableNumBytes(cumulative_size),
                    100. * cumulative_size / total_size,
                    xla::HumanReadableNumBytes(offset_info.offset_size.size),
                    offset_info.offset_size.offset, offset_info.values.size());

    // Count the number of values with the same shape and append them at the end
    // of the line.
    absl::flat_hash_map<std::string, int64_t> shapes;
    for (auto& value : offset_info.values) shapes[value->shape().ToString()]++;

    StrAppend(
        &output,
        absl::StrJoin(shapes, ", ", [](std::string* out, const auto& pair) {
          if (pair.second == 1) {
            return absl::StrAppend(out, pair.first);
          }
          return absl::StrAppend(out, pair.second, "×", pair.first);
        }));

    StrAppend(&output, "\n");

    // Skip the rest of the values if they are less than percentile of the
    // total size and not more than k.
    if (++index > more_than_k &&
        total_size - cumulative_size < total_size * percentile) {
      StrAppendFormat(
          &output,
          "%sThe rest %d values are less than %d%% of the total size and not "
          "shown.\n",
          prefix, sorted_offset_infos.size() - index,
          static_cast<int>(percentile * 100));
      break;
    }
  }

  return output;
}

// Returns the largest k buffers present at the point of peak memory usage
// across allocations as a vector of pairs with their corresponding sizes.
std::vector<std::pair<int64_t, const HloValue*>> TopKPeakBuffers(
    uint64_t k, const std::vector<BufferAllocation> allocations) {
  absl::btree_multimap<int64_t, const HloValue*> topk;
  for (const BufferAllocation& allocation : allocations) {
    for (const HloValue* value : allocation.PeakMemoryLogicalBuffers()) {
      int64_t size = allocation.assigned_buffers().at(value).size;
      if (topk.size() < k) {
        topk.insert({size, value});
      } else {
        auto it = topk.begin();
        if (size > it->first) {
          topk.erase(it);
          topk.insert({size, value});
        }
      }
    }
  }

  // map will iterate smallest first, so reverse it.
  std::vector<std::pair<int64_t, const HloValue*>> topk_descending;
  topk_descending.reserve(topk.size());
  absl::c_reverse_copy(topk, std::back_inserter(topk_descending));
  return topk_descending;
}

std::string BufferAssignment::ToVerboseString(
    size_t max_buffers_to_show) const {
  std::string output =
      absl::StrCat("BufferAssignment OOM Debugging.\n",
                   StatsString(/*report_total_fragmentation=*/true));

  std::vector<std::pair<int64_t, const HloValue*>> peak_buffers =
      TopKPeakBuffers(max_buffers_to_show, allocations_);
  std::vector<std::string> buf_strs;
  for (size_t i = 0; i < std::min(max_buffers_to_show, peak_buffers.size());
       ++i) {
    const HloValue* value = peak_buffers[i].second;
    const HloInstruction* instr = value->instruction();
    int64_t size = peak_buffers[i].first;
    buf_strs.push_back(absl::StrCat("\n\tBuffer ", i + 1, ":\n\t\tSize: ",
                                    xla::HumanReadableNumBytes(size)));
    if (!instr->metadata().op_name().empty()) {
      buf_strs.push_back(absl::StrCat(
          "\n\t\tOperator: ", xla::OpMetadataToString(instr->metadata())));
    }
    if (instr->opcode() == HloOpcode::kParameter &&
        (instr->parent() == instr->GetModule()->entry_computation())) {
      // Special case on entry parameters as they sometimes have hundreds of
      // indices in their shapes, and overwhelm the output.
      buf_strs.push_back(absl::StrCat(
          "\n\t\tEntry Parameter Subshape: ",
          ShapeUtil::GetSubshape(instr->shape(), value->index()).ToString()));
    } else {
      // TODO(loreno): change this to a truncated string of the instruction.
      buf_strs.push_back(
          absl::StrCat("\n\t\tXLA Label: ", HloOpcodeString(instr->opcode()),
                       "\n\t\tShape: ", value->shape().ToString()));
    }
    buf_strs.push_back("\n\t\t==========================\n");
  }
  absl::StrAppend(&output, "Peak buffers:", absl::StrJoin(buf_strs, ""));
  return output;
}

std::string BufferAssignment::BufferInfoString() const {
  std::string binfo;
  // Columns in buffer information:
  // buffer_id: int. This value can be used to match the allocation in
  // allocation information.
  // buffer_name: string.
  // offset: int. Starting position of the buffer in the memory space.
  // size: int. Size of the buffer in bytes.
  // definition_time: int. Position in the schedule where the buffer starts
  // being live (inclusive).
  // end_time: int. Position in the schedule where the buffer stops being live
  // (exclusive).
  // num_uses: int. Number of uses of the buffer.
  // use_names: string. This is a semicolon-separated list of string
  // representation of uses.
  // Append the column names.
  absl::StrAppend(&binfo,
                  "buffer_id,buffer_name,offset,size,"
                  "definition_time,end_time,num_uses,use_times,use_names\n");
  const HloLiveRange& live_ranges = hlo_live_range();
  const auto& instruction_schedule = live_ranges.instruction_schedule();
  const auto& buffer_live_ranges = live_ranges.buffer_live_ranges();
  // Sort the buffers by Id.
  std::vector<std::pair<const HloValue*, BufferAllocation::OffsetSize>> buffers;
  for (const BufferAllocation& allocation : allocations_) {
    absl::c_copy(allocation.assigned_buffers(), std::back_inserter(buffers));
  }
  absl::c_sort(
      buffers,
      [](const std::pair<const HloValue*, BufferAllocation::OffsetSize>& b1,
         const std::pair<const HloValue*, BufferAllocation::OffsetSize>& b2) {
        return b1.first->id() < b2.first->id();
      });
  for (const auto& buffer_pair : buffers) {
    const HloValue& buffer = *buffer_pair.first;
    const BufferAllocation::OffsetSize& offset_size = buffer_pair.second;
    if (!buffer_live_ranges.contains(&buffer)) {
      continue;
    }
    // Ordering uses by their use position.
    std::vector<std::pair<int64_t, std::string>> uses;
    uses.reserve(buffer.GetUses().size());
    for (const HloUse& use : buffer.GetUses()) {
      uses.emplace_back(instruction_schedule.at(use.instruction),
                        use.ToString());
    }
    absl::c_sort(uses);
    std::vector<int64_t> use_positions;
    std::vector<std::string> use_names;
    use_positions.reserve(uses.size());
    use_names.reserve(uses.size());
    for (const auto& use : uses) {
      use_positions.push_back(use.first);
      use_names.push_back(use.second);
    }
    const int64_t definition_time =
        instruction_schedule.at(buffer.defining_position().instruction);
    const int64_t end_t = buffer_live_ranges.at(&buffer).end;
    absl::StrAppend(&binfo, buffer.id(), ",");
    absl::StrAppend(&binfo, "\"", buffer.ToShortString(), "\",");
    absl::StrAppend(&binfo, offset_size.offset, ",");
    absl::StrAppend(&binfo, offset_size.size, ",");
    absl::StrAppend(&binfo, definition_time, ",");
    absl::StrAppend(&binfo, end_t, ",");
    absl::StrAppend(&binfo, use_positions.size(), ",");
    absl::StrAppend(&binfo, "\"", absl::StrJoin(use_positions, ";"), "\",");
    absl::StrAppend(&binfo, "\"", absl::StrJoin(use_names, ";"), "\"");
    absl::StrAppend(&binfo, "\n");
  }
  return binfo;
}

BufferAssignmentProto BufferAssignment::ToProto() const {
  BufferAssignmentProto proto;
  // NOTE: DataflowAnalysis state is serialized here in BufferAssignment,
  // because we need to do the HasAllocation check for each buffer. Otherwise
  // the buffer_size_ call might fail for some backends.
  const HloDataflowAnalysis& dataflow = this->dataflow_analysis();
  for (BufferValue::Id id = 0; id < dataflow.values().size(); id++) {
    auto& value = dataflow.values().at(id);
    if (HasAllocation(*value)) {
      LogicalBufferProto proto_buffer = value->ToProto(buffer_size_);
      proto.add_logical_buffers()->Swap(&proto_buffer);

      // Fill buffer aliases.
      for (const HloValue* alias :
           alias_analysis().GetBufferContainingValue(*value).values()) {
        if (alias->instruction() == value->instruction() &&
            alias->index() == value->index()) {
          continue;  // skip self-aliases
        }
        BufferAssignmentProto::BufferAlias* proto_alias =
            proto.add_buffer_aliases();
        LogicalBufferProto::Location proto_alias_location =
            BufferValue::ToLocationProto(*alias->instruction(), alias->index());
        proto_alias->set_source_buffer_id(value->id());
        proto_alias->mutable_location()->Swap(&proto_alias_location);
      }
    }
  }
  for (const BufferAllocation& allocation : Allocations()) {
    BufferAllocationProto proto_allocation = allocation.ToProto();
    proto.add_buffer_allocations()->Swap(&proto_allocation);
    for (const HeapSimulatorTrace& heap_trace : allocation.HeapTraces()) {
      *proto.add_heap_simulator_traces() = heap_trace;
    }
  }
  return proto;
}

/* static */
absl::StatusOr<std::unique_ptr<BufferAssignment>> BufferAssignment::FromProto(
    const BufferAssignmentProto& proto, const HloModule* module,
    BufferValue::SizeFunction buffer_size,
    HloDataflowAnalysis::CanShareBuffer can_share_buffer) {
  // Create alias and dataflow analysis.
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                      HloAliasAnalysis::Run(module, can_share_buffer));

  // Build a map from a unique_id to corresponding HloInstruction in the module.
  auto id_to_hlo_instruction = BuildIdToHloInstructionMap(module);

  // Build a map from logical buffer id in the proto to hlo value in the
  // existing dataflow analysis.
  absl::flat_hash_map<int64_t, const HloValue*> id_to_logical_buffer;
  TF_ASSIGN_OR_RETURN(
      id_to_logical_buffer,
      BuildIdToLogicalBufferMap(proto, id_to_hlo_instruction, alias_analysis));

  std::unique_ptr<BufferAssignment> buffer_assignment =
      absl::WrapUnique(new BufferAssignment(
          module, /*hlo_ordering=*/nullptr, std::move(buffer_size),
          /*color_alignment=*/nullptr, std::move(alias_analysis),
          /*hlo_live_range=*/nullptr));

  // Process each buffer allocation entry in the proto to create a new
  // allocation.
  for (const auto& alloc_proto : proto.buffer_allocations()) {
    BufferAllocation* allocation = buffer_assignment->NewEmptyAllocation(
        alloc_proto.size(), alloc_proto.color());

    // We don't copy allocation index as it gets automatically assigned.
    CHECK(allocation->index() == alloc_proto.index())
        << "Expected allocations in BufferAssignment proto to be sorted by "
           "index.";

    // Set allocation properties for a newly constructed BufferAllocation.
    allocation->set_is_thread_local(alloc_proto.is_thread_local());
    allocation->set_is_tuple(alloc_proto.is_tuple());
    allocation->set_constant(alloc_proto.is_constant());

    // If allocation corresponds to an entry computation parameter, copy
    // parameter properties to a BufferAllocation.
    if (alloc_proto.is_entry_computation_parameter()) {
      std::vector<int64_t> shape_idx_vals;
      absl::c_copy(alloc_proto.parameter_shape_index(),
                   std::back_inserter(shape_idx_vals));
      ShapeIndex shape_index(shape_idx_vals);
      const bool parameter_has_alias =
          module->input_output_alias_config().ParameterHasAlias(
              alloc_proto.parameter_number(), shape_index);
      allocation->set_entry_computation_parameter(
          alloc_proto.parameter_number(), shape_index, parameter_has_alias);
    }

    // Process each logical buffer assigned to the current allocation and create
    // buffer assignment entries.
    for (const auto& assignee : alloc_proto.assigned()) {
      HloValue::Id logical_buffer_id = assignee.logical_buffer_id();
      const auto& buffer_val = id_to_logical_buffer[logical_buffer_id];
      buffer_assignment->AddAssignment(allocation, *buffer_val,
                                       assignee.offset(), assignee.size());
    }

    // We don't set `maybe_live_out` as it is inferred automatically by
    // buffer assignment when we call `AddAssignment` above.
    CHECK_EQ(allocation->maybe_live_out(), alloc_proto.maybe_live_out())
        << "Dataflow analysis differs from proto.";
  }

  // Ensure each buffer in the proto has an allocation assigned.
  TF_RET_CHECK(proto.logical_buffers_size() ==
               buffer_assignment->allocation_index_for_value_.size());
  for (auto& logical_buffer_proto : proto.logical_buffers()) {
    TF_RET_CHECK(buffer_assignment->HasAllocation(
        *id_to_logical_buffer[logical_buffer_proto.id()]));
  }
  return buffer_assignment;
}

/* static */
absl::StatusOr<std::unique_ptr<BufferAssignment>> BufferAssigner::Run(
    const HloModule* module, std::unique_ptr<HloOrdering> hlo_ordering,
    BufferValue::SizeFunction buffer_size,
    LogicalBuffer::AlignmentFunction color_alignment,
    bool allocate_buffers_for_constants, BufferAssigner::Colorer colorer,
    std::optional<BufferAssigner::MustNotLiveOut> must_not_live_out,
    HloDataflowAnalysis::CanShareBuffer can_share_buffer,
    std::unique_ptr<PresetAssignments> preset_assignments,
    const PrivateStacks& private_stacks,
    GlobalDecreasingSizeBestFitHeap<HloValue>::BufferIntervalCompare
        heap_buffer_interval_compare,
    std::optional<BufferAssignment::BufferIsolationOptions> isolation_options,
    std::optional<BufferValue::Color> temp_buffer_color) {
  BufferAssigner assigner(allocate_buffers_for_constants, std::move(colorer),
                          must_not_live_out, std::move(preset_assignments));
  return assigner.CreateAssignment(
      module, std::move(hlo_ordering), std::move(buffer_size),
      std::move(color_alignment), std::move(can_share_buffer), private_stacks,
      heap_buffer_interval_compare, isolation_options, temp_buffer_color);
}

bool BufferAssigner::LiveRangeInterferes(const HloValue* buffer1,
                                         const HloValue* buffer2,
                                         BufferAssignment* assignment) {
  CHECK((assignment->hlo_live_range().total_order_scheduled()));
  const HloLiveRange& hlo_live_range = assignment->hlo_live_range();

  const auto& buffer_live_ranges = hlo_live_range.buffer_live_ranges();

  auto live_range_it1 = buffer_live_ranges.find(buffer1);
  CHECK(live_range_it1 != buffer_live_ranges.end())
      << "Buffer doesn't have a proper live range:" << buffer1->ToString();

  auto live_range_it2 = buffer_live_ranges.find(buffer2);
  CHECK(live_range_it2 != buffer_live_ranges.end())
      << "Buffer doesn't have a proper live range:" << buffer2->ToString();

  // Check if a user value can share the same buffer as its operand.
  auto can_share_as_operand =
      [&assignment](const HloValue* user_value, const HloValue* operand_value,
                    const HloLiveRange::TimeBound& operand_live_range) {
        // An hlo value can hold multiple instructions during its life time. We
        // only look at the last instruction and check if it can be shared with
        // the operand.
        HloPosition operand_end_position = operand_live_range.end_position;
        return user_value->instruction()->opcode() != HloOpcode::kCopy &&
               user_value->instruction()->IsUserOf(
                   operand_end_position.instruction) &&
               assignment->dataflow_analysis().CanShareOperandBufferWithUser(
                   operand_end_position.instruction, operand_end_position.index,
                   user_value->instruction(), user_value->index());
      };

  const auto& live_range_1 = live_range_it1->second;
  const auto& live_range_2 = live_range_it2->second;

  if (!(live_range_1.start > live_range_2.end ||
        live_range_2.start > live_range_1.end)) {
    if (live_range_1.end == live_range_2.start) {
      auto operand_value = buffer1;
      auto user_value = buffer2;
      if (!can_share_as_operand(user_value, operand_value, live_range_1)) {
        VLOG(4) << "End of live range of " << buffer1->ToShortString()
                << " is equal to the start of live range of "
                << buffer2->ToShortString() << ", buffer cannot be shared.";
        return true;
      }
    } else if (live_range_2.end == live_range_1.start) {
      auto operand_value = buffer2;
      auto user_value = buffer1;
      if (!can_share_as_operand(user_value, operand_value, live_range_2)) {
        VLOG(4) << "End of live range of " << buffer2->ToShortString()
                << " is equal to the start of live range of "
                << buffer1->ToShortString() << ", buffer cannot be shared.";
        return true;
      }
    } else {
      VLOG(4) << "Can't assign: assignee " << *buffer1 << " may interfere with "
              << *buffer2;
      VLOG(4) << "assigned_buffer.start: " << live_range_1.start;
      VLOG(4) << "assigned_buffer.end: " << live_range_1.end;
      VLOG(4) << "live_range_2.start" << live_range_2.start;
      VLOG(4) << "live_range_2.end" << live_range_2.end;
      return true;
    }
  }
  return false;
}

bool BufferAssigner::MaybeAssignBuffer(BufferAllocation* allocation,
                                       const HloBuffer& hlo_buffer,
                                       BufferAssignment* assignment) {
  CHECK(!assignment->HasAllocation(hlo_buffer))
      << "buffer " << hlo_buffer << " already has an allocation assigned.";

  VLOG(4) << "Trying to assign " << hlo_buffer << " size "
          << assignment->HloBufferSize(hlo_buffer)
          << " to allocation: " << *allocation;

  if (hlo_buffer.color() != allocation->color()) {
    VLOG(4) << "Can't assign: buffer has color " << hlo_buffer.color()
            << " and allocation has color " << allocation->color() << ".";
    return false;
  }

  if (assignment->HloBufferSize(hlo_buffer) > allocation->size()) {
    VLOG(4) << "Can't assign: buffer is larger than allocation ("
            << assignment->HloBufferSize(hlo_buffer) << " > "
            << allocation->size() << ")";
    return false;
  }

  if (allocation->is_readonly()) {
    VLOG(4) << "Can't assign: allocation is readonly";
    return false;
  }

  if (must_not_live_out_.has_value()) {
    if (allocation->maybe_live_out()) {
      // If a buffer maybe live out, the allocation cannot contain any node
      // where must_not_live_out_ returns true.
      for (const HloValue* value : hlo_buffer.values()) {
        if ((*must_not_live_out_)(assignment->alias_analysis(),
                                  value->instruction(), value->index())) {
          VLOG(4) << "Can't assign: " << value->instruction()->ToString()
                  << " cannot live out of the module";
          return false;
        }
      }
    }
    // The above check is not enough -- There could be the case where an
    // allocation can be not live out and contains an instruction where
    // must_not_live_out_ returns true, but assigning a live out buffer to
    // that allocation makes the allocation live out and also contain an
    // instruction where ust_not_live_out_ returns true.
    if (assignment->alias_analysis().BufferLivesOut(hlo_buffer)) {
      for (const auto& buffer_offset_size : allocation->assigned_buffers()) {
        const HloValue* value = buffer_offset_size.first;
        if ((*must_not_live_out_)(assignment->alias_analysis(),
                                  value->instruction(), value->index())) {
          VLOG(4) << "Can't assign: " << value->instruction()
                  << " cannot live out of the module";
          return false;
        }
      }
    }
  }

  if (!allocation->is_reusable()) {
    VLOG(4) << "Can't assign: allocation is not reusable";
    return false;
  }

  for (const auto& buffer_offset_size : allocation->assigned_buffers()) {
    // Pairwise compare.
    const HloValue& assigned_buffer =
        *CHECK_NOTNULL(dynamic_cast<const HloValue*>(buffer_offset_size.first));
    for (const HloValue* new_value : hlo_buffer.values()) {
      if (assignment->hlo_live_range().total_order_scheduled()) {
        if (LiveRangeInterferes(new_value, &assigned_buffer, assignment)) {
          VLOG(4) << "Can't assign: assignee " << assigned_buffer
                  << " live range interferes with "
                  << new_value->ToShortString();
          return false;
        }
      } else if (assignment->hlo_ordering().MayInterfere(
                     assigned_buffer, *new_value,
                     assignment->dataflow_analysis())) {
        // Fallback to partial order based interference detection (slower) when
        // we don't have a total order scheduled module.
        VLOG(4) << "Can't assign: assignee " << assigned_buffer
                << " may interfere with " << new_value->ToShortString();
        return false;
      }

      // Copy instruction don't share a buffer with their input operand.
      if (new_value->instruction()->opcode() == HloOpcode::kCopy) {
        for (const HloPosition& assigned_buffer_position :
             assigned_buffer.positions()) {
          if (new_value->instruction()->IsUserOf(
                  assigned_buffer_position.instruction)) {
            VLOG(4) << "Can't assign: assignee " << assigned_buffer
                    << " is used at copy instruction "
                    << new_value->ToShortString();
            return false;
          }
        }
      }
    }
  }

  // If the buffer is live out of the computation then it should only be
  // assigned a buffer which exactly fits the result to avoid wasting memory
  // (result buffers can have arbitrary lifetimes).
  if (assignment->alias_analysis().BufferLivesOut(hlo_buffer) &&
      allocation->size() != assignment->HloBufferSize(hlo_buffer)) {
    VLOG(4) << "Can't assign: buffer " << hlo_buffer
            << "is live out and size not the same as allocation";
    return false;
  }

  assignment->AddAssignment(allocation, hlo_buffer, /*offset=*/0,
                            assignment->HloBufferSize(hlo_buffer));
  return true;
}  // namespace xla

absl::Status BufferAssigner::AssignSingleHloBuffer(
    const HloBuffer* hlo_buffer, bool is_thread_local,
    absl::flat_hash_map<const HloComputation*,
                        absl::flat_hash_set<const HloValue*>>*
        buffers_to_assign_sequentially,
    std::vector<BufferAllocation::Index>* allocation_indices,
    BufferAssignment* assignment) {
  const int64_t buffer_size = assignment->HloBufferSize(*hlo_buffer);
  for (const HloValue* value : hlo_buffer->values()) {
    if (value->instruction()->opcode() == HloOpcode::kConstant) {
      if (allocate_buffers_for_constants_) {
        BufferAllocation* allocation =
            assignment->NewAllocation(*hlo_buffer, buffer_size);
        allocation->set_constant(true);
        VLOG(3) << "New allocation #" << allocation->index() << " for constant "
                << *hlo_buffer << " value ptr: " << value;
      }
      VLOG(3) << "Not allocating buffer for constant";
      return absl::OkStatus();
    }

    const HloInstruction* instruction = value->instruction();
    const bool is_entry_parameter =
        instruction->opcode() == HloOpcode::kParameter &&
        instruction->parent() == instruction->GetModule()->entry_computation();

    if (is_entry_parameter) {
      bool parameter_has_alias =
          assignment->module().input_output_alias_config().ParameterHasAlias(
              instruction->parameter_number(), value->index());
      // If the hlo buffer is part of an external parameter, creates a new
      // allocation and sets its parameter number. Parameters of non-entry
      // computations do not need special allocations because they live inside
      // callers.
      BufferAllocation* allocation =
          assignment->NewAllocation(*hlo_buffer, buffer_size);

      allocation->set_entry_computation_parameter(
          instruction->parameter_number(), value->index(), parameter_has_alias);
      if (parameter_has_alias) {
        allocation_indices->push_back(allocation->index());
      }
      VLOG(3) << "New allocation #" << allocation->index()
              << " marked as entry computation parameter: " << *hlo_buffer;
      return absl::OkStatus();
    }
  }

  if (is_thread_local) {
    BufferAllocation* allocation =
        assignment->NewAllocation(*hlo_buffer, buffer_size);
    allocation->set_is_thread_local(true);
    VLOG(3) << "New allocation #" << allocation->index()
            << " for thread-local: " << *hlo_buffer;
    return absl::OkStatus();
  }

  for (const HloValue* value : hlo_buffer->values()) {
    if (value->shape().IsTuple()) {
      BufferAllocation* allocation =
          assignment->NewAllocation(*hlo_buffer, buffer_size);
      allocation->set_is_tuple(true);
      VLOG(3) << "New allocation #" << allocation->index()
              << " for tuple-shaped buffer: " << *hlo_buffer;
      return absl::OkStatus();
    }

    if (value->IsTopLevel() && !value->IsTuple()) {
      const HloInstruction* instruction = value->instruction();
      for (auto* operand : instruction->operands()) {
        for (const auto& operand_slice :
             assignment->GetAllSlices(operand, /*index=*/{})) {
          BufferAllocation* allocation =
              assignment->GetMutableAllocation(operand_slice.index());
          if (MaybeAssignBuffer(allocation, *hlo_buffer, assignment)) {
            VLOG(3) << "Reusing (operand) allocation #" << allocation->index()
                    << " for: " << *hlo_buffer;
            return absl::OkStatus();
          }
        }
      }
    }
  }

  // Find the smallest buffer which can be reused iterating from end of
  // allocation_indices (smallest) to beginning (largest).
  for (int allocation_index = allocation_indices->size() - 1;
       allocation_index >= 0; allocation_index--) {
    BufferAllocation* allocation = assignment->GetMutableAllocation(
        allocation_indices->at(allocation_index));
    if (MaybeAssignBuffer(allocation, *hlo_buffer, assignment)) {
      VLOG(3) << "Reusing allocation #" << allocation->index()
              << " for: " << *hlo_buffer;
      return absl::OkStatus();
    }
  }

  if (!assignment->HasAllocation(*hlo_buffer) &&
      !assignment->alias_analysis().BufferLivesOut(*hlo_buffer)) {
    bool all_computations_have_sequential_order = true;
    for (const HloValue* hlo_value : hlo_buffer->values()) {
      HloComputation* computation = hlo_value->instruction()->parent();
      const bool has_sequential_order =
          assignment->hlo_ordering().SequentialOrder(*computation) != nullptr;
      all_computations_have_sequential_order &= has_sequential_order;
    }

    if (all_computations_have_sequential_order) {
      for (const HloValue* hlo_value : hlo_buffer->values()) {
        HloComputation* computation = hlo_value->instruction()->parent();
        // There is a sequential instruction ordering, so we delay assignment
        // of temp buffers until after the loop. We do this right before we
        // decide to create a new allocation, to ensure we've exhausted all
        // the buffer re-use cases above.
        //
        // Entry parameters and thread local buffers were already handled
        // earlier in this loop iteration.  See
        // BufferAllocation::IsPreallocatedTempBuffer for the definition of
        // temp buffers.
        (*buffers_to_assign_sequentially)[computation].insert(hlo_value);
        VLOG(3) << "Delaying assignment of temp buffer: " << *hlo_value;
      }
      return absl::OkStatus();
    }
  }

  if (!assignment->HasAllocation(*hlo_buffer)) {
    BufferAllocation* allocation =
        assignment->NewAllocation(*hlo_buffer, buffer_size);
    allocation_indices->push_back(allocation->index());
    VLOG(3) << "New allocation #" << allocation->index()
            << " for: " << *hlo_buffer;
  }

  TF_RET_CHECK(assignment->HasAllocation(*hlo_buffer));
  return absl::OkStatus();
}

absl::Status BufferAssigner::AssignBuffersForComputations(
    const std::vector<const HloComputation*>& computations,
    bool is_thread_local,
    absl::flat_hash_map<const HloComputation*,
                        absl::flat_hash_set<const HloValue*>>*
        buffers_to_assign_sequentially,
    BufferAssignment* assignment) {
  if (computations.empty()) {
    return absl::OkStatus();
  }
  std::vector<const HloBuffer*> sorted_buffers;

  // First assign the preset allocations.
  absl::flat_hash_set<const HloBuffer*> preset_assigned_buffers;

  TF_RETURN_IF_ERROR(AssignPresetBuffers(&preset_assigned_buffers, assignment));

  const HloAliasAnalysis& alias_analysis = assignment->alias_analysis();

  for (const HloBuffer& buffer : alias_analysis.buffers()) {
    // Skip if the buffer is already assigned since it had a preset allocation.
    if (preset_assigned_buffers.find(&buffer) !=
        preset_assigned_buffers.end()) {
      VLOG(3) << "Skip allocation for buffer: " << buffer;
      continue;
    }
    TF_RET_CHECK(!buffer.values().empty());
    const HloComputation* comp = buffer.values()[0]->instruction()->parent();
    if (absl::c_linear_search(computations, comp)) {
      sorted_buffers.push_back(&buffer);
    }
  }

  // Generate a post order sort of instructions for sorting of the
  // HloBuffers.
  flat_hash_map<const HloInstruction*, int> post_order_position;
  int position = 0;
  std::vector<const HloComputation*> reverse_post_order_computations;
  std::unique_ptr<CallGraph> call_graph =
      CallGraph::Build(computations[0]->parent());
  TF_RETURN_IF_ERROR(call_graph->VisitNodes([&](const CallGraphNode& node) {
    if (absl::c_linear_search(computations, node.computation())) {
      reverse_post_order_computations.push_back(node.computation());
    }
    return absl::OkStatus();
  }));
  absl::c_reverse(reverse_post_order_computations);
  for (auto* computation : reverse_post_order_computations) {
    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      post_order_position.emplace(instruction, position);
      position++;
    }
  }

  HloSchedule schedule(&assignment->module());

  for (const HloComputation* computation : computations) {
    const HloInstructionSequence* instruction_sequence =
        assignment->hlo_ordering().SequentialOrder(*computation);
    const bool has_sequential_order = instruction_sequence != nullptr;
    if (has_sequential_order && buffers_to_assign_sequentially != nullptr) {
      // Every sequential computation must get an entry in the
      // buffers_to_assign_sequentially map, even if we end up with an empty
      // set of buffers. This ensures we can correctly determine whether to
      // run whole-module heap simulation.
      buffers_to_assign_sequentially->emplace(computation,
                                              flat_hash_set<const HloValue*>());

      schedule.set_sequence(computation, *instruction_sequence);
    }
  }

  absl::c_sort(
      sorted_buffers, [&post_order_position, &alias_analysis, assignment](
                          const HloBuffer* a, const HloBuffer* b) {
        // Primary sort is by decreasing buffer size.
        const int64_t a_size = assignment->HloBufferSize(*a);
        const int64_t b_size = assignment->HloBufferSize(*b);
        if (a_size != b_size) {
          return a_size > b_size;  // use ">" for decreasing size.
        }

        const bool a_live_out = alias_analysis.BufferLivesOut(*a);
        const bool b_live_out = alias_analysis.BufferLivesOut(*b);
        if (a_live_out != b_live_out) {
          return a_live_out;
        }
        auto compare = [&post_order_position](const HloValue* value1,
                                              const HloValue* value2) {
          return post_order_position.at(value1->instruction()) <
                 post_order_position.at(value2->instruction());
        };
        const HloValue* a_min = *absl::c_min_element(a->values(), compare);
        const HloValue* b_min = *absl::c_min_element(b->values(), compare);
        if (post_order_position.at(a_min->instruction()) <
            post_order_position.at(b_min->instruction())) {
          return true;
        } else if (post_order_position.at(a_min->instruction()) >
                   post_order_position.at(b_min->instruction())) {
          return false;
        }

        // Use buffer ids to break ties and ensure a stable ordering.
        return a->id() < b->id();
      });

  std::vector<BufferAllocation::Index> allocation_indices;

  for (const HloBuffer* buffer : sorted_buffers) {
    VLOG(3) << "=================================================";
    VLOG(3) << "Assigning buffer for " << *buffer;
    TF_RETURN_IF_ERROR(AssignSingleHloBuffer(buffer, is_thread_local,
                                             buffers_to_assign_sequentially,
                                             &allocation_indices, assignment));
  }
  return absl::OkStatus();
}

flat_hash_map<LogicalBuffer::Color, flat_hash_set<const HloValue*>>
BufferAssigner::SplitBuffersByColor(
    const flat_hash_set<const HloValue*>& buffers) const {
  flat_hash_map<LogicalBuffer::Color, flat_hash_set<const HloValue*>> color_map;
  for (auto buffer : buffers) {
    color_map[buffer->color()].insert(buffer);
  }
  return color_map;
}

absl::flat_hash_map<const HloComputation*, absl::flat_hash_set<const HloValue*>>
BufferAssigner::SplitBuffersByPrivateStackComputation(
    const absl::flat_hash_set<const HloValue*>& buffers,
    absl::Span<const HloComputation* const> private_stack_computations,
    const CallGraph& call_graph) const {
  absl::flat_hash_map<const HloComputation*,
                      absl::flat_hash_set<const HloValue*>>
      computation_map;
  for (const HloValue* value : buffers) {
    bool found_computation = false;
    for (const HloComputation* computation : private_stack_computations) {
      if (call_graph.InstructionIsNestedIn(value->instruction(), computation)) {
        found_computation = true;
        computation_map[computation].insert(value);
        break;
      }
    }
    CHECK(found_computation);
  }
  return computation_map;
}

absl::Status BufferAssigner::AssignPresetBuffers(
    absl::flat_hash_set<const HloBuffer*>* assigned_buffers,
    BufferAssignment* assignment) {
  if (!preset_assignments_) {
    return absl::OkStatus();
  }

  // Create an allocation for each preset color.
  absl::flat_hash_map<LogicalBuffer::Color, BufferAllocation*>
      preset_allocations;
  for (auto& color_and_info : preset_assignments_->assignment_informations()) {
    LogicalBuffer::Color color(color_and_info.first);
    auto inserted = preset_allocations.emplace(
        color,
        assignment->NewEmptyAllocation(color_and_info.second.size, color));
    BufferAllocation* inserted_allocation = inserted.first->second;
    inserted_allocation->AddHeapTrace(
        color_and_info.second.heap_simulator_trace);
    VLOG(3) << "Created preset buffer allocation "
            << inserted_allocation->index()
            << ", color: " << inserted_allocation->color()
            << ", size: " << inserted_allocation->size();
  }

  const HloAliasAnalysis& alias_analysis = assignment->alias_analysis();

  for (auto& position_and_chunk : preset_assignments_->chunks()) {
    const HloPosition& defining_position = position_and_chunk.first;
    const HloBuffer& buffer = alias_analysis.GetUniqueBufferAt(
        defining_position.instruction, defining_position.index);
    for (const HloValue* value : buffer.values()) {
      VLOG(3) << "Preset allocation for value: " << value->ToShortString();
      const HeapSimulator::Chunk& chunk = position_and_chunk.second;
      auto preset_allocations_iter = preset_allocations.find(value->color());
      CHECK(preset_allocations_iter != preset_allocations.end())
          << "No preset value allocation for color " << value->color()
          << " for " << value->ToShortString() << " found.";
      preset_allocations_iter->second->AddAssignment(*value, chunk.offset,
                                                     chunk.size);
    }

    assigned_buffers->insert(&buffer);
  }

  // Upon consumption of the preset assignments, delete it so that if this
  // method is called again, it does not assign the same buffers multiple times.
  preset_assignments_ = {};

  return absl::OkStatus();
}

absl::Status BufferAssigner::AssignBuffersWithSequentialOrdering(
    const flat_hash_map<const HloComputation*, flat_hash_set<const HloValue*>>&
        buffers_to_assign_sequentially,
    bool run_whole_module_heap_simulation, BufferAssignment* assignment,
    const PrivateStacks& private_stacks,
    GlobalDecreasingSizeBestFitHeap<HloValue>::BufferIntervalCompare
        heap_buffer_interval_compare,
    std::optional<BufferAssignment::BufferIsolationOptions> isolation_options) {
  // Run the sequence of instructions through the heap simulator.  The
  // heuristic that seems to give the best results is lazy-best-fit, with all
  // runs of alloc / free calls sorted in decreasing size order.
  const HloOrdering& hlo_ordering = assignment->hlo_ordering();

  // Returns a heap algorithm that chooses the best result from several
  // algorithms.
  auto get_heap_algorithm =
      [&](int64_t alignment) -> std::unique_ptr<HeapAlgorithm<HloValue>> {
    if (heap_buffer_interval_compare) {
      return std::make_unique<ConstrainedGlobalDecreasingSizeBestFitHeap>(
          assignment->multiheap_size_constraint_per_heap(), alignment,
          GlobalDecreasingSizeBestFitHeap<HloValue>::kCustom,
          heap_buffer_interval_compare);
    }
    auto algorithms = std::make_unique<
        std::vector<std::unique_ptr<HeapAlgorithm<HloValue>>>>();
    algorithms->push_back(
        std::make_unique<ConstrainedGlobalDecreasingSizeBestFitHeap>(
            assignment->multiheap_size_constraint_per_heap(), alignment,
            GlobalDecreasingSizeBestFitHeap<HloValue>::kSpatial));
    algorithms->push_back(
        std::make_unique<ConstrainedGlobalDecreasingSizeBestFitHeap>(
            assignment->multiheap_size_constraint_per_heap(), alignment,
            GlobalDecreasingSizeBestFitHeap<HloValue>::kTemporal));
    return std::make_unique<ChooseBestHeapAlgorithm<HloValue>>(
        std::move(algorithms));
  };

  if (run_whole_module_heap_simulation) {
    // Run the heap simulation over the whole module. This reduces memory
    // usage, since buffers for kCall, kWhile, and kConditional
    // sub-computations are only live for the duration of their calling
    // instructions.
    VLOG(1) << "Running whole-module heap simulation";
    HloSchedule schedule(&assignment->module());
    flat_hash_set<const HloValue*> all_buffers_to_assign;
    for (const auto& pair : buffers_to_assign_sequentially) {
      const HloComputation* computation = pair.first;
      const flat_hash_set<const HloValue*>& buffers_to_assign = pair.second;
      const HloInstructionSequence* instruction_sequence =
          hlo_ordering.SequentialOrder(*computation);
      CHECK(instruction_sequence != nullptr) << computation->name();
      schedule.set_sequence(computation, *instruction_sequence);
      all_buffers_to_assign.insert(buffers_to_assign.begin(),
                                   buffers_to_assign.end());
    }
    auto color_map = SplitBuffersByColor(all_buffers_to_assign);
    std::vector<LogicalBuffer::Color> sorted_colors;
    sorted_colors.reserve(color_map.size());
    for (auto& single_colored_set : color_map) {
      auto color = single_colored_set.first;
      sorted_colors.emplace(sorted_colors.end(), color);
    }
    absl::c_sort(sorted_colors);
    for (auto color : sorted_colors) {
      VLOG(2) << "Simulating heap for color " << color;
      int64_t alignment = assignment->color_alignment_(color);
      HeapSimulator::Options options;
      options.alloc_constants = allocate_buffers_for_constants_;
      auto private_stacks_it = private_stacks.find(color);
      if (private_stacks_it != private_stacks.end()) {
        // For private stack colors, we collect all of the buffers that are
        // dominated by the private stack computation and run heap simulation on
        // that computation. The reason why we don't perform a whole-module heap
        // simulation is that all buffers that participate in an async operation
        // are treated as live for the duration of the async operation in
        // whole-module heap simulation. Performing heap simulation from the
        // private stack computation allows better temporal reuse of buffers.
        auto computation_map = SplitBuffersByPrivateStackComputation(
            color_map[color], private_stacks_it->second,
            assignment->alias_analysis().dataflow_analysis().call_graph());
        for (const HloComputation* private_stack_computation :
             private_stacks_it->second) {
          VLOG(2) << "private stack computation: "
                  << private_stack_computation->name();
          auto computation_map_it =
              computation_map.find(private_stack_computation);
          CHECK(computation_map_it != computation_map.end());
          options.buffers_to_assign = &computation_map_it->second;
          const HloInstructionSequence* instruction_sequence =
              hlo_ordering.SequentialOrder(*private_stack_computation);
          TF_ASSIGN_OR_RETURN(
              HeapSimulator::Result<HloValue> result,
              HeapSimulator::Run(
                  get_heap_algorithm(alignment), *private_stack_computation,
                  *instruction_sequence, assignment->alias_analysis(),
                  assignment->buffer_size_, &schedule, options));
          AssignBuffersFromHeapSimulator(result, assignment, color,
                                         isolation_options);
        }
      } else {
        options.buffers_to_assign = &color_map[color];
        TF_ASSIGN_OR_RETURN(
            HeapSimulator::Result<HloValue> result,
            HeapSimulator::Run(get_heap_algorithm(alignment),
                               assignment->module(), schedule,
                               assignment->alias_analysis(),
                               assignment->buffer_size_, options));
        AssignBuffersFromHeapSimulator(result, assignment, color,
                                       isolation_options);
      }
    }
  } else {
    // Run the heap-simulation on a per-computation basis. Buffers for
    // sub-computations are assigned disjoint BufferAllocations, assuming the
    // worst-case that they may all be live concurrently.
    VLOG(1) << "Running per-computation heap simulation";
    for (const auto& pair : buffers_to_assign_sequentially) {
      const HloComputation* computation = pair.first;
      const flat_hash_set<const HloValue*>& buffers_to_assign = pair.second;
      const HloInstructionSequence* instruction_sequence =
          hlo_ordering.SequentialOrder(*computation);
      CHECK(instruction_sequence != nullptr) << computation->name();
      auto color_map = SplitBuffersByColor(buffers_to_assign);
      std::vector<LogicalBuffer::Color> sorted_colors;
      sorted_colors.reserve(color_map.size());
      for (auto& single_colored_set : color_map) {
        auto color = single_colored_set.first;
        sorted_colors.emplace(sorted_colors.end(), color);
      }
      absl::c_sort(sorted_colors);
      for (auto color : sorted_colors) {
        VLOG(2) << "Simulating heap for color " << color;
        int64_t alignment = assignment->color_alignment_(color);
        HeapSimulator::Options options;
        options.buffers_to_assign = &color_map[color];
        TF_ASSIGN_OR_RETURN(
            HeapSimulator::Result<HloValue> result,
            HeapSimulator::Run(get_heap_algorithm(alignment), *computation,
                               *instruction_sequence,
                               assignment->alias_analysis(),
                               assignment->buffer_size_, options));
        AssignBuffersFromHeapSimulator(result, assignment, color,
                                       isolation_options);
      }
    }
  }
  return absl::OkStatus();
}

namespace {
// Computes and returns the set of logical buffers live at the point of
// maximal liveness in the given heap trace. LogicalBuffers are (stabily)
// sorted by id.
std::vector<const HloValue*> ComputePeakMemoryLogicalBuffers(
    const BufferAllocation& allocation, const HeapSimulatorTrace& heap_trace) {
  // Create a map from LogicalBuffer::Id to LogicalBuffer* for the logical
  // buffers in this allocation.
  absl::flat_hash_map<BufferValue::Id, const HloValue*> id_to_value;
  absl::flat_hash_map<const HloValue*, int64_t> buffer_sizes;
  for (const auto& pair : allocation.assigned_buffers()) {
    const HloValue* value = pair.first;
    const BufferAllocation::OffsetSize& offset_size = pair.second;
    id_to_value[value->id()] = value;
    buffer_sizes[value] = offset_size.size;
  }
  VLOG(1) << "Compute peak memory logical buffers";

  // To properly account for shared buffers, we keep track of the number of
  // instances of the same shared buffer are currently live, their canonical ids
  // and the size we had returned when allocating the buffer so that we can
  // return the -size when freeing the buffer.
  absl::flat_hash_map<int64_t, int> num_outstanding_shared_buffers;
  absl::flat_hash_map<int64_t, int64_t> shared_canonical_ids;
  absl::flat_hash_map<int64_t, int64_t> allocated_sizes;
  // Returns how much the given event increases the total size of live
  // buffers. Can be negative.
  auto memory_delta = [&](const HeapSimulatorTrace::Event& event) -> int64_t {
    const HloValue* buffer = id_to_value.at(event.buffer_id());
    const int64_t buffer_size = buffer_sizes.at(buffer);
    if (event.kind() == HeapSimulatorTrace::Event::ALLOC) {
      num_outstanding_shared_buffers[event.buffer_id()] = 1;
      allocated_sizes[event.buffer_id()] = buffer_size;
      return buffer_size;
    } else if (event.kind() == HeapSimulatorTrace::Event::SHARE_WITH) {
      shared_canonical_ids[event.buffer_id()] = event.share_with_canonical_id();
      if (++num_outstanding_shared_buffers[event.share_with_canonical_id()] ==
          1) {
        // This shared buffer is currently the only instance of the buffer with
        // the canonical id. So we return the buffer size.
        allocated_sizes[event.buffer_id()] = buffer_size;
        return buffer_size;
      }
      // There are multiple instances of this buffer, so return 0.
      allocated_sizes[event.buffer_id()] = 0;
      return 0;
    } else if (event.kind() == HeapSimulatorTrace::Event::FREE) {
      auto shared_canonical_id_it =
          shared_canonical_ids.find(event.buffer_id());
      // Decrement the outstanding instances of this buffer and return the
      // -size.
      int64_t buffer_id = (shared_canonical_id_it == shared_canonical_ids.end())
                              ? event.buffer_id()
                              : shared_canonical_id_it->second;
      --num_outstanding_shared_buffers[buffer_id];
      return -1 * allocated_sizes[event.buffer_id()];
    }
    LOG(FATAL) << "Unknown event kind: " << event.kind();
  };

  // First compute the size of the maximal live set.
  int64_t max_live_size = 0;
  int64_t live_size = 0;
  for (const auto& event : heap_trace.events()) {
    if (!id_to_value.contains(event.buffer_id())) {
      // Skip as the buffer associated with this trace event is not placed into
      // this allocation. This can happen when size constraints are given to the
      // heap simulator.
      continue;
    }
    live_size += memory_delta(event);
    if (max_live_size < live_size) {
      max_live_size = live_size;
    }
  }

  // Next gather the set of logical buffers live at the earliest point of
  // maximal live set size.
  absl::flat_hash_set<const HloValue*> live_values;
  live_size = 0;
  num_outstanding_shared_buffers.clear();
  for (const auto& event : heap_trace.events()) {
    if (!id_to_value.contains(event.buffer_id())) {
      // Skip as the buffer associated with this trace event is not placed into
      // this allocation. This can happen when size constraints are given to the
      // heap simulator.
      continue;
    }
    const HloValue* value = id_to_value.at(event.buffer_id());
    int64_t delta = memory_delta(event);
    // To avoid including buffers that are aliases of each other to the peak
    // buffers list, only add the buffers that memory_delta returns non-zero
    // positive sizes. memory_delta returns 0 as the size for the buffer already
    // has a live alias of itself.
    if (delta > 0) {
      InsertOrDie(&live_values, value);
    } else if (delta < 0) {
      CHECK(ContainsKey(live_values, value));
      live_values.erase(value);
    }
    live_size += delta;

    if (live_size == max_live_size) {
      break;
    }
  }
  CHECK_EQ(live_size, max_live_size);

  std::vector<const HloValue*> live_values_vector;
  live_values_vector.insert(live_values_vector.end(), live_values.begin(),
                            live_values.end());

  // Stabily sort the live buffers.
  absl::c_sort(live_values_vector, [](const HloValue* a, const HloValue* b) {
    return a->id() < b->id();
  });
  VLOG(4) << "Peak memory buffer:";
  for (auto value : live_values_vector) {
    VLOG(4) << "  " << value->ToString();
  }
  return live_values_vector;
}

}  // namespace

void BufferAssigner::IsolateHeapBuffers(
    std::optional<BufferAssignment::BufferIsolationOptions> isolation_options,
    const BufferAssignment* assignment, LogicalBuffer::Color color,
    HeapSimulator::Result<HloValue>& result) const {
  if (!isolation_options) {
    return;
  }

  result.heap_size = 0;
  for (HeapSimulator::HeapResult<HloValue>& heap_result : result.heap_results) {
    if (absl::c_find(isolation_options->config.isolation_colors(), color) !=
        isolation_options->config.isolation_colors().end()) {
      VLOG(1) << "Isolating color: " << color;
      int64_t alignment = assignment->color_alignment_(color);
      // First, sort the values by the provided comparison function.
      std::vector<const HloValue*> sorted_values;
      sorted_values.reserve(heap_result.chunk_map.size());
      for (const auto& [value, chunk] : heap_result.chunk_map) {
        sorted_values.push_back(value);
      }
      absl::c_sort(sorted_values, isolation_options->hlo_value_compare);

      // Calculate the offset to place the next isolated buffer.
      int64_t isolation_offset =
          RoundUpTo(isolation_options->config.base_offset_bytes() +
                        heap_result.heap_size +
                        isolation_options->config.isolation_padding_bytes(),
                    alignment);
      int64_t value_index;
      // Iterate on the buffers up to the size of values or the fuel, whichever
      // is smaller.
      for (value_index = 0;
           value_index < std::min(static_cast<int64_t>(sorted_values.size()),
                                  isolation_options->config.isolation_fuel());
           ++value_index) {
        const HloValue* value = sorted_values[value_index];
        HeapSimulator::Chunk& chunk = heap_result.chunk_map.at(value);
        VLOG(1) << "Isolating " << value->ToShortString() << " from "
                << chunk.offset << " to " << isolation_offset;
        chunk.offset = isolation_offset;
        isolation_offset += RoundUpTo(
            chunk.size + isolation_options->config.isolation_padding_bytes(),
            alignment);
      }
      // For the values that aren't isolated, adjust the offset by the provided
      // base offset.
      for (; value_index < sorted_values.size(); ++value_index) {
        const HloValue* value = sorted_values[value_index];
        HeapSimulator::Chunk& chunk = heap_result.chunk_map.at(value);
        int64_t new_offset = RoundUpTo(
            chunk.offset + isolation_options->config.base_offset_bytes(),
            alignment);
        VLOG(1) << "Not isolating " << value->ToShortString() << ", from "
                << chunk.offset << " to " << new_offset;
        chunk.offset = new_offset;
      }
      heap_result.heap_size = isolation_offset;
    }
    result.heap_size += heap_result.heap_size;
  }
}

void BufferAssigner::AssignBuffersFromHeapSimulator(
    HeapSimulator::Result<HloValue>& result, BufferAssignment* assignment,
    BufferValue::Color color,
    std::optional<BufferAssignment::BufferIsolationOptions> isolation_options) {
  IsolateHeapBuffers(isolation_options, assignment, color, result);
  if (assignment->stats_.preallocated_temp_fragmentation_bytes == -1) {
    assignment->stats_.preallocated_temp_fragmentation_bytes =
        result.fragmentation_size;
  } else {
    assignment->stats_.preallocated_temp_fragmentation_bytes +=
        result.fragmentation_size;
  }
  VLOG(1) << "Result size from heap simulator: " << result.heap_size;

  // Iterate through heap_results. For each heap_result, create a new allocation
  // in `assignment`.
  for (const HeapSimulator::HeapResult<HloValue>& heap_result :
       result.heap_results) {
    BufferAllocation* allocation =
        assignment->NewEmptyAllocation(heap_result.heap_size, color);

    for (const auto& [value, chunk] : heap_result.chunk_map) {
      assignment->AddAssignment(allocation, *value, chunk.offset, chunk.size);
    }
    allocation->peak_buffers_ =
        ComputePeakMemoryLogicalBuffers(*allocation, result.debug_trace);

    XLA_VLOG_LINES(2, allocation->ToString());

    allocation->AddHeapTrace(result.debug_trace);
  }
}

absl::StatusOr<std::unique_ptr<BufferAssignment>>
BufferAssigner::CreateAssignment(
    const HloModule* module, std::unique_ptr<HloOrdering> hlo_ordering,
    BufferValue::SizeFunction buffer_size,
    LogicalBuffer::AlignmentFunction color_alignment,
    HloDataflowAnalysis::CanShareBuffer can_share_buffer,
    const PrivateStacks& private_stacks,
    GlobalDecreasingSizeBestFitHeap<HloValue>::BufferIntervalCompare
        heap_buffer_interval_compare,
    std::optional<BufferAssignment::BufferIsolationOptions> isolation_options,
    std::optional<BufferValue::Color> temp_buffer_color) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                      HloAliasAnalysis::Run(module, can_share_buffer));

  // Set up a schedule for each computation.
  HloSchedule schedule(module);
  for (const HloComputation* computation : module->computations()) {
    const HloInstructionSequence* instruction_sequence =
        hlo_ordering->SequentialOrder(*computation);
    const bool has_sequential_order = instruction_sequence != nullptr;
    if (has_sequential_order) {
      schedule.set_sequence(computation, *instruction_sequence);
    }
  }

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloLiveRange> hlo_live_range,
                      HloLiveRange::Run(schedule, *alias_analysis,
                                        module->entry_computation(), true));

  VLOG(1) << "Assigning buffers to module " << module->name();
  XLA_VLOG_LINES(3, module->ToString());
  XLA_VLOG_LINES(3, alias_analysis->ToString());
  XLA_VLOG_LINES(3, alias_analysis->dataflow_analysis().ToString());
  VLOG(1) << "Number of buffers to assign: "
          << alias_analysis->buffers().size();

  // Can't use std::make_unique because BufferAssignment constructor is
  // private.
  std::unique_ptr<BufferAssignment> assignment(new BufferAssignment(
      module, std::move(hlo_ordering), std::move(buffer_size),
      std::move(color_alignment), std::move(alias_analysis),
      std::move(hlo_live_range)));

  TF_RETURN_IF_ERROR(
      colorer_(&assignment->alias_analysis(), assignment->hlo_ordering()));
  VLOG(3) << "After coloring:";
  XLA_VLOG_LINES(3,
                 assignment->alias_analysis().dataflow_analysis().ToString());

  std::vector<const HloComputation*> thread_local_computations;
  std::vector<const HloComputation*> global_computations;
  TF_RETURN_IF_ERROR(GatherComputationsByAllocationType(
      module, &thread_local_computations, &global_computations));

  // First assign buffers for global computations. Temporary buffers for
  // sequential computations are collected in
  // 'buffers_to_assign_sequentially'.
  flat_hash_map<const HloComputation*, flat_hash_set<const HloValue*>>
      buffers_to_assign_sequentially;
  TF_RETURN_IF_ERROR(AssignBuffersForComputations(
      global_computations,
      /*is_thread_local=*/false, &buffers_to_assign_sequentially,
      assignment.get()));
  // Assign buffers with sequential ordering, if any. If all global
  // computations are sequential, we can run heap simulation on the whole
  // module, which reduces memory usage.
  const bool run_whole_module_heap_simulation =
      buffers_to_assign_sequentially.size() == global_computations.size();
  VLOG(2) << "Running whole module heap simulation: "
          << run_whole_module_heap_simulation;
  const int32_t multiheap_size_constraint_per_heap =
      module->config().debug_options().xla_multiheap_size_constraint_per_heap();
  VLOG(2) << "Multiheap per heap size limit: "
          << multiheap_size_constraint_per_heap;
  TF_RETURN_IF_ERROR(AssignBuffersWithSequentialOrdering(
      buffers_to_assign_sequentially, run_whole_module_heap_simulation,
      assignment.get(), private_stacks, heap_buffer_interval_compare,
      isolation_options));

  std::vector<const HloComputation*> thread_local_computations_no_fusion;
  // Now assign buffers for thread-local computations. All LogicalBuffers get
  // their own BufferAllocation.

  for (auto* computation : thread_local_computations) {
    TF_RET_CHECK(computation != module->entry_computation());
    if (computation->IsFusionComputation()) {
      continue;
    }
    thread_local_computations_no_fusion.push_back(computation);
  }

  TF_RETURN_IF_ERROR(AssignBuffersForComputations(
      thread_local_computations_no_fusion,
      /*is_thread_local=*/true,
      /*buffers_to_assign_sequentially=*/nullptr, assignment.get()));

  // Mark all buffers which may be live out of the entry computation as
  // "liveout".
  for (const HloBuffer* buffer :
       assignment->alias_analysis().LiveOutBuffers()) {
    VLOG(3) << "maybe_live_out LogicalBuffer: " << *buffer;
    if (assignment->HasAllocation(*buffer)) {
      BufferAllocation* alloc =
          assignment->GetMutableAssignedAllocation(*buffer);
      alloc->set_maybe_live_out(true);
      VLOG(3) << "maybe_live_out BufferAllocation: " << *alloc;
    }
  }

  // Combines allocations of temporary buffers into big BufferAllocations
  // subject to the buffer allocation size constraint. This can only be
  // performed after all buffers have been assigned, and after maybe_live_out
  // is marked, since it is used to determine whether an allocation contains
  // temporary buffers or not.
  absl::flat_hash_set<BufferValue::Color> private_stack_colors;
  for (const auto& [color, computations] : private_stacks) {
    private_stack_colors.insert(color);
  }

  assignment->CombineTempAllocations(private_stack_colors, temp_buffer_color);

  XLA_VLOG_LINES(2, assignment->ToString());
  assignment->ComputeSummaryStats();
  XLA_VLOG_LINES(1,
                 assignment->StatsString(/*report_total_fragmentation=*/true));
  VLOG(1) << "Buffer assignment done.";
  return std::move(assignment);
}

}  // namespace xla

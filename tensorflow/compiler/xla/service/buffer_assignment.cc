/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/buffer_assignment.h"

#include <algorithm>
#include <deque>
#include <ostream>
#include <utility>

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/buffer_value_containers.h"
#include "tensorflow/compiler/xla/service/heap_simulator.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_scheduling.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

namespace xla {

using ::tensorflow::gtl::FlatMap;
using ::tensorflow::gtl::FlatSet;
using ::tensorflow::strings::Appendf;
using ::tensorflow::strings::HumanReadableNumBytes;
using ::tensorflow::strings::Printf;
using ::tensorflow::strings::StrAppend;

namespace {

template <typename T>
string ColocatedBufferSetsToString(const T& container, const char* title) {
  string result;
  StrAppend(&result, title, "\n");
  for (const auto& it : container) {
    StrAppend(&result, "\t", it->ToString(), "\n");
  }
  return result;
}

// Walk the call graph of the HLO module and place each computation into either
// thread_local_computations or global_computations depending upon whether the
// computation requires thread-local allocations or global allocations. The
// elements in thread_local_computations and global_computations are in post
// order (if computation A has an instruction which calls computation B, then A
// will appear after B in the vector).
Status GatherComputationsByAllocationType(
    const HloModule* module,
    std::vector<const HloComputation*>* thread_local_computations,
    std::vector<const HloComputation*>* global_computations) {
  // Create a worklist of computations paired with whether the allocation must
  // be thread-local.
  std::deque<std::pair<const HloComputation*, bool>> worklist;
  worklist.push_back(std::make_pair(module->entry_computation(),
                                    /*is_thread_local*/ false));

  // Sets for quickly checking membership. Computations are returned in vectors
  // for stable iteration.
  FlatSet<const HloComputation*> thread_local_set;
  FlatSet<const HloComputation*> global_set;

  while (!worklist.empty()) {
    auto worklist_front = worklist.front();
    worklist.pop_front();
    const HloComputation* computation = worklist_front.first;
    bool is_thread_local = worklist_front.second;
    bool in_thread_local_set = thread_local_set.count(computation) > 0;
    bool in_global_set = global_set.count(computation) > 0;

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
          computation->name().c_str());
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
            // Call and while must be called from a computation with global
            // allocations as they may return references to buffers inside the
            // called computation which cannot be thread-local.
            if (is_thread_local) {
              return InvalidArgument(
                  "computation %s cannot contain call/while op because it "
                  "requires thread-local buffer allocations",
                  computation->name().c_str());
            }
            worklist.push_back(std::make_pair(subcomputation,
                                              false));  // Not thread local.
            break;
          case HloOpcode::kCrossReplicaSum:
          case HloOpcode::kMap:
          case HloOpcode::kReduce:
          case HloOpcode::kReduceWindow:
          case HloOpcode::kSelectAndScatter:
          case HloOpcode::kFusion:
            // Map/reduce etc computations are always thread-local.
            worklist.push_back(std::make_pair(subcomputation,
                                              true));  // Thread local.
            break;
          default:
            return InternalError(
                "Unexpected calling opcode: %s",
                HloOpcodeString(instruction->opcode()).c_str());
        }
      }
    }
  }

  // Add the computations to the vectors in post order.
  for (auto* computation : module->MakeComputationPostOrder()) {
    if (thread_local_set.count(computation) > 0) {
      thread_local_computations->push_back(computation);
    } else if (global_set.count(computation) > 0) {
      global_computations->push_back(computation);
    }
    // If the computation is not reachable from the entry computation, then it
    // will not appear in either thread_local_set or global_set. We don't bother
    // assigning buffers for these.
  }
  return Status::OK();
}

// Checks that points-to set of 'instruction' is unambiguous and distinct
// (ensured by CopyInsertion), then adds the buffer from the points-to set at
// 'index' to 'colocated_set'.
const LogicalBuffer* AddBufferToColocatedSet(
    const HloInstruction* instruction, const ShapeIndex& index,
    const TuplePointsToAnalysis& points_to_analysis,
    std::vector<const LogicalBuffer*>* colocated_set) {
  // CopyInsertion ensures root points-to set is unambiguous and distinct.
  const auto& points_to = points_to_analysis.GetPointsToSet(instruction);
  DCHECK(!points_to.IsAmbiguous());
  colocated_set->push_back(points_to.element(index)[0]);
  return colocated_set->back();
}

// Given the interference map of a graph (the list of interfering node indices
// for each node), perform graph coloring such that interfering nodes are
// assigned to different colors. Returns the assigned color of the nodes, where
// the colors are represented as integer values [0, color_count).
std::vector<int64> ColorInterferenceGraph(
    const std::vector<std::vector<int64>>& interference_map) {
  const int64 node_count = interference_map.size();

  // Sort the nodes such that we assign nodes with more interference first. This
  // relies on the common heuristic of assigning the most constrained node
  // first, but it would be good to investigate other ordering heuristics too.
  std::vector<int64> nodes(node_count);
  std::iota(nodes.begin(), nodes.end(), 0);
  std::sort(nodes.begin(), nodes.end(),
            [&interference_map](const int64 i, const int64 j) {
              return interference_map[i].size() > interference_map[j].size();
            });

  const int64 kColorUnassigned = -1;
  std::vector<int64> assigned_colors(node_count, kColorUnassigned);
  for (int64 node : nodes) {
    // Mark the colors that are already assigned to the neighbors.
    std::vector<bool> available_colors(node_count, true);
    for (int64 neighbor : interference_map[node]) {
      int64 color = assigned_colors[neighbor];
      if (color != kColorUnassigned) {
        available_colors[color] = false;
      }
    }

    // Find the color that is not yet assigned to the neighbors.
    int64 color = kColorUnassigned;
    for (color = 0; color < available_colors.size(); ++color) {
      if (available_colors[color]) {
        break;
      }
    }
    CHECK_NE(color, kColorUnassigned);
    assigned_colors[node] = color;
  }
  return assigned_colors;
}

}  // namespace

size_t BufferAllocation::Slice::Hasher::operator()(Slice s) const {
  uint64 h = std::hash<int64>()(s.index());
  h = tensorflow::Hash64Combine(h, std::hash<int64>()(s.offset()));
  h = tensorflow::Hash64Combine(h, std::hash<int64>()(s.size()));
  return h;
}

string BufferAllocation::Slice::ToString() const {
  return tensorflow::strings::StrCat("{index:", index(), ", offset:", offset_,
                                     ", size:", size_, "}");
}

BufferAllocation::Slice BufferAllocation::GetSlice(
    const LogicalBuffer& buffer) const {
  const OffsetSize os = FindOrDie(assigned_buffers_, &buffer);
  return Slice(this, os.offset, os.size);
}

void BufferAllocation::AddAssignment(const LogicalBuffer& buffer, int64 offset,
                                     int64 size) {
  VLOG(4) << "Trying to add " << buffer << " to " << this;
  CHECK(assigned_buffers_.count(&buffer) == 0)
      << "LogicalBuffer " << buffer << " already assigned to allocation "
      << index_;
  CHECK_LE(offset, size_) << "LogicalBuffer " << buffer
                          << " offset out of range";
  CHECK_LE(offset + size, size_)
      << "LogicalBuffer " << buffer
      << " size out of range at offset: " << offset << " with size: " << size;
  CHECK_EQ(buffer.color(), color())
      << "Buffer color " << buffer.color() << " for buffer " << buffer
      << " does not match allocation color " << color() << ".";
  OffsetSize offset_size;
  offset_size.offset = offset;
  offset_size.size = size;
  assigned_buffers_.emplace(&buffer, offset_size);
}

BufferAllocationProto BufferAllocation::ToProto() const {
  BufferAllocationProto proto;
  proto.set_index(index_);
  proto.set_size(size_);
  proto.set_is_thread_local(is_thread_local_);
  proto.set_is_tuple(is_tuple_);
  proto.set_color(color_.value());
  if (is_entry_computation_parameter_) {
    proto.set_is_entry_computation_parameter(true);
    for (int64 idx : param_shape_index()) {
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
  std::sort(proto.mutable_assigned()->begin(), proto.mutable_assigned()->end(),
            [](const BufferAllocationProto::Assigned& assign1,
               const BufferAllocationProto::Assigned& assign2) {
              return assign1.logical_buffer_id() < assign2.logical_buffer_id();
            });
  return proto;
}

string BufferAllocation::ToString() const {
  string output;
  Appendf(&output, "allocation %lld: %p, size %lld", index_, this, size());
  if (color().value() != 0) {
    StrAppend(&output, ", color ", color().value());
  }
  if (is_entry_computation_parameter()) {
    StrAppend(&output, ", parameter ", parameter_number(), " at ShapeIndex ",
              param_shape_index().ToString());
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
  // Dump the assigned buffers ordered by id.
  std::vector<const LogicalBuffer*> sorted_buffers;
  for (const auto& buffer_offset_size : assigned_buffers_) {
    sorted_buffers.push_back(buffer_offset_size.first);
  }
  std::sort(sorted_buffers.begin(), sorted_buffers.end(),
            [](const LogicalBuffer* a, const LogicalBuffer* b) {
              return a->id() < b->id();
            });
  for (const LogicalBuffer* buffer : sorted_buffers) {
    const OffsetSize& offset_size = FindOrDie(assigned_buffers_, buffer);
    StrAppend(&output,
              tensorflow::strings::Printf(
                  "  %s [%lld,%lld]: %s\n", buffer->ToString().c_str(),
                  offset_size.offset, offset_size.size,
                  ShapeUtil::HumanStringWithLayout(buffer->shape()).c_str()));
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

const PointsToSet& BufferAssignment::GetPointsToSet(
    const HloInstruction* instruction) const {
  return points_to_analysis().GetPointsToSet(instruction);
}

bool BufferAssignment::HasAllocation(const LogicalBuffer& buffer) const {
  TF_CHECK_OK(points_to_analysis().VerifyBuffer(buffer));
  return allocation_index_for_buffer_.count(&buffer) > 0;
}

const BufferAllocation& BufferAssignment::GetAssignedAllocation(
    const LogicalBuffer& buffer) const {
  CHECK(HasAllocation(buffer));
  return GetAllocation(allocation_index_for_buffer_.at(&buffer));
}

BufferAllocation* BufferAssignment::GetMutableAssignedAllocation(
    const LogicalBuffer& buffer) {
  return const_cast<BufferAllocation*>(&GetAssignedAllocation(buffer));
}

std::set<BufferAllocation::Slice> BufferAssignment::GetAllSlices(
    const HloInstruction* instruction, const ShapeIndex& index) const {
  std::set<BufferAllocation::Slice> result;
  for (const LogicalBuffer* buffer : GetSourceBuffers(instruction, index)) {
    if (HasAllocation(*buffer)) {
      result.insert(GetAssignedAllocation(*buffer).GetSlice(*buffer));
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

BufferAllocation* BufferAssignment::GetMutableAllocation(
    BufferAllocation::Index index) {
  return const_cast<BufferAllocation*>(&GetAllocation(index));
}

bool BufferAssignment::HasAllocationAt(const HloInstruction* instruction,
                                       const ShapeIndex& index) const {
  for (const LogicalBuffer* buffer :
       GetPointsToSet(instruction).element(index)) {
    if (allocation_index_for_buffer_.count(buffer) > 0) {
      return true;
    }
  }
  return false;
}

bool BufferAssignment::HasTopLevelAllocation(
    const HloInstruction* instruction) const {
  return HasAllocationAt(instruction, /*index=*/{});
}

StatusOr<BufferAllocation::Slice> BufferAssignment::GetUniqueSlice(
    const HloInstruction* instruction, const ShapeIndex& index) const {
  VLOG(3) << "Trying to find unique slice for " << instruction->name() << " ["
          << index << "]";
  BufferAllocation::Slice result;
  for (const LogicalBuffer* buffer :
       GetPointsToSet(instruction).element(index)) {
    VLOG(3) << "Examining buffer " << *buffer;
    if (HasAllocation(*buffer)) {
      VLOG(3) << "Has allocation";
      const BufferAllocation::Slice slice =
          GetAssignedAllocation(*buffer).GetSlice(*buffer);
      if (result.allocation() == nullptr) {
        result = slice;
      } else if (result != slice) {
        return FailedPrecondition(
            "BufferAllocation::Slice for instruction %s at index %s cannot "
            "be determined at compile-time.",
            instruction->name().c_str(), index.ToString().c_str());
      }
    } else {
      VLOG(3) << "No allocation";
    }
  }
  if (result.allocation() == nullptr) {
    return FailedPrecondition(
        "BufferAllocation::Slice not assigned for instruction %s at index %s",
        instruction->name().c_str(), index.ToString().c_str());
  }
  return result;
}

StatusOr<BufferAllocation::Slice> BufferAssignment::GetUniqueTopLevelSlice(
    const HloInstruction* instruction) const {
  return GetUniqueSlice(instruction, /*index=*/{});
}

bool BufferAssignment::SharesSliceAtIndex(
    const HloInstruction* hlo_a, const ShapeIndex& shape_index_a,
    const HloInstruction* hlo_b, const ShapeIndex& shape_index_b) const {
  return GetUniqueSlice(hlo_a, shape_index_a).ConsumeValueOrDie() ==
         GetUniqueSlice(hlo_b, shape_index_b).ConsumeValueOrDie();
}

bool BufferAssignment::HaveDisjointSlices(const HloInstruction* hlo_a,
                                          const HloInstruction* hlo_b) const {
  using SliceSet =
      FlatSet<BufferAllocation::Slice, BufferAllocation::Slice::Hasher>;
  // Gets the slices all of instr's subshapes.  If any subshape doesn't have an
  // assigned slice, returns the empty set.
  auto collect_slices = [&](const HloInstruction* instr) -> SliceSet {
    SliceSet slices;
    Status status = ShapeUtil::ForEachSubshapeWithStatus(
        instr->shape(),
        [&](const Shape& /*subshape*/, const ShapeIndex& index) {
          auto shape_slices = GetAllSlices(instr, index);
          if (shape_slices.empty()) {
            return InvalidArgument("No slices assigned to part of instr.");
          }
          slices.insert(shape_slices.begin(), shape_slices.end());
          return Status::OK();
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
         std::none_of(slices_a.begin(), slices_a.end(),
                      [&](const BufferAllocation::Slice& slice) {
                        return slices_b.count(slice) > 0;
                      });
}

StatusOr<BufferAllocation::Slice>
BufferAssignment::GetUniqueTopLevelOutputSlice() const {
  return GetUniqueTopLevelSlice(
      module_->entry_computation()->root_instruction());
}

BufferAllocation* BufferAssignment::NewEmptyAllocation(
    int64 size, LogicalBuffer::Color color) {
  BufferAllocation::Index index = allocations_.size();
  allocations_.emplace_back(index, size, color);
  BufferAllocation* allocation = &allocations_.back();
  return allocation;
}

BufferAllocation* BufferAssignment::NewAllocation(const LogicalBuffer& buffer,
                                                  int64 size) {
  BufferAllocation* allocation = NewEmptyAllocation(size, buffer.color());
  AddAssignment(allocation, buffer, /*offset=*/0, size);
  allocation->peak_buffers_.push_back(&buffer);
  return allocation;
}

// Adds an instruction to the set assigned to the given buffer.
void BufferAssignment::AddAssignment(BufferAllocation* allocation,
                                     const LogicalBuffer& buffer, int64 offset,
                                     int64 size) {
  CHECK_EQ(0, allocation_index_for_buffer_.count(&buffer))
      << "LogicalBuffer " << buffer << " already has an allocation.";
  CHECK(allocation->is_reusable() || allocation->assigned_buffers().empty())
      << "Non-reusable allocation already assigned a buffer: "
      << allocation->ToString();

  TF_CHECK_OK(points_to_analysis().VerifyBuffer(buffer));

  allocation->AddAssignment(buffer, offset, size);
  allocation_index_for_buffer_[&buffer] = allocation->index();
}

// Combines allocations of temporary buffers of the same color into one big
// BufferAllocation.
void BufferAssignment::CombineTempAllocations() {
  VLOG(1) << "CombineTempAllocations()";
  FlatMap<LogicalBuffer::Color, BufferAllocation, LogicalBuffer::Color::Hasher>
      combined_allocation_map;

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
      const BufferAllocation& temp_allocation = *it;
      LogicalBuffer::Color color = temp_allocation.color();
      auto combined_it = combined_allocation_map.find(color);
      if (combined_it == combined_allocation_map.end()) {
        // We have found the first temp allocation of this color. Collect
        // the other temp allocations of the same color into it.
        VLOG(1) << "Combined temp allocation for color " << color
                << " is: " << temp_allocation;
        combined_allocation_map.emplace(color, temp_allocation);
        continue;
      }

      auto* combined_allocation = &combined_it->second;
      VLOG(1) << "Combined allocation absorbing temp allocation: "
              << temp_allocation;

      // Each temp allocation is placed end-to-end, accounting for alignment.
      // The offset of each buffer in the combined allocation is computed from
      // the base offset of the allocation.
      int64 alignment = color_alignment_(color);
      const int64 base =
          RoundUpToNearest(combined_allocation->size(), alignment);
      combined_allocation->set_size(base + temp_allocation.size());
      for (const auto& buffer_offset_size : temp_allocation.assigned_buffers_) {
        const LogicalBuffer* buffer = buffer_offset_size.first;
        const int64 offset = buffer_offset_size.second.offset;
        const int64 size = buffer_offset_size.second.size;
        combined_allocation->AddAssignment(*buffer, base + offset, size);
      }
      if (!temp_allocation.HeapTraces().empty()) {
        CHECK_EQ(temp_allocation.HeapTraces().size(), 1);
        combined_allocation->AddHeapTrace(temp_allocation.HeapTraces().front());
      }
      combined_allocation->peak_buffers_.insert(
          combined_allocation->peak_buffers_.end(),
          temp_allocation.peak_buffers_.begin(),
          temp_allocation.peak_buffers_.end());
    }
    // Replace all existing temporary allocations with the new combined
    // allocations.
    allocations_.erase(first_temp_it, allocations_.end());
    for (auto& combined : combined_allocation_map) {
      allocations_.push_back(combined.second);
      temp_allocation_total_size_ += combined.second.size();
    }
  }

  // Update allocation indices to their new positions.
  allocation_index_for_buffer_.clear_no_resize();
  for (size_t index = 0; index < allocations_.size(); ++index) {
    BufferAllocation* allocation = &allocations_[index];
    allocation->set_index(index);
    for (const auto& buffer_offset_size : allocation->assigned_buffers_) {
      const LogicalBuffer* buffer = buffer_offset_size.first;
      allocation_index_for_buffer_[buffer] = index;
    }
  }
}

Status BufferAssignment::ComputeSummaryStats() {
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

  // Only compute total fragmentation if all computations are sequential.
  SequentialHloOrdering::HloModuleSequence module_sequence;
  for (const auto& computation : module_->computations()) {
    const std::vector<const HloInstruction*>* sequence =
        liveness_->hlo_ordering().SequentialOrder(*computation);
    if (sequence != nullptr) {
      module_sequence.emplace(computation, *sequence);
    }
  }
  if (module_sequence.size() == module_->computation_count()) {
    TF_ASSIGN_OR_RETURN(
        const int64 min_size,
        HeapSimulator::MinimumMemoryForModule(module_sequence, buffer_size_));
    stats_.total_fragmentation_bytes = stats_.total_allocation_bytes - min_size;
  }

  return Status::OK();
}

string BufferAssignment::Stats::ToString() const {
  string s;
  Appendf(&s, "BufferAssignment stats:\n");
  Appendf(&s, "             parameter allocation: %10s\n",
          HumanReadableNumBytes(parameter_allocation_bytes).c_str());
  Appendf(&s, "              constant allocation: %10s\n",
          HumanReadableNumBytes(constant_allocation_bytes).c_str());
  Appendf(&s, "        maybe_live_out allocation: %10s\n",
          HumanReadableNumBytes(maybe_live_out_allocation_bytes).c_str());
  Appendf(&s, "     preallocated temp allocation: %10s\n",
          HumanReadableNumBytes(preallocated_temp_allocation_bytes).c_str());
  if (preallocated_temp_fragmentation_bytes >= 0) {
    const double percent = 100. * preallocated_temp_fragmentation_bytes /
                           preallocated_temp_allocation_bytes;
    Appendf(
        &s, "  preallocated temp fragmentation: %10s (%.2f%%)\n",
        HumanReadableNumBytes(preallocated_temp_fragmentation_bytes).c_str(),
        percent);
  }
  Appendf(&s, "                 total allocation: %10s\n",
          HumanReadableNumBytes(total_allocation_bytes).c_str());
  if (total_fragmentation_bytes >= 0) {
    const double percent =
        100. * total_fragmentation_bytes / total_allocation_bytes;
    Appendf(&s, "              total fragmentation: %10s (%.2f%%)\n",
            HumanReadableNumBytes(total_fragmentation_bytes).c_str(), percent);
  }
  return s;
}

string BufferAssignment::ToString() const {
  string output;
  tensorflow::strings::StrAppend(&output, "BufferAssignment:\n");
  for (auto& allocation : allocations_) {
    tensorflow::strings::StrAppend(&output, allocation.ToString());
  }
  return output;
}

BufferAssignmentProto BufferAssignment::ToProto() const {
  BufferAssignmentProto proto;
  // NOTE: TuplePointsToAnalysis state is serialized here in BufferAssigment,
  // because we need to do the HasAllocation check for each buffer. Otherwise
  // the buffer_size_ call might fail for some backends.
  const TuplePointsToAnalysis& points_to_analysis =
      liveness_->points_to_analysis();
  for (LogicalBuffer::Id id = 0; id < points_to_analysis.num_logical_buffers();
       id++) {
    auto& buffer = points_to_analysis.logical_buffer(id);
    if (HasAllocation(buffer)) {
      LogicalBufferProto proto_buffer = buffer.ToProto(buffer_size_);
      proto.add_logical_buffers()->Swap(&proto_buffer);

      // Fill buffer aliases.
      for (const BufferAlias& alias :
           points_to_analysis.GetBufferAliases(buffer)) {
        if (alias.instruction() == buffer.instruction() &&
            alias.index() == buffer.index()) {
          continue;  // skip self-aliases
        }
        BufferAssignmentProto::BufferAlias* proto_alias =
            proto.add_buffer_aliases();
        LogicalBufferProto::Location proto_alias_location =
            BufferValue::ToLocationProto(*alias.instruction(), alias.index());
        proto_alias->set_source_buffer_id(buffer.id());
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
StatusOr<std::unique_ptr<BufferAssignment>> BufferAssigner::Run(
    const HloModule* module, std::unique_ptr<HloOrdering> hlo_ordering,
    LogicalBuffer::SizeFunction buffer_size,
    LogicalBuffer::AlignmentFunction color_alignment,
    bool allow_input_output_aliasing, bool allocate_buffers_for_constants,
    BufferLiveness::Colorer colorer) {
  BufferAssigner assigner(allow_input_output_aliasing,
                          allocate_buffers_for_constants, std::move(colorer));
  return assigner.CreateAssignment(module, std::move(hlo_ordering),
                                   std::move(buffer_size),
                                   std::move(color_alignment));
}

bool BufferAssigner::MaybeAssignBuffer(BufferAllocation* allocation,
                                       const LogicalBuffer& buffer,
                                       BufferAssignment* assignment) {
  const LogicalBuffer::SizeFunction& buffer_size = assignment->buffer_size_;

  CHECK(!assignment->HasAllocation(buffer))
      << "buffer " << buffer << " already has an allocation assigned.";

  VLOG(4) << "Trying to assign " << buffer << " to allocation: " << *allocation;

  if (buffer.color() != allocation->color()) {
    VLOG(4) << "Can't assign: buffer has color" << buffer.color()
            << " and allocation has color " << allocation->color() << ".";
    return false;
  }

  if (buffer_size(buffer) > allocation->size()) {
    VLOG(4) << "Can't assign: buffer is larger than allocation ("
            << buffer_size(buffer) << " > " << allocation->size() << ")";
    return false;
  }

  if (allocation->is_readonly()) {
    VLOG(4) << "Can't assign: allocation is readonly";
    return false;
  }

  if (!allocation->is_reusable()) {
    VLOG(4) << "Can't assign: allocation is not reusable";
    return false;
  }

  for (const auto& buffer_offset_size : allocation->assigned_buffers()) {
    const LogicalBuffer& assigned_buffer = *buffer_offset_size.first;
    if (assignment->liveness().MayInterfere(assigned_buffer, buffer)) {
      VLOG(4) << "Can't assign: assignee " << assigned_buffer
              << " may interfere with " << buffer;
      return false;
    }
    // Copy instruction don't share a buffer with their input operand.
    if (buffer.instruction()->IsUserOf(assigned_buffer.instruction()) &&
        buffer.instruction()->opcode() == HloOpcode::kCopy) {
      VLOG(4) << "Can't assign: assignee " << assigned_buffer
              << " is used at copy instruction " << buffer;
      return false;
    }
  }

  if (allow_input_output_aliasing_ && allocation->maybe_live_out()) {
    const HloComputation* entry_computation =
        assignment->module_->entry_computation();
    for (auto param : entry_computation->parameter_instructions()) {
      for (auto& param_buffer :
           assignment->points_to_analysis().GetBuffersDefinedByInstruction(
               param)) {
        if (assignment->liveness().MayInterfere(*param_buffer, buffer)) {
          VLOG(4) << "Can't assign: Parameter interference with result";
          return false;
        }
      }
    }
  }

  // If the buffer is live out of the computation then it should only be
  // assigned a buffer which exactly fits the result to avoid wasting memory
  // (result buffers can have arbitrary lifetimes).
  if (assignment->liveness().MaybeLiveOut(buffer) &&
      allocation->size() != buffer_size(buffer)) {
    VLOG(4) << "Can't assign: buffer " << buffer
            << "is live out and size not the same as allocation";
    return false;
  }

  assignment->AddAssignment(allocation, buffer, /*offset=*/0,
                            buffer_size(buffer));
  return true;
}

Status BufferAssigner::AssignBuffersForComputation(
    const HloComputation* computation, const DebugOptions& debug_options,
    bool is_thread_local,
    const FlatSet<const LogicalBuffer*>& colocated_buffers,
    const FlatSet<BufferAllocation::Index>& colocated_allocations,
    FlatMap<const HloComputation*, FlatSet<const LogicalBuffer*>>*
        buffers_to_assign_sequentially,
    BufferAssignment* assignment) {
  // Buffers are sorted and assigned to BufferAllocations in decreasing order of
  // size.
  std::vector<const LogicalBuffer*> sorted_buffers;
  for (auto* instruction : computation->instructions()) {
    // Add all buffers which this instruction defines. Instruction which don't
    // define buffers (eg, bitcast which just forwards a pointer) don't need
    // any allocations.
    for (const LogicalBuffer* buffer :
         assignment->points_to_analysis().GetBuffersDefinedByInstruction(
             instruction)) {
      sorted_buffers.push_back(buffer);
    }
  }

  // Generate a post order sort of instructions for sorting of the
  // LogicalBuffers.
  FlatMap<const HloInstruction*, int> post_order_position;
  int position = 0;
  for (auto* instruction : computation->MakeInstructionPostOrder()) {
    post_order_position.emplace(instruction, position);
    position++;
  }

  // If there is a sequential instruction ordering, we'll delay assignment of
  // temp buffers until after the main assignment loop.
  const BufferLiveness& liveness = assignment->liveness();
  const bool has_sequential_order =
      liveness.hlo_ordering().SequentialOrder(*computation) != nullptr;
  if (has_sequential_order && buffers_to_assign_sequentially != nullptr) {
    // Every sequential computation must get an entry in the
    // buffers_to_assign_sequentially map, even if we end up with an empty set
    // of buffers. This ensures we can correctly determine whether to run
    // whole-module heap simulation.
    buffers_to_assign_sequentially->emplace(computation,
                                            FlatSet<const LogicalBuffer*>());
  }

  // Sort the LogicalBuffers first by size. We assign the larger LogicalBuffers
  // first for simplicity. This means any previously created BufferAllocation is
  // necessarily large enough to hold the output of the current Buffer in
  // consideration.
  //
  // As a secondary sorting criteria, if the instructions are sequentially
  // ordered, we assign live-out buffers before others. Note that for sequential
  // computations, we'll take temp buffers that can't re-use any allocations and
  // assign them via a heap scheduler. By assigning live-out buffers first, we
  // increase the odds that temp buffers can re-use an allocation.
  //
  // As a final tiebreaker use post order position of the HLO instruction which
  // defines the buffer. This means an instruction will appear after its
  // operands (assuming operands are the same/larger size) enabling the
  // important reuse case where an elementwise instruction reuses one of its
  // operand's buffer. This improves locality.
  std::sort(sorted_buffers.begin(), sorted_buffers.end(),
            [this, has_sequential_order, &liveness, &post_order_position,
             assignment](const LogicalBuffer* a, const LogicalBuffer* b) {
              // Primary sort is by decreasing buffer size.
              const int64 a_size = assignment->buffer_size_(*a);
              const int64 b_size = assignment->buffer_size_(*b);
              if (a_size != b_size) {
                return a_size > b_size;  // use ">" for decreasing size.
              }
              // Otherwise live out buffers come before others, if the
              // instructions are sequentially ordered.
              if (has_sequential_order) {
                const bool a_live_out = liveness.MaybeLiveOut(*a);
                const bool b_live_out = liveness.MaybeLiveOut(*b);
                if (a_live_out != b_live_out) {
                  return a_live_out;
                }
              }
              // Final tiebreaker is in instruction post order.
              return post_order_position.at(a->instruction()) <
                     post_order_position.at(b->instruction());
            });

  // BufferAllocations are necessarily created in decreasing size order. Keep
  // indices of previously created BufferAllocations in allocation_indices.
  std::vector<BufferAllocation::Index> allocation_indices;
  for (const LogicalBuffer* buffer : sorted_buffers) {
    VLOG(3) << "Assigning allocation to: " << *buffer;
    if (colocated_buffers.count(buffer) > 0) {
      // Colocated buffers are currently assigned in an earlier pass.
      VLOG(3) << "Skipping colocated buffer: " << *buffer;
      continue;
    }

    TF_RET_CHECK(!assignment->HasAllocation(*buffer));

    const HloInstruction* instruction = buffer->instruction();
    const int64 buffer_size = assignment->buffer_size_(*buffer);

    if (instruction->opcode() == HloOpcode::kConstant) {
      if (allocate_buffers_for_constants_) {
        BufferAllocation* allocation =
            assignment->NewAllocation(*buffer, buffer_size);
        allocation->set_constant(true);
        VLOG(3) << "New allocation #" << allocation->index() << " for constant "
                << *buffer;
      }
      continue;
    }

    const bool is_entry_parameter =
        instruction->opcode() == HloOpcode::kParameter &&
        computation == computation->parent()->entry_computation();
    if (is_entry_parameter) {
      // If the LogicalBuffer is part of an external parameter, creates a new
      // allocation and sets its parameter number. Parameters of non-entry
      // computations do not need special allocations because they live inside
      // callers.
      BufferAllocation* allocation =
          assignment->NewAllocation(*buffer, buffer_size);
      allocation->set_entry_computation_parameter(
          instruction->parameter_number(), buffer->index());
      VLOG(3) << "New allocation #" << allocation->index()
              << " for entry computation parameter: " << *buffer;
      continue;
    }

    if (is_thread_local) {
      BufferAllocation* allocation =
          assignment->NewAllocation(*buffer, buffer_size);
      allocation->set_is_thread_local(true);
      VLOG(3) << "New allocation #" << allocation->index()
              << " for thread-local: " << *buffer;
      continue;
    }

    if (ShapeUtil::IsTuple(buffer->shape())) {
      BufferAllocation* allocation =
          assignment->NewAllocation(*buffer, buffer_size);
      allocation->set_is_tuple(true);
      VLOG(3) << "New allocation #" << allocation->index()
              << " for tuple-shaped buffer: " << *buffer;
      continue;
    }

    // First try to assign a LogicalBuffer to one of its operand allocations to
    // improve locality. This is only possible with elementwise operations
    // (checked in liveness analysis) which are necessarily top-level
    // array-shaped buffers.
    if (buffer->IsTopLevel() && !buffer->IsTuple()) {
      for (auto* operand : instruction->operands()) {
        bool assigned_operand = false;
        for (const auto& operand_slice :
             assignment->GetAllSlices(operand, /*index=*/{})) {
          BufferAllocation* allocation =
              assignment->GetMutableAllocation(operand_slice.index());
          if (colocated_allocations.count(allocation->index()) == 0) {
            // TODO(b/32491382) Colocated buffers are currently assigned in an
            // earlier pass, and so can break the "increasing allocation size"
            // invariant in this function (causing this CHECK to fail). However,
            // the call to MaybeAssignBuffer is safe as it returns false if
            // allocation.size < buffer.size.
            CHECK_GE(allocation->size(), buffer_size);
          }
          if (MaybeAssignBuffer(allocation, *buffer, assignment)) {
            VLOG(3) << "Reusing (operand) allocation #" << allocation->index()
                    << " for: " << *buffer;
            assigned_operand = true;
            break;
          }
        }
        if (assigned_operand) {
          break;
        }
      }
    }

    if (!assignment->HasAllocation(*buffer)) {
      // Find the smallest buffer which can be reused iterating from end of
      // allocation_indices (smallest) to beginning (largest).
      for (int allocation_index = allocation_indices.size() - 1;
           allocation_index >= 0; allocation_index--) {
        BufferAllocation* allocation = assignment->GetMutableAllocation(
            allocation_indices[allocation_index]);
        // Instructions are iterated in increasing buffer size, so any
        // previously create allocation must be large enough to hold this
        // instruction's output (with the exception of colocated buffers).
        if (colocated_allocations.count(allocation->index()) == 0) {
          // TODO(b/32491382) Colocated buffers are currently assigned in an
          // earlier pass, and so can break the "increasing allocation size"
          // invariant in this function (causing this CHECK to fail). However,
          // the call to MaybeAssignBuffer is safe as it returns false if
          // allocation.size < buffer.size.
          CHECK_GE(allocation->size(), buffer_size);
        }

        if (MaybeAssignBuffer(allocation, *buffer, assignment)) {
          VLOG(3) << "Reusing allocation #" << allocation->index()
                  << " for: " << *buffer;
          break;
        }
      }
    }

    if (!assignment->HasAllocation(*buffer) && has_sequential_order &&
        !liveness.MaybeLiveOut(*buffer)) {
      // There is a sequential instruction ordering, so we delay assignment of
      // temp buffers until after the loop. We do this right before we decide to
      // create a new allocation, to ensure we've exhausted all the buffer
      // re-use cases above.
      //
      // Entry parameters and thread local buffers were already handled earlier
      // in this loop iteration.  See BufferAllocation::IsPreallocatedTempBuffer
      // for the definition of temp buffers.
      CHECK(!is_entry_parameter) << *buffer;
      CHECK(!is_thread_local) << *buffer;
      (*buffers_to_assign_sequentially)[computation].insert(buffer);
      VLOG(3) << "Delaying assignment of temp buffer: " << *buffer;
      continue;
    }

    if (!assignment->HasAllocation(*buffer)) {
      BufferAllocation* allocation =
          assignment->NewAllocation(*buffer, buffer_size);
      allocation_indices.push_back(allocation->index());
      VLOG(3) << "New allocation #" << allocation->index()
              << " for: " << *buffer;
    }
  }

  return Status::OK();
}

FlatMap<LogicalBuffer::Color, FlatSet<const LogicalBuffer*>,
        LogicalBuffer::Color::Hasher>
BufferAssigner::SplitBuffersByColor(
    const FlatSet<const LogicalBuffer*>& buffers) {
  FlatMap<LogicalBuffer::Color, FlatSet<const LogicalBuffer*>,
          LogicalBuffer::Color::Hasher>
      color_map;
  for (auto buffer : buffers) {
    color_map[buffer->color()].insert(buffer);
  }
  return color_map;
}

Status BufferAssigner::AssignBuffersWithSequentialOrdering(
    const FlatMap<const HloComputation*, FlatSet<const LogicalBuffer*>>&
        buffers_to_assign_sequentially,
    bool run_whole_module_heap_simulation, BufferAssignment* assignment) {
  // Run the sequence of instructions through the heap simulator.  The heuristic
  // that seems to give the best results is lazy-best-fit, with all runs of
  // alloc / free calls sorted in decreasing size order.
  const HloOrdering& hlo_ordering = assignment->liveness().hlo_ordering();
  if (run_whole_module_heap_simulation) {
    // Run the heap simulation over the whole module. This reduces memory usage,
    // since buffers for kCall, kWhile, and kConditional sub-computations are
    // only live for the duration of their calling instructions.
    VLOG(1) << "Running whole-module heap simulation";
    SequentialHloOrdering::HloModuleSequence module_sequence;
    FlatSet<const LogicalBuffer*> all_buffers_to_assign;
    for (const auto& pair : buffers_to_assign_sequentially) {
      const HloComputation* computation = pair.first;
      const FlatSet<const LogicalBuffer*>& buffers_to_assign = pair.second;
      const std::vector<const HloInstruction*>* instruction_sequence =
          hlo_ordering.SequentialOrder(*computation);
      CHECK(instruction_sequence != nullptr) << computation->name();
      module_sequence[computation] = *instruction_sequence;
      all_buffers_to_assign.insert(buffers_to_assign.begin(),
                                   buffers_to_assign.end());
    }
    auto color_map = SplitBuffersByColor(all_buffers_to_assign);
    for (auto& single_colored_set : color_map) {
      auto color = single_colored_set.first;
      VLOG(2) << "Simulating heap for color " << color;
      int64 alignment = assignment->color_alignment_(color);
      HeapSimulator::Options options;
      options.alloc_constants = allocate_buffers_for_constants_;
      BufferValueFlatSet buffer_value_set =
          ToBufferValueFlatSet(single_colored_set.second);
      options.buffers_to_assign = &buffer_value_set;
      TF_ASSIGN_OR_RETURN(
          const HeapSimulator::Result result,
          HeapSimulator::Run(MakeUnique<DecreasingSizeRunsHeap>(
                                 MakeUnique<LazyBestFitHeap>(alignment)),
                             assignment->module(), module_sequence,
                             assignment->points_to_analysis(),
                             assignment->buffer_size_, options));
      AssignBuffersFromHeapSimulator(result, assignment,
                                     single_colored_set.first);
    }
  } else {
    // Run the heap-simulation on a per-computation basis. Buffers for
    // sub-computations are assigned disjoint BufferAllocations, assuming the
    // worst-case that they may all be live concurrently.
    VLOG(1) << "Running per-computation heap simulation";
    for (const auto& pair : buffers_to_assign_sequentially) {
      const HloComputation* computation = pair.first;
      const FlatSet<const LogicalBuffer*>& buffers_to_assign = pair.second;
      const std::vector<const HloInstruction*>* instruction_sequence =
          hlo_ordering.SequentialOrder(*computation);
      CHECK(instruction_sequence != nullptr) << computation->name();
      auto color_map = SplitBuffersByColor(buffers_to_assign);
      for (auto& single_colored_set : color_map) {
        auto color = single_colored_set.first;
        VLOG(2) << "Simulating heap for color " << color;
        int64 alignment = assignment->color_alignment_(color);
        HeapSimulator::Options options;
        BufferValueFlatSet buffer_value_set =
            ToBufferValueFlatSet(single_colored_set.second);
        options.buffers_to_assign = &buffer_value_set;
        TF_ASSIGN_OR_RETURN(
            const HeapSimulator::Result result,
            HeapSimulator::Run(MakeUnique<DecreasingSizeRunsHeap>(
                                   MakeUnique<LazyBestFitHeap>(alignment)),
                               *computation, *instruction_sequence,
                               assignment->points_to_analysis(),
                               assignment->buffer_size_, options));
        AssignBuffersFromHeapSimulator(result, assignment,
                                       single_colored_set.first);
      }
    }
  }
  return Status::OK();
}

namespace {

// Computes and returns the set of logical buffers live at the point of maximal
// liveness in the given heap trace. LogicalBuffers are (stabily) sorted by id.
std::vector<const LogicalBuffer*> ComputePeakMemoryLogicalBuffers(
    const BufferAllocation& allocation, const HeapSimulatorTrace& heap_trace) {
  // Create a map from LogicalBuffer::Id to LogicalBuffer* for the logical
  // buffers in this allocation.
  tensorflow::gtl::FlatMap<LogicalBuffer::Id, const LogicalBuffer*>
      id_to_buffer;
  tensorflow::gtl::FlatMap<const LogicalBuffer*, int64> buffer_sizes;
  for (const auto& pair : allocation.assigned_buffers()) {
    const LogicalBuffer* buffer = pair.first;
    const BufferAllocation::OffsetSize& offset_size = pair.second;
    id_to_buffer[buffer->id()] = buffer;
    buffer_sizes[buffer] = offset_size.size;
  }

  // Returns how much the given event increases the total size of live
  // buffers. Can be negative.
  auto memory_delta = [&id_to_buffer, &buffer_sizes](
                          const HeapSimulatorTrace::Event& event) -> int64 {
    const LogicalBuffer* buffer = id_to_buffer.at(event.buffer_id());
    const int64 buffer_size = buffer_sizes.at(buffer);
    if (event.kind() == HeapSimulatorTrace::Event::ALLOC) {
      return buffer_size;
    } else if (event.kind() == HeapSimulatorTrace::Event::SHARE_WITH) {
      // Sharing a buffer does not change the live set size for the purposes of
      // the heap simulator. Even though the shared-with buffer may be smaller,
      // the entire allocation remains live.
      return 0;
    } else if (event.kind() == HeapSimulatorTrace::Event::FREE) {
      return -1 * buffer_size;
    }
    LOG(FATAL) << "Unknown event kind: " << event.kind();
  };

  // First compute the size of the maximal live set.
  int64 max_live_size = 0;
  int64 live_size = 0;
  for (const auto& event : heap_trace.events()) {
    live_size += memory_delta(event);
    if (max_live_size < live_size) {
      max_live_size = live_size;
    }
  }

  // Next gather the set of logical buffers live at the earliest point of
  // maximal live set size.
  tensorflow::gtl::FlatSet<const LogicalBuffer*> live_buffers;
  live_size = 0;
  for (const auto& event : heap_trace.events()) {
    const LogicalBuffer* buffer = id_to_buffer.at(event.buffer_id());
    if (event.kind() == HeapSimulatorTrace::Event::ALLOC) {
      InsertOrDie(&live_buffers, buffer);
    } else if (event.kind() == HeapSimulatorTrace::Event::SHARE_WITH) {
      // Nothing to do.
    } else if (event.kind() == HeapSimulatorTrace::Event::FREE) {
      CHECK(ContainsKey(live_buffers, buffer));
      live_buffers.erase(buffer);
    }

    live_size += memory_delta(event);
    if (live_size == max_live_size) {
      break;
    }
  }
  CHECK_EQ(live_size, max_live_size);

  std::vector<const LogicalBuffer*> live_buffers_vector;
  live_buffers_vector.insert(live_buffers_vector.end(), live_buffers.begin(),
                             live_buffers.end());

  // Stabily sort the live buffers.
  std::sort(live_buffers_vector.begin(), live_buffers_vector.end(),
            [](const LogicalBuffer* a, const LogicalBuffer* b) {
              return a->id() < b->id();
            });
  return live_buffers_vector;
}

}  // namespace

void BufferAssigner::AssignBuffersFromHeapSimulator(
    const HeapSimulator::Result& result, BufferAssignment* assignment,
    LogicalBuffer::Color color) {
  if (assignment->stats_.preallocated_temp_fragmentation_bytes == -1) {
    assignment->stats_.preallocated_temp_fragmentation_bytes =
        result.fragmentation_size;
  } else {
    assignment->stats_.preallocated_temp_fragmentation_bytes +=
        result.fragmentation_size;
  }

  BufferAllocation* allocation =
      assignment->NewEmptyAllocation(result.heap_size, color);
  for (const auto& buffer_chunk : result.chunk_map) {
    // TODO(lauj) Remove this down_cast after downstream users of
    // BufferAllocation::assigned_buffers() are updated to use BufferValue.
    const LogicalBuffer& buffer =
        *CHECK_NOTNULL(dynamic_cast<const LogicalBuffer*>(buffer_chunk.first));
    const HeapSimulator::Chunk& chunk = buffer_chunk.second;
    assignment->AddAssignment(allocation, buffer, chunk.offset, chunk.size);
  }
  allocation->peak_buffers_ =
      ComputePeakMemoryLogicalBuffers(*allocation, result.debug_trace);

  VLOG(1) << "Ran heap simulation for allocation: " << allocation->ToString();
  allocation->AddHeapTrace(result.debug_trace);
}

// Adds the 'colocated_set' of buffers to 'colocated_buffer_sets', maintaining
// the invariant that all sets in 'colocated_buffer_sets' are disjoint.
//
// A practical example of when this is necessary is a chain of kCall ops:
//   computation.entry
//     %a = call() -> computation.1
//   computation.1
//     %b = call() -> computation.2
//   computation.2
//     %c = parameter()
// This yields the logical sets {%a,%b} {%b,%c} {%c}, which need to be merged
// into a single set {%a,%b,%c}
void BufferAssigner::AddSetToColocatedBufferSets(
    const std::vector<const LogicalBuffer*>& colocated_set,
    std::vector<ColocatedBufferSet>* colocated_buffer_sets) {
  if (colocated_set.empty()) {
    return;
  }
  VLOG(5) << ColocatedBufferSetsToString(colocated_set,
                                         "Adding colocated buffer set");
  // Find existing sets that overlap with at least one buffer from the
  // colocated_set. The resulting 'overlap_set_indices' will have at most
  // colocated_buffer_sets->size() entries, and will be in increasing order.
  std::vector<size_t> overlap_set_indices;
  for (size_t index = 0; index < colocated_buffer_sets->size(); ++index) {
    for (const LogicalBuffer* buffer : colocated_set) {
      if ((*colocated_buffer_sets)[index].count(buffer) > 0) {
        VLOG(5) << "Found overlap with existing set on buffer "
                << buffer->ToString() << "\n"
                << ColocatedBufferSetsToString((*colocated_buffer_sets)[index],
                                               "Overlapping set");
        overlap_set_indices.push_back(index);
        break;
      }
    }
  }

  // If there is no overlap with existing sets, create a new set.
  if (overlap_set_indices.empty()) {
    colocated_buffer_sets->emplace_back();
    colocated_buffer_sets->back().insert(colocated_set.begin(),
                                         colocated_set.end());
    VLOG(5) << "No overlap found, new group created";
    return;
  }

  // Merge all overlap sets and the colocated set into the first overlap set.
  ColocatedBufferSet* first = &(*colocated_buffer_sets)[overlap_set_indices[0]];
  for (size_t index = 1; index < overlap_set_indices.size(); ++index) {
    const ColocatedBufferSet& overlap_set =
        (*colocated_buffer_sets)[overlap_set_indices[index]];
    first->insert(overlap_set.begin(), overlap_set.end());
  }
  first->insert(colocated_set.begin(), colocated_set.end());
  VLOG(5) << ColocatedBufferSetsToString(
      *first, "Result of the colocated buffer set merging");

  // Remove overlap sets that we just merged. The offset accounts for the fact
  // that as elements are erased, the indices need to be adjusted. Keep in mind
  // that overlap_set_indices is in increasing order.
  for (size_t index = 1; index < overlap_set_indices.size(); ++index) {
    const size_t offset = overlap_set_indices[index] - index + 1;
    colocated_buffer_sets->erase(colocated_buffer_sets->begin() + offset);
  }
}

std::vector<BufferAssigner::ColocatedBufferSet>
BufferAssigner::MergeColocatedBufferSets(
    const std::vector<ColocatedBufferSet>& colocated_buffer_sets,
    const BufferLiveness& buffer_liveness,
    const LogicalBuffer::SizeFunction& buffer_size) {
  VLOG(1) << "colocation sets count before coalescing:"
          << colocated_buffer_sets.size();

  // Returns true if the given buffer is for the entry parameter.
  auto is_entry_parameter = [](const LogicalBuffer& buffer) {
    auto* instruction = buffer.instruction();
    auto* computation = instruction->parent();
    auto* module = computation->parent();
    return instruction->opcode() == HloOpcode::kParameter &&
           computation == module->entry_computation();
  };

  // Returns true if the two colocated buffer sets (specified by their indices
  // into the colocated_buffer_sets) can be merged into a single set.
  auto cannot_merge_buffer_sets = [&colocated_buffer_sets, &buffer_liveness,
                                   &buffer_size,
                                   &is_entry_parameter](int64 i, int64 j) {
    // Do not merge if one of the sets includes live outs or entry parameters.
    for (int64 key : {i, j}) {
      for (auto& buffer : colocated_buffer_sets[key]) {
        if (buffer_liveness.MaybeLiveOut(*buffer) ||
            is_entry_parameter(*buffer)) {
          return true;
        }
      }
    }

    // Colocated sets satisfy the invariant that all buffers within a set have
    // the same size. That means we need to check whether the size is the same
    // between the two sets, but also that it's enough to look at just one
    // buffer within each set.
    if (buffer_size(**colocated_buffer_sets[i].begin()) !=
        buffer_size(**colocated_buffer_sets[j].begin())) {
      return true;
    }

    // Do not merge if some pair of buffers interferes with each other.
    for (auto& buffer_a : colocated_buffer_sets[i]) {
      for (auto& buffer_b : colocated_buffer_sets[j]) {
        if (buffer_a->id() != buffer_b->id() &&
            buffer_liveness.MayInterfere(*buffer_a, *buffer_b)) {
          return true;
        }
      }
    }

    return false;
  };

  // Build the interference map among the colocated buffer sets (nodes), by
  // adding an edge between any two nodes that cannot be merged into a single
  // colocated buffer set.
  std::vector<std::vector<int64>> interference_map(
      colocated_buffer_sets.size());
  for (int64 i = 0; i < colocated_buffer_sets.size(); ++i) {
    for (int64 j = i + 1; j < colocated_buffer_sets.size(); ++j) {
      if (cannot_merge_buffer_sets(i, j)) {
        interference_map[i].push_back(j);
        interference_map[j].push_back(i);
      }
    }
  }

  // Assign a color to each colocation set in colocated_buffer_sets, such that
  // the sets that can be merged are assigned with the same color.
  auto assigned_colors = ColorInterferenceGraph(interference_map);

  // Merge the buffer sets with the same color.
  CHECK(!assigned_colors.empty());
  int64 num_sets =
      *std::max_element(assigned_colors.begin(), assigned_colors.end()) + 1;
  std::vector<ColocatedBufferSet> new_colocated_buffer_sets(num_sets);
  for (int64 i = 0; i < colocated_buffer_sets.size(); ++i) {
    const auto& buffer_set = colocated_buffer_sets[i];
    new_colocated_buffer_sets[assigned_colors[i]].insert(buffer_set.begin(),
                                                         buffer_set.end());
  }

  VLOG(1) << "colocation sets count after coalescing:"
          << colocated_buffer_sets.size();
  return new_colocated_buffer_sets;
}

// Builds sets of buffers in 'colocated_buffer_sets' which should be colocated
// in the same allocation (currently just supports kWhile, kCall, and
// kConditional).
void BufferAssigner::BuildColocatedBufferSets(
    const HloModule* module, const BufferLiveness& buffer_liveness,
    const LogicalBuffer::SizeFunction& buffer_size,
    std::vector<ColocatedBufferSet>* colocated_buffer_sets) {
  const TuplePointsToAnalysis& points_to_analysis =
      buffer_liveness.points_to_analysis();
  for (const HloComputation* computation : module->MakeComputationPostOrder()) {
    if (computation->IsFusionComputation()) {
      continue;
    }
    for (const HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      const HloOpcode opcode = instruction->opcode();
      if (opcode == HloOpcode::kWhile) {
        const HloInstruction* while_hlo = instruction;
        ShapeUtil::ForEachSubshape(
            while_hlo->shape(),
            [this, while_hlo, &points_to_analysis, &buffer_liveness,
             buffer_size, computation, colocated_buffer_sets](
                const Shape& /*subshape*/, const ShapeIndex& index) {
              std::vector<const LogicalBuffer*> colocated_set;
              // Add while.init.
              AddBufferToColocatedSet(while_hlo->operand(0), index,
                                      points_to_analysis, &colocated_set);
              // Add while.result.
              AddBufferToColocatedSet(while_hlo, index, points_to_analysis,
                                      &colocated_set);
              // Add while.cond.parameter.
              AddBufferToColocatedSet(
                  while_hlo->while_condition()->parameter_instruction(0), index,
                  points_to_analysis, &colocated_set);
              // Add while.body.parameter.
              AddBufferToColocatedSet(
                  while_hlo->while_body()->parameter_instruction(0), index,
                  points_to_analysis, &colocated_set);
              // Add while.body.root.
              AddBufferToColocatedSet(
                  while_hlo->while_body()->root_instruction(), index,
                  points_to_analysis, &colocated_set);
              AddSetToColocatedBufferSets(colocated_set, colocated_buffer_sets);
            });
      } else if (opcode == HloOpcode::kCall) {
        const HloInstruction* call_hlo = instruction;
        const HloComputation* callee = call_hlo->to_apply();
        const HloInstruction* root_hlo = callee->root_instruction();
        for (int64 i = 0; i < call_hlo->operand_count(); i++) {
          const HloInstruction* call_param = callee->parameter_instruction(i);
          const HloInstruction* call_operand = call_hlo->operand(i);
          ShapeUtil::ForEachSubshape(
              call_operand->shape(),
              [&](const Shape& /*subshape*/, const ShapeIndex& index) {
                std::vector<const LogicalBuffer*> colocated_set;
                AddBufferToColocatedSet(call_param, index, points_to_analysis,
                                        &colocated_set);
                AddBufferToColocatedSet(call_operand, index, points_to_analysis,
                                        &colocated_set);
                AddSetToColocatedBufferSets(colocated_set,
                                            colocated_buffer_sets);
              });
        }
        ShapeUtil::ForEachSubshape(
            call_hlo->shape(),
            [this, call_hlo, root_hlo, &points_to_analysis,
             colocated_buffer_sets](const Shape& /*subshape*/,
                                    const ShapeIndex& index) {
              std::vector<const LogicalBuffer*> colocated_set;
              // Add call.result.
              AddBufferToColocatedSet(call_hlo, index, points_to_analysis,
                                      &colocated_set);
              // Add call.subcomputation.root.
              AddBufferToColocatedSet(root_hlo, index, points_to_analysis,
                                      &colocated_set);
              AddSetToColocatedBufferSets(colocated_set, colocated_buffer_sets);
            });
      } else if (opcode == HloOpcode::kConditional) {
        const HloInstruction* conditional_hlo = instruction;
        ShapeUtil::ForEachSubshape(
            conditional_hlo->shape(),
            [this, conditional_hlo, &points_to_analysis, colocated_buffer_sets](
                const Shape& /*subshape*/, const ShapeIndex& index) {
              std::vector<const LogicalBuffer*> colocated_set;
              // Add conditional.result.
              AddBufferToColocatedSet(conditional_hlo, index,
                                      points_to_analysis, &colocated_set);
              // Add conditional.true_computation.root.
              AddBufferToColocatedSet(
                  conditional_hlo->true_computation()->root_instruction(),
                  index, points_to_analysis, &colocated_set);
              // Add conditional.false_computation.root.
              AddBufferToColocatedSet(
                  conditional_hlo->false_computation()->root_instruction(),
                  index, points_to_analysis, &colocated_set);
              AddSetToColocatedBufferSets(colocated_set, colocated_buffer_sets);
            });

        // Add true_operand and conditional.true_computation.parameter(0) as a
        // colocated buffer set. Note that this has to be done for each subshape
        // in the true_operand of the conditional.
        ShapeUtil::ForEachSubshape(
            conditional_hlo->operand(1)->shape(),
            [this, conditional_hlo, &points_to_analysis, colocated_buffer_sets](
                const Shape& /*subshape*/, const ShapeIndex& index) {
              std::vector<const LogicalBuffer*> true_set;
              // Add conditional.true_operand.
              AddBufferToColocatedSet(conditional_hlo->operand(1), index,
                                      points_to_analysis, &true_set);
              // Add conditional.true_computation.parameter_instruction(0).
              AddBufferToColocatedSet(
                  conditional_hlo->true_computation()->parameter_instruction(0),
                  index, points_to_analysis, &true_set);
              AddSetToColocatedBufferSets(true_set, colocated_buffer_sets);
            });

        // Add false_operand and conditional.false_computation.parameter(0) as a
        // colocated buffer set. Note that this has to be done for each subshape
        // in the false_operand of the conditional.
        ShapeUtil::ForEachSubshape(
            conditional_hlo->operand(2)->shape(),
            [this, conditional_hlo, &points_to_analysis, colocated_buffer_sets](
                const Shape& /*subshape*/, const ShapeIndex& index) {
              std::vector<const LogicalBuffer*> false_set;
              // Add conditional.false_operand.
              AddBufferToColocatedSet(conditional_hlo->operand(2), index,
                                      points_to_analysis, &false_set);
              // Add conditional.false_computation.parameter_instruction(0).
              AddBufferToColocatedSet(
                  conditional_hlo->false_computation()->parameter_instruction(
                      0),
                  index, points_to_analysis, &false_set);
              AddSetToColocatedBufferSets(false_set, colocated_buffer_sets);
            });
      }
    }
  }

  if (colocated_buffer_sets->empty()) {
    return;
  }

  // Try to find more coalescing opportunities among the colocated buffer sets.
  //
  // TODO(b/32491382): We should be able to remove this by using the
  // module-level liveness analysis, which would let us directly detect buffer
  // sharing opportunities between the while instruction buffer and the buffers
  // from the predicate and body computation, as well as sharing across
  // different while instructions.
  std::vector<ColocatedBufferSet> new_colocated_buffer_sets =
      MergeColocatedBufferSets(*colocated_buffer_sets, buffer_liveness,
                               buffer_size);
  std::swap(*colocated_buffer_sets, new_colocated_buffer_sets);
}

// Assigns all colocated buffer sets in 'colocated_buffer_sets' to the same
// allocation in 'assignment'.
void BufferAssigner::AssignColocatedBufferSets(
    const std::vector<ColocatedBufferSet>& colocated_buffer_sets,
    BufferAssignment* assignment,
    FlatSet<const LogicalBuffer*>* colocated_buffers,
    FlatSet<BufferAllocation::Index>* colocated_allocations) {
  for (const ColocatedBufferSet& colocated_buffer_set : colocated_buffer_sets) {
    BufferAllocation* allocation = nullptr;
    // Set 'entry_parameter_number' and 'entry_parameter_shape_idx' if entry
    // param in 'colocated_buffer_set'.
    int64 entry_parameter_number = -1;
    const ShapeIndex* entry_parameter_shape_idx = nullptr;
    bool is_constant = false;
    for (const LogicalBuffer* buffer : colocated_buffer_set) {
      const HloInstruction* instruction = buffer->instruction();
      const HloComputation* computation = instruction->parent();
      if (instruction->opcode() == HloOpcode::kParameter &&
          computation == computation->parent()->entry_computation()) {
        entry_parameter_number = instruction->parameter_number();
        entry_parameter_shape_idx = &buffer->index();
      } else if (instruction->opcode() == HloOpcode::kConstant) {
        is_constant = true;
      }
    }

    CHECK(!is_constant || entry_parameter_number == -1)
        << "Copy insertion should have inserted copies to prevent this.";

    for (const LogicalBuffer* buffer : colocated_buffer_set) {
      const int64 buffer_size = assignment->buffer_size_(*buffer);
      if (allocation == nullptr) {
        // TODO(b/32491382) Avoid current trivial solution of using new
        // allocations for each colocated buffer set. When liveness has
        // module-level scope, we can allow buffers to be shared across
        // computations (in some cases).
        allocation = assignment->NewAllocation(*buffer, buffer_size);
        if (entry_parameter_number >= 0) {
          allocation->set_entry_computation_parameter(
              entry_parameter_number, *entry_parameter_shape_idx);
        }
        if (is_constant) {
          allocation->set_constant(true);
        }
        colocated_allocations->insert(allocation->index());
      } else {
        CHECK_EQ(buffer_size, allocation->size())
            << "Buffer: " << *buffer << " size mismatch in colocated buffer "
            << "allocation: " << *allocation;
        assignment->AddAssignment(allocation, *buffer, /*offset=*/0,
                                  buffer_size);
      }
      colocated_buffers->insert(buffer);
    }
  }
}

StatusOr<std::unique_ptr<BufferAssignment>> BufferAssigner::CreateAssignment(
    const HloModule* module, std::unique_ptr<HloOrdering> hlo_ordering,
    LogicalBuffer::SizeFunction buffer_size,
    LogicalBuffer::AlignmentFunction color_alignment) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<BufferLiveness> liveness,
                      BufferLiveness::Run(module, std::move(hlo_ordering)));

  VLOG(1) << "Assigning buffers to module " << module->name();
  XLA_VLOG_LINES(2, module->ToString());
  XLA_VLOG_LINES(3, liveness->ToString());
  XLA_VLOG_LINES(3, liveness->points_to_analysis().ToString());

  // Can't use MakeUnique because BufferAssignment constructor is private.
  std::unique_ptr<BufferAssignment> assignment(
      new BufferAssignment(module, std::move(liveness), std::move(buffer_size),
                           std::move(color_alignment)));

  // Assign buffers with the tightest constraints first (colocated buffer sets).
  // Once b/32491382 enables module-level liveness analysis, we may be able
  // to assign colocated buffers (or at least reuse their allocation for
  // buffers outside of the set) in AssignBuffersForComputation.
  FlatSet<const LogicalBuffer*> colocated_buffers;
  FlatSet<BufferAllocation::Index> colocated_allocations;
  std::vector<ColocatedBufferSet> colocated_buffer_sets;
  BuildColocatedBufferSets(module, assignment->liveness(),
                           assignment->buffer_size_, &colocated_buffer_sets);
  TF_RETURN_IF_ERROR(colorer_(assignment->liveness()));
  VLOG(3) << "After coloring:";
  XLA_VLOG_LINES(3, assignment->points_to_analysis().ToString());

  AssignColocatedBufferSets(colocated_buffer_sets, assignment.get(),
                            &colocated_buffers, &colocated_allocations);

  std::vector<const HloComputation*> thread_local_computations;
  std::vector<const HloComputation*> global_computations;
  TF_RETURN_IF_ERROR(GatherComputationsByAllocationType(
      module, &thread_local_computations, &global_computations));

  // First assign buffers for global computatations. Temporary buffers for
  // sequential computations are collected in 'buffers_to_assign_sequentially'.
  FlatMap<const HloComputation*, FlatSet<const LogicalBuffer*>>
      buffers_to_assign_sequentially;
  for (auto* computation : global_computations) {
    TF_RETURN_IF_ERROR(AssignBuffersForComputation(
        computation, module->config().debug_options(),
        /*is_thread_local=*/false, colocated_buffers, colocated_allocations,
        &buffers_to_assign_sequentially, assignment.get()));
  }
  // Assign buffers with sequential ordering, if any. If all global computations
  // are sequential, we can run heap simuation on the whole module, which
  // reduces memory usage.
  const bool run_whole_module_heap_simulation =
      buffers_to_assign_sequentially.size() == global_computations.size();
  TF_RETURN_IF_ERROR(AssignBuffersWithSequentialOrdering(
      buffers_to_assign_sequentially, run_whole_module_heap_simulation,
      assignment.get()));

  // Now assign buffers for thread-local computations. All LogicalBuffers get
  // their own BufferAllocation.
  for (auto* computation : thread_local_computations) {
    TF_RET_CHECK(computation != module->entry_computation());
    if (computation->IsFusionComputation()) {
      continue;
    }
    TF_RETURN_IF_ERROR(AssignBuffersForComputation(
        computation, module->config().debug_options(),
        /*is_thread_local=*/true, colocated_buffers, colocated_allocations,
        /*buffers_to_assign_sequentially=*/nullptr, assignment.get()));
  }

  // Mark all buffers which may be live out of the entry computation as
  // "liveout".
  for (const LogicalBuffer* buffer :
       assignment->liveness().maybe_live_out_buffers()) {
    VLOG(3) << "maybe_live_out LogicalBuffer: " << *buffer;
    if (assignment->HasAllocation(*buffer)) {
      BufferAllocation* alloc =
          assignment->GetMutableAssignedAllocation(*buffer);
      alloc->set_maybe_live_out(true);
      VLOG(3) << "maybe_live_out BufferAllocation: " << *alloc;
    }
  }

  // Combines allocations of temporary buffers into one big BufferAllocation.
  // This can only be performed after all buffers have been assigned, and after
  // maybe_live_out is marked, since it is used to determine whether an
  // allocation contains temporary buffers or not.
  assignment->CombineTempAllocations();

  XLA_VLOG_LINES(2, assignment->ToString());
  TF_RETURN_IF_ERROR(assignment->ComputeSummaryStats());
  XLA_VLOG_LINES(1, assignment->GetStats().ToString());
  return std::move(assignment);
}

}  // namespace xla

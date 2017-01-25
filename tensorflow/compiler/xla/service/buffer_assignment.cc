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

#include "tensorflow/compiler/xla/legacy_flags/buffer_assignment_flags.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

namespace xla {

void BufferAllocation::AddAssignment(const LogicalBuffer& buffer) {
  DCHECK(std::find(assigned_buffers_.begin(), assigned_buffers_.end(),
                   &buffer) == assigned_buffers_.end())
      << "LogicalBuffer " << buffer.ToString()
      << " already assigned to allocation " << index();
  assigned_buffers_.push_back(&buffer);
}

string BufferAllocation::ToString() const {
  string output;
  tensorflow::strings::StrAppend(
      &output, tensorflow::strings::Printf("allocation %lld: %p, size %lld",
                                           index_, this, size()));
  if (is_entry_computation_parameter()) {
    tensorflow::strings::StrAppend(&output, ", parameter ", parameter_number());
  }
  if (is_thread_local()) {
    tensorflow::strings::StrAppend(&output, ", thread-local");
  }
  tensorflow::strings::StrAppend(&output, ":\n");
  for (const auto& buffer : assigned_buffers()) {
    tensorflow::strings::StrAppend(
        &output,
        tensorflow::strings::Printf(
            "  %s::%s : %s\n", buffer->instruction()->parent()->name().c_str(),
            buffer->ToString().c_str(),
            ShapeUtil::HumanStringWithLayout(buffer->shape()).c_str()));
  }
  return output;
}

std::ostream& operator<<(std::ostream& out, const BufferAllocation& buffer) {
  out << buffer.ToString();
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

std::set<BufferAllocation> BufferAssignment::GetAllocations(
    const HloInstruction* instruction, const ShapeIndex& index) const {
  std::set<BufferAllocation> allocations;
  for (const LogicalBuffer* buffer : GetSourceBuffers(instruction, index)) {
    if (allocation_index_for_buffer_.count(buffer) > 0) {
      allocations.insert(
          GetAllocation(allocation_index_for_buffer_.at(buffer)));
    }
  }
  return allocations;
}

const BufferAllocation& BufferAssignment::GetAllocation(
    BufferAllocation::Index index) const {
  CHECK(index >= 0 && index < allocations_.size())
      << "Allocation index " << index << "is out of range.";
  return allocations_[index];
}

BufferAllocation* BufferAssignment::GetMutableAllocation(
    BufferAllocation::Index index) {
  return const_cast<BufferAllocation*>(&GetAllocation(index));
}

bool BufferAssignment::HasTopLevelAllocation(
    const HloInstruction* instruction) const {
  for (const LogicalBuffer* buffer :
       GetPointsToSet(instruction).element(/*index=*/{})) {
    if (allocation_index_for_buffer_.count(buffer) > 0) {
      return true;
    }
  }
  return false;
}

StatusOr<const BufferAllocation*> BufferAssignment::GetUniqueAllocation(
    const HloInstruction* instruction, const ShapeIndex& index) const {
  const BufferAllocation* allocation = nullptr;
  for (const LogicalBuffer* buffer :
       GetPointsToSet(instruction).element(index)) {
    if (HasAllocation(*buffer)) {
      if (allocation != nullptr &&
          *allocation != GetAssignedAllocation(*buffer)) {
        return FailedPrecondition(
            "LogicalBuffer allocation for instruction %s at index {%s} cannot "
            "be determined at compile-time.",
            instruction->name().c_str(),
            tensorflow::str_util::Join(index, ",").c_str());
      }
      allocation = &GetAssignedAllocation(*buffer);
    }
  }
  if (allocation == nullptr) {
    return FailedPrecondition(
        "instruction %s has no buffer allocation at index {%s}",
        instruction->name().c_str(),
        tensorflow::str_util::Join(index, ",").c_str());
  }
  return allocation;
}

StatusOr<const BufferAllocation*> BufferAssignment::GetUniqueTopLevelAllocation(
    const HloInstruction* instruction) const {
  return GetUniqueAllocation(instruction, /*index=*/{});
}

StatusOr<const BufferAllocation*>
BufferAssignment::GetUniqueTopLevelOutputAllocation() const {
  return GetUniqueTopLevelAllocation(
      module_->entry_computation()->root_instruction());
}

BufferAllocation* BufferAssignment::NewAllocation(const LogicalBuffer& buffer,
                                                  int64 size,
                                                  bool is_thread_local,
                                                  bool is_reusable) {
  BufferAllocation::Index index = allocations_.size();
  allocations_.emplace_back(index, size, is_thread_local, is_reusable);
  BufferAllocation* allocation = &allocations_.back();
  AddAssignment(buffer, allocation, /*colocated_buffer=*/false);
  allocation_index_for_buffer_[&buffer] = index;
  return allocation;
}

// Adds an instruction to the set assigned to the given buffer.
void BufferAssignment::AddAssignment(const LogicalBuffer& buffer,
                                     BufferAllocation* allocation,
                                     bool colocated_buffer) {
  CHECK_EQ(0, allocation_index_for_buffer_.count(&buffer))
      << "LogicalBuffer " << buffer << " already has an allocation.";
  CHECK(allocation->is_reusable() || allocation->assigned_buffers().empty() ||
        colocated_buffer)
      << "Non-reusable allocation already assigned a buffer";

  TF_CHECK_OK(points_to_analysis().VerifyBuffer(buffer));

  allocation->AddAssignment(buffer);
  allocation_index_for_buffer_[&buffer] = allocation->index();
}

string BufferAssignment::ToString() const {
  string output;
  tensorflow::strings::StrAppend(&output, "BufferAssignment:\n");
  for (auto& allocation : allocations_) {
    tensorflow::strings::StrAppend(&output, allocation.ToString());
  }
  return output;
}

namespace {

// Walk the call graph of the HLO module and place each computation into either
// thread_local_computations or global_computations depending upon whether the
// computation requires thread-local allocations or global allocations. The
// elements in thread_local_computations and global_computations are in post
// order (if computation A has an instruction which calls computation B, then A
// will appear after B in the vector).
tensorflow::Status GatherComputationsByAllocationType(
    const HloModule* module,
    std::vector<const HloComputation*>* thread_local_computations,
    std::vector<const HloComputation*>* global_computations) {
  // Create a worklist of computations paired with whether the allocation must
  // be thread-local.
  std::deque<std::pair<HloComputation*, bool>> worklist;
  worklist.push_back(std::make_pair(module->entry_computation(),
                                    /*is_thread_local*/ false));

  // Sets for quickly checking membership. Computations are returned in vectors
  // for stable iteration.
  std::unordered_set<HloComputation*> thread_local_set;
  std::unordered_set<HloComputation*> global_set;

  while (!worklist.empty()) {
    auto worklist_front = worklist.front();
    worklist.pop_front();
    HloComputation* computation = worklist_front.first;
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

    for (auto& instruction : computation->instructions()) {
      for (auto* subcomputation : instruction->MakeCalledComputationsSet()) {
        switch (instruction->opcode()) {
          case HloOpcode::kCall:
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
  return tensorflow::Status::OK();
}

}  // namespace

/* static */
StatusOr<std::unique_ptr<BufferAssignment>> BufferAssigner::Run(
    const HloModule* module, std::unique_ptr<HloOrdering> hlo_ordering,
    BufferSizeFunction buffer_size, bool colocate_related_buffers,
    const std::vector<const HloInstruction*>* hlos_to_allocate) {
  BufferAssigner assigner(std::move(buffer_size), colocate_related_buffers);
  return assigner.CreateAssignment(module, std::move(hlo_ordering),
                                   hlos_to_allocate);
}

/* static */
StatusOr<std::unique_ptr<BufferAssignment>> BufferAssigner::Run(
    const HloModule* module, std::unique_ptr<HloOrdering> hlo_ordering,
    int64 pointer_size) {
  return BufferAssigner::Run(module, std::move(hlo_ordering),
                             [pointer_size](const LogicalBuffer& buffer) {
                               return ShapeUtil::IsOpaque(buffer.shape())
                                          ? 0
                                          : ShapeUtil::ByteSizeOf(
                                                buffer.shape(), pointer_size);
                             },
                             /*colocate_related_buffers=*/true);
}

bool BufferAssigner::MaybeAssignBuffer(BufferAllocation* allocation,
                                       const LogicalBuffer& buffer,
                                       BufferAssignment* assignment) {
  CHECK(!assignment->HasAllocation(buffer))
      << "buffer " << buffer << " already has an allocation assigned.";

  VLOG(4) << "Trying to assign " << buffer.ToString()
          << " to allocation: " << allocation->ToString();

  if (buffer_size_(buffer) > allocation->size()) {
    VLOG(4) << "Can't assign: buffer is larger than allocation ("
            << buffer_size_(buffer) << " > " << allocation->size() << ")";
    return false;
  }

  if (allocation->is_entry_computation_parameter()) {
    VLOG(4) << "Can't assign: allocation holds parameter";
    return false;
  }

  if (!allocation->is_reusable()) {
    VLOG(4) << "Can't assign: allocation is not reusable";
    return false;
  }

  for (const LogicalBuffer* assigned_buffer : allocation->assigned_buffers()) {
    if (assignment->liveness().MayInterfere(*assigned_buffer, buffer)) {
      VLOG(4) << "Can't assign: assignee " << assigned_buffer->ToString()
              << " may interfere with " << buffer.ToString();
      return false;
    }
  }

  // If the buffer is live out of the computation then it should only be
  // assigned a buffer which exactly fits the result to avoid wasting memory
  // (result buffers can have arbitrary lifetimes).
  if (assignment->liveness().MaybeLiveOut(buffer) &&
      allocation->size() != buffer_size_(buffer)) {
    VLOG(4) << "Can't assign: buffer " << buffer.ToString()
            << "is live out and size not the same as allocation";
    return false;
  }

  assignment->AddAssignment(buffer, allocation, /*colocated_buffer=*/false);
  return true;
}

tensorflow::Status BufferAssigner::AssignBuffersForComputation(
    const HloComputation* computation, bool is_thread_local,
    const std::unordered_set<const HloInstruction*>* hlos_to_allocate,
    BufferAssignment* assignment) {
  // Buffers are sorted and assigned to BufferAllocations in decreasing order of
  // size.
  std::vector<const LogicalBuffer*> sorted_buffers;
  for (auto& instruction : computation->instructions()) {
    if (hlos_to_allocate == nullptr ||
        hlos_to_allocate->count(instruction.get()) > 0) {
      // Add all buffers which this instruction defines. Instruction which don't
      // define buffers (eg, bitcast which just forwards a pointer) don't need
      // any allocations.
      for (const LogicalBuffer* buffer :
           assignment->points_to_analysis().GetBuffersDefinedByInstruction(
               instruction.get())) {
        sorted_buffers.push_back(buffer);
      }
    }
  }

  // Generate a post order sort of instructions for sorting of the
  // LogicalBuffers.
  std::unordered_map<const HloInstruction*, int> post_order_position;
  int position = 0;
  for (auto* instruction : computation->MakeInstructionPostOrder()) {
    post_order_position.emplace(instruction, position);
    position++;
  }

  // Sort the LogicalBuffers first by size. We assign the larger LogicalBuffers
  // first for simplicity. This means any previously created BufferAllocation is
  // necessarily large enough to hold the output of the current Buffer in
  // consideration.
  //
  // As a secondary sorting criteria, use post order position of the HLO
  // instruction which defines the buffer. This means an instruction will appear
  // after its operands (assuming operands are the same/larger size) enabling
  // the important reuse case where an elementwise instruction reuses one of its
  // operand's buffer. This improves locality.
  std::sort(sorted_buffers.begin(), sorted_buffers.end(),
            [this, &post_order_position](const LogicalBuffer* a,
                                         const LogicalBuffer* b) {
              int64 a_size = buffer_size_(*a);
              int64 b_size = buffer_size_(*b);
              if (a_size == b_size) {
                // For instructions with the same size buffers, sort them in
                // post order.
                return post_order_position.at(a->instruction()) <
                       post_order_position.at(b->instruction());
              } else {
                // We want the HLOs sorted in reverse order by size so use ">".
                return a_size > b_size;
              }
            });

  // BufferAllocations are necessarily created in decreasing size order. Keep
  // indices of previously created BufferAllocations in allocation_indices.
  std::vector<BufferAllocation::Index> allocation_indices;
  for (const auto* buffer : sorted_buffers) {
    VLOG(3) << "Assigning allocation to: " << buffer->ToString();
    if (colocated_buffers_.find(buffer) != colocated_buffers_.end()) {
      // Colocated buffers are currently assigned in an earlier pass.
      continue;
    }

    TF_RET_CHECK(!assignment->HasAllocation(*buffer));

    if (buffer->instruction()->opcode() == HloOpcode::kConstant) {
      // No BufferAllocations for constants.
      // TODO(b/32248867): For consistency, constants should get allocations.
      continue;
    }

    if (buffer->instruction()->opcode() == HloOpcode::kParameter &&
        computation == computation->parent()->entry_computation()) {
      // If the LogicalBuffer is part of an external parameter, creates a new
      // allocation and sets its parameter number. Parameters of non-entry
      // computations do not need special allocations because they live inside
      // callers.
      BufferAllocation* allocation =
          assignment->NewAllocation(*buffer, buffer_size_(*buffer),
                                    /*is_thread_local=*/false,
                                    /*is_reusable=*/false);
      allocation->set_entry_computation_parameter(
          buffer->instruction()->parameter_number());
      VLOG(3) << "New allocation for entry computation parameter: "
              << buffer->ToString();
      continue;
    }

    legacy_flags::BufferAssignmentFlags* flags =
        legacy_flags::GetBufferAssignmentFlags();
    if (!flags->xla_enable_buffer_reuse || is_thread_local ||
        buffer->instruction()->opcode() == HloOpcode::kCustomCall) {
      // Custom call operations never have reusable buffers. Also we do not
      // reuse thread-local buffers for now, because they are dynamically
      // allocated and their lifetimes are hard to compute.
      assignment->NewAllocation(*buffer, buffer_size_(*buffer), is_thread_local,
                                /*is_reusable=*/false);
      continue;
    }

    if (buffer->instruction()->opcode() == HloOpcode::kCall &&
        buffer->IsTopLevel()) {
      // Assign the kCall instruction the same allocation as the root of the
      // called computation. The points-to set of the root of the called
      // computation must be unambigous so we know statically the allocation for
      // the root.
      //
      // TODO(b/32491382): This is a hack. To properly handle this case
      // points-to analysis, liveness analysis, and buffer assignment need to
      // module-scope rather than computation-scope.
      HloInstruction* call = buffer->instruction();
      HloInstruction* computation_root = call->to_apply()->root_instruction();

      // The buffer of the root of the called computation must be unambiguous.
      const auto& root_points_to = assignment->GetPointsToSet(computation_root);
      if (root_points_to.IsAmbiguous()) {
        return Unimplemented(
            "kCall of a computation with an ambiguous root points-to set");
      }
      CHECK_EQ(1, root_points_to.element(/*index=*/{}).size());
      const LogicalBuffer* root_buffer =
          root_points_to.element(/*index=*/{})[0];
      BufferAllocation* root_allocation =
          assignment->GetMutableAssignedAllocation(*root_buffer);

      // Can't use MaybeAssignBuffer here because buffer liveness conservatively
      // assumes buffers in different computations always interfere.
      CHECK_GE(root_allocation->size(), buffer_size_(*buffer));
      assignment->AddAssignment(*buffer, root_allocation,
                                /*colocated_buffer=*/true);
      continue;
    }

    if (ShapeUtil::IsTuple(buffer->shape())) {
      // TODO(b/34669761): Don't reuse tuple buffers because the GPU backend
      // assumes longer buffer liveness than indicated by the analysis.
      assignment->NewAllocation(*buffer, buffer_size_(*buffer), is_thread_local,
                                /*is_reusable=*/false);
      continue;
    }

    // First try to assign a LogicalBuffer to one of its operand allocations to
    // improve locality. This is only possible with elementwise operations
    // (checked in liveness analysis) which are necessarily top-level
    // array-shaped buffers.
    if (buffer->IsTopLevel() && !buffer->IsTuple()) {
      for (auto* operand : buffer->instruction()->operands()) {
        bool assigned_operand = false;
        for (const auto& operand_allocation :
             assignment->GetAllocations(operand, /*index=*/{})) {
          BufferAllocation* allocation =
              assignment->GetMutableAllocation(operand_allocation.index());
          if (colocated_buffer_allocations_.find(allocation->index()) ==
              colocated_buffer_allocations_.end()) {
            // TODO(b/32491382) Colocated buffers are currently assigned in an
            // earlier pass, and so can break the "increasing allocation size"
            // invariant in this function (causing this CHECK to fail). However,
            // the call to MaybeAssignBuffer is safe as it returns false if
            // allocation.size < buffer.size.
            CHECK_GE(allocation->size(), buffer_size_(*buffer));
          }
          if (MaybeAssignBuffer(allocation, *buffer, assignment)) {
            VLOG(3) << "Reusing (operand) allocation for: "
                    << buffer->ToString();
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
        if (colocated_buffer_allocations_.find(allocation->index()) ==
            colocated_buffer_allocations_.end()) {
          // TODO(b/32491382) Colocated buffers are currently assigned in an
          // earlier pass, and so can break the "increasing allocation size"
          // invariant in this function (causing this CHECK to fail). However,
          // the call to MaybeAssignBuffer is safe as it returns false if
          // allocation.size < buffer.size.
          CHECK_GE(allocation->size(), buffer_size_(*buffer));
        }

        if (MaybeAssignBuffer(allocation, *buffer, assignment)) {
          VLOG(3) << "Reusing buffer for: " << buffer->ToString();
          break;
        }
      }
    }
    if (!assignment->HasAllocation(*buffer)) {
      auto* allocation =
          assignment->NewAllocation(*buffer, buffer_size_(*buffer),
                                    is_thread_local, /*is_reusable=*/true);
      VLOG(3) << "New allocation for: " << buffer->ToString();
      allocation_indices.push_back(allocation->index());
    }
  }
  return tensorflow::Status::OK();
}

void BufferAssigner::AddBufferToColocatedBufferSet(
    const HloInstruction* instruction, const ShapeIndex& index,
    const TuplePointsToAnalysis& points_to_analysis,
    BufferAssigner::ColocatedBufferSet* colocated_buffer_set) {
  const auto& points_to = points_to_analysis.GetPointsToSet(instruction);
  // CopyInsertion ensures root points-to set is unambiguous and distinct.
  CHECK(!points_to.IsAmbiguous());
  CHECK(points_to.IsDistinct());
  colocated_buffer_set->push_back(points_to.element(index)[0]);
}

// Builds sets of buffers in 'colocated_buffer_sets' which should be colocated
// in the same allocation (currently just supports kWhile).
std::vector<BufferAssigner::ColocatedBufferSet>
BufferAssigner::BuildColocatedBufferSets(
    const HloModule* module, const TuplePointsToAnalysis& points_to_analysis) {
  std::vector<ColocatedBufferSet> colocated_buffer_sets;
  for (auto& computation : module->computations()) {
    for (auto& instruction : computation->instructions()) {
      if (instruction->opcode() != HloOpcode::kWhile) {
        continue;
      }
      HloInstruction* while_hlo = instruction.get();
      TF_CHECK_OK(ShapeUtil::ForEachSubshape(
          while_hlo->shape(),
          [this, &points_to_analysis, &while_hlo, &colocated_buffer_sets](
              const Shape& /*subshape*/, const ShapeIndex& index) {
            ColocatedBufferSet colocated_buffer_set;
            // Add while.init.
            AddBufferToColocatedBufferSet(while_hlo->operand(0), index,
                                          points_to_analysis,
                                          &colocated_buffer_set);
            // Add while.result.
            AddBufferToColocatedBufferSet(while_hlo, index, points_to_analysis,
                                          &colocated_buffer_set);
            // Add while.cond.parameter.
            AddBufferToColocatedBufferSet(
                while_hlo->while_condition()->parameter_instruction(0), index,
                points_to_analysis, &colocated_buffer_set);
            // Add while.body.parameter.
            AddBufferToColocatedBufferSet(
                while_hlo->while_body()->parameter_instruction(0), index,
                points_to_analysis, &colocated_buffer_set);
            // Add while.body.root.
            AddBufferToColocatedBufferSet(
                while_hlo->while_body()->root_instruction(), index,
                points_to_analysis, &colocated_buffer_set);

            colocated_buffer_sets.push_back(std::move(colocated_buffer_set));
            return tensorflow::Status::OK();
          }));
    }
  }
  return colocated_buffer_sets;
}

// Assigns all colocated buffer sets in 'colocated_buffer_sets' to the same
// allocation in 'assignment'.
void BufferAssigner::AssignColocatedBufferSets(
    const std::vector<ColocatedBufferSet>& colocated_buffer_sets,
    BufferAssignment* assignment) {
  for (const auto& colocated_buffer_set : colocated_buffer_sets) {
    BufferAllocation* allocation = nullptr;
    for (const auto& buffer : colocated_buffer_set) {
      if (colocated_buffers_.find(buffer) != colocated_buffers_.end()) {
        // ColocatedBufferSet duplicates can occur if a buffer is forwarded
        // from one instruction to another (i.e. while.body param to root).
        continue;
      }
      if (allocation == nullptr) {
        // TODO(b/32491382) Avoid current trivial solution of using new
        // allocations for each colocated buffer set. When liveness has
        // module-level scope, we can allow buffers to be shared across
        // computations (in some cases).
        allocation = assignment->NewAllocation(*buffer, buffer_size_(*buffer),
                                               /*is_thread_local=*/false,
                                               /*is_reusable=*/true);
        colocated_buffer_allocations_.insert(allocation->index());
      } else {
        assignment->AddAssignment(*buffer, allocation,
                                  /*colocated_buffer=*/true);
      }
      colocated_buffers_.insert(buffer);
    }
  }
}

StatusOr<std::unique_ptr<BufferAssignment>> BufferAssigner::CreateAssignment(
    const HloModule* module, std::unique_ptr<HloOrdering> hlo_ordering,
    const std::vector<const HloInstruction*>* hlos_to_allocate) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<BufferLiveness> liveness,
                      BufferLiveness::Run(module, std::move(hlo_ordering)));

  std::vector<const HloComputation*> thread_local_computations;
  std::vector<const HloComputation*> global_computations;
  VLOG(1) << "Assigning buffers to module " << module->name();
  if (hlos_to_allocate != nullptr) {
    VLOG(3) << "LogicalBuffer assignment restricted to hlos: ";
    for (auto hlo : *hlos_to_allocate) {
      VLOG(3) << "  " << hlo->parent()->name() << "::" << hlo->name();
    }
  }
  XLA_VLOG_LINES(3, module->ToString());
  XLA_VLOG_LINES(3, liveness->ToString());
  XLA_VLOG_LINES(3, liveness->points_to_analysis().ToString());

  TF_RETURN_IF_ERROR(GatherComputationsByAllocationType(
      module, &thread_local_computations, &global_computations));

  // Set of HLO's to allocate if hlos_to_allocate is given. Passed as a set to
  // AssignBuffersForComputation for fast membership testing.
  std::unique_ptr<std::unordered_set<const HloInstruction*>> hlo_set;
  if (hlos_to_allocate != nullptr) {
    hlo_set = MakeUnique<std::unordered_set<const HloInstruction*>>(
        hlos_to_allocate->begin(), hlos_to_allocate->end());
  }

  // Can't use MakeUnique because BufferAssignment constructor is private.
  std::unique_ptr<BufferAssignment> assignment(
      new BufferAssignment(module, std::move(liveness)));

  // Assign buffers with the tightest constraints first (colocated buffer sets).
  // Once b/32491382 enables module-level liveness analysis, we may be able
  // to assign colocated buffers (or at least reuse their allocation for
  // buffers outside of the set) in AssignBuffersForComputation.
  if (colocate_related_buffers_) {
    std::vector<ColocatedBufferSet> colocated_buffer_sets =
        BuildColocatedBufferSets(module, assignment->points_to_analysis());
    AssignColocatedBufferSets(colocated_buffer_sets, assignment.get());
  }

  for (auto* computation : global_computations) {
    TF_RETURN_IF_ERROR(AssignBuffersForComputation(
        computation,
        /*is_thread_local=*/false, hlo_set.get(), assignment.get()));
  }
  for (auto* computation : thread_local_computations) {
    TF_RET_CHECK(computation != module->entry_computation());
    TF_RETURN_IF_ERROR(AssignBuffersForComputation(
        computation,
        /*is_thread_local=*/true, hlo_set.get(), assignment.get()));
  }

  // Mark all buffers which may be live out of the entry computation as
  // "liveout".
  auto entry = module->entry_computation();
  auto root_instruction = entry->root_instruction();
  const PointsToSet& root_points_to =
      assignment->GetPointsToSet(root_instruction);
  TF_RETURN_IF_ERROR(root_points_to.ForEachElement([&assignment](
      const ShapeIndex& /*index*/, bool /*is_leaf*/,
      const std::vector<const LogicalBuffer*>& buffers) {
    for (auto buffer : buffers) {
      if (assignment->HasAllocation(*buffer)) {
        assignment->GetMutableAssignedAllocation(*buffer)->set_maybe_live_out(
            true);
      }
    }
    return tensorflow::Status::OK();
  }));

  XLA_VLOG_LINES(2, assignment->ToString());

  // Compute sizes of various kinds of buffers for logging.
  int64 total_size = 0;
  int64 parameter_size = 0;
  for (auto& allocation : assignment->Allocations()) {
    if (allocation.is_entry_computation_parameter()) {
      parameter_size += allocation.size();
    }
    total_size += allocation.size();
  }

  // Compute the total size of the output. Iterate over the subshapes and sum up
  // the sizes of the buffers for each subshape.
  int64 output_size = 0;
  HloInstruction* root = module->entry_computation()->root_instruction();
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshape(
      root->shape(), [this, &output_size, root, &assignment](
                         const Shape& /*subshape*/, const ShapeIndex& index) {
        const auto& allocations = assignment->GetAllocations(root, index);
        if (allocations.size() > 0) {
          output_size += allocations.begin()->size();
        }
        return tensorflow::Status::OK();
      }));

  VLOG(1) << "Allocation sizes for module " << module->name() << ":";
  VLOG(1) << "  parameter allocation total size: "
          << tensorflow::strings::HumanReadableNumBytes(parameter_size);
  VLOG(1) << "     output allocation total size: "
          << tensorflow::strings::HumanReadableNumBytes(output_size);
  VLOG(1) << "       temp allocation total size: "
          << tensorflow::strings::HumanReadableNumBytes(
                 total_size - parameter_size - output_size);
  VLOG(1) << "            total allocation size: "
          << tensorflow::strings::HumanReadableNumBytes(total_size);
  return std::move(assignment);
}

}  // namespace xla

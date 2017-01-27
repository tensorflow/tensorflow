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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_BUFFER_ASSIGNMENT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_BUFFER_ASSIGNMENT_H_

#include <functional>
#include <iosfwd>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/compiler/xla/service/buffer_liveness.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// This class abstracts an allocation of contiguous memory which can hold the
// values described by LogicalBuffers. A BufferAllocation may hold different
// LogicalBuffers at different times, but currently never more than one
// LogicalBuffer simultaneously. The abstraction includes information required
// by the backends for allocation, use, and deallocation of the buffer. This
// includes the LogicalBuffers which are held in this allocation through the
// execution of the computation.
class BufferAllocation {
 public:
  // Holds a unique identifier for each allocation. Values are assigned
  // contiguously and can be used as array indexes.
  using Index = int64;

  BufferAllocation(Index index, int64 size, bool is_thread_local,
                   bool is_reusable)
      : index_(index),
        size_(size),
        is_thread_local_(is_thread_local),
        is_reusable_(is_reusable) {}
  ~BufferAllocation() {}

  // Adds a LogicalBuffer to the set assigned to this buffer.
  void AddAssignment(const LogicalBuffer& buffer);

  // Whether this allocation is used in a parallel calling context such as
  // inside of a map or reduce computation. Such allocations need to be thread
  // local.
  bool is_thread_local() const { return is_thread_local_; }

  // Whether this allocation can be used by more than one logical buffer.
  bool is_reusable() const { return is_reusable_; }

  // Whether this allocation holds a LogicalBuffer from a parameter of the entry
  // computation. These buffers have lifetimes which may be longer than the
  // XLA computation.
  bool is_entry_computation_parameter() const {
    return is_entry_computation_parameter_;
  }
  // If this allocation holds a Buffer from a parameter of the entry
  // computation, this methods returns the parameter number. CHECKs otherwise.
  int64 parameter_number() const {
    CHECK(is_entry_computation_parameter_);
    return parameter_number_;
  }
  // Sets that this allocation holds a LogicalBuffer from a parameter of the
  // entry computation.
  void set_entry_computation_parameter(int64 parameter_number) {
    is_entry_computation_parameter_ = true;
    parameter_number_ = parameter_number;
  }

  // Returns/sets whether this allocation is assigned a LogicalBuffer which may
  // be live out of the entry computation.
  bool maybe_live_out() const { return maybe_live_out_; }
  void set_maybe_live_out(bool value) { maybe_live_out_ = value; }

  // Returns the size of the allocation. Necessarily this must be at least as
  // large as any LogicalBuffer assigned to this allocation.
  int64 size() const { return size_; }

  // Access to the logical buffers assigned to this allocation.
  const std::vector<const LogicalBuffer*>& assigned_buffers() const {
    return assigned_buffers_;
  }

  Index index() const { return index_; }

  string ToString() const;

  // Whether the buffer is a parameter to or live out of the entry computation.
  bool IsInputOrOutput() const {
    return is_entry_computation_parameter() || maybe_live_out();
  }

  // Whether the buffer is a temporary buffer allocated before
  // Executable::ExecuteOnStream.
  bool IsPreallocatedTempBuffer() const {
    // Parameters do not need temporary buffers.
    return !is_entry_computation_parameter() &&
           // LogicalBuffers that maybe pointed to by the output should live out
           // of the computation.
           !maybe_live_out() &&
           // Thread-local buffers are allocated using `alloca`s.
           !is_thread_local();
  }

  bool operator==(const BufferAllocation& other) const {
    return index_ == other.index_;
  }
  bool operator!=(const BufferAllocation& other) const {
    return !(*this == other);
  }
  bool operator<(const BufferAllocation& other) const {
    return index() < other.index();
  }

 private:
  // The index of the allocation in the BufferAssignment.
  Index index_;

  // Size of the allocation in bytes.
  int64 size_;

  // Whether this buffer needs to be thread-local.
  bool is_thread_local_;

  // Whether this buffer is usable by more than one logical buffer.
  bool is_reusable_;

  // Whether this allocation holds an entry computation parameter. Entry
  // computation parameters are special be cause they have lifetimes which may
  // outlast the computation.
  bool is_entry_computation_parameter_ = false;

  // If this allocation holds an entry computation parameter, this field
  // indicates the index (starting from 0) of the parameter.
  int64 parameter_number_ = 0;

  // Whether the allocation contains a LogicalBuffer which may be live-out of
  // the entry computation. Note that this flag is conservatively computed by
  // TuplePointsToAnalysis.  That is, an allocation marked `maybe_live_out_`
  // might not actually escape.
  bool maybe_live_out_ = false;

  // The set of buffers assigned to this allocation.
  std::vector<const LogicalBuffer*> assigned_buffers_;
};

// Add stream operator for nicer output of CHECK/RET_CHECK failures.
std::ostream& operator<<(std::ostream& out, const BufferAllocation& s);

// This class encapsulates an assignment of the LogicalBuffers in an XLA
// module to a set of BufferAllocations.
class BufferAssignment {
 public:
  // Returns the vector containing all buffer allocations in this assignment.
  const std::vector<BufferAllocation>& Allocations() const {
    return allocations_;
  }

  // Returns whether the given buffer has been assigned an allocation.
  bool HasAllocation(const LogicalBuffer& buffer) const;

  // Returns the allocation that a particular LogicalBuffer has been assigned
  // to. CHECKs if buffer has not been assigned an allocation.
  const BufferAllocation& GetAssignedAllocation(
      const LogicalBuffer& buffer) const;

  // Returns the allocation with the given index. CHECKs if no allocation exists
  // with the given index.
  const BufferAllocation& GetAllocation(BufferAllocation::Index index) const;

  // Builds and returns a vector containing the allocations which might contain
  // the subvalue at the given index of given instruction.
  std::set<BufferAllocation> GetAllocations(const HloInstruction* instruction,
                                            const ShapeIndex& index) const;

  // Convenience function which returns whether the top-level buffer of the
  // instruction (index == {}) is assigned an allocation.
  bool HasTopLevelAllocation(const HloInstruction* instruction) const;

  // Convenience function which returns the unique buffer allocation containing
  // the buffer at the given index of the given instruction. If an allocation is
  // not assigned or the allocation cannot be determined at compile time then an
  // error is returned.
  StatusOr<const BufferAllocation*> GetUniqueAllocation(
      const HloInstruction* instruction, const ShapeIndex& index) const;
  // Like GetUniqueAllocation but fixes the index to the top-level of the shape
  // (index = {}).
  StatusOr<const BufferAllocation*> GetUniqueTopLevelAllocation(
      const HloInstruction* instruction) const;
  // Like GetUniqueTopLevelAllocation but returns the allocation for the output
  // of the entry computation of the HLO module (ie, the result of the XLA
  // computation).
  StatusOr<const BufferAllocation*> GetUniqueTopLevelOutputAllocation() const;

  // Returns the set LogicalBuffers which may be the source of the value at the
  // given index and instruction.
  const std::vector<const LogicalBuffer*>& GetSourceBuffers(
      const HloInstruction* instruction, const ShapeIndex& index) const {
    return GetPointsToSet(instruction).element(index);
  }

  // Returns the underlying points-to analysis used for this assignment.
  const TuplePointsToAnalysis& points_to_analysis() const {
    return liveness_->points_to_analysis();
  }

  string ToString() const;

 private:
  // Only BufferAssigner can build or modify BufferAssignments.
  friend class BufferAssigner;

  explicit BufferAssignment(const HloModule* module,
                            std::unique_ptr<BufferLiveness> liveness)
      : module_(module), liveness_(std::move(liveness)) {}

  // Creates and returns a new BufferAllocation. Ownership is maintained
  // internally. The allocation initially has only the given LogicalBuffer
  // assigned to it. `is_thread_local` indicates whether this buffer needs to be
  // thread-local.
  BufferAllocation* NewAllocation(const LogicalBuffer& buffer, int64 size,
                                  bool is_thread_local, bool is_reusable);

  // Adds a LogicalBuffer to the set assigned to the given allocation. If
  // colocated_buffer is true, then the logical buffer is an alias of another
  // buffer assigned to this allocation.
  void AddAssignment(const LogicalBuffer& buffer, BufferAllocation* allocation,
                     bool colocated_buffer);

  // Returns the BufferLiveness object used to construct this assignment.
  const BufferLiveness& liveness() { return *liveness_; }

  // Convenience function which returns the PointsToSet for the given
  // instruction. Extracted from the liveness object.
  const PointsToSet& GetPointsToSet(const HloInstruction* instruction) const;

  // Mutable accessors for allocations.
  BufferAllocation* GetMutableAssignedAllocation(const LogicalBuffer& buffer);
  BufferAllocation* GetMutableAllocation(BufferAllocation::Index index);

  // The vector of buffer allocations. Indexed by BufferAllocation::Index.
  std::vector<BufferAllocation> allocations_;

  // Maps Buffers to the index of the BufferAllocation which holds the buffer.
  std::map<const LogicalBuffer*, BufferAllocation::Index>
      allocation_index_for_buffer_;

  const HloModule* module_;
  std::unique_ptr<BufferLiveness> liveness_;

  TF_DISALLOW_COPY_AND_ASSIGN(BufferAssignment);
};

// A class which constructs a buffer assignment.
class BufferAssigner {
 public:
  // Build and return a BufferAssignment for the given module. The given
  // HloOrdering is used to determine buffer liveness. buffer_size is a function
  // which returns the size of a LogicalBuffer. If hlos_to_allocate is not null
  // then only instructions in this vector are considered for buffer
  // assignment. If hlos_to_allocate is null then all instructions are
  // considered. If 'colocate_related_buffers' is true, related LogicalBuffers
  // will be colocated in the same allocation (i.e buffers for while result
  // will share an allocation with buffers related to that same while
  // instruction: init operand, condition/body parameter and body result).
  static StatusOr<std::unique_ptr<BufferAssignment>> Run(
      const HloModule* module, std::unique_ptr<HloOrdering> hlo_ordering,
      LogicalBuffer::SizeFunction buffer_size, bool colocate_related_buffers,
      const std::vector<const HloInstruction*>* hlos_to_allocate = nullptr);

  // Overload of Run which uses ShapeUtil::ByteSizeOf to determine buffer size
  // and assigns buffers to all HLO instructions in the module.
  static StatusOr<std::unique_ptr<BufferAssignment>> Run(
      const HloModule* module, std::unique_ptr<HloOrdering> hlo_ordering,
      int64 pointer_size);

 private:
  explicit BufferAssigner(LogicalBuffer::SizeFunction buffer_size,
                          bool colocate_related_buffers)
      : buffer_size_(std::move(buffer_size)),
        colocate_related_buffers_(colocate_related_buffers) {}
  virtual ~BufferAssigner() = default;

  // Create a buffer assignment.
  StatusOr<std::unique_ptr<BufferAssignment>> CreateAssignment(
      const HloModule* module, std::unique_ptr<HloOrdering> hlo_ordering,
      const std::vector<const HloInstruction*>* hlos_to_allocate = nullptr);

  // Assigns buffers to the instructions in the given computation. "assignment"
  // is modified to reflect the new buffer assignments. If is_thread_local is
  // true, then all assigned buffers have the is_thread_local flag set to
  // true. If hlos_to_allocate is not null it indicates which HLOs to include in
  // buffer assignment. If null, all instructions in the computation are
  // included.
  tensorflow::Status AssignBuffersForComputation(
      const HloComputation* computation, bool is_thread_local,
      const tensorflow::gtl::FlatSet<const HloInstruction*>* hlos_to_allocate,
      const tensorflow::gtl::FlatSet<const LogicalBuffer*>& colocated_buffers,
      const tensorflow::gtl::FlatSet<BufferAllocation::Index>&
          colocated_allocations,
      BufferAssignment* assignment);

  // Tries to assign the given instruction to the given buffer. Returns if the
  // assignment was successful.
  bool MaybeAssignBuffer(BufferAllocation* allocation,
                         const LogicalBuffer& buffer,
                         BufferAssignment* assignment);

  // Colocated buffers are logical buffers from different computations which
  // alias. Explicitly handling these colocated buffers is necessary because
  // points-to analysis is computation level scope and does not recognize
  // aliasing across computations (b/32491382).
  using ColocatedBufferSet = tensorflow::gtl::FlatSet<const LogicalBuffer*>;

  // Returns a vector of ColocatedBufferSet objects, where each
  // ColocatedBufferSet aggregates a set of related LogicalBuffers from 'module'
  // which should be colocated in the same buffer allocation.
  void BuildColocatedBufferSets(
      const HloModule* module, const TuplePointsToAnalysis& points_to_analysis,
      std::vector<ColocatedBufferSet>* colocated_buffer_sets);

  // For each buffer set in 'colocated_buffer_sets', assigns all buffers in the
  // same set to the same buffer allocation in 'assignment'.
  void AssignColocatedBufferSets(
      const std::vector<ColocatedBufferSet>& colocated_buffer_sets,
      BufferAssignment* assignment,
      tensorflow::gtl::FlatSet<const LogicalBuffer*>* colocated_buffers,
      tensorflow::gtl::FlatSet<BufferAllocation::Index>* colocated_allocations);

  // Adds the 'colocated_set' of buffers to 'colocated_buffer_sets', maintaining
  // the invariant that all sets in 'colocated_buffer_sets' are disjoint.
  void AddSetToColocatedBufferSets(
      const std::vector<const LogicalBuffer*>& colocated_set,
      std::vector<ColocatedBufferSet>* colocated_buffer_sets);

  const HloModule* module_;

  // Function which returns the buffer size for a given shape.
  LogicalBuffer::SizeFunction buffer_size_;

  // Indicates whether related buffers should share the same buffer allocation.
  const bool colocate_related_buffers_;

  TF_DISALLOW_COPY_AND_ASSIGN(BufferAssigner);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_BUFFER_ASSIGNMENT_H_

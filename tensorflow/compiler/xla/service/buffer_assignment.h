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
#include <vector>

#include "tensorflow/compiler/xla/service/buffer_liveness.h"
#include "tensorflow/compiler/xla/service/heap_simulator.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// This class abstracts an allocation of contiguous memory which can hold the
// values described by LogicalBuffers. Each LogicalBuffer occupies a sub-range
// of the allocation, represented by a Slice. A single BufferAllocation may hold
// LogicalBuffers with disjoint liveness, which may have overlapping Slices. A
// single BufferAllocation may also hold LogicalBuffers with overlapping
// liveness, which must have disjoint Slices.
//
// The abstraction includes information required by the backends for allocation,
// use, and deallocation of the buffer. This includes the LogicalBuffers which
// are held in this allocation through the execution of the computation.
class BufferAllocation {
 public:
  // Holds a unique identifier for each allocation. Values are assigned
  // contiguously and can be used as array indexes.
  using Index = int64;

  BufferAllocation(Index index, int64 size, bool is_thread_local,
                   bool is_reusable, LogicalBuffer::Color color)
      : index_(index),
        size_(size),
        is_thread_local_(is_thread_local),
        is_reusable_(is_reusable),
        color_(color) {}
  ~BufferAllocation() {}

  // Returns the index of this allocation.
  Index index() const { return index_; }

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

  // Returns whether this allocation is assigned a LogicalBuffer which may
  // be live out of the entry computation.
  bool maybe_live_out() const { return maybe_live_out_; }

  // Returns the size of the allocation. Necessarily this must be at least as
  // large as any LogicalBuffer assigned to this allocation.
  int64 size() const { return size_; }

  // Returns the color of the allocation. Only logical buffers with a matching
  // color can reside in this allocation.
  LogicalBuffer::Color color() const { return color_; }

  struct OffsetSize {
    int64 offset = 0;
    int64 size = 0;
  };

  // Access to the logical buffers assigned to this allocation, and their
  // associated logical offsets and sizes.
  const tensorflow::gtl::FlatMap<const LogicalBuffer*, OffsetSize>&
  assigned_buffers() const {
    return assigned_buffers_;
  }

  // A Slice represents a contiguous portion of a memory allocation. It is used
  // to identify the memory range that a LogicalBuffer corresponds to.
  class Slice {
   public:
    Slice() {}
    Slice(const BufferAllocation* allocation, int64 offset, int64 size)
        : allocation_(allocation), offset_(offset), size_(size) {}

    const BufferAllocation* allocation() const { return allocation_; }
    Index index() const { return allocation_->index(); }
    int64 offset() const { return offset_; }
    int64 size() const { return size_; }

    bool operator==(const Slice& other) const {
      return index() == other.index() && offset_ == other.offset_ &&
             size_ == other.size_;
    }
    bool operator!=(const Slice& other) const { return !(*this == other); }
    bool operator<(const Slice& other) const {
      if (index() != other.index()) return index() < other.index();
      if (offset_ != other.offset_) return offset_ < other.offset_;
      return size_ < other.size_;
    }

    // Returns true iff this slice's memory range has a non-empty intersection
    // with the other slice's memory range.
    bool OverlapsWith(const Slice& other) const {
      const int64 end = offset_ + size_;
      const int64 other_end = other.offset_ + other.size_;
      return index() == other.index() && offset_ < other_end &&
             end > other.offset_;
    }

    struct Hasher {
      size_t operator()(Slice s) const;
    };

    string ToString() const;

   private:
    const BufferAllocation* allocation_ = nullptr;
    int64 offset_ = 0;
    int64 size_ = 0;
  };

  // GetSlice returns the Slice of contiguous memory that holds the value
  // described by the given 'buffer'.
  // REQUIRES: 'buffer' must be assigned to this allocation.
  Slice GetSlice(const LogicalBuffer& buffer) const;

  string ToString() const;
  BufferAllocationProto ToProto() const;

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
  // Only BufferAssigner and BufferAssignment can modify BufferAllocation.
  friend class BufferAssigner;
  friend class BufferAssignment;

  // Adds a LogicalBuffer to the set assigned to this buffer.
  void AddAssignment(const LogicalBuffer& buffer, int64 offset, int64 size);

  void set_entry_computation_parameter(int64 parameter_number) {
    is_entry_computation_parameter_ = true;
    parameter_number_ = parameter_number;
  }
  void set_maybe_live_out(bool value) { maybe_live_out_ = value; }
  void set_index(Index index) { index_ = index; }
  void set_size(int64 size) { size_ = size; }

  // The index of the allocation in the BufferAssignment.
  Index index_;

  // Size of the allocation in bytes.
  int64 size_;

  // Whether this buffer needs to be thread-local.
  bool is_thread_local_;

  // Whether this buffer is usable by more than one logical buffer.
  bool is_reusable_;

  // Color of the allocation.
  LogicalBuffer::Color color_;

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

  // Mapping from the set of buffers assigned to this allocation to their
  // logical offsets and sizes.
  tensorflow::gtl::FlatMap<const LogicalBuffer*, OffsetSize> assigned_buffers_;
};

// Add stream operators for nicer output of CHECK/RET_CHECK failures.
std::ostream& operator<<(std::ostream& out, const BufferAllocation& s);
std::ostream& operator<<(std::ostream& out, const BufferAllocation::Slice& s);

// This class encapsulates an assignment of the LogicalBuffers in an XLA
// module to a set of BufferAllocations.
class BufferAssignment {
 public:
  // Returns the vector containing all buffer allocations in this assignment.
  const std::vector<BufferAllocation>& Allocations() const {
    return allocations_;
  }

  // Returns the total size allocation holding all temporary buffers.
  int64 temp_allocation_total_size() const {
    return temp_allocation_total_size_;
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

  // Builds and returns a vector containing the slices which might contain the
  // subvalue at the given index of given instruction.
  std::set<BufferAllocation::Slice> GetAllSlices(
      const HloInstruction* instruction, const ShapeIndex& index) const;

  // Convenience function which returns whether the top-level buffer of the
  // instruction (index == {}) is assigned an allocation.
  bool HasTopLevelAllocation(const HloInstruction* instruction) const;

  // Convenience function which returns the unique slice containing the buffer
  // at the given index of the given instruction. If a slice is not assigned or
  // the slice cannot be determined at compile time then an error is returned.
  StatusOr<BufferAllocation::Slice> GetUniqueSlice(
      const HloInstruction* instruction, const ShapeIndex& index) const;
  // Like GetUniqueSlice but fixes the index to the top-level of the shape
  // (index = {}).
  StatusOr<BufferAllocation::Slice> GetUniqueTopLevelSlice(
      const HloInstruction* instruction) const;
  // Like GetUniqueTopLevelSlice but returns the slice for the output of the
  // entry computation of the HLO module (ie, the result of the XLA
  // computation).
  StatusOr<BufferAllocation::Slice> GetUniqueTopLevelOutputSlice() const;

  // Returns the set LogicalBuffers which may be the source of the value at the
  // given index and instruction.
  const std::vector<const LogicalBuffer*>& GetSourceBuffers(
      const HloInstruction* instruction, const ShapeIndex& index) const {
    return GetPointsToSet(instruction).element(index);
  }

  // Returns true if 'hlo_a{shape_index_a}' and 'hlo_b{shape_index_b}'
  // share the same BufferAllocation::Slice.
  // Returns false otherwise.
  // REQUIRES: BufferAssignment assigned allocations to both instructions.
  bool SharesSliceAtIndex(const HloInstruction* hlo_a,
                          const ShapeIndex& shape_index_a,
                          const HloInstruction* hlo_b,
                          const ShapeIndex& shape_index_b) const;

  // Returns the underlying points-to analysis used for this assignment.
  const TuplePointsToAnalysis& points_to_analysis() const {
    return liveness_->points_to_analysis();
  }

  // Returns the BufferLiveness object used to construct this assignment.
  const BufferLiveness& liveness() const { return *liveness_; }

  string ToString() const;
  BufferAssignmentProto ToProto() const;

  // Statistics for the assignment.  Values initialized to -1 are not always
  // collected; fragmentation is only collected for instructions that have a
  // sequential total ordering.
  struct Stats {
    int64 parameter_allocation_count = 0;
    int64 parameter_allocation_bytes = 0;
    int64 maybe_live_out_allocation_count = 0;
    int64 maybe_live_out_allocation_bytes = 0;
    int64 preallocated_temp_allocation_count = 0;
    int64 preallocated_temp_allocation_bytes = 0;
    int64 preallocated_temp_fragmentation_bytes = -1;
    int64 total_allocation_count = 0;
    int64 total_allocation_bytes = 0;
    int64 total_fragmentation_bytes = -1;

    string ToString() const;
  };
  const Stats& GetStats() const { return stats_; }

 private:
  // Only BufferAssigner can build or modify BufferAssignments.
  friend class BufferAssigner;

  explicit BufferAssignment(const HloModule* module,
                            std::unique_ptr<BufferLiveness> liveness,
                            int64 alignment,
                            LogicalBuffer::SizeFunction buffer_size)
      : module_(module),
        liveness_(std::move(liveness)),
        alignment_(alignment),
        buffer_size_(std::move(buffer_size)) {}

  // Creates and returns a new BufferAllocation, with no assigned
  // LogicalBuffers. Ownership is maintained internally.
  BufferAllocation* NewEmptyAllocation(int64 size, bool is_thread_local,
                                       bool is_reusable,
                                       LogicalBuffer::Color color);

  // Helper that calls NewEmptyAllocation and AddAssignment in one call,
  // creating an allocation containing a single LogicalBuffer.
  BufferAllocation* NewAllocation(const LogicalBuffer& buffer, int64 size,
                                  bool is_thread_local, bool is_reusable);

  // Adds a LogicalBuffer to the set assigned to the given allocation.
  void AddAssignment(BufferAllocation* allocation, const LogicalBuffer& buffer,
                     int64 offset, int64 size);

  // Returns the HloModule used to construct this assignment.
  const HloModule& module() const { return *module_; }

  // Convenience function which returns the PointsToSet for the given
  // instruction. Extracted from the liveness object.
  const PointsToSet& GetPointsToSet(const HloInstruction* instruction) const;

  // Mutable accessors for allocations.
  BufferAllocation* GetMutableAssignedAllocation(const LogicalBuffer& buffer);
  BufferAllocation* GetMutableAllocation(BufferAllocation::Index index);

  // Combines allocations of temporary buffers into one big BufferAllocation.
  void CombineTempAllocations();

  // Computes stats for the assignment, to be retrieved by GetStats.
  Status ComputeSummaryStats();

  // The vector of buffer allocations. Indexed by BufferAllocation::Index.
  std::vector<BufferAllocation> allocations_;

  // The total size of all temporary buffers.
  int64 temp_allocation_total_size_ = 0;

  // Maps Buffers to the index of the BufferAllocation which holds the buffer.
  tensorflow::gtl::FlatMap<const LogicalBuffer*, BufferAllocation::Index>
      allocation_index_for_buffer_;

  const HloModule* module_;
  const std::unique_ptr<BufferLiveness> liveness_;
  const int64 alignment_;

  // Function which returns the buffer size for a given logical buffer (shape).
  LogicalBuffer::SizeFunction buffer_size_;

  Stats stats_;
  std::vector<HeapSimulatorTrace> heap_simulator_traces_;

  TF_DISALLOW_COPY_AND_ASSIGN(BufferAssignment);
};

// A class which constructs a buffer assignment.
class BufferAssigner {
 public:
  // Build and return a BufferAssignment for the given module. The given
  // HloOrdering is used to determine buffer liveness. buffer_size is a function
  // which returns the size of a LogicalBuffer. Alignment is the minimum
  // alignment of any buffer. allow_input_output_aliasing specifies whether
  // input buffer are allowed to be reused as outbut buffers by the client code.
  static StatusOr<std::unique_ptr<BufferAssignment>> Run(
      const HloModule* module, std::unique_ptr<HloOrdering> hlo_ordering,
      LogicalBuffer::SizeFunction buffer_size, int64 alignment,
      bool allow_input_output_aliasing = false,
      TuplePointsToAnalysis::Colorer colorer =
          TuplePointsToAnalysis::DefaultColorer());

 private:
  BufferAssigner(int64 alignment, bool allow_input_output_aliasing,
                 TuplePointsToAnalysis::Colorer colorer)
      : alignment_(alignment),
        allow_input_output_aliasing_(allow_input_output_aliasing),
        colorer_(colorer) {}
  virtual ~BufferAssigner() = default;

  // Create a buffer assignment.
  StatusOr<std::unique_ptr<BufferAssignment>> CreateAssignment(
      const HloModule* module, std::unique_ptr<HloOrdering> hlo_ordering,
      LogicalBuffer::SizeFunction buffer_size);

  // Assigns buffers to the instructions in the given computation. "assignment"
  // is modified to reflect the new buffer assignments. If is_thread_local is
  // true, then all assigned buffers have the is_thread_local flag set to
  // true.
  Status AssignBuffersForComputation(
      const HloComputation* computation, bool is_thread_local,
      const tensorflow::gtl::FlatSet<const LogicalBuffer*>& colocated_buffers,
      const tensorflow::gtl::FlatSet<BufferAllocation::Index>&
          colocated_allocations,
      tensorflow::gtl::FlatMap<const HloComputation*,
                               tensorflow::gtl::FlatSet<const LogicalBuffer*>>*
          buffers_to_assign_sequentially,
      BufferAssignment* assignment);

  // Assigns 'buffers_to_assign_sequentially' using heap simulation, assuming
  // the HLO instructions will be executed in the sequential order given by
  // assignment->liveness().hlo_ordering().SequentialOrder. If
  // 'run_whole_module_heap_simulation' is true, the heap simulation will be run
  // assuming all global computations are sequentially ordered.
  Status AssignBuffersWithSequentialOrdering(
      const tensorflow::gtl::FlatMap<
          const HloComputation*,
          tensorflow::gtl::FlatSet<const LogicalBuffer*>>&
          buffers_to_assign_sequentially,
      bool run_whole_module_heap_simulation, BufferAssignment* assignment);

  // Uses the results of the heap simulator to create a single allocation, with
  // LogicalBuffers packed to specific offsets.
  void AssignBuffersFromHeapSimulator(const HeapSimulator::Result& result,
                                      BufferAssignment* assignment,
                                      LogicalBuffer::Color color);

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
      const HloModule* module, const BufferLiveness& buffer_liveness,
      const LogicalBuffer::SizeFunction& buffer_size,
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

  // Conceptually the same as AddSetToColocatedBufferSets, but specific to the
  // colocated buffers for while instructions.
  void AddWhileSetToColocatedBufferSets(
      const std::vector<const LogicalBuffer*>& colocated_set,
      const LogicalBuffer* while_init_buffer, const HloInstruction* while_hlo,
      const HloComputation& computation, const BufferLiveness& buffer_liveness,
      const LogicalBuffer::SizeFunction& buffer_size,
      std::vector<ColocatedBufferSet>* colocated_buffer_sets);

  // Split a set of buffers into several sets, each of which contains buffers
  // colored with the same color.
  tensorflow::gtl::FlatMap<LogicalBuffer::Color,
                           tensorflow::gtl::FlatSet<const LogicalBuffer*>,
                           LogicalBuffer::Color::Hasher>
  SplitBuffersByColor(
      const tensorflow::gtl::FlatSet<const LogicalBuffer*>& buffers);

  // Minimum alignment of any buffer.
  int64 alignment_;

  // If true, buffer assignments assumes that input parameter buffers and output
  // buffers can be shared if their sizes match.
  bool allow_input_output_aliasing_;

  // Functor used to assign colors to newly allocated logical buffers.
  TuplePointsToAnalysis::Colorer colorer_;

  TF_DISALLOW_COPY_AND_ASSIGN(BufferAssigner);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_BUFFER_ASSIGNMENT_H_

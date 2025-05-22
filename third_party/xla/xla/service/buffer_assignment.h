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

#ifndef XLA_SERVICE_BUFFER_ASSIGNMENT_H_
#define XLA_SERVICE_BUFFER_ASSIGNMENT_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/buffer_assignment.pb.h"
#include "xla/service/buffer_value.h"
#include "xla/service/call_graph.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_value.h"
#include "xla/service/logical_buffer.h"
#include "xla/service/memory_space_assignment/memory_space_assignment.h"
#include "xla/shape_util.h"
#include "tsl/platform/logging.h"

namespace xla {

// Walk the call graph of the HLO module and place each computation into either
// thread_local_computations or global_computations depending upon whether the
// computation requires thread-local allocations or global allocations. The
// elements in thread_local_computations and global_computations are in post
// order (if computation A has an instruction which calls computation B, then A
// will appear after B in the vector).
absl::Status GatherComputationsByAllocationType(
    const HloModule* module,
    std::vector<const HloComputation*>* thread_local_computations,
    std::vector<const HloComputation*>* global_computations);

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
  using Index = int64_t;

  BufferAllocation(Index index, int64_t size, LogicalBuffer::Color color)
      : index_(index), size_(size), color_(color) {}

  // Returns the index of this allocation.
  Index index() const { return index_; }

  // Whether this allocation is used in a parallel calling context such as
  // inside of a map or reduce computation. Such allocations need to be thread
  // local.
  bool is_thread_local() const { return is_thread_local_; }
  void set_is_thread_local(bool is_thread_local) {
    is_thread_local_ = is_thread_local;
  }

  // Whether this allocation can be used by more than one logical buffer.
  bool is_reusable() const {
    // We do not reuse thread-local buffers for now, because they are
    // dynamically allocated and their lifetimes are hard to compute.
    //
    // TODO(b/34669761): Don't reuse tuple buffers because the GPU backend
    // assumes longer buffer liveness than indicated by the analysis.
    return !is_thread_local() && !is_tuple();
  }

  // Whether this allocation is read-only i.e. backed by memory we cannot write
  // to.
  bool is_readonly() const {
    // Entry parameters are generally readonly, except when they are aliased
    // with any output.
    return (is_entry_computation_parameter() &&
            !is_parameter_aliased_with_output_) ||
           is_constant();
  }

  bool is_tuple() const { return is_tuple_; }
  void set_is_tuple(bool is_tuple) { is_tuple_ = is_tuple; }

  // Whether this allocation holds a LogicalBuffer from a parameter of the entry
  // computation. These buffers have lifetimes which may be longer than the
  // XLA computation.
  bool is_entry_computation_parameter() const {
    return is_entry_computation_parameter_;
  }

  bool is_parameter_aliased_with_output() const {
    return is_parameter_aliased_with_output_;
  }

  // Whether this allocation holds a constant.  On the CPU and GPU backends
  // constant allocations are not allocated dynamically, instead we resolve
  // references to these buffer allocations to a global in the readonly section
  // of the binary.
  bool is_constant() const { return is_constant_; }

  // If this allocation holds a Buffer from a parameter of the entry
  // computation, this methods returns the parameter number. CHECKs otherwise.
  int64_t parameter_number() const {
    CHECK(is_entry_computation_parameter_);
    return parameter_number_;
  }

  // If this allocation is for a parameter of the entry computation, this
  // function returns which subshape of the parameter the allocation is for.
  const ShapeIndex& param_shape_index() const {
    CHECK(is_entry_computation_parameter_);
    return param_shape_index_;
  }

  // Returns whether this allocation is assigned a LogicalBuffer which may
  // be live out of the entry computation.
  bool maybe_live_out() const { return maybe_live_out_; }

  void set_maybe_live_out(bool value) { maybe_live_out_ = value; }

  // Returns the size of the allocation. Necessarily this must be at least as
  // large as any LogicalBuffer assigned to this allocation.
  int64_t size() const { return size_; }

  // Returns the color of the allocation. Only logical buffers with a matching
  // color can reside in this allocation.
  LogicalBuffer::Color color() const { return color_; }

  void set_color(LogicalBuffer::Color color) { color_ = color; }
  struct OffsetSize {
    int64_t offset = 0;
    int64_t size = 0;
  };

  // Access to the logical buffers assigned to this allocation, and their
  // associated logical offsets and sizes.
  const absl::flat_hash_map<const HloValue*, OffsetSize>& assigned_buffers()
      const {
    return assigned_buffers_;
  }

  // A Slice represents a contiguous portion of a memory allocation. It is used
  // to identify the memory range that a LogicalBuffer corresponds to.
  class Slice {
   public:
    Slice() = default;
    Slice(const BufferAllocation* allocation, int64_t offset, int64_t size)
        : allocation_(allocation), offset_(offset), size_(size) {}

    const BufferAllocation* allocation() const { return allocation_; }
    Index index() const { return allocation_->index(); }
    int64_t offset() const { return offset_; }
    int64_t size() const { return size_; }

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
      const int64_t end = offset_ + size_;
      const int64_t other_end = other.offset_ + other.size_;
      return index() == other.index() && offset_ < other_end &&
             end > other.offset_;
    }

    template <typename H>
    friend H AbslHashValue(H h, const Slice& s) {
      return H::combine(std::move(h), s.index(), s.offset(), s.size());
    }

    std::string ToString() const;

    absl::StatusOr<xla::buffer_assignment::BufferAllocationSliceProto> ToProto()
        const;

    static absl::StatusOr<BufferAllocation::Slice> FromProto(
        const xla::buffer_assignment::BufferAllocationSliceProto& proto,
        absl::Span<const BufferAllocation> buffer_allocations);

   private:
    const BufferAllocation* allocation_ = nullptr;
    int64_t offset_ = 0;
    int64_t size_ = 0;
  };

  // GetSlice returns the Slice of contiguous memory that holds the value
  // described by the given 'buffer'.
  // REQUIRES: 'buffer' must be assigned to this allocation.
  Slice GetSlice(const HloValue& buffer) const;

  std::string ToString() const;
  std::string ToShortString(bool human_readable_size = false) const;
  std::string ValuesToString() const;

  // The function returns memory usage report for the values belonging to the
  // buffer allocation. The values are grouped by their offset in the
  // allocation. The groups are sorted by the max size(Z-A) of the values in the
  // group. Percentile and more_than_k are used to control the number of groups
  // being reported.
  std::string MemoryUsageReport(const std::string& prefix,
                                float percentile = 0.05,
                                int64_t more_than_k = 50) const;

  BufferAllocationProto ToProto() const;
  static BufferAllocation FromProto(const BufferAllocationProto&);

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
           !is_thread_local() &&
           // Constant buffers are allocated as global values.
           !is_constant();
  }

  // Add a heap trace which was used to assign slices to logical buffers in this
  // allocation. A single BufferAllocation may include multiple heap traces
  // in the case of the temporary block where there is a heap trace per
  // computation.
  void AddHeapTrace(const HeapSimulatorTrace& heap_trace) {
    heap_traces_.push_back(heap_trace);
    heap_traces_.back().set_buffer_allocation_index(index());
  }

  // Return the set of heap traces used to assign slices to logical buffers in
  // this allocation.
  std::vector<HeapSimulatorTrace> HeapTraces() const { return heap_traces_; }

  // Returns the LogicalBuffers which are live at the point of peak memory usage
  // for this allocation. The point of peak memory usage is the point at which
  // the total size of all live logical buffers is maximal. If peak memory is
  // reached at multiple points, the set of logical buffers live at the earliest
  // maximal point is returned. The vector is stably sorted by
  // BufferValue::Index.
  const std::vector<const HloValue*>& PeakMemoryLogicalBuffers() const {
    return peak_buffers_;
  }

  // Get the number of bytes lost to fragmentation. This is equal to the
  // difference between the size of the allocation and the size of the maximal
  // live set.
  int64_t fragmentation_bytes() const { return fragmentation_bytes_; }

  bool operator==(const BufferAllocation& other) const {
    return index_ == other.index_;
  }
  bool operator!=(const BufferAllocation& other) const {
    return !(*this == other);
  }
  bool operator<(const BufferAllocation& other) const {
    return index() < other.index();
  }

  void set_entry_computation_parameter(int64_t parameter_number,
                                       ShapeIndex param_shape_index,
                                       bool parameter_aliased_with_output) {
    is_entry_computation_parameter_ = true;
    is_parameter_aliased_with_output_ = parameter_aliased_with_output;
    parameter_number_ = parameter_number;
    param_shape_index_ = std::move(param_shape_index);
  }

  void set_constant(bool is_constant) { is_constant_ = is_constant; }

 private:
  // Only BufferAssigner and BufferAssignment can modify BufferAllocation.
  friend class BufferAssigner;
  friend class BufferAssignment;

  // Adds a LogicalBuffer to the set assigned to this buffer.
  void AddAssignment(const HloValue& buffer, int64_t offset, int64_t size);

  void set_index(Index index) { index_ = index; }
  void set_size(int64_t size) { size_ = size; }

  // The index of the allocation in the BufferAssignment.
  Index index_;

  // Size of the allocation in bytes.
  int64_t size_;

  // Whether this buffer needs to be thread-local.
  bool is_thread_local_ = false;

  // Whether this buffer holds a tuple.
  bool is_tuple_ = false;

  // Color of the allocation.
  LogicalBuffer::Color color_;

  // Whether this allocation holds an entry computation parameter. Entry
  // computation parameters are special because they have lifetimes which may
  // outlast the computation.
  bool is_entry_computation_parameter_ = false;

  // Whether this entry computation parameter is aliased with output.
  bool is_parameter_aliased_with_output_ = false;

  // If this allocation holds an entry computation parameter, this field
  // indicates the index (starting from 0) of the parameter.
  int64_t parameter_number_ = 0;

  // If this buffer is for an entry computation parameter, which subshape of the
  // parameter is it for?
  ShapeIndex param_shape_index_;

  // Whether the allocation contains a LogicalBuffer which may be live-out of
  // the entry computation. Note that this flag is conservatively computed by
  // TuplePointsToAnalysis.  That is, an allocation marked `maybe_live_out_`
  // might not actually escape.
  bool maybe_live_out_ = false;

  // See comment on the is_constant() accessor.
  bool is_constant_ = false;

  // Mapping from the set of buffers assigned to this allocation to their
  // logical offsets and sizes.
  absl::flat_hash_map<const HloValue*, OffsetSize> assigned_buffers_;

  int64_t fragmentation_bytes_ = 0;
  std::vector<HeapSimulatorTrace> heap_traces_;

  // Set of buffers live at the point of peak memory usage for this allocation.
  std::vector<const HloValue*> peak_buffers_;
};

// Add stream operators for nicer output of CHECK/RET_CHECK failures.
std::ostream& operator<<(std::ostream& out, const BufferAllocation& buffer);
std::ostream& operator<<(std::ostream& out, const BufferAllocation::Slice& s);

// This class encapsulates an assignment of the LogicalBuffers in an XLA
// module to a set of BufferAllocations.
class BufferAssignment {
 public:
  // This is a think wrapper around BufferIsolationConfig. Please see the
  // documentation for BufferIsolationConfig for details on how buffer isolation
  // works. hlo_value_compare is the concrete implementation of the HloValue
  // comparison that uses the isolation_order_salt value in the
  // BufferIsolationConfig.
  struct BufferIsolationOptions {
    std::function<bool(const HloValue*, const HloValue*)> hlo_value_compare;
    buffer_assignment::BufferIsolationConfig config;
  };

  // Returns the vector containing all buffer allocations in this assignment.
  const std::vector<BufferAllocation>& Allocations() const {
    return allocations_;
  }

  // Returns the total size allocation holding all temporary buffers.
  int64_t temp_allocation_total_size() const {
    return temp_allocation_total_size_;
  }

  uint64_t multiheap_size_constraint_per_heap() const {
    return multiheap_size_constraint_per_heap_;
  }

  // Returns whether the given buffer has been assigned an allocation.
  bool HasAllocation(const HloValue& value) const;

  // Returns whether the given (logical) buffer with the id has been assigned an
  // allocation.
  bool HasAllocation(HloValue::Id value_id) const;

  bool HasAllocation(const HloBuffer& buffer) const;

  // Returns the allocation that a particular LogicalBuffer has been assigned
  // to. CHECKs if buffer has not been assigned an allocation.
  const BufferAllocation& GetAssignedAllocation(const HloValue& value) const;

  const BufferAllocation& GetAssignedAllocation(
      const HloBuffer& hlo_buffer) const;

  // Returns the allocation with the given index. CHECKs if no allocation exists
  // with the given index.
  const BufferAllocation& GetAllocation(BufferAllocation::Index index) const;

  // Returns the allocation with the given instruction and shape index. nullptr
  // if no allocation exists.
  const BufferAllocation* GetInstructionAllocation(
      const HloInstruction* hlo, const ShapeIndex& shape_index) const;

  // Builds and returns a vector containing the slices which might contain the
  // subvalue at the given index of given instruction.
  std::set<BufferAllocation::Slice> GetAllSlices(
      const HloInstruction* instruction, const ShapeIndex& index) const;

  // Convenience function which returns whether the buffer of the
  // instruction at the given index is assigned an allocation.
  bool HasAllocationAt(const HloInstruction* instruction,
                       const ShapeIndex& index) const;

  // Convenience function which returns whether the top-level buffer of the
  // instruction (index == {}) is assigned an allocation.
  bool HasTopLevelAllocation(const HloInstruction* instruction) const;

  // Convenience function which returns the unique slice containing the buffer
  // at the given index of the given instruction. If a slice is not assigned or
  // the slice cannot be determined at compile time then an error is returned.
  absl::StatusOr<BufferAllocation::Slice> GetUniqueSlice(
      const HloInstruction* instruction, const ShapeIndex& index) const;
  // Like GetUniqueSlice but fixes the index to the top-level of the shape
  // (index = {}).
  absl::StatusOr<BufferAllocation::Slice> GetUniqueTopLevelSlice(
      const HloInstruction* instruction) const;
  // Like GetUniqueTopLevelSlice but returns the slice for the output of the
  // entry computation of the HLO module (ie, the result of the XLA
  // computation).
  absl::StatusOr<BufferAllocation::Slice> GetUniqueTopLevelOutputSlice() const;

  // Returns the set BufferValues which may be the source of the value at the
  // given index and instruction.
  const std::vector<const HloValue*>& GetSourceBuffers(
      const HloInstruction* instruction, const ShapeIndex& index) const {
    return dataflow_analysis().GetValueSet(instruction, index).values();
  }

  // Returns true if 'hlo_a{shape_index_a}' and 'hlo_b{shape_index_b}'
  // share the same BufferAllocation::Slice.
  // Returns false otherwise.
  // REQUIRES: BufferAssignment assigned allocations to both instructions.
  bool SharesSliceAtIndex(const HloInstruction* hlo_a,
                          const ShapeIndex& shape_index_a,
                          const HloInstruction* hlo_b,
                          const ShapeIndex& shape_index_b) const;

  // Returns true if the top-level buffers of hlo_a and hlo_b are the same.
  // REQUIRES: HasTopLevelAllocation(hlo_a) && HasTopLevelAllocation(hlo_b).
  bool SharesTopLevelSlice(const HloInstruction* hlo_a,
                           const HloInstruction* hlo_b) const {
    return SharesSliceAtIndex(hlo_a, {}, hlo_b, {});
  }

  // Returns true if hlo_a and hlo_b both have at least one buffer assigned for
  // their top-level and each of their nested shape indices, and if hlo_a's
  // buffers are all different from hlo_b's buffers.
  bool HaveDisjointSlices(const HloInstruction* hlo_a,
                          const HloInstruction* hlo_b) const;

  const HloDataflowAnalysis& dataflow_analysis() const {
    return alias_analysis_->dataflow_analysis();
  }

  HloAliasAnalysis& alias_analysis() const { return *alias_analysis_; }

  const HloOrdering& hlo_ordering() const { return *hlo_ordering_; }

  // Returns the HloLiveRange object used to construct this assignment.
  const HloLiveRange& hlo_live_range() const { return *hlo_live_range_; }

  // Is in use by many compilers to dump the buffer-assignment info.
  std::string ToString() const;

  // Returns a memory usage report with the list of buffer allocations ordered
  // by the size(Z-A) and the values assigned to each buffer allocation.
  std::string MemoryUsageReport(float percentile = 0.05,
                                int64_t more_than_k = 50) const;
  // Verbose string tailored to debugging OOMs, includes the Hlo op metadata for
  // every buffer associated with each allocation.
  std::string ToVerboseString(size_t max_buffers_to_show) const;

  // Is in use by tpu compiler to dump the buffer info.
  std::string BufferInfoString() const;

  // Convert BufferAssignment to or from a proto.
  BufferAssignmentProto ToProto() const;
  static absl::StatusOr<std::unique_ptr<BufferAssignment>> FromProto(
      const BufferAssignmentProto& proto, const HloModule* module,
      BufferValue::SizeFunction buffer_size,
      HloDataflowAnalysis::CanShareBuffer can_share_buffer);

  // Returns string representation of buffer assignment statistics. Also
  // calculates and returns the total fragmentation if
  // report_total_fragmentation is true.
  std::string StatsString(bool report_total_fragmentation = false) const;

  // Statistics for the assignment.  Values initialized to -1 are not always
  // collected; fragmentation is only collected for instructions that have a
  // sequential total ordering.
  struct Stats {
    int64_t parameter_allocation_count = 0;
    int64_t parameter_allocation_bytes = 0;
    int64_t constant_allocation_count = 0;
    int64_t constant_allocation_bytes = 0;
    int64_t maybe_live_out_allocation_count = 0;
    int64_t maybe_live_out_allocation_bytes = 0;
    int64_t preallocated_temp_allocation_count = 0;
    int64_t preallocated_temp_allocation_bytes = 0;
    int64_t preallocated_temp_fragmentation_bytes = -1;
    int64_t total_allocation_count = 0;
    int64_t total_allocation_bytes = 0;
  };
  const Stats& GetStats() const { return stats_; }

 private:
  // Only BufferAssigner can build or modify BufferAssignments.
  friend class BufferAssigner;

  BufferAssignment(const HloModule* module,
                   std::unique_ptr<HloOrdering> hlo_ordering,
                   BufferValue::SizeFunction buffer_size,
                   LogicalBuffer::AlignmentFunction color_alignment,
                   std::unique_ptr<HloAliasAnalysis> alias_analysis,
                   std::unique_ptr<HloLiveRange> hlo_live_range)
      : module_(module),
        hlo_ordering_(std::move(hlo_ordering)),
        buffer_size_(std::move(buffer_size)),
        color_alignment_(std::move(color_alignment)),
        alias_analysis_(std::move(alias_analysis)),
        hlo_live_range_(std::move(hlo_live_range)) {
    int32_t raw_value = module->config()
                            .debug_options()
                            .xla_multiheap_size_constraint_per_heap();
    // -1 means no constraint.
    multiheap_size_constraint_per_heap_ =
        (raw_value == -1) ? UINT64_MAX : raw_value;
  }

  // Creates and returns a new BufferAllocation, with no assigned
  // LogicalBuffers. Ownership is maintained internally.
  BufferAllocation* NewEmptyAllocation(int64_t size,
                                       LogicalBuffer::Color color);

  // Helper that calls NewEmptyAllocation and AddAssignment in one call,
  // creating an allocation containing a single LogicalBuffer.
  BufferAllocation* NewAllocation(const HloBuffer& buffer, int64_t size);

  // Adds a LogicalBuffer to the set assigned to the given allocation.
  void AddAssignment(BufferAllocation* allocation, const HloBuffer& buffer,
                     int64_t offset, int64_t size);

  void AddAssignment(BufferAllocation* allocation, const HloValue& value,
                     int64_t offset, int64_t size);

  // Returns the HloModule used to construct this assignment.
  const HloModule& module() const { return *module_; }

  // Mutable accessors for allocations.
  BufferAllocation* GetMutableAssignedAllocation(const HloBuffer& buffer);
  BufferAllocation* GetMutableAllocation(BufferAllocation::Index index);

  int64_t HloBufferSize(const HloBuffer& buffer) {
    auto iter = cached_buffer_sizes_.find(buffer.id());
    if (iter != cached_buffer_sizes_.end()) return iter->second;
    int64_t result = 0;
    for (const HloValue* value : buffer.values()) {
      result = std::max(result, buffer_size_(*value));
    }
    cached_buffer_sizes_.insert({buffer.id(), result});
    return result;
  }

  // Combines allocations of temporary buffers into one big BufferAllocation.
  void CombineTempAllocations(
      const absl::flat_hash_set<BufferValue::Color>& private_stack_colors,
      std::optional<BufferValue::Color> temp_buffer_color);

  // Computes stats for the assignment, to be retrieved by GetStats.
  void ComputeSummaryStats();

  // Calculates and returns the total fragmentation in bytes.
  absl::StatusOr<int64_t> ComputeTotalFragmentationBytes() const;

  // The vector of buffer allocations. Indexed by BufferAllocation::Index.
  std::vector<BufferAllocation> allocations_;

  // The total size of all temporary buffers.
  int64_t temp_allocation_total_size_ = 0;

  uint64_t multiheap_size_constraint_per_heap_;

  // Maps Buffers to the index of the BufferAllocation which holds the buffer.
  absl::flat_hash_map<const HloValue*, BufferAllocation::Index>
      allocation_index_for_value_;

  const HloModule* module_;

  const std::unique_ptr<HloOrdering> hlo_ordering_;

  // Function which returns the buffer size for a given logical buffer (shape).
  BufferValue::SizeFunction buffer_size_;

  // Function which returns the alignment for a given logical buffer color.
  LogicalBuffer::AlignmentFunction color_alignment_;

  std::unique_ptr<HloAliasAnalysis> alias_analysis_;

  std::unique_ptr<HloLiveRange> hlo_live_range_;

  Stats stats_;

  absl::flat_hash_map<HloBuffer::Id, int64_t> cached_buffer_sizes_;

  BufferAssignment(const BufferAssignment&) = delete;
  BufferAssignment& operator=(const BufferAssignment&) = delete;
};

// A class which constructs a buffer assignment.
class BufferAssigner {
 public:
  using Colorer =
      std::function<absl::Status(HloAliasAnalysis*, const HloOrdering&)>;
  using MustNotLiveOut = std::function<bool(
      const HloAliasAnalysis&, const HloInstruction*, const ShapeIndex&)>;
  using PrivateStacks = absl::flat_hash_map<BufferValue::Color,
                                            std::vector<const HloComputation*>>;

  static Colorer DefaultColorer() {
    return [](HloAliasAnalysis* alias_analysis, const HloOrdering&) {
      for (HloValue* value : alias_analysis->dataflow_analysis().values()) {
        const HloPosition& defining_position = value->defining_position();
        if (defining_position.shape().has_layout()) {
          value->set_color(BufferValue::Color(
              defining_position.shape().layout().memory_space()));
        } else {
          value->set_color(BufferValue::Color(0));
        }
      }
      return absl::OkStatus();
    };
  }

  // Returns false if a buffer cannot be assigned to given allocation.

  // Build and return a BufferAssignment for the given module. The given
  // HloOrdering is used to determine buffer liveness. buffer_size and
  // color_alignment are functions which returns the size and alignment of a
  // LogicalBuffer. If preset_assignments is provided, those pre-set assignment
  // offsets will be used. The caller guarantees that those assignments are
  // valid and they do not overwrite each other.
  static absl::StatusOr<std::unique_ptr<BufferAssignment>> Run(
      const HloModule* module, std::unique_ptr<HloOrdering> hlo_ordering,
      BufferValue::SizeFunction buffer_size,
      LogicalBuffer::AlignmentFunction color_alignment,
      bool allocate_buffers_for_constants = false,
      Colorer colorer = DefaultColorer(),
      std::optional<MustNotLiveOut> must_not_live_out = std::nullopt,
      HloDataflowAnalysis::CanShareBuffer can_share_buffer = nullptr,
      std::unique_ptr<memory_space_assignment::PresetAssignments>
          preset_assignments = {},
      const PrivateStacks& private_stacks = {},
      GlobalDecreasingSizeBestFitHeap<HloValue>::BufferIntervalCompare
          heap_buffer_interval_compare = nullptr,
      std::optional<BufferAssignment::BufferIsolationOptions>
          isolation_options = std::nullopt,
      std::optional<BufferValue::Color> temp_buffer_color = std::nullopt);

 private:
  BufferAssigner(bool allocate_buffers_for_constants, Colorer colorer,
                 std::optional<MustNotLiveOut> must_not_live_out,
                 std::unique_ptr<memory_space_assignment::PresetAssignments>
                     preset_assignments)
      : allocate_buffers_for_constants_(allocate_buffers_for_constants),
        colorer_(colorer),
        must_not_live_out_(must_not_live_out),
        preset_assignments_(std::move(preset_assignments)) {}
  virtual ~BufferAssigner() = default;

  // Create a buffer assignment.
  absl::StatusOr<std::unique_ptr<BufferAssignment>> CreateAssignment(
      const HloModule* module, std::unique_ptr<HloOrdering> hlo_ordering,
      BufferValue::SizeFunction buffer_size,
      LogicalBuffer::AlignmentFunction color_alignment,
      HloDataflowAnalysis::CanShareBuffer can_share_buffer,
      const PrivateStacks& private_stacks,
      GlobalDecreasingSizeBestFitHeap<HloValue>::BufferIntervalCompare
          heap_buffer_interval_compare,
      std::optional<BufferAssignment::BufferIsolationOptions> isolation_options,
      std::optional<BufferValue::Color> temp_buffer_color);

  // Assigns buffers to the instructions in the given computations. "assignment"
  // is modified to reflect the new buffer assignments. If is_thread_local is
  // true, then all assigned buffers have the is_thread_local flag set to
  // true.
  absl::Status AssignBuffersForComputations(
      const std::vector<const HloComputation*>& computations,
      bool is_thread_local,
      absl::flat_hash_map<const HloComputation*,
                          absl::flat_hash_set<const HloValue*>>*
          buffers_to_assign_sequentially,
      BufferAssignment* assignment);

  // Returns true if buffer's live range interferences with buffer2's.
  bool LiveRangeInterferes(const HloValue* buffer1, const HloValue* buffer2,
                           BufferAssignment* assignment);

  // Assigns pre-set assignments, if provided. These assignments will be added
  // to assigned_buffers and skip buffer allocation.
  absl::Status AssignPresetBuffers(
      absl::flat_hash_set<const HloBuffer*>* assigned_buffers,
      BufferAssignment* assignment);

  // Assigns a single hlo buffer to an HLO allocation.
  absl::Status AssignSingleHloBuffer(
      const HloBuffer* hlo_buffer, bool is_thread_local,
      absl::flat_hash_map<const HloComputation*,
                          absl::flat_hash_set<const HloValue*>>*
          buffers_to_assign_sequentially,
      std::vector<BufferAllocation::Index>* allocation_indices,
      BufferAssignment* assignment);

  // Assigns 'buffers_to_assign_sequentially' using heap simulation, assuming
  // the HLO instructions will be executed in the sequential order given by
  // assignment->liveness().hlo_ordering().SequentialOrder. If
  // 'run_whole_module_heap_simulation' is true, the heap simulation will be run
  // assuming all global computations are sequentially ordered.
  absl::Status AssignBuffersWithSequentialOrdering(
      const absl::flat_hash_map<const HloComputation*,
                                absl::flat_hash_set<const HloValue*>>&
          buffers_to_assign_sequentially,
      bool run_whole_module_heap_simulation, BufferAssignment* assignment,
      const PrivateStacks& private_stacks,
      GlobalDecreasingSizeBestFitHeap<HloValue>::BufferIntervalCompare
          heap_buffer_interval_compare,
      std::optional<BufferAssignment::BufferIsolationOptions>
          isolation_options);

  // Isolates the buffers packed by heap simulator using the provided isolation
  // options. Please see the documentation for BufferIsolationConfig for more
  // details.
  void IsolateHeapBuffers(
      std::optional<BufferAssignment::BufferIsolationOptions> isolation_options,
      const BufferAssignment* assignment, LogicalBuffer::Color color,
      HeapSimulator::Result<HloValue>& result) const;

  // Uses the results of the heap simulator to create a single allocation, with
  // LogicalBuffers packed to specific offsets.
  void AssignBuffersFromHeapSimulator(
      HeapSimulator::Result<HloValue>& result, BufferAssignment* assignment,
      LogicalBuffer::Color color,
      std::optional<BufferAssignment::BufferIsolationOptions>
          isolation_options);

  // Tries to assign the given instruction to the given buffer. Returns if the
  // assignment was successful.
  bool MaybeAssignBuffer(BufferAllocation* allocation, const HloBuffer& buffer,
                         BufferAssignment* assignment);

  // Split a set of buffers into several sets, each of which contains buffers
  // colored with the same color.
  absl::flat_hash_map<LogicalBuffer::Color,
                      absl::flat_hash_set<const HloValue*>>
  SplitBuffersByColor(
      const absl::flat_hash_set<const HloValue*>& buffers) const;

  // Split a set of buffers into several sets, each of which contains buffers
  // with defining instructions that are dominated by the given private stack
  // computation. This function CHECK-fails if there are outstanding buffers
  // that do not have a dominating private stack computation.
  absl::flat_hash_map<const HloComputation*,
                      absl::flat_hash_set<const HloValue*>>
  SplitBuffersByPrivateStackComputation(
      const absl::flat_hash_set<const HloValue*>& buffers,
      absl::Span<const HloComputation* const> private_stack_computations,
      const CallGraph& call_graph) const;

  // If true, allocate buffers for constant instructions.
  bool allocate_buffers_for_constants_;

  // Functor used to assign colors to newly allocated logical buffers.
  Colorer colorer_;

  // An optional function that returns true if the given instruction can't live
  // out of a computation.
  std::optional<MustNotLiveOut> must_not_live_out_;

  // Description of any buffer offsets that are already set by an earlier pass.
  std::unique_ptr<memory_space_assignment::PresetAssignments>
      preset_assignments_;

  BufferAssigner(const BufferAssigner&) = delete;
  BufferAssigner& operator=(const BufferAssigner&) = delete;
};

}  // namespace xla

#endif  // XLA_SERVICE_BUFFER_ASSIGNMENT_H_

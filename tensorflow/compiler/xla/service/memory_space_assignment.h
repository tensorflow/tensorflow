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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_H_

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>

// TODO(b/210891274): Use btree_map after build issue in Windows is resolved.
#if defined(__GNUC__) || defined(__clang__)
#include "absl/container/btree_map.h"
#endif
#include "absl/functional/function_ref.h"
#include "tensorflow/compiler/xla/service/heap_simulator.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/memory_space_assignment_repacking.h"

namespace xla {

namespace memory_space_assignment {
// Forward Declaration of Options.
class Options;

// This class contains pre-set assignments determined by memory space
// assignment. It contains two data structures: (1) a chunks vector that maps a
// defining HloPosition to a Chunk (offset and size), and (2) an assignment_info
// vector that maps the memory space to information like its allocated size and
// heap memory trace. If there is only one alternate memory space like there is
// currently, there will be one entry in assignment_info.
class PresetAssignments {
 public:
  // Contains per-memory-space information like the allocated size and heap
  // simulator trace.
  struct AssignmentInformation {
    int64_t size;
    HeapSimulatorTrace heap_simulator_trace;
  };

  PresetAssignments() = default;

  void add_chunk(const HloPosition& position,
                 const HeapSimulator::Chunk& chunk) {
    chunks_.emplace_back(position, chunk);
  }

  void add_scoped_allocation_chunk(HloInstruction* instruction,
                                   const HeapSimulator::Chunk& chunk) {
    scoped_allocation_chunks_.emplace_back(instruction, chunk);
  }

  AssignmentInformation* assignment_information_for_space(
      int64_t memory_space) {
    for (auto& space_and_info : assignment_info_) {
      if (space_and_info.first == memory_space) {
        return &space_and_info.second;
      }
    }
    assignment_info_.emplace_back(memory_space, AssignmentInformation());
    return &assignment_info_.back().second;
  }

  absl::Span<const std::pair<HloPosition, HeapSimulator::Chunk>> chunks()
      const {
    return chunks_;
  }

  absl::Span<const std::pair<HloInstruction*, HeapSimulator::Chunk>>
  scoped_allocation_chunks() const {
    return scoped_allocation_chunks_;
  }

  absl::Span<const std::pair<int64_t, AssignmentInformation>>
  assignment_informations() const {
    return assignment_info_;
  }

  // Get debugging information.
  std::string buffer_info_str() const { return buffer_info_str_; }
  std::string allocation_info_str() const { return allocation_info_str_; }

 private:
  std::vector<std::pair<HloPosition, HeapSimulator::Chunk>> chunks_;
  std::vector<std::pair<HloInstruction*, HeapSimulator::Chunk>>
      scoped_allocation_chunks_;
  std::vector<std::pair<int64_t, AssignmentInformation>> assignment_info_;
  std::string buffer_info_str_;
  std::string allocation_info_str_;
};

// A wrapper class around HloCostAnalysis with additional knowledge about the
// bandwidths of different memory spaces.
class MemorySpaceAssignmentCostAnalysis {
 public:
  // An optional Cache object may be provided to some of the methods below to
  // speed up the lookup.
  struct Cache {
    absl::flat_hash_map<const HloInstruction*, float> while_nest_multiplier;
  };

  // Function type that can be used to indicate which input/output values are in
  // the alternate memory.
  using IsInAlternateMemoryFun = absl::FunctionRef<bool(
      std::optional<int> /*operand_num*/, const ShapeIndex& /*index*/,
      const Shape& /*shape*/)>;

  virtual ~MemorySpaceAssignmentCostAnalysis() = default;

  static StatusOr<std::unique_ptr<MemorySpaceAssignmentCostAnalysis>> Create(
      const HloCostAnalysis& cost_analysis, const Options& options,
      const HloModule& module);

  const HloCostAnalysis& cost_analysis() const { return cost_analysis_; }

  // Returns a heuristic value that captures how much putting this tensor to the
  // alternate memory would help if the op is memory bound, or otherwise how far
  // off is the op to memory boundedness. The larger this number, the higher
  // priority it will be placed in the alternate memory.
  float GetAlternateMemoryBenefit(const HloInstruction& instruction,
                                  float elapsed_time_due_to_alternate_mem,
                                  Cache* cache = nullptr) const;
  // Like above, return the benefit of putting the output tensor in the
  // alternate memory.
  float GetAlternateMemoryBenefit(const HloPosition& position,
                                  Cache* cache = nullptr) const;
  // Like above, return the benefit of putting the input tensor in the alternate
  // memory.
  float GetAlternateMemoryBenefit(const HloUse& use,
                                  Cache* cache = nullptr) const;

  // Returns a heuristic value of memory boundedness for the given
  // BufferInterval.  The larger this number, the higher priority it will be
  // placed in the alternate memory.
  float GetMemoryBoundedness(
      const GlobalDecreasingSizeBestFitHeap<HloValue>::BufferInterval& interval,
      Cache* cache = nullptr) const;

  // Returns the elapsed time in seconds due to compute only.
  float GetInstructionElapsedDueToCompute(
      const HloInstruction& instruction) const;

  // Returns the elapsed time in seconds due to memory only. If
  // operands_in_alternate_mem or outputs_in_alternate_mem is provided, it will
  // assume that the corresponding operands or output will be in the alternate
  // memory space. This is useful for calculating the benefit of placing the
  // buffer in alternate memory.
  float GetInstructionElapsedDueToMemory(
      const HloInstruction& instruction,
      absl::Span<const std::pair<int64_t, ShapeIndex>>
          operands_in_alternate_mem = {},
      absl::Span<const ShapeIndex> outputs_in_alternate_mem = {}) const;

  // Like above, only the inputs/outputs indicated by is_in_alternate_mem are in
  // the alternate memory.
  float GetInstructionElapsedDueToMemory(
      const HloInstruction& instruction,
      IsInAlternateMemoryFun is_in_alternate_mem) const;

  // Returns the estimated elapsed duration of the instruction in seconds.  It
  // assumes all operands and outputs of the instruction are in the default
  // memory.
  virtual float GetInstructionElapsed(const HloInstruction& instruction) const;

  // Returns the estimated elapsed duration of the instruction in seconds.  It
  // assumes all operands and outputs of the instruction are in the default
  // memory, except for the operands and outputs specified to be in the
  // alternate memory.
  virtual float GetInstructionElapsedInAlternateMemory(
      const HloInstruction& instruction,
      absl::Span<const std::pair<int64_t, ShapeIndex>>
          operands_in_alternate_mem,
      absl::Span<const ShapeIndex> outputs_in_alternate_mem) const;

  // Like above, only the inputs/outputs indicated by is_in_alternate_mem are in
  // the alternate memory.
  float GetInstructionElapsedInAlternateMemory(
      const HloInstruction& instruction,
      IsInAlternateMemoryFun is_in_alternate_mem) const;

  // Returns the elapsed time it would take to asynchronously copy the shape
  // from default to alternate memory space (or vice versa).
  virtual float GetAsyncCopyElapsed(const Shape& shape) const;

  int64_t GetScheduleEndTime() const;

  // Returns the number of nested computation levels this instruction resides
  // in. If while_only is true, it returns the while loop nest level and 0
  // means the instruction is not in a while loop.
  int CalculateComputationNestLevel(const HloInstruction* instruction,
                                    bool while_only) const;

  const HloLiveRange& hlo_live_range() const { return *hlo_live_range_; }
  const Options& options() const { return options_; }

 protected:
  MemorySpaceAssignmentCostAnalysis(
      const HloCostAnalysis& cost_analysis, const Options& options,
      std::unique_ptr<HloAliasAnalysis> alias_analysis,
      std::unique_ptr<HloLiveRange> hlo_live_range,
      std::unique_ptr<CallGraph> call_graph)
      : cost_analysis_(cost_analysis),
        options_(options),
        alias_analysis_(std::move(alias_analysis)),
        hlo_live_range_(std::move(hlo_live_range)),
        call_graph_(std::move(call_graph)) {}

 private:
  const HloCostAnalysis& cost_analysis_;
  const Options& options_;
  std::unique_ptr<HloAliasAnalysis> alias_analysis_;
  std::unique_ptr<HloLiveRange> hlo_live_range_;
  std::unique_ptr<CallGraph> call_graph_;
};

// Abstract base class that memory space assignment uses to pick prefetch
// intervals.
class PrefetchIntervalPicker {
 public:
  PrefetchIntervalPicker() = default;
  virtual ~PrefetchIntervalPicker() = default;

  // Returns true if the buffer can be allocated in alternate memory space
  // without any copies (prefetches).
  virtual bool CanAllocateInAlternateMemoryNoCopy(const Shape& shape,
                                                  int64_t start_time,
                                                  int64_t end_time) const = 0;

  // Returns the preferred end time for an eviction that starts at a given time
  // and must end by the given end time.
  virtual int64_t PreferredEvictionEndTime(const Shape& shape,
                                           int64_t start_time,
                                           int64_t latest_end_time) const = 0;

  // Returns the latest time that a prefetch can start.
  virtual int64_t LatestPrefetchStartTime(const Shape& shape,
                                          int64_t start_time, int64_t end_time,
                                          const HloUse* use) const = 0;

  // Returns the preferred time that a prefetch can start.
  virtual int64_t PreferredPrefetchStartTime(
      const Shape& shape, int64_t earliest_prefetch_start_time,
      int64_t latest_prefetch_start_time, int64_t prefetch_end_time) const = 0;

  // Returns the latest time that a prefetch can end that is less than or equal
  // to proposed_prefetch_end_time.
  virtual int64_t LatestPrefetchEndTime(
      int64_t original_prefetch_end_time,
      int64_t proposed_prefetch_end_time) const {
    return proposed_prefetch_end_time;
  }

  // Returns the estimated end time of a prefetch that starts at the given time.
  virtual int64_t EstimatedPrefetchEndTime(const Shape& shape,
                                           int64_t start_time,
                                           int64_t end_time) const = 0;

  // Returns the elapsed time in seconds between the logical interval that
  // corresponds to the instruction schedule.
  virtual float GetLogicalIntervalElapsed(int64_t start_time,
                                          int64_t end_time) const = 0;

  // Begins the iterator for the first start time of the prefetch.
  virtual void Begin(const HloUse& use, int64_t start_time, int64_t end_time,
                     std::optional<int64_t> preferred_time) = 0;

  // Advances the start time of the prefetch and returns that value.
  virtual int64_t Next() = 0;

  // Returns true if the available prefetch intervals have been exhausted.
  virtual bool Done() const = 0;

  // Returns the latest time the prefetch interval picker will have pick.
  virtual int64_t latest_time() const = 0;

  // The retry number can be used to modify the interval picking policies. The
  // first attempt will have a retry_number of 0, then 1, etc.
  virtual void SetRetryNumber(int retry_number) {
    retry_number_ = retry_number;
  }
  int retry_number() const { return retry_number_; }

  // Returns a debug string for the current state of the prefetch interval
  // picker.
  virtual std::string ToDebugString() const = 0;

  // Returns a debug string for no-copy allocation.
  virtual std::string ToNoCopyDebugString(const Shape& shape,
                                          int64_t start_time,
                                          int64_t end_time) const = 0;

  // Prefetch interval pickers may return a value corresponding to the benefit
  // of placing the BufferInterval in the alternate memory. The larger value,
  // the more beneficial.
  virtual std::optional<float> BufferIntervalAlternateMemoryBenefit(
      const GlobalDecreasingSizeBestFitHeap<HloValue>::BufferInterval& interval)
      const {
    return std::nullopt;
  }

 protected:
  const absl::flat_hash_map<const HloInstruction*, int64_t>*
      instruction_schedule_ = nullptr;
  int retry_number_ = 0;
};

// Prefetch interval picker that uses instruction count to overlap asynchronous
// copies with independent computation. The min and max overlap counts describe
// the number of independent HLOs overlapped while a value is being prefetched
// into the alternate memory (between CopyStart and CopyDone HLO instructions).
// max_overlap_count attempts to prevent bringing tensors into the alternate
// memory too eagerly and hence occupying the space for other tensors which
// might use it.  min_overlap_count attempts to prevent cases where tensors are
// prefetched into the alternate memory without sufficient time for the copy to
// take place.  In those cases, it's just better to keep the tensor in the
// default memory instead of hurting the critical path with this copy that
// likely won't finish in time.
class InstructionCountPrefetchIntervalPicker : public PrefetchIntervalPicker {
 public:
  InstructionCountPrefetchIntervalPicker(int64_t min_overlap_count,
                                         int64_t max_overlap_count)
      : min_overlap_count_(min_overlap_count),
        max_overlap_count_(max_overlap_count) {}

  bool CanAllocateInAlternateMemoryNoCopy(const Shape& shape,
                                          int64_t start_time,
                                          int64_t end_time) const override;

  int64_t PreferredEvictionEndTime(const Shape& shape, int64_t start_time,
                                   int64_t latest_end_time) const override;

  int64_t LatestPrefetchStartTime(const Shape& shape, int64_t start_time,
                                  int64_t end_time,
                                  const HloUse* use) const override;

  int64_t PreferredPrefetchStartTime(const Shape& shape,
                                     int64_t earliest_prefetch_start_time,
                                     int64_t latest_prefetch_start_time,
                                     int64_t prefetch_end_time) const override;

  int64_t EstimatedPrefetchEndTime(const Shape& shape, int64_t start_time,
                                   int64_t end_time) const override;
  float GetLogicalIntervalElapsed(int64_t start_time,
                                  int64_t end_time) const override;

  void Begin(const HloUse& use, int64_t start_time, int64_t end_time,
             std::optional<int64_t> preferred_time) override;

  int64_t Next() override;
  bool Done() const override;

  int64_t latest_time() const override;

  std::string ToDebugString() const override;
  std::string ToNoCopyDebugString(const Shape& shape, int64_t start_time,
                                  int64_t end_time) const override;

 private:
  int64_t min_overlap_count_;
  int64_t max_overlap_count_;
  int64_t end_time_;
  int64_t current_prefetch_time_;
};

// Forward Declaration of MemorySpaceAssignmentCostAnalysis
class MemorySpaceAssignmentCostAnalysis;
// Prefetch interval picker that uses cost analysis to overlap asynchronous
// copies with independent computation. It uses min (independent computation
// duration) / (asynchronous copy duration) ratio to guide whether the prefetch
// is within the lower bound. For the upper bound, it restricts the maximum
// duration that a buffer may occupy the alternate memory space as a multiple of
// the time it would take to copy a buffer that is the size of the alternate
// memory. It starts with the preferred ratio in Begin() and works its way for
// alternately earlier and later prefetches until hitting min and max ratios.
// The value for buffer size for max async copy is a mechanism to prevent
// copying small buffers between the two memories unnecessarily. For calculating
// the max time that the buffer can reside in alternate memory, we use the
// larger of this value and the actual size of the buffer. A shape override can
// also be provided which causes the interval picker to use that shape for async
// copy durations instead of the actual shape of the copy.
class CostAnalysisPrefetchIntervalPicker : public PrefetchIntervalPicker {
 public:
  CostAnalysisPrefetchIntervalPicker(
      const MemorySpaceAssignmentCostAnalysis& cost_analysis,
      float min_overlap_to_async_copy_ratio,
      float preferred_overlap_to_async_copy_ratio,
      float max_overlap_to_mem_size_async_copy_ratio, int64_t mem_size_bytes,
      const Shape* shape_override = nullptr);

  bool CanAllocateInAlternateMemoryNoCopy(const Shape& shape,
                                          int64_t start_time,
                                          int64_t end_time) const override;

  int64_t PreferredEvictionEndTime(const Shape& shape, int64_t start_time,
                                   int64_t latest_end_time) const override;

  int64_t LatestPrefetchEndTime(
      int64_t original_prefetch_end_time,
      int64_t proposed_prefetch_end_time) const override;

  int64_t LatestPrefetchStartTime(const Shape& shape, int64_t start_time,
                                  int64_t end_time,
                                  const HloUse* use) const override;

  int64_t PreferredPrefetchStartTime(const Shape& shape,
                                     int64_t earliest_prefetch_start_time,
                                     int64_t latest_prefetch_start_time,
                                     int64_t prefetch_end_time) const override;

  int64_t EstimatedPrefetchEndTime(const Shape& shape, int64_t start_time,
                                   int64_t end_time) const override;
  float GetLogicalIntervalElapsed(int64_t start_time,
                                  int64_t end_time) const override;

  void Begin(const HloUse& use, int64_t start_time, int64_t end_time,
             std::optional<int64_t> preferred_time) override;

  int64_t Next() override;
  bool Done() const override;

  int64_t latest_time() const override;

  void SetRetryNumber(int retry_number) override;

  std::string ToDebugString() const override;
  std::string ToNoCopyDebugString(const Shape& shape, int64_t start_time,
                                  int64_t end_time) const override;

  std::optional<float> BufferIntervalAlternateMemoryBenefit(
      const GlobalDecreasingSizeBestFitHeap<HloValue>::BufferInterval& interval)
      const override;

 private:
  // Finds the minimum nest level in the given interval.
  int GetMinWhileNestLevel(int64_t start_time, int64_t end_time) const;

  // Given the elapsed time to copy this buffer to the alternate memory, returns
  // the longest time that this buffer may reside in the alternate memory space.
  float GetMaxElapsedInAlternateMemory(float async_copy_elapsed) const;

  // For each instruction in the flattened schedule, maintain their elapsed time
  // (in cumulative sum) and while nesting level.
  std::vector<float> elapsed_time_cumsum_;
  std::vector<int> while_nest_level_;
  std::vector<int> computation_nest_level_;
  // Maintain the index of the most recent (before this instruction) nest level
  // change in order to efficiently determine the minimum nest level in an
  // interval.
  std::vector<int> while_nest_level_change_;

  const MemorySpaceAssignmentCostAnalysis& cost_analysis_;
  float min_overlap_to_async_copy_ratio_;
  float preferred_overlap_to_async_copy_ratio_;
  float max_async_copy_elapsed_;
  float max_overlap_multiplier_ = 1.0;

  float async_copy_elapsed_;
  float inst_elapsed_reduction_;
  int64_t end_logical_time_;
  int64_t earliest_prefetch_time_;
  int64_t latest_prefetch_time_;
  bool using_increasing_prefetch_time_iterator_ = true;
  int64_t increasing_prefetch_time_iterator_;
  int64_t decreasing_prefetch_time_iterator_;

  std::vector<float> while_execution_counts_;
  // Shape override is used to override the shape of the shape of the async copy
  // to treat all async copies the same duration. Having an override forces
  // prefetches to be scheduled roughly in FIFO order.
  std::optional<Shape> shape_override_;
};

// MemorySpaceAssignment assigns memory spaces (default or alternate) to each
// instruction in the module. It will greedily try placing as as many values in
// the alternate memory space as possible. It uses the heap simulator to
// determine the actual allocation offsets of values in the alternate memory
// space to account for fragmentation. The default memory space is assumed to be
// large enough to hold the values that could not be placed in the alternate
// memory space.
class MemorySpaceAssignment {
 public:
  using Chunk = HeapSimulator::Chunk;
  using BufferInterval =
      GlobalDecreasingSizeBestFitHeap<HloValue>::BufferInterval;
  using BufferIntervalCompare =
      GlobalDecreasingSizeBestFitHeap<HloValue>::BufferIntervalCompare;
  using IsAllowedInAlternateMemoryFunction =
      std::function<bool(const HloValue&)>;
  using IsUseAllowedInAlternateMemoryFunction =
      std::function<bool(const HloUse&)>;
  using IsPositionAllowedInAlternateMemoryFunction =
      std::function<bool(const HloPosition&)>;
  using ReservedScopedMemoryFunction =
      std::function<int64_t(const HloInstruction*)>;

  // MemorySpaceAssignment uses a notion of a slow and large default memory
  // space and a fast and small alternate memory space.
  enum class MemorySpace { kDefault, kAlternate };

  // Forward declaration for Allocation.
  class Allocation;
  class ParentAllocation;

  // This class represents an allocation that might either be in the default or
  // alternate memory. An HloValue might live in multiple different allocations
  // over its lifetime. The lifetimes of the allocations are defined using
  // start_time and end_time, which corresponds to the instruction indexes in
  // the flattened schedule. Each of these allocations might partially overlap
  // with each other. CopyAllocation defined below represents asynchronous
  // copies between Allocations.
  //
  // Consider an instruction Foo, and its users Bar and Baz, and the times given
  // in terms of the flattened schedule of the entire module:
  //
  //      Foo:10
  //       /   \
  //    Bar:14  \
  //           Baz:25
  //
  // A valid memory space assignment could be like the following:
  //
  //  Time:         10 ... 14        ...      25
  //                Foo    Bar                Baz
  //  Alternate     +-------+           +-----+
  //  Default           +---------------------+
  //                    ^   ^           ^     ^
  //                    |   |           |     |
  //                evict   evict  prefetch  prefetch
  //                start    end    start      end
  //
  // This would be represented with:
  //   - Allocation(memory_space=kAlternate, start_time=10, end_time=14)
  //   - CopyAllocation(memory_space=kDefault, start_time=12, end_time=25)
  //   - CopyAllocation(memory_space=kAlternate, start_time=22, end_time=25)
  class Allocation {
    friend class ParentAllocation;

   public:
    Allocation(HloPosition defining_position, MemorySpace memory_space,
               std::optional<Chunk> chunk, int64_t start_time, int64_t end_time,
               bool is_scoped_allocation)
        : defining_position_(defining_position),
          memory_space_(memory_space),
          chunk_(chunk),
          start_time_(start_time),
          end_time_(end_time),
          is_scoped_allocation_(is_scoped_allocation) {
      CHECK(!is_scoped_allocation || defining_position.index == ShapeIndex({}));
    }
    virtual ~Allocation() = default;

    virtual bool is_copy_allocation() const { return false; }

    // Adds a use to this allocation.
    void AddUse(HloUse use);

    // Extends the end time of this allocation.
    void Extend(int64_t end_time) { end_time_ = end_time; }

    // After all of the time ranges for the allocations have been assigned,
    // Process morphs the instructions affected to assign the memory spaces and
    // insert asynchronous copy instructions if necessary.
    virtual Status Process();

    // An optional post-process step that will be called after all allocations
    // have been processed.
    virtual Status PostProcess() { return OkStatus(); }

    // Marks (adds this allocation to needed_allocations) if this allocation is
    // needed. Allocation and CopyAllocations are always needed and
    // ParentAllocations are needed if they have any uses or if other
    // CopyAllocation or ParentAllocations depend on them.
    virtual void MarkIfNeeded(
        absl::flat_hash_set<const Allocation*>& needed_allocations) const;

    // Marks this allocation as needed.
    virtual void MarkNeeded(
        absl::flat_hash_set<const Allocation*>& needed_allocations) const;

    // Returns the defining position for this allocation.
    virtual HloPosition defining_position() const { return defining_position_; }

    // Returns the time the buffer is first available to be used. For
    // Allocation, this is start_time.
    virtual int64_t earliest_available_time() const { return start_time_; }

    const std::vector<HloUse>& uses() const { return uses_; }
    MemorySpace memory_space() const { return memory_space_; }
    // Returns the associated chunk that may be a nullopt if the allocation is
    // in the default memory space.
    std::optional<Chunk> maybe_chunk() const { return chunk_; }
    // Returns the associated chunk. The caller should ensure that the chunk is
    // defined (the allocation should be in the alternate memory space).
    Chunk chunk() const {
      CHECK(chunk_.has_value());
      return *chunk_;
    }
    Chunk* mutable_chunk() { return &*chunk_; }
    void set_start_time(int64_t start_time) { start_time_ = start_time; }
    int64_t start_time() const { return start_time_; }
    int64_t end_time() const { return end_time_; }
    bool is_scoped_allocation() const { return is_scoped_allocation_; }
    virtual std::optional<int64_t> cross_program_prefetch_index() const {
      return std::nullopt;
    }

    bool operator==(const Allocation& other) const;
    virtual std::string ToString() const;

   protected:
    // Recursively create kGetTupleElement instructions if the defining position
    // shape is not an array. Returns the new instruction that has array shape.
    HloInstruction* AddGetTupleElements() const;

    HloPosition defining_position_;
    std::vector<HloUse> uses_;
    MemorySpace memory_space_;
    std::optional<Chunk> chunk_;
    int64_t start_time_;
    int64_t end_time_;
    const bool is_scoped_allocation_;
  };

  // This class represents an allocation as a result of an asynchronous copy.
  // Note: CopyStart instructions are inserted after `start_time` or later,
  // while CopyDone instructions are inserted before
  // `copy_done_schedule_before_time` or earlier.
  class CopyAllocation : public Allocation {
   public:
    CopyAllocation(
        const Allocation& prev_allocation, MemorySpace memory_space,
        std::optional<Chunk> chunk, int64_t start_time, int64_t end_time,
        int64_t copy_done_schedule_before_time,
        std::optional<int64_t> cross_program_prefetch_index = std::nullopt)
        : Allocation(/*defining_position=*/{nullptr, {}}, memory_space, chunk,
                     start_time, end_time, /*is_scoped_allocation=*/false),
          prev_allocation_(prev_allocation),
          copy_start_schedule_after_(start_time),
          copy_done_schedule_before_(copy_done_schedule_before_time),
          cross_program_prefetch_index_(cross_program_prefetch_index) {}

    bool is_copy_allocation() const override { return true; }

    Status Process() override;

    void MarkNeeded(absl::flat_hash_set<const Allocation*>& needed_allocations)
        const override;

    HloPosition defining_position() const override {
      // Unless explicitly set, the defining position of a copy allocation in
      // retrieved from the previous allocation. This is because we don't create
      // new CopyStart/CopyDone instructions until later and the position should
      // point to the previous (copy or otherwise) allocation's position for the
      // original defining position.
      if (defining_position_.instruction == nullptr) {
        return prev_allocation_.defining_position();
      }
      return defining_position_;
    }

    HloInstruction* copy_start() const { return copy_start_; }
    HloInstruction* copy_done() const { return copy_done_; }

    // Returns the time the buffer is first available to be used. For
    // CopyAllocation, this is when the copy ends, which is
    // copy_done_schedule_before.
    int64_t earliest_available_time() const override {
      return copy_done_schedule_before_;
    }

    int64_t copy_start_schedule_after() const {
      return copy_start_schedule_after_;
    }
    int64_t copy_done_schedule_before() const {
      return copy_done_schedule_before_;
    }

    void set_copy_start_schedule_after(int64_t copy_start_schedule_after) {
      copy_start_schedule_after_ = copy_start_schedule_after;
    }

    void set_copy_done_schedule_before(int64_t copy_done_schedule_before) {
      copy_done_schedule_before_ = copy_done_schedule_before;
    }

    std::optional<int64_t> cross_program_prefetch_index() const override {
      return cross_program_prefetch_index_;
    }

    bool operator==(const CopyAllocation& other) const;
    std::string ToString() const override;

   private:
    const Allocation& prev_allocation_;
    // These variables define the scheduling boundaries where CopyStart and
    // CopyDone can be scheduled. The earliest CopyStart can be scheduled is
    // after copy_start_schedule_after_ and the latest CopyDone can be scheduled
    // is before copy_done_schedule_before_.
    int64_t copy_start_schedule_after_;
    int64_t copy_done_schedule_before_;
    HloInstruction* copy_start_;
    HloInstruction* copy_done_;
    std::optional<int64_t> cross_program_prefetch_index_;
  };

  // An allocation in the default memory space that mirrors another Allocation
  // object. This is useful to model an eviction that happens before a while op
  // so that we don't need to redundantly evict the buffer after the while op as
  // well.
  class MirroredAllocation : public Allocation {
   public:
    MirroredAllocation(const Allocation& original_allocation, int64_t time)
        : Allocation(original_allocation.defining_position(),
                     MemorySpace::kDefault, original_allocation.maybe_chunk(),
                     /*start_time=*/time,
                     /*end_time=*/time, /*is_scoped_allocation=*/false),
          original_allocation_(original_allocation) {}

    Status Process() override;

    void MarkNeeded(absl::flat_hash_set<const Allocation*>& needed_allocations)
        const override;

    std::string ToString() const override;

   private:
    const Allocation& original_allocation_;
  };

  // An allocation in default memory space that is defined in the parent
  // computation. If a value has a copy in the default memory space in the
  // parent computation, we don't need to evict this buffer in a while loop.
  class ParentAllocation : public Allocation {
   public:
    ParentAllocation(const Allocation& original_allocation,
                     HloInstruction* calling_instruction, HloPosition position,
                     int64_t time)
        : Allocation(position, MemorySpace::kDefault,
                     original_allocation.maybe_chunk(), /*start_time=*/time,
                     /*end_time=*/time, /*is_scoped_allocation=*/false),
          original_allocation_(original_allocation),
          calling_instruction_(calling_instruction) {}

    Status Process() override;
    Status PostProcess() override;

    void MarkIfNeeded(absl::flat_hash_set<const Allocation*>&
                          needed_allocations) const override;
    void MarkNeeded(absl::flat_hash_set<const Allocation*>& needed_allocations)
        const override;

    std::string ToString() const override;

   private:
    const Allocation& original_allocation_;
    HloInstruction* calling_instruction_;
  };

  using AllocationSequence = std::vector<std::unique_ptr<Allocation>>;
  // AllocationValue is used to break up HloValues for each non-trivial position
  // (trivial positions are considered Tuple, GetTupleElement, and Bitcast). An
  // HloValue may include positions and uses that alias with each other across
  // multiple computations. We use this class to break these HloValues such that
  // every AllocationValue has one defining position (that may alias with other
  // AllocationValues). The uses field of the AllocationValue contains only the
  // direct uses of the AllocationValue's defining position.
  //
  // For example, consider the following HLO snippet:
  //
  // Body {
  //   body_param = (f32[4,3]{1,0}, f32[]) parameter(0)
  //   get-tuple-element.3 = f32[4,3]{1,0} get-tuple-element(body_param),
  //   index=0
  //   ...
  //   ROOT tuple = (f32[4,3]{1,0}, f32[]) tuple(get-tuple-element.3, ...)
  // }
  //
  // Cond {
  //   cond_param = (f32[4,3]{1,0}, f32[]) parameter(0)
  //   ...
  // }
  //
  // add.4 = f32[4,3]{1,0} add(...)
  // tuple.1 = (f32[4,3]{1,0}, f32[]) tuple(add.4, ...)
  // while = (f32[4,3]{1,0}, f32[]) while(tuple.1), body=Body, condition=Cond
  // get-tuple-element.5 = f32[4,3]{1,0} get-tuple-element(while), index=0
  // add.5 = f32[4,3]{1,0} add(get-tuple-element.5, ...)
  //
  // This contains an HloValue that looks like the following:
  // positions:
  //  add.4
  //  body_param {0}
  //  get-tuple-element.3
  //  tuple {0}
  //  cond_param {0}
  //  tuple.1 {0}
  //  while {0}
  //  get-tuple-element.5
  // uses:
  //  add.1, operand 0
  //  tuple, operand 0
  //  while, operand 0 {0}
  //  add.5, operand 0
  //
  // We break this HloValue up into the following AllocationValues for each
  // non-trivial position:
  // AllocationValue1: computation = Entry
  //  position:
  //   add.4
  //  uses:
  //   while, operand 0 {0}
  // AllocationValue2: computation = Cond
  //  position:
  //   cond_param {0}
  //  uses:
  // AllocationValue3: computation = Body
  //  position:
  //   body_param {0}
  //  uses:
  //   add.1, operand 0
  //   tuple, operand 0
  // AllocationValue4: computation = Entry
  //  position:
  //   while {0}
  //  uses:
  //   add.5, operand 0
  class AllocationValue {
   public:
    // This data structure wraps an HloUse and adds additional metadata that are
    // useful for allocation.
    struct Use {
      // The wrapped HloUse object.
      HloUse hlo_use;
      // The logical time this use is scheduled.
      int64_t time;
      // All the positions where this use aliases with. The aliased positions
      // must get the same allocation.
      std::vector<HloPosition> aliases;

      bool operator==(const Use& other) const {
        return hlo_use == other.hlo_use && time == other.time &&
               aliases == other.aliases;
      }

      template <typename H>
      friend H AbslHashValue(H h, const Use& s) {
        return H::combine(std::move(h), s.hlo_use, s.time, s.aliases);
      }
    };

    AllocationValue(const HloValue* value, const HloPosition& position,
                    int64_t size)
        : value_(value),
          defining_position_(position),
          size_(size),
          requires_contiguous_allocation_(false) {}

    const HloPosition& defining_position() const { return defining_position_; }
    const HloInstruction* defining_instruction() const {
      return defining_position().instruction;
    }
    int64_t size() const { return size_; }
    const std::vector<Use>& uses() const { return uses_; }
    std::vector<Use>& uses() { return uses_; }
    const HloValue* value() const { return value_; }
    const HloComputation* computation() const {
      return defining_instruction()->parent();
    }
    AllocationSequence* allocation_sequence() { return &allocation_sequence_; }

    // Sets/gets whether this AllocationValue requires allocating it
    // contiguously throughout its live range (without any copies).
    bool requires_contiguous_allocation() const {
      return requires_contiguous_allocation_;
    }
    void set_requires_contiguous_allocation(
        bool requires_contiguous_allocation) {
      requires_contiguous_allocation_ = requires_contiguous_allocation;
    }

    void AddUse(const HloUse& use, int64_t use_time) {
      uses_.push_back({use, use_time, {}});
    }

    std::string ToString() const;
    std::string ToShortString() const;

   private:
    const HloValue* value_;
    HloPosition defining_position_;
    int64_t size_;
    // If true, there must be a contiguous allocation for this buffer without
    // any copies.
    bool requires_contiguous_allocation_;
    std::vector<Use> uses_;
    AllocationSequence allocation_sequence_;
  };

  // Statistics of asynchronous copies.
  struct AsyncCopyStats {
    int64_t max_outstanding_async_copies;
    int64_t num_prefetches;
    int64_t prefetch_bytes;
    int64_t num_evictions;
    int64_t eviction_bytes;
  };

  virtual ~MemorySpaceAssignment() = default;

  // Runs the MemorySpaceAssignment pass.
  static StatusOr<std::unique_ptr<PresetAssignments>> Run(
      HloModule* module, const HloLiveRange& hlo_live_range,
      const HloAliasAnalysis& alias_analysis, const Options& options);

  // Calculates asynchronous copy statistics.
  StatusOr<AsyncCopyStats> CalculateAsyncCopyStats() const;

  static BufferIntervalCompare GetMemoryBoundednessBufferIntervalCompare(
      const MemorySpaceAssignmentCostAnalysis& cost_analysis,
      MemorySpaceAssignmentCostAnalysis::Cache* cache = nullptr);

  // Verify that the memory space assignment is free of overlapping buffers and
  // export heap simulator trace to be used by buffer_assignment.
  Status VerifyAndExportHeapSimulatorTrace();

 protected:
  // Main driver of the memory space assignment pass.
  virtual StatusOr<std::unique_ptr<PresetAssignments>> RunMemorySpaceAssignment(
      const HloLiveRange& hlo_live_range,
      const HloAliasAnalysis& alias_analysis);

  // Finds an AllocationSequence for placing buffers in alternate memory using
  // the AlternateMemoryBestFitHeap algorithm. Must be set before Process() is
  // called.
  virtual Status FindAllocationSequence(const HloLiveRange& hlo_live_range,
                                        const HloAliasAnalysis& alias_analysis);

  const Options& options() const { return options_; }

  MemorySpaceAssignment(HloModule* module, const Options& options,
                        const HloLiveRange& hlo_live_range)
      : module_(module),
        options_(options),
        flattened_instructions_(hlo_live_range.flattened_instruction_sequence()
                                    .instructions()
                                    .begin(),
                                hlo_live_range.flattened_instruction_sequence()
                                    .instructions()
                                    .end()),
        computations_in_schedule_(),
        preset_assignments_(std::make_unique<PresetAssignments>()) {
    for (const auto& computation_and_bound :
         hlo_live_range.computation_span_times()) {
      computations_in_schedule_.insert(computation_and_bound.first);
    }
  }

  AllocationSequence allocations_;

  HloModule* module() { return module_; }

 private:
  // Process calls Process methods of the allocations after the allocations have
  // been finalized.
  Status Process();

  // Process() might have altered the computation graph by inserting kTuple and
  // kGetTupleElement instructions. SimplifyGraph performs a simple DCE and
  // tuple simplification operation (e.g., given GetTupleElement(Tuple(a, b),
  // 1), simply forwards b). Runs to fixed point.
  Status SimplifyGraph();

  // FixSchedule inserts asynchronous copies in the schedule.
  Status FixSchedule();

  // Export the alternate memory assignments to the PresetAssignments and color
  // the HLO graph with the determined memory spaces.
  Status ExportAndColorBuffers();

  // Schedules asynchronous copies and ensures that the CopyStarts and their
  // corresponding CopyDones follow the same order.
  void ScheduleAsynchronousCopies();

  // Remove the positions and chunks associated with the instruction from
  // alternate_memory_assignments_.
  void RemoveAssignmentForInstruction(const HloInstruction* instruction);

  // Returns the estimated elapsed duration of the hlo module in seconds. It
  // uses the 'allocations' argument to determine the location (default memory
  // or alternate memory) of each operand and output of an instruction.
  float ComputeEstimatedElapsedTime(const HloLiveRange& hlo_live_range,
                                    const AllocationSequence& allocations);

  HloModule* module_;
  const Options& options_;
  std::vector<HloInstruction*> flattened_instructions_;
  absl::flat_hash_set<const HloComputation*> computations_in_schedule_;
  std::unique_ptr<PresetAssignments> preset_assignments_;
  std::vector<std::pair<HloPosition, Chunk>> alternate_memory_assignments_;
  std::vector<std::pair<HloInstruction*, Chunk>> scoped_memory_assignments_;
  int64_t alternate_memory_size_ = 0;

  // These maps hold vectors of new instructions that need to be scheduled after
  // (or before) the instruction index in the key. FixSchedule uses these maps
  // to modify and fix the schedule.
  absl::flat_hash_map<int64_t, std::vector<HloInstruction*>> schedule_after_;
  absl::flat_hash_map<int64_t, std::vector<HloInstruction*>> schedule_before_;
};

// The different options to be passed to the Run() API.
struct Options {
  // Backend-specific integer value that describes the alternate memory.
  int64_t alternate_memory_space = 0;

  // Maximum size of the alternate memory space.
  int64_t max_size_in_bytes = 0;

  // Memory alignment of the alternate memory space.
  int64_t alignment_in_bytes = 1;

  // If provided, we sort the buffers using this comparison function
  // otherwise, we use GlobalDecreasingSizeBestFitHeap::kSpatial.
  std::optional<MemorySpaceAssignment::BufferIntervalCompare>
      buffer_interval_compare = std::nullopt;

  // This object determines how early and how late prefetches can occur.
  PrefetchIntervalPicker* prefetch_interval_picker = nullptr;

  // This object is used to determine the benefit of a particular allocation.
  MemorySpaceAssignmentCostAnalysis* cost_analysis = nullptr;

  // Size function for buffer values.
  BufferValue::SizeFunction size_fn;

  // This function can be used to prevent certain HloValues (e.g., based on
  // the opcode) to be placed on the alternate memory.
  MemorySpaceAssignment::IsAllowedInAlternateMemoryFunction
      is_allowed_in_alternate_mem_fn;

  // This function can be used to prevent certain HloUses (e.g., based on
  // the opcode) to be placed on the alternate memory.
  MemorySpaceAssignment::IsUseAllowedInAlternateMemoryFunction
      is_use_allowed_in_alternate_mem_fn = [](const HloUse&) { return true; };

  // Specifies if the given position is allowed in the alternate memory.
  MemorySpaceAssignment::IsPositionAllowedInAlternateMemoryFunction
      is_position_allowed_in_alternate_mem_fn =
          [](const HloPosition&) { return true; };

  // This function returns the amount of scoped memory in bytes that should be
  // reserved during the execution of this instruction.
  MemorySpaceAssignment::ReservedScopedMemoryFunction
      reserved_scoped_memory_fn = [](const HloInstruction*) { return 0; };

  // If true, we allocate the reserved scoped memory at the same offset. This
  // is useful to enable more deduplication between HLOs that have reserved
  // scoped memories, but may result in less efficient memory packing.
  bool allocate_reserved_scoped_memory_at_same_offset = true;

  // Specifies the upper bound for number of outstanding prefetches and
  // evictions, -1 for unlimited.
  int64_t max_outstanding_prefetches = -1;
  int64_t max_outstanding_evictions = -1;

  // Extra outstanding prefetch limit for while uses (in addition to
  // max_outstanding_prefetches).
  int64_t while_use_extra_outstanding_prefetch_limit = 0;

  // Specifies the maximum number of retries that will be performed for each
  // value in case prefetching failed due to running out of asynchronous
  // copies or asynchronous copy resource.
  int64_t max_retries = 1;

  // The maximum number of repacks that we are willing to perform in case we
  // can't allocate a buffer due to running out of memory. If this value is
  // greater than 0, repacker must be non-nullptr.
  int64_t max_repacks = 0;

  // This variable is used by the cost analysis in estimating how many times
  // each while loop will execute. Nested loops will be assumed to have
  // executed pow(while_execution_count, nesting_level) times.
  uint64_t xla_tpu_memory_space_assignment_while_execution_count = 5ULL;

  float async_copy_bandwidth_bytes_per_second = 0.0f;

  float alternate_mem_bandwidth_bytes_per_second = 0.0f;

  // The repacking algorithm to reduce fragmentation. Must be non-null if
  // max_repacks is greater than 0.
  MemorySpaceAssignmentRepacker* repacker = nullptr;

  // This is only useful for testing, repack after every allocation.
  bool repack_after_every_allocation = false;

  // If true, tries allocating buffers across (e.g., before and inside a while
  // loop body) sequential calls (kWhile, kCall, and kConditional).
  bool allocate_across_sequential_calls = false;

  // If true, verifies the memory space assignment against overlapping
  // buffers.
  bool verify = false;

  // If not nullptr, this function is called to dump debugging information.
  // The first argument is appended to the file name and the second argument
  // is the contents of the file.
  std::function<void(absl::string_view, absl::string_view)> dump_fn = nullptr;

  // Enable prefetching buffers into preferred memory across program
  // boundaries
  bool enable_cross_program_prefetch = true;

  // If true, use buffer_interval_compare to determine which buffers to
  // prefetch across program boundaries.
  bool default_cross_program_prefetch_heuristic = false;

  // Enable cross-program prefetch freeing optimization where the
  // cross-program-prefetched buffer can be reused.
  bool enable_cross_program_prefetch_freeing = true;

  // The maximum number of cross program prefetches.
  // TODO(tjablin): Use a heuristic to determine this automatically.
  int max_cross_program_prefetches = 1;

  // Enable redundant eviction optimization in/around while loops. If enabled,
  // this optimization would keep a copy of the buffer in the default memory in
  // addition to alternate memory to eliminate redundant evictions.
  bool enable_while_redundant_eviction_elimination = true;

  // An optional memory space assignment autotuning config, which is used
  // to sort allocated buffers.
  std::optional<std::vector<uint64_t>> autotuning_config = std::nullopt;

  // Scales effective bandwidth for async copies. Valid range is (0, 1].
  float async_copy_bandwidth_scaling_factor = 1.0;

  // If true, uses the earlier instance of the same instruction to use as
  // preferred prefetch start time.
  bool use_repeated_instance_for_preferred_prefetch_time = false;

  // If true, enforces the FIFO order for prefetches.
  bool enforce_prefetch_fifo_order = false;
};

// A struct representing an asynchronous copy with its logical start and end
// time (time that copy done is scheduled), the resource this copy would use,
// its destination memory space, and a unique ID.
struct AsynchronousCopy {
  int64_t start_time;
  int64_t end_time;
  float resource;
  MemorySpaceAssignment::MemorySpace destination;
  int64_t id;

  std::tuple<int64_t, int64_t, float, MemorySpaceAssignment::MemorySpace,
             int64_t>
  AsTuple() const {
    return std::make_tuple(start_time, end_time, resource, destination, id);
  }
};

// Compare asynchronous copies such that an earlier start time has the same or
// earlier end time and an earlier end time has the same or earlier start time.
bool operator<(const AsynchronousCopy& a, const AsynchronousCopy& b);

bool operator==(const AsynchronousCopy& a, const AsynchronousCopy& b);
bool operator!=(const AsynchronousCopy& a, const AsynchronousCopy& b);

// Helper class to enforce asynchronous copy ordering. If the appropriate option
// is enabled, we only allow asynchronous copies that are pipelined: if an
// asynchronous copy ends earlier than another asynchronous copy, it must start
// the same time or earlier than the other asynchronous copy; and if an
// asynchronous copy starts earlier than another asynchronous copy, it must end
// the same time or earlier than the other asynchronous copy.
class AsynchronousCopyOrdering {
 public:
  AsynchronousCopyOrdering() = default;

  // Adds an asynchronous copy.
  void AddCopy(const AsynchronousCopy& copy);

  // Removes an asynchronous copy. CHECKs that it is removed.
  void RemoveCopy(const AsynchronousCopy& copy);

  // Returns true if the addition of an asynchronous copy in the given time
  // interval would violate the asynchronous copy ordering. E.g., consider the
  // following scenario:
  //                                  CS          CD
  //  already committed async copy:   +-----------+
  //                new async copy:     +--------+
  //
  // The new asynchronous copy would violate the ordering guarantee because the
  // copy start is after an already committed asynchronous copy while its copy
  // done is before the committed copy.
  bool ViolatesOrdering(int64_t start_time, int64_t end_time) const;

 private:
  // We use this data structure for keys into the map that has a custom
  // comparator for the ordering guarantees.
  struct Interval {
    int64_t start_time;
    int64_t end_time;

    // We allow multiple prefetches that have one or both of the same start and
    // end times. std::map considers two values as equal if neither are less
    // than the other.  Using this comparator, we can ensure that the only
    // intervals that evaluate to be equal are those with the same start and end
    // times or those with intervals that violate the FIFO order.
    bool operator<(const Interval& other) const {
      return (start_time < other.start_time && end_time <= other.end_time) ||
             (start_time <= other.start_time && end_time < other.end_time);
    }
  };
  // Stores asynchronous copies in a tree set respecting the pipelining order.
  std::map<Interval, std::set<AsynchronousCopy>> ranges_;
};

// Helper class to enforce asynchronous copy resources by keeping track of
// available copy bandwidth and elapsed times of overlapped operations. It
// maintains a list of initial resources that correspond to the elapsed times of
// overlapped operations. As asynchronous copies are added, the available
// resource is subtracted to keep track of the current state.
class AsynchronousCopyResource {
 public:
  AsynchronousCopyResource() = default;

  // The constructor needs the initial resources.
  explicit AsynchronousCopyResource(absl::Span<const float> initial_resources)
      : initial_resources_(initial_resources.begin(), initial_resources.end()),
        delay_(initial_resources.size(), 0) {}

  // Adds the given asynchronous copy and updates the current resources. CHECK
  // fails if there aren't enough resources to satisfy this copy (the caller
  // should use HasEnoughResource first to ensure there is enough resource).
  void AddCopy(const AsynchronousCopy& copy);

  // Removes the given copy and frees the resource.
  void RemoveCopy(const AsynchronousCopy& copy);

  // Returns true if a copy with the given start and end times and resource can
  // be satisfied.
  bool HasEnoughResource(int64_t start_time, int64_t end_time, float resource);

  // This is only used for debugging and testing purposes, it returns the
  // currently available resource at each logical time.
  std::vector<float> GetCurrentResources() const {
    std::vector<float> current_resources(initial_resources_.begin(),
                                         initial_resources_.end());
    for (int i = 0; i < current_resources.size(); ++i) {
      current_resources[i] -= std::min(current_resources[i], delay_[i]);
    }
    return current_resources;
  }

 private:
  // Internal helper method to implement adding/removing/checking resources.
  // Only updates the current resources if update_current_resource is true. The
  // current_copy points to an iterator in async_copies_ and this
  bool ConsumeResource(
      int64_t start_time, int64_t end_time, float resource,
      bool update_current_resource,
      const std::list<AsynchronousCopy>::iterator* current_copy = nullptr,
      float resource_to_free = 0.0);

  // Same as the public RemoveCopy except it works on the async_copies_
  // iterator. Assumes copy_it points to the last copy for its start time;
  // otherwise the public RemoveCopy method is supposed to temporarily remove
  // these later copies that share the same start time before removing the
  // requested copy.
  void RemoveCopy(std::list<AsynchronousCopy>::iterator& copy_it);

  // We maintain a linked list of asynchronous copies sorted by the start times.
  // This allows us to efficiently find the copy that starts right after another
  // one because adding a copy might push a copy further into the future.
  std::list<AsynchronousCopy> async_copies_;
// To make the lookups into async_copies_ more efficient, we also maintain a
// binary tree that is indexed by the start time, containing iterators into
// async_copies_.
// TODO(b/210891274): Use btree_map after build issue in Windows is resolved.
#if defined(__GNUC__) || defined(__clang__)
  absl::btree_map<int64_t, std::list<AsynchronousCopy>::iterator>
      async_copy_time_map_;
#else
  std::map<int64_t, std::list<AsynchronousCopy>::iterator> async_copy_time_map_;
#endif
  std::vector<float> initial_resources_;
  std::vector<float> delay_;
};

// This class inherits from GlobalDecreasingSizeBestFitHeap with a notion of
// maximum size.
class AlternateMemoryBestFitHeap
    : public GlobalDecreasingSizeBestFitHeap<HloValue> {
 public:
  using MemorySpace = MemorySpaceAssignment::MemorySpace;
  using AllocationValue = MemorySpaceAssignment::AllocationValue;

  AlternateMemoryBestFitHeap(
      MemorySpaceAssignment::AllocationSequence* allocations,
      const Options& options, const HloAliasAnalysis& alias_analysis,
      const HloLiveRange& hlo_live_range);

  // Allocates a buffer in preferred memory with whole program lifetime and
  // enables prefetching prefetch_candidate from default memory across program
  // boundaries.
  void AllocateCrossProgramPrefetchBuffer(
      HloModule* module, const BufferInterval& prefetch_candidate);

  HeapSimulator::Result<HloValue> Finish() override;

 protected:
  // Given a buffer interval, returns the colocated intervals. Unlike the
  // similar GlobalDecreasingSizeBestFitHeap::GetTransitiveColocations, it
  // returns the colocated intervals sorted by scheduled time.
  std::vector<const BufferInterval*> GetSortedColocatedIntervals(
      const BufferInterval& interval) const;

  // Given a BufferInterval, creates AllocationValue objects and corresponding
  // AllocationSequences and appends them into allocation_sequence_list_.
  void CreateAllocationValues(
      const BufferInterval& buffer_interval,
      std::vector<AllocationValue>& allocation_values) const;

  // Given colocated intervals, populates allocation_values with the
  // corresponding AllocationValue objects.
  virtual void CreateAllocationValuesFromColocatedIntervals(
      absl::Span<const AlternateMemoryBestFitHeap::BufferInterval* const>
          colocated_intervals,
      std::vector<MemorySpaceAssignment::AllocationValue>& allocation_values);

  // Go through all the uses in the AllocationValues and find the aliasing
  // positions.
  void FindAliases(std::vector<AllocationValue>* allocation_values) const;

  MemorySpaceAssignment::AllocationSequence* allocations() {
    return allocations_;
  }
  const Options& options() const { return options_; }
  const HloAliasAnalysis& alias_analysis() { return alias_analysis_; }
  const HloLiveRange& hlo_live_range() { return hlo_live_range_; }

 private:
  // We inherit AllocationBlock struct to attach the Allocation information to
  // make importing repacked offsets easier.
  struct RepackAllocationBlock
      : MemorySpaceAssignmentRepacker::AllocationBlock {
    MemorySpaceAssignment::Allocation* allocation;
  };

  // A data structure we use to associate Allocation objects that are aliased
  // and must get the same offset.
  struct AliasedOffset {
    int64_t offset;
    absl::flat_hash_set<const MemorySpaceAssignment::Allocation*> allocations;
  };

  // An allocation request for a use segment. A use segment is the time segment
  // between the definition and the first use, and the time segment between the
  // uses of a buffer. For example, the time between the definition and Use1, is
  // the first segment, and the time between Use1 and Use2 is the second segment
  // and so on:
  //
  //        +------+----------+-------+
  //       /        \          \       \
  //      /          v          v       v
  //    Def         Use1       Use2    Use3
  //     <----------> <--------> <----->
  //        Segment    Segment   Segment
  //
  // start_time and end_time are the start and end logical times of the segment.
  // use_times is a sorted sequence of the times of all uses.
  // latest_prefetch_time is the latest time we can schedule the CopyDone for a
  // prefetch.
  // If allow_no_copy_alternate_mem_allocation is false, an eviction is forced.
  // If earliest_prefetch_time is set, prefetches cannot start before this
  // value.
  struct AllocationRequest {
    int64_t start_time;
    int64_t end_time;
    int64_t latest_prefetch_time;
    int64_t size;
    bool allow_no_copy_alternate_mem_allocation;
    std::optional<int64_t> earliest_prefetch_time;
    std::optional<int64_t> preferred_prefetch_time;
    AliasedOffset* preferred_offset;
    const MemorySpaceAssignment::AllocationValue::Use* use;
    MemorySpaceAssignment::AllocationValue* allocation_value;
    absl::Span<const int64_t> all_use_times;
  };

  // This struct contains mandatory memory assignments at a given time. E.g., an
  // input's required memory assignment time would correspond to the definition
  // time of the parameter instruction, and an output's time would correspond to
  // the time of last use.
  struct RequiredMemoryAssignment {
    MemorySpaceAssignment::MemorySpace memory_space;
    int64_t time;
    AliasedOffset* offset;

    bool equals_ignoring_time(const RequiredMemoryAssignment& other) const {
      return memory_space == other.memory_space && offset == other.offset;
    }

    bool operator==(const RequiredMemoryAssignment& other) const {
      return memory_space == other.memory_space && time == other.time &&
             offset == other.offset;
    }

    bool operator!=(const RequiredMemoryAssignment& other) const {
      return !(*this == other);
    }
  };

  // Result of an allocation, prefetch, eviction etc. request.  The result is
  // either kSuccess or a bitwise OR of one or more failures. The values are
  // unique powers of two. To check if a result contains a particular failure,
  // use the result_is method. To add a new failure to a result, use the
  // result_mark method.
  enum class Result {
    // Successful allocation.
    kSuccess = 0,
    // Allocation failed because we ran out of alternate memory.
    kFailOutOfMemory = 1,
    // A no-copy allocation couldn't be performed because the previous
    // allocation wasn't in the alternate memory space.
    kFailPrevAllocationNotInAlternateMem = 2,
    // A no-copy allocation couldn't be performed because the live range was too
    // long.
    kFailLiveRangeTooLong = 4,
    // A prefetching couldn't be performed because the live range was too short.
    kFailLiveRangeTooShort = 8,
    // Ran out of outstanding asynchronous copy limit either during prefetching
    // or eviction.
    kFailOutOfAsyncCopies = 16,
    // A prefetching couldn't be performed because the asynchronous copy
    // resource was violated.
    kFailViolatesAsyncCopyResource = 32,
    // An allocation failure happened that requires uncommitting all the pending
    // allocations. Usually this is due to a situation requiring an eviction but
    // the eviction couldn't be performed.
    kFailRequiresUncommit = 64
  };

  // Return true if the result belongs to a failure.
  static bool result_is(Result result, Result failure) {
    return static_cast<int>(result) & static_cast<int>(failure);
  }

  // Mark (bitwise OR) a failure to the result.
  static Result result_mark(Result failure, Result& result) {
    result = static_cast<Result>(static_cast<int>(result) |
                                 static_cast<int>(failure));
    return result;
  }

  // Return true if the result is a failure that requires us to uncommit pending
  // chunks.
  static bool result_requires_uncommit(Result result) {
    return result_is(result, Result::kFailRequiresUncommit);
  }

  // Return true if the result is a failure either due to running out of
  // outstanding asynchronous copies or due to violating asynchronous copy
  // ordering.
  static bool result_failed_because_of_async_copy(Result result) {
    return result_is(result, Result::kFailOutOfAsyncCopies) ||
           result_is(result, Result::kFailViolatesAsyncCopyResource);
  }

  // Allocates buffers for instructions that need reserved scoped allocations in
  // the alternate memory space.
  void AllocateReservedScopedAllocations();

  // Returns the AliasedOffset object associated with the allocation.
  AliasedOffset* GetAliasedOffset(
      const MemorySpaceAssignment::Allocation& allocation);

  // If aliased_offset is non-null, this method adds the allocation to
  // aliased_offset. Otherwise, it creates a new AliasedOffset object and adds
  // the allocation to this new AliasedOffset.
  void CreateOrAddToAliasedOffset(
      const MemorySpaceAssignment::Allocation& allocation,
      AliasedOffset* aliased_offset);

  // Given an allocation sequence, returns the live allocation at time with a
  // preference towards allocations in alternate memory. Returns nullptr if no
  // allocation is alive at that time.
  static MemorySpaceAssignment::Allocation* GetLiveAllocationAt(
      const MemorySpaceAssignment::AllocationSequence& allocations,
      int64_t time);

  // Returns true if the use is allowed in the alternate memory.
  bool IsUseAllowedInAlternateMemory(const AllocationValue& value,
                                     const HloUse& use) const;

  // Finds allocations for allocation values generated from colocated intervals.
  // All of the allocation values have a must-alias relationship with each
  // other. Returns either kSuccess if all of the sites could be placed in the
  // alternate memory or a bitwise OR of failure reasons why they couldn't
  Result AllocateAllocationValues(
      absl::Span<AllocationValue> allocation_values);

  // Finds an allocation for an allocation request for a segment (see the
  // documentation for AllocationRequest above how a segment is defined).
  //
  // It performs three things in the following order:
  //  1- Allocate the allocation request entirely in the alternate memory, if
  //     there is enough space and if the prefetch interval picker allows.
  //  2- If (1) was unsuccessful, and the only allocation for
  //     this buffer was in the alternate memory, we try to perform a prefetch.
  //  3- If (1) was unsuccessful, prefetch the buffer into the alternate memory,
  //     if there is enough space and if the prefetch interval picker allows.
  //
  // If an eviction (2) was requested and was unsuccessful, this method returns
  // Result::kFailRequiresUncommit. This means we could not find a suitable
  // allocation, so all previous allocations for this buffer must be removed and
  // allocated in the default memory. Otherwise, this method may return
  // Result::kSuccess if the buffer could be placed in alternate memory or some
  // other Result with an OR of reasons why the buffer couldn't be placed in
  // alternate memory.
  Result AllocateSegment(const AllocationRequest& request);

  // Try allocating in alternate memory without any copies.
  Result AllocateInAlternateMemoryNoCopy(const AllocationRequest& request);

  // Try evicting to default memory space.
  Result Evict(const AllocationRequest& request);

  // Returns the time a copy done of a prefetch should be scheduled.
  int64_t FindPrefetchEndTime(const AllocationRequest& request,
                              int64_t earliest_prefetch_time) const;

  // Try prefetching to alternate memory space.
  Result Prefetch(
      const AllocationRequest& request,
      const MemorySpaceAssignment::Allocation& prev_allocation_in_default_mem);

  // Find the best possible chunk candidate, where it has the longest possible
  // availability if no preferred offset is given, or at the preferred_offset if
  // it is given.
  std::optional<Chunk> FindBestChunkCandidate(
      const AllocationRequest& request, const AliasedOffset* preferred_offset,
      BufferInterval* alternate_mem_interval) const;

  // Returns the required assignment at a particular time, if available.
  std::optional<RequiredMemoryAssignment> RequiredMemoryAssignmentAt(
      const HloValue* buffer, int64_t time) const;

  // Searches for aliases in the use for a required assignment, and returns it
  // if found.
  std::optional<RequiredMemoryAssignment> AliasedRequiredAssignmentForUse(
      const AllocationValue::Use& use) const;

  // Goes through the colocated intervals and adds any required assignment.
  void AddRequiredAssignmentsForColocatedIntervals(
      absl::Span<const AlternateMemoryBestFitHeap::BufferInterval* const>
          colocated_intervals);

  // Propagates aliased required assignment for a given position.
  void AddAliasedRequiredAssignment(
      const HloInstruction* instruction, ShapeIndex index,
      const MemorySpaceAssignment::Allocation* aliased_allocation);

  // This sets a required assignment. CHECK fails if there is a conflicting
  // required assignment at the same time.
  void AddRequiredAssignment(const HloValue* value,
                             const HloInstruction* instruction,
                             MemorySpace memory_space, int64_t time,
                             AliasedOffset* offset = nullptr,
                             bool add_to_pending = true);
  void AddRequiredAssignment(const HloInstruction* instruction,
                             ShapeIndex index, MemorySpace memory_space,
                             AliasedOffset* offset = nullptr);

  // Adds input and outputs as required assignments.
  void AddInputAndOutputRequiredAssignments();

  // Returns true if the colocated intervals in the argument are in a parameter
  // or root instruction of the entry computation and are reserved by the user
  // to be in the alternate memory space.
  bool AreIntervalsReservedInAlternateMemory(
      absl::Span<const BufferInterval* const> colocated_intervals) const;

  // Since the allocations are recorded to the AllocationSequence, we don't
  // maintain result_ in GlobalDecreasingSizeBestFitHeap. Override AddToChunkMap
  // to avoid unnecessarily adding the chunk to the chunk map.
  void AddToChunkMap(const HloValue* buffer, Chunk chunk) override {}

  // Returns true if the addition of an asynchronous copy in the given time
  // interval would violate the maximum number of asynchronous copies. An extra
  // async copy limit can be provided to increase the limit of asynchronous
  // copies for this instance.
  bool ViolatesMaximumOutstandingAsyncCopies(
      int64_t start_time, int64_t end_time, bool is_prefetch,
      int64_t extra_async_copy_limit = 0) const;

  // Exports the allocations for repacking and puts them into the vector in the
  // parameter.
  void ExportAllocationsForRepacking(
      std::vector<MemorySpaceAssignmentRepacker::AllocationBlock*>&
          allocations);

  // Imports repacked allocations and updates the internal data structures
  // consistent with the new packing.
  void ImportRepackedAllocations();

  // Adds an asynchronous copy to the allocations.
  void AddAsyncCopy(
      const MemorySpaceAssignment::Allocation& prev_allocation,
      MemorySpace memory_space, std::optional<Chunk> chunk, int64_t start_time,
      int64_t end_time, int64_t copy_done_schedule_before_time,
      MemorySpaceAssignment::AllocationSequence* allocations,
      AliasedOffset* aliased_offset, float resource,
      std::optional<int> cross_program_prefetch_index = std::nullopt);

  // This method is used for committing the chunk candidate but adding it to
  // pending_chunks_ so that we can "uncommit" them in case we need to roll back
  // this allocation sequence.
  void AddToPendingChunks(const BufferInterval& buffer_interval,
                          const Chunk& chunk);
  // If we need to remove the allocations for this allocation sequence, this
  // removes pending chunks and asynchronous copies in the respective pending
  // buffers from the interval trees. If an allocation request returns
  // kFailRequiresUncommit, this method must be called.
  void UncommitPendingChunks(absl::Span<AllocationValue> allocation_values);

  // Finalizes the allocations where they can no longer be uncommitted.
  void FinalizeAllocations(absl::Span<AllocationValue> allocation_values);

  // Clears all pending chunks and asynchronous copies.
  void ClearPendingChunks();

  // Append buffer and allocation infos for debugging and dump it into a file,
  // if enabled.
  void AppendBufferInfoDebugString(const BufferInterval& interval,
                                   std::string* debug_str) const;
  void AppendScopedAllocationBufferInfoDebugString(
      const HloInstruction* instruction, int64_t time, int64_t size,
      std::string& debug_str) const;
  void AppendAllocationInfoDebugString(
      const MemorySpaceAssignment::Allocation& allocation,
      std::string& debug_str) const;
  void DumpDebugStringsIfEnabled() const;

  // Returns the available heap size in the alternate memory.
  int64_t available_heap_size() const {
    return options_.max_size_in_bytes - reserved_in_bytes_;
  }

  // Returns the earliest time in the [start_time, end_time] range that a new
  // allocation with the given size would fit in the alternate memory. If it
  // doesn't fit, it returns nullopt.
  std::optional<int> FindEarliestTimeToSatisfyPeakMemory(int start_time,
                                                         int end_time,
                                                         int64_t size) const;

  // Creates and returns a RepackAllocationBlock.
  static RepackAllocationBlock MakeRepackAllocationBlock(
      int64_t start_time, int64_t end_time, int64_t size,
      int64_t initial_offset, int64_t id,
      MemorySpaceAssignment::Allocation* allocation) {
    RepackAllocationBlock allocation_block;
    allocation_block.start_time = start_time;
    allocation_block.end_time = end_time;
    allocation_block.size = size;
    allocation_block.offset = -1;
    allocation_block.initial_offset = initial_offset;
    allocation_block.id = id;
    allocation_block.colocations = {};
    allocation_block.allocation = allocation;
    return allocation_block;
  }

  // Returns a vector of instructions that have the same fingerprint as this
  // instruction.
  const std::vector<const HloInstruction*>* GetRepeatedInstructionList(
      const HloInstruction* instruction) const;

  MemorySpaceAssignment::AllocationSequence* allocations_;
  const Options& options_;
  const HloAliasAnalysis& alias_analysis_;
  const HloLiveRange& hlo_live_range_;
  // We use a interval tree to keep track of the number of outstanding
  // prefetches and evictions.
  BufferIntervalTree prefetch_interval_tree_;
  BufferIntervalTree eviction_interval_tree_;
  AsynchronousCopyOrdering async_copy_ordering_;
  AsynchronousCopyResource prefetch_async_copy_resource_;
  AsynchronousCopyResource eviction_async_copy_resource_;
  // A list of RepackAllocationBlock objects that mirrors allocation sequences,
  // used for repacking. We use a list here because we need pointer stability
  // for aliased allocations.
  std::list<RepackAllocationBlock> repack_allocation_blocks_;
  int64_t num_repacks_ = 0;
  std::vector<std::pair<BufferInterval, Chunk>> pending_chunks_;
  std::vector<AsynchronousCopy> pending_async_copies_;
  std::vector<std::pair<const HloValue*, RequiredMemoryAssignment>>
      pending_required_assignments_;
  // A cache to keep the peak memory usage at each point in the graph. We use
  // this to see if the proposed allocation in the alternate memory would fit
  // ignoring fragmentation, and if not, we can skip the more expensive lookup
  // in the BufferIntervalTree, which also considers fragmentation.
  std::vector<int64_t> peak_memory_usage_;
  // The data structure that contains AliasedOffset objects and Allocation to
  // AliasedOffset map for efficient lookup.
  std::list<AliasedOffset> aliased_offsets_;
  absl::flat_hash_map<const MemorySpaceAssignment::Allocation*, AliasedOffset*>
      aliased_offset_map_;
  // This map contains required memory assignments for HloValues (e.g., input
  // and outputs).
  absl::flat_hash_map<const HloValue*, std::vector<RequiredMemoryAssignment>>
      required_assignments_;
  // Number of bytes reserved in alternate memory space.
  int64_t reserved_in_bytes_ = 0;
  // A rough measure of the memory pressure of the model, in bytes. Note that
  // this is pressure for memory capacity (and not accessed bytes), and for
  // alternate memory (not default memory).
  int64_t memory_pressure_ = 0;
  int64_t next_async_copy_id_ = 0;
  // Fingerprint cache.
  absl::flat_hash_map<const HloInstruction*, std::string> fingerprint_map_;
  // Vector of repeated instructions (that have the same fingerprint) indexed by
  // fingerprint.
  absl::flat_hash_map<std::string, std::vector<const HloInstruction*>>
      repeated_inst_map_;
  // Debug strings.
  std::string buffer_info_str_;
  std::string allocation_info_str_;
};
}  // namespace memory_space_assignment
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_H_

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

#ifndef XLA_SERVICE_HEAP_SIMULATOR_HEAP_SIMULATOR_H_
#define XLA_SERVICE_HEAP_SIMULATOR_HEAP_SIMULATOR_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// TODO(b/210891274): Use btree_map after build issue in Windows is resolved.
#if defined(__GNUC__) || defined(__clang__)
#include "absl/container/btree_map.h"
#endif
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/buffer_value.h"
#include "xla/service/heap_simulator/allocation_block.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_alias_analysis.h"
#include "xla/service/hlo_value.h"
#include "xla/service/logical_buffer.h"

namespace xla {

// Forward declare classes defined below.
template <typename BufferType>
class HeapAlgorithm;
template <typename BufferType>
class NoFragmentationStatsHeap;

// HeapSimulator assigns buffer offsets by running a simulation of a regular
// memory heap with Alloc and Free calls.  It only works for completely
// sequential instruction sequences.  Unlike regular heaps, we have the
// advantage that the sequence of Alloc and Free calls is known up-front; we
// don't need to return the assignment of buffer offsets until the very end.
class HeapSimulator {
 public:
  // Chunk represents a contiguous piece of memory.  Each BufferValue will be
  // associated with a chunk in the assignment result.
  struct Chunk {
    static Chunk FromOffsetEnd(int64_t offset, int64_t end);
    static Chunk FromOffsetSize(int64_t offset, int64_t size);
    Chunk() : Chunk(-1, 0) {}

    std::string ToString() const;

    int64_t offset;
    int64_t size;

    int64_t chunk_end() const { return offset + size; }

    bool OverlapsWith(Chunk other_chunk) const;

    bool operator==(const Chunk& other) const {
      return offset == other.offset && size == other.size;
    }

   private:
    Chunk(int64_t offset, int64_t size) : offset(offset), size(size) {}

    friend std::ostream& operator<<(std::ostream& stream, const Chunk& chunk);
  };

  template <typename BufferType>
  struct HeapResult {
    // Returns the updated heap size if `chunk` is added to the heap.
    int64_t UpdatedHeapSize(const Chunk& chunk) const {
      return std::max(heap_size, chunk.chunk_end());
    }

    // The assignment of buffers to chunks.
    absl::flat_hash_map<const BufferType*, Chunk> chunk_map;

    // The total size in bytes of the heap, containing all assigned chunks.
    int64_t heap_size = 0;
  };
  // Result represents the result of the heap simulation.
  template <typename BufferType>
  struct Result {
    // Heap results.
    std::vector<HeapResult<BufferType>> heap_results;

    // The total size in bytes of the heaps.
    // heap_size == sum([hr.heap_size for hr in heap_results]).
    int64_t heap_size = 0;

    // The total size in bytes of heap fragmentation.
    int64_t fragmentation_size = 0;

    // A trace of heap simulation events.
    HeapSimulatorTrace debug_trace;
  };

  // The different options to be passed to the Run() APIs.
  struct Options {
    Options()
        : may_reuse_operand_buffers(true),
          alloc_constants(false),
          buffers_to_assign(nullptr) {}

    // Whether a buffer about to be Free()-ed, can be recycled for a new born
    // one, hence collapsing Free()+Alloc() calls (default true).
    bool may_reuse_operand_buffers;
    // Whether to issue Alloc() and Free() calls for constants (default false).
    bool alloc_constants;
    // If 'buffers_to_assign' is provided, only those buffers are assigned
    // offsets, otherwise all buffers defined by the instructions are assigned.
    const absl::flat_hash_set<const HloValue*>* buffers_to_assign;
  };

  // Returns the minimum memory required to compute an HLO module where all
  // computations have been scheduled (represented by the given
  // schedule), assuming no fragmentation.
  static absl::StatusOr<int64_t> MinimumMemoryForModule(
      const HloSchedule& schedule,
      const LogicalBuffer::SizeFunction& size_function);

  // Returns the minimum memory required to compute the given computation,
  // assuming no fragmentation.
  static absl::StatusOr<int64_t> MinimumMemoryForComputation(
      const HloComputation& computation, const HloInstructionSequence& sequence,
      const HloAliasAnalysis& alias_analysis,
      const LogicalBuffer::SizeFunction& size_function);

  static absl::StatusOr<int64_t> MinimumMemoryForComputation(
      const HloComputation& computation, const HloInstructionSequence& sequence,
      const HloAliasAnalysis& alias_analysis,
      const LogicalBuffer::SizeFunction& size_function,
      const HloSchedule* schedule);

  // Run the heap simulation with the given algorithm, assuming the given
  // schedule, which must contain a topologically-consistent total
  // ordering of all instructions within each computation. The result is invalid
  // if instructions are not run in exactly this sequence.
  //
  // Running heap simulation on the whole module tends to save memory, compared
  // to running on a per-computation basis, since we can re-use buffer space for
  // called sub-computations.
  //
  static absl::StatusOr<Result<HloValue>> Run(
      std::unique_ptr<HeapAlgorithm<HloValue>> algorithm,
      const HloModule& module, const HloSchedule& schedule,
      const HloAliasAnalysis& alias_analysis,
      const BufferValue::SizeFunction& size_fn,
      const Options& options = Options());

  // Same as above, but runs on a single computation. The 'instruction_sequence'
  // must contain a topologically-consistent total ordering of all instructions
  // in the computation. The result is invalid if instructions are not run in
  // exactly this sequence.
  static absl::StatusOr<Result<HloValue>> Run(
      std::unique_ptr<HeapAlgorithm<HloValue>> algorithm,
      const HloComputation& computation,
      const HloInstructionSequence& instruction_sequence,
      const HloAliasAnalysis& alias_analysis,
      const BufferValue::SizeFunction& size_fn,
      const Options& options = Options());

  // Same as above, but runs on with a schedule that covers all nested
  // computations.
  static absl::StatusOr<Result<HloValue>> Run(
      std::unique_ptr<HeapAlgorithm<HloValue>> algorithm,
      const HloComputation& computation,
      const HloInstructionSequence& instruction_sequence,
      const HloAliasAnalysis& alias_analysis,
      const BufferValue::SizeFunction& size_fn, const HloSchedule* schedule,
      const Options& options = Options());

 private:
  // If 'schedule' is non-null, it is used to find kCall and kWhile
  // sub-computations, and the heap simulation for those sub-computations will
  // be run recursively. I.e. the simulation is run over the whole module.
  HeapSimulator(std::unique_ptr<HeapAlgorithm<HloValue>> algorithm,
                const BufferValue::SizeFunction& size_fn,
                const Options& options, const HloSchedule* schedule = nullptr);
  ~HeapSimulator();

  absl::Status RunComputation(
      const HloComputation& computation,
      const HloInstructionSequence& instruction_sequence,
      const HloAliasAnalysis& alias_analysis, HloLiveRange* live_range);

  bool IgnoreBuffer(const HloValue* buffer) const;
  void Alloc(const HloValue* buffer, const HloInstruction* instruction);
  void Free(const HloValue* buffer, const HloInstruction* instruction);
  // ShareBuffer indicates that a new buffer is defined and it has to be the
  // same address as the shared one.
  void ShareBuffer(const HloValue* buffer, const HloValue* shared,
                   const HloInstruction* instruction);

  // Returns the size of the HloValue, which is the max size of the HloValues
  // that are part of the HloBuffer.
  int64_t GetBufferSize(const HloValue* buffer) const;

  // Returns true if:
  //  Two buffers belong to the same shared group.
  //  Eight of the buffer has no shared group assigned.
  bool InSameSharedGroup(const HloValue* left, const HloValue* right);
  absl::StatusOr<Result<HloValue>> Finish();

  void FillDebugTrace(HeapSimulatorTrace::Event::Kind kind,
                      const HloValue* buffer, const HloInstruction* instruction,
                      const HloValue* share_with_canonical);

  // Counterintuitive: the algorithm_ itself can be a NoFragmentationStatsHeap,
  // in which case we are calculating the same allocs/frees twice in the
  // simulation.
  const std::unique_ptr<NoFragmentationStatsHeap<HloValue>>
      no_fragmentation_stats_;
  const std::unique_ptr<HeapAlgorithm<HloValue>> algorithm_;
  const BufferValue::SizeFunction size_fn_;
  const Options options_;
  // schedule_ is set by buffer assignment. Then, in RunComputation, we check
  // both in order to handle subcomputations. It would be good to unify the
  // handling of subcomputations, but it's not clear how.
  const HloSchedule* schedule_;

  // Hold some sets for error-checking the sequence of Alloc and Free calls.
  absl::flat_hash_set<const HloValue*> allocated_buffers_;
  absl::flat_hash_set<const HloValue*> freed_buffers_;

  absl::flat_hash_map<const HloValue*, int64_t> buffer_sizes_;

  // Debugging information filled in while the heap simulator runs.
  HeapSimulatorTrace debug_trace_;
};

// Abstract base class describing a heap simulation algorithm that assigns
// offsets to buffers.  A sequence of Alloc / Free calls will be made, with the
// same semantics as a regular memory heap.  Finish will be called at the end to
// collect the simulation results.
template <typename BufferType>
class HeapAlgorithm {
 public:
  using Chunk = HeapSimulator::Chunk;
  using Result = HeapSimulator::Result<BufferType>;
  using HeapResult = HeapSimulator::HeapResult<BufferType>;

  virtual ~HeapAlgorithm() = default;

  // Alloc allocates a buffer of 'size' bytes.
  virtual void Alloc(const BufferType* buffer, int64_t size) = 0;

  // Takes memory usage of subcomputations into account when calculating the
  // memory usage of a computation. Currently, we don't handle buffer aliasing
  // between computations entirely correctly. We are careful to not double count
  // for the output buffers of whiles/conds/calls. But we don't take into
  // account other aliases, such as for the while init. A more thorough solution
  // would require something like BufferAssignment::BuildColocatedBufferSets.
  // TODO(b/65835246):
  // Since TuplePointsToAnalysis is being replaced with a module-aware alias
  // analysis, it's not worth making major changes to HeapSimulator now.
  virtual void AccountForSubcomputationMemory(
      const HloInstruction* instruction,
      // The total number of bytes allocated by instruction.
      int64_t alloc_size_by_instruction) {}

  // Free de-allocates a previously allocated buffer.
  virtual void Free(const BufferType* buffer, int64_t size) = 0;

  // Indicates that a buffer has to be collocated with another buffer. In
  // addition to Alloc and Free, the heap simulator exposes a concept of buffer
  // sharing.  When ShareBuffer is called, instead of allocating new space for
  // the buffer, it associates the buffer with a previously allocated (or
  // shared) buffer.  Each group of mutually-shared buffers points to a single
  // SharedGroup instance, which is a shared control block.
  virtual void ShareWith(const BufferType* buffer, const BufferType* share_with,
                         int64_t size) {
    Alloc(buffer, size);
  }

  // Finish collects the buffer offset assignment results.  Finish may only be
  // called once, after all Alloc and Free calls.
  virtual absl::StatusOr<Result> Finish() = 0;
};

// NoFragmentationStatsHeap computes the heap size assuming no fragmentation;
// this is the absolute minimum size for a given instruction sequence.  The
// result.chunk_map returned in Finish is always empty, since we only collect
// stats, and don't actually compute chunk assignments.
template <typename BufferType>
class NoFragmentationStatsHeap : public HeapAlgorithm<BufferType> {
 public:
  using Result = HeapSimulator::Result<BufferType>;

  NoFragmentationStatsHeap() = default;
  ~NoFragmentationStatsHeap() override = default;

  void Alloc(const BufferType* buffer, int64_t size) override;

  void AccountForSubcomputationMemory(
      const HloInstruction* instruction,
      int64_t alloc_size_by_instruction) override;

  void Free(const BufferType* buffer, int64_t size) override;

  absl::StatusOr<Result> Finish() override;

 private:
  int64_t current_heap_size_ = 0;
  int64_t max_heap_size_ = 0;
};

// Node in BufferIntervalTree that stores the alloc and free times of a buffer,
// and the chunk assigned to it.
struct BufferIntervalTreeNode {
  // Alloc time.
  int64_t start;
  // Free time.
  int64_t end;
  // Maximum free time of all nodes in the subtree where this node is the root.
  int64_t subtree_end;
  // Allocated chunk for the buffer.
  HeapSimulator::Chunk chunk;
  // Left child.
  BufferIntervalTreeNode* left;
  // Right child.
  BufferIntervalTreeNode* right;
  // parent
  BufferIntervalTreeNode* parent;

  std::string ToString() const;
};

// An interval tree that can query buffers overlapping in time.
class BufferIntervalTree {
 public:
  using Chunk = HeapSimulator::Chunk;
  // Adds a buffer to the interval tree, with the time interval and allocated
  // chunk specified.
  void Add(int64_t start, int64_t end, const Chunk& chunk);

  // Remove the interval from the tree. Returns true if the chunk is removed.
  bool Remove(int64_t start, int64_t end, const Chunk& chunk);

  // Returns vector of allocated chunks that overlap with the given time
  // interval.
  std::vector<Chunk> ChunksOverlappingInTime(int64_t start, int64_t end) const;

  BufferIntervalTreeNode* GetRoot() { return root_; }

  // Returns a compact 2D view of memory usage over time.
  // X axis is time, Y axis is memory.
  //
  // Say there are 3 buffers in the heap:
  // - Buffer 1: memory block [0, 16), time interval [15, 25]
  // - Buffer 2: memory block [16, 48), time interval [15, 19]
  // - Buffer 3: memory block [32, 64), time interval [20, 22]
  //
  // NodesOverlappingInTimeToAsciiArt(/*start=*/18, /*end=*/23,
  // /*group_size=*/3) returns:
  //
  // Memory map for time: [18,23], memory_block_size: 16, group_size: 3
  //
  //  ..# ##. 64
  //  ### ##. 48
  //  ##. ... 32
  //  ### ### 16
  //  890 123
  //
  // Explanation:
  //
  // The functions decides a memory block size of 16 would be most compact to
  // display all the buffers.
  // '#' indicates used and '.' indicates free memory.
  //
  // ..# ##. 64      "64" indicates memory block [48,64)
  // ### ##. 48      "48" indicates memory block [32,48)
  // ##. ... 32      "32" indicates memory block [16,32)
  // ### ### 16      "16" indicates memory block [0,16)
  // 890 123
  //
  // "890 123" indicate the last digits of time instants 18, 19, 20, 21, 22, 23.
  // Only the last digit is shown for compactness.
  // `group_size=3` inserts spaces after every 3 columns (time instants).
  // All the memory blocks beyond 64 are free for time interval [18,23].
  std::string NodesOverlappingInTimeToAsciiArt(int64_t start, int64_t end,
                                               int64_t group_size = 0) const;

  // Returns a vector of size `end - start + 1` where the element at index i is
  // the memory used at the time instant `start + i`. Both `start` and `end` are
  // inclusive.
  std::vector<int64_t> MemoryUsedInInterval(int64_t start, int64_t end) const;

 private:
  std::vector<const BufferIntervalTreeNode*> NodesOverlappingInTime(
      int64_t start, int64_t end) const;

  BufferIntervalTreeNode* root_ = nullptr;
  std::list<BufferIntervalTreeNode> node_storage_;
};

// An iterator that is passed to
// GlobalDecreasingSizeBestFitHeap::CreateSlicedAllocationFinder() when trying
// to place a buffer, telling the finder which permutations of starting slice
// times to try (and in which order to try them).
// * The set of slice times is the set {x : x ∈ [0, num_slices - 1]}. If a
//   buffer is not sliced, it will only have 1 permutation, containing slice
//   time 0.
// * The ith value in a permutation is the slice time for the slice at the
//   ith smallest offset.
// * Iterators skip permutations that are equivalent to previously emitted
//   permutations. The ith smallest slice time corresponds to the ith smallest
//   inclusive start time. Let the start_time_permutation be the mapping of a
//   permutation to its corresponding start times. Two permutations are
//   equivalent if their start_time_permutations are equivalent. For example,
//   let's say slice time 0 and slice time 1 both map to inclusive start time
//   1000. There is no difference in permutation [0, 1, x] and [1, 0, x]
//   because the first two slices map to the same inclusive start time.
// * When repacking slice data is provided, iterators skip invalid
//   permutations. A permutation is invalid if the mapping from inclusive
//   start times to slice sizes is not maintained from before the repack.
// * Begin() must be called to initialize the iterator before it can be used.
class SliceTimePermutationIterator {
 public:
  enum class Ty : std::int8_t {
    // Include all valid permutations
    kAll,
    // Only include perferred valid permutations. Heap simulator is trying to
    // optimize fitting allocations into a grid of (heap) space by time. The
    // preferred permutation iterator only allows the following triagular
    // shapes:
    //
    //     Smaller offsets      Smaller offsets      Slice times are
    //    get smaller slice     get larger slice   distributed around
    //         times                  times         the middle offset
    //
    // space                space                space
    //   ^                    ^                    ^
    //   |             +--+   | +--------------+   |             +--+
    //   |          +--+  |   | +--+           |   |       +-----+  |
    //   |       +--+     |   |    +--+        |   | +-----+        |
    //   |    +--+        |   |       +--+     |   | +--+           |
    //   | +--+           |   |          +--+  |   |    +-----+     |
    //   | +--------------+   |             +--+   |          +-----+
    //   +------------------> +------------------> +------------------> time
    //
    // We deviate from those shapes as needed to make valid permutations.
    kPreferred,
  };

  // A new iterator is typically created for each buffer to be placed.
  // - num_slices: number of slices in the buffer. 1 if not sliced.
  // - original_sliced_allocation: For a repacking scenario, the original
  //   details of each slice in a sliced buffer. nullptr is used if the buffer
  //   was not sliced. (Note, if the repacker has no slicing data, it is
  //   treated as unsliced in the repacker and by this iterator.)
  static std::unique_ptr<SliceTimePermutationIterator> CreateForNewAllocation(
      Ty ty, absl::Span<const int64_t> inclusive_slice_start_times);
  static std::unique_ptr<SliceTimePermutationIterator> CreateForRepack(
      Ty ty, const SlicedAllocationData* original_sliced_allocation);

  virtual ~SliceTimePermutationIterator() = default;

  virtual void Begin() = 0;
  virtual bool Done() const = 0;
  virtual void Next() = 0;

  // A permutation of starting slice times.
  virtual absl::Span<const int64_t> Get() const = 0;

 protected:
  SliceTimePermutationIterator() = default;
};

// GlobalDecreasingSizeBestFitHeap collects the live intervals of all buffers,
// then allocates them in decreasing spatial or temporal size regardless of the
// alloc/free time. It internally tracks the allocated buffers and their live
// intervals; when allocating a buffer, it finds the best-fit free chunk during
// its live interval.
template <typename BufferType>
class GlobalDecreasingSizeBestFitHeap : public HeapAlgorithm<BufferType> {
 public:
  using HeapResult = HeapSimulator::HeapResult<BufferType>;
  using Result = HeapSimulator::Result<BufferType>;
  using Chunk = HeapSimulator::Chunk;

  // A mapping from a free chunk offset to the end of that chunk (exclusive).
#if defined(__GNUC__) || defined(__clang__)
  using FreeChunks = absl::btree_map<int64_t, int64_t, std::greater<int64_t>>;
#else
  using FreeChunks = std::map<int64_t, int64_t, std::greater<int64_t>>;
#endif

  enum Type {
    kSpatial = 0,
    kTemporal,
    // Custom uses a custom BufferIntervalCompare function provided in the
    // constructor.
    kCustom
  };

  // BufferInterval stores a buffer's size and time interval.
  struct BufferInterval {
    // Convenience method for use with debugging and logging.
    std::string ToString() const;

    const BufferType* buffer = nullptr;
    int64_t size = -1;
    // Alloc time of the buffer.
    int64_t start = -1;
    // Free time of the buffer.
    int64_t end = -1;

    // Colocation buffers that need to be collocated with this one.
    absl::InlinedVector<const BufferType*, 2> colocations;

    // True if this buffer needs an allocation. False if it is collocated with
    // other buffer.
    bool need_allocation = false;
  };

  // Comparison function that is used to store buffer intervals.
  using BufferIntervalCompare =
      std::function<bool(const BufferInterval&, const BufferInterval&)>;

  // SlicedBufferInterval is a wrapper around BufferInterval with parameters
  // indicating whether the BufferInterval should be allocated in slices. (If
  // NumSlices() is 1, the allocation will not be sliced.) This class is used as
  // input to GlobalDecreasingSizeBestFitHeap::FindChunkCandidates().
  //
  // For example, instead of allocating A in space and time as illustrated on
  // the left, we may wish to allocate A0 and A1 overlapping in time, contiguous
  // in memory, (as illustrated on the right). Doing so allows us to free up
  // allocation space between [s,i], but we only have the full allocation for A
  // from [i,e].
  //
  //   ^
  // s | +-----------+                 s |       +-----+
  // p | |           |                 p |       |  A1 |
  // a | |     A     |                 a | +-----+-----+
  // c | |           |                 c | |     A0    |
  // e | +-----------+                 e | +-----------+
  //   --|-----------|------->           --|-----|-----|------->
  //     s           e   time              s     i     e   time
  class SlicedBufferInterval {
   public:
    // Factory constructors.
    static const SlicedBufferInterval CreateConstInterval(
        const BufferInterval& full_buffer_interval);
    static SlicedBufferInterval CreateMutableInterval(
        BufferInterval& full_buffer_interval);

    SlicedBufferInterval() = delete;

    // Updates the number of slices, and slice sizes. An empty
    // slice_sizes_sorted_by_offset is treated the same as setting the number of
    // slices to 1. Every time Slice() is called with a set of sizes > 1, it
    // should be followed at some point by a call to UpdateSliceStartTimes, to
    // update slice start times. Otherwise, the slice start times are
    // meaningless.
    //
    // REQUIRES:
    // - sum(slice_sizes_sorted_by_offset) == full_buffer_interval_.size
    void Slice(absl::Span<const int64_t> slice_sizes_sorted_by_offset);

    // Updates the times at which we will start each slice. However, we have not
    // yet decided which slice size will correspond to which start time.
    //
    // Mutates mutable_full_buffer_interval_.
    //
    // REQUIRES:
    // - The SlicedBufferInterval was constructed using CreateMutableInterval.
    // - *_start_times.size() == NumSlices()
    // - *_start_times should be set such that it is permissible for any
    //   slice size to map to any start time.
    void UpdateExclusiveSliceStartTimes(
        const std::vector<int64_t>& exclusive_start_times);
    void UpdateInclusiveSliceStartTimes(
        const std::vector<int64_t>& inclusive_start_times);

    // Updates the free time for all the slices.
    //
    // Mutates mutable_full_buffer_interval_.
    //
    // REQUIRES:
    // - The SlicedBufferInterval was constructed using CreateMutableInterval.
    void UpdateEndTime(int64_t end_time);

    const BufferInterval& full_buffer_interval() const;
    size_t num_slices() const { return slice_sizes_sorted_by_offset_.size(); }
    const std::vector<int64_t>& SliceSizesSortedByOffset() const;
    std::vector<int64_t> inclusive_start_times() const;

    // Returns a BufferInterval with the requirements to call
    // GlobalDecreasingSizeBestFitHeap::MakeFreeChunks at the specified slice
    // time. The requirements are:
    // - At the latest slice time, we need a contiguous buffer that is big
    //   enough to fit all slices. In addition, that contiguous buffer will have
    //   the same colocation requirements as the full_buffer_interval().
    // - At other slice times, required chunks may be as small as the smallest
    //   slice. Furthermore, their colocation requirements are empty.
    // - The logical start time of the interval at slice time i is the end time
    //   of the interval at slice time i-1.
    const BufferInterval& IntervalForMakeFreeChunks(int64_t slice_time) const;

    // Convenience method for use with debugging and logging.
    std::string ToString() const;

   private:
    explicit SlicedBufferInterval(
        const BufferInterval& full_buffer_interval,
        BufferInterval* mutable_full_buffer_interval = nullptr);

    const BufferInterval& full_buffer_interval_;
    BufferInterval* mutable_full_buffer_interval_ = nullptr;
    std::vector<int64_t> slice_sizes_sorted_by_offset_;
    // make_free_chunks_intervals are indexed by slice time.
    std::vector<BufferInterval> make_free_chunks_intervals_;
  };

  // A class for finding locations to allocate a sliced allocation. A sliced
  // allocation is an allocation of a buffer, in which slices of the buffer are
  // allocated at different times, called slice times. Slice time is a logical
  // time. For example, a requestor may ask for 15 Mib, allocated 5 MiB at a
  // time, at 3 slices times t0, t1, and t2.
  //
  // The primary data structure inside this class is free_chunks_. free_chunks_
  // is a sorted map of the chunks of memory that are free at the latest
  // requested slice time. For each memory offset within each of those chunks,
  // we track the earliest slice time t, such that the memory offset is
  // continuously free during [t, latest requested slice time].
  //
  // For example, the following depiction of free_chunks_ indicates that
  // at slice time t2, we have 2 free chunks, [5,15) and [20, 25). At slice time
  // t1, the free chunk [5,15) is still free at [6,8) and [10,12). At slice time
  // t0, the free chunk [5,15) is still free at [7,8). The free chunk [20, 25)
  // is also free at slice times t0 and t1. (In the depicition, `x` indicates
  // used space and ` ` indicates free space.)
  //
  //    ^
  // t2 |xxxxx          xxxxx     xxxxxx
  // t1 |xxxxxx  xx  xxxxxxxx     xxxxxx
  // t0 |xxxxxxx xxxxxxxxxxxx     xxxxxx
  //    +!----|----!----|----!----|----!>
  //          space
  class SlicedAllocationFinder {
   public:
    // The chunk at index i is the chunk that should be allocated at slice time
    // i.
    using ChunksSortedBySliceTime = std::vector<Chunk>;

    // A structure representing a piece of a free chunk that is continuously
    // free in [piece.earliest_free_slice_time, LatestSliceTime()].
    struct FreeChunkPiece {
      std::string ToString() const;

      int64_t earliest_free_slice_time;
      Chunk dimensions;
    };

    // A sorted map (indexed by starting offset) describing how far back in
    // slice time different pieces of a FreeChunkRoot are free.
#if defined(__GNUC__) || defined(__clang__)
    using FreeChunkPieces =
        absl::btree_map<int64_t, FreeChunkPiece, std::greater<int64_t>>;
#else
    using FreeChunkPieces =
        std::map<int64_t, FreeChunkPiece, std::greater<int64_t>>;
#endif

    // A free chunk that has been split into FreeChunkPieces.
    struct FreeChunkRoot {
      FreeChunkRoot(const Chunk& free_chunk, int64_t free_chunk_slice_time);

      std::string ToString() const;

      // Update pieces in accordance with the knowledge that free_chunk is
      // free at free_chunk_slice_time.
      //
      // REQUIRES:
      // - We must process all updates at free_chunk_slice_time x before
      //   processing those at free time x-1.
      void Update(const Chunk& free_chunk, int64_t free_chunk_slice_time);

      Chunk chunk;
      FreeChunkPieces pieces;
    };

    // A sorted map (indexed by starting offset) of FreeChunkRoots.
#if defined(__GNUC__) || defined(__clang__)
    using FreeChunkRoots =
        absl::btree_map<int64_t, FreeChunkRoot, std::greater<int64_t>>;
#else
    using FreeChunkRoots =
        std::map<int64_t, FreeChunkRoot, std::greater<int64_t>>;
#endif

    // A method that can be passed to the is_offset_allowed parameter for
    // SlicedAllocationFinder() that permits placement at any offset.
    static bool AllOffsetsAllowed(int64_t offset) { return true; }

    // Arguments:
    // - free_chunks_per_slice_time[i]: Describes free chunks at slice time i.
    // - sorted_slice_sizes: A sliced allocation request. In space, the i+1th
    //   slice immediately follows the ith slice.
    // - max_colocation_size: The max size of any buffer that will be colocated
    //   with the fully allocated sliced allocation.
    // - preferred_offset: The preferred starting offset for the fully allocated
    //   sliced allocation.
    // - slice_time_permutation_iterator: An iterator for iterating over the
    //   different slice time permutations for slices. Users may specify the
    //   order in which different permutations are tried by the HeapSimulator.
    //   Users are also responsbile for ensuring that returned permutations are
    //   legal.
    // - is_offset_allowed: Indicates if a the entire sliced allocation is
    //   allowed to be allocated at a given offset.
    //
    // REQUIRES:
    // - sorted_slice_sizes.size() == free_chunks_per_slice_time.size()
    // - any slice can be allocated at any slice time
    // - alignment >= 1
    //
    // In the future, if we want to restrict certain slices to be fetched at
    // certain slice times (e.g., because certain slices don't represent enough
    // real time to allocate a larger slice), we can take a lambda to indicate
    // what is permitted.
    SlicedAllocationFinder(
        absl::Span<const FreeChunks> free_chunks_per_slice_time,
        std::vector<int64_t> sorted_slice_sizes, int64_t max_colocation_size,
        int64_t preferred_offset, int64_t alignment,
        std::unique_ptr<SliceTimePermutationIterator>
            slice_time_permutation_iterator,
        absl::AnyInvocable<bool(int64_t) const> is_offset_allowed =
            &AllOffsetsAllowed);

    std::string FreeChunksToAsciiArt() const;
    std::string ToString() const;

    // Finds a set of chunks in which to allocate the sliced allocation request.
    // Returns a vector of chunks in which the ith element is the chunk that
    // should be allocated at slice time i. If no such chunks can be found, an
    // empty vector is returned.
    //
    // The returned vector will always be 1 larger than the initial request,
    // with a chunk to represent any additional allocation needed for
    // max_colocation_size_. This extra chunk will always come at the end of
    // the returned vector and will be present even if its size is 0.
    ChunksSortedBySliceTime Find() const;

    // Similar to Find(), but only checks placement at the specified offset. If
    // the sliced allocation can not be placed at the specified offset, an
    // empty vector is returned.
    ChunksSortedBySliceTime FindForOffset(int64_t offset) const;

   private:
    // The earliest slice time for the specified sliced allocation request.
    int64_t EarliestSliceTime() const { return 0; }

    // The latest slice time for the specified sliced allocation request.
    int64_t LatestSliceTime() const { return sorted_slice_sizes_.size() - 1; }

    // Returns ok if the given permutation of slice times results in an
    // allocation of free space in root, at the specified offset. Otherwise,
    // returns the reason such an allocation would not fit.
    //
    // permutation_of_slice_times[i] is the slice time that the ith slice
    // (spatially) should be allocated. Such a slice has size
    // sorted_slice_sizes_[i] and would be allocated at offset +
    // sum(sorted_slice_sizes[j], for j in [0, i-1]).
    absl::Status DoesPermutationFit(
        absl::Span<const int64_t> permutation_of_slice_times,
        const FreeChunkRoot& root, int64_t offset) const;

    // Only DoesSlicedPermutationFit() should call this method directly. Other
    // callers should call DoesSlicedPermutationFit(), which contains some
    // wrapper VLOGGING.
    absl::Status DoesPermutationFitImpl(
        absl::Span<const int64_t> permutation_of_slice_times,
        const FreeChunkRoot& root, int64_t offset) const;

    // Same as Find() except only checks root, to see if it can hold the sliced
    // allocation request. If only_try_this_offset is set, we only evaluate the
    // specified offset, when trying to find a fit for the sliced allocation
    // request.
    ChunksSortedBySliceTime FindInRoot(
        const FreeChunkRoot& root,
        std::optional<int64_t> only_try_this_offset = std::nullopt) const;

    // Given a permutation of slice times (see DoesSlicedPermutationFit()),
    // return a vector of chunks, in which the ith chunk should be allocated at
    // slice time i, with size sorted_slice_sizes_[i] and at offset +
    // sum(sorted_slice_sizes[j], for j in [0, i-1]).
    //
    // PermutationToChunks() does the additional job of adding a Chunk to the
    // end of the result to account for an additional colocation space that
    // need to be allocated. This Chunk is added, even if it is of size 0.
    ChunksSortedBySliceTime PermutationToChunks(
        absl::Span<const int64_t> permutation_of_slice_times,
        int64_t offset) const;

    std::vector<int64_t> sorted_slice_sizes_;
    int64_t slice_size_sum_;
    int64_t max_colocation_size_;
    int64_t preferred_offset_;
    int64_t alignment_;
    FreeChunkRoots free_chunks_;
    std::unique_ptr<SliceTimePermutationIterator>
        slice_time_permutation_iterator_;
    absl::AnyInvocable<bool(int64_t) const> is_offset_allowed_;
  };

  explicit GlobalDecreasingSizeBestFitHeap(
      int64_t alignment, Type type = kSpatial,
      BufferIntervalCompare buffer_interval_compare = nullptr,
      SliceTimePermutationIterator::Ty slice_time_permutation_iterator_type =
          SliceTimePermutationIterator::Ty::kAll);
  ~GlobalDecreasingSizeBestFitHeap() override {}

  void Alloc(const BufferType* buffer, int64_t size) override;
  void Free(const BufferType* buffer, int64_t size) override;

  void ShareWith(const BufferType* buffer, const BufferType* share_with,
                 int64_t size) override;

  absl::StatusOr<Result> Finish() override;

  // Return a BufferIntervalCompare function that sort by spatial size. We don't
  // look at co-locates as they should have the same size.
  static BufferIntervalCompare GetSpatialBufferIntervalCompare();

 protected:
  // Returns the buffer intervals sorted according to buffer_interval_compare_.
  std::vector<BufferInterval> GetSortedBufferIntervals() const;

  // Compute free chunks as all memory - chunks that are allocated at some time
  // during the lifetime of buffer_interval or during the lifetime of buffers
  // that are colocated with buffer_interval.
  //
  // Imagine that we've already allocated three chunks : a, b and c.  And now
  // we want to allocate d. Since e is colocated with d, we have to allocate
  // chunks for them together at the same address. To do this, we first gather
  // all chunks that overlap with d and e on the time dimension, in this case
  // the overlapped chunks are a and b (c doesn't overlap with either of d and
  // e), then find create a new chunk that doesn't overlap with a and b on the
  // space dimension.
  //
  // space
  //   ^
  //   |+--d---+      +---e---+
  //   |
  //   |+---+  +---------------+  +-------+
  //   ||   |  |               |  |       |
  //   ||   |  |               |  |       |
  //   |+-a-+  +-------b-------+  +---c---+
  //   ----------------------------------------> time
  //
  // MakeFreeChunks imposes the following additional constraints on its output:
  // - The chunks in the result will start on alignment_ boundaries.
  // - A free chunk will not be returned if it does not have enough space to fit
  //   max_colocation_size.
  FreeChunks MakeFreeChunks(const BufferInterval& buffer_interval,
                            int64_t max_colocation_size) const;

  // These two methods below are exposed to other heap algorithms that inherit
  // from this class. The Finish() method tries to find a candidate chunk for
  // each BufferInterval, after calling GetSortedBufferIntervals. If a
  // non-negative preferred_offset is provided, FindChunkCandidate attempts
  // finding a chunk at this offset. The Finish() method can then call
  // CommitChunk to associate the chunk with the BufferInterval, if the final
  // heap size is within the limits.
  Chunk FindChunkCandidate(const BufferInterval& buffer_interval,
                           int64_t preferred_offset = -1) const;
  // FindChunkCandidates is the same as FindChunkCandidate, except it finds
  // spatially contiguous chunks candidates for a sliced buffer interval.
  // Returned chunk i should be copied at slice time i.
  //
  // Given the way that FindChunkCandidates and MakeFreeChunks interact, the
  // following properties are guaranteed about colocations.
  // - The returned spatially contiguous chunks have enough space for every
  //   colocation specified in sliced_buffer_interval.
  // - The returned spatially contiguous chunks will be free for the entire
  //   lifetime of each colocation. If a colocation is sliced, the returned
  //   chunks will be free for the lifetime of the longest-lived slice.
  std::vector<Chunk> FindChunkCandidates(
      const SlicedBufferInterval& sliced_buffer_interval,
      int64_t preferred_offset = -1) const;
  // The following 3 methods are used to implement FindChunkCandidates.
  int64_t GetMaxColocationSize(const BufferInterval& buffer_interval) const;
  SlicedAllocationFinder CreateSlicedAllocationFinder(
      const SlicedBufferInterval& sliced_interval, int64_t max_colocation_size,
      int64_t preferred_offset,
      std::unique_ptr<SliceTimePermutationIterator>
          slice_time_permutation_iterator,
      absl::AnyInvocable<bool(int64_t) const> is_offset_allowed =
          &SlicedAllocationFinder::AllOffsetsAllowed) const;
  std::vector<Chunk> PostProcessFindChunkCandidatesResult(
      const SlicedBufferInterval& sliced_interval,
      std::vector<Chunk> chunks) const;

  void CommitChunk(const BufferInterval& buffer_interval, Chunk chunk);

  // Adds the buffer and the chunk to the result chunk map.
  virtual void AddToChunkMap(const BufferType* buffer, Chunk chunk);

  // Return a BufferIntervalCompare function that sorts by live ranges.  A live
  // range is defined by the range between the start of the first buffer and the
  // end of the last co-located buffer.  There could be "holes" in the live
  // ranges of each co-located buffers, but in this heuristics we think they are
  // contiguous.
  BufferIntervalCompare GetTemporalBufferIntervalCompare() const;

  SliceTimePermutationIterator::Ty slice_time_permutation_iterator_type() const;

  absl::flat_hash_map<const BufferType*, BufferInterval> buffer_intervals_;
  HeapResult result_;
  BufferIntervalCompare buffer_interval_compare_;
  BufferIntervalTree interval_tree_;

 private:
  int64_t alignment_;

  // The current time represented as an integer. It increments by 1 at each
  // Alloc or Free call.
  int64_t current_time_ = 0;

  SliceTimePermutationIterator::Ty slice_time_permutation_iteration_type_ =
      SliceTimePermutationIterator::Ty::kAll;

 protected:
  // Returns all transitive colocated buffers of this buffer interval. I.e., If
  // a buffer A is colocated with B and B is colocated with C, this function
  // returns all three of them.
  absl::flat_hash_set<const BufferType*> GetTransitiveColocations(
      const BufferInterval& interval) const;
};

// This class implements an algorithm that will produce multiple heaps, where
// each heap size is constrained by a given limit. Note that the constraint is
// soft, meaning that a valid heap result is generated even if there are some
// buffer sizes larger than the given constraint size.
//
// Pseudocode:
//   while( `buffers` is not empty ) {
//     create a new heap `h`
//     for (each buffer `buf` in `buffers` in the size-decreasing order) {
//       if (buf.size() is larger than the heap size limit &&
//           `h` is empty) {
//         h.place(buf)
//         buffers.remove(buf)
//       } else if (placing `buf` into `h` does not violate size
//           constraint) {
//         h.place(buf)
//         buffers.remove(buf)
//       }
//     }
//   }
class ConstrainedGlobalDecreasingSizeBestFitHeap
    : public GlobalDecreasingSizeBestFitHeap<HloValue> {
 public:
  explicit ConstrainedGlobalDecreasingSizeBestFitHeap(
      uint64_t size_limit_per_heap, int64_t alignment, Type type = kSpatial,
      BufferIntervalCompare buffer_interval_compare = nullptr)
      : GlobalDecreasingSizeBestFitHeap<HloValue>(alignment, type,
                                                  buffer_interval_compare),
        size_limit_per_heap_(size_limit_per_heap) {}
  ~ConstrainedGlobalDecreasingSizeBestFitHeap() override {}

  absl::StatusOr<Result> Finish() override;

 private:
  uint64_t size_limit_per_heap_;
};

// A heap algorithm that chooses the best results from other algorithms added to
// it.
template <typename BufferType>
class ChooseBestHeapAlgorithm : public HeapAlgorithm<BufferType> {
 public:
  using Result = HeapSimulator::Result<BufferType>;

  ChooseBestHeapAlgorithm(
      std::unique_ptr<std::vector<std::unique_ptr<HeapAlgorithm<BufferType>>>>
          algorithms)
      : algorithms_(std::move(*algorithms)) {}
  ~ChooseBestHeapAlgorithm() override {}

  void Alloc(const BufferType* buffer, int64_t size) override {
    for (auto& algorithm : algorithms_) {
      algorithm->Alloc(buffer, size);
    }
  }

  void ShareWith(const BufferType* buffer, const BufferType* share_with,
                 int64_t size) override {
    for (auto& algorithm : algorithms_) {
      algorithm->ShareWith(buffer, share_with, size);
    }
  }

  void Free(const BufferType* buffer, int64_t size) override {
    for (auto& algorithm : algorithms_) {
      algorithm->Free(buffer, size);
    }
  }

  absl::StatusOr<Result> Finish() override;

 private:
  std::vector<std::unique_ptr<HeapAlgorithm<BufferType>>> algorithms_;
};

extern template class GlobalDecreasingSizeBestFitHeap<HloValue>;
extern template class GlobalDecreasingSizeBestFitHeap<AllocationBlock>;
extern template class ChooseBestHeapAlgorithm<HloValue>;

}  // namespace xla

#endif  // XLA_SERVICE_HEAP_SIMULATOR_HEAP_SIMULATOR_H_

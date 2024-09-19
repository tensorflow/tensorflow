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

#ifndef XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_MEMORY_BOUND_LOOP_OPTIMIZER_H_
#define XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_MEMORY_BOUND_LOOP_OPTIMIZER_H_

#include <algorithm>
#include <cstdint>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/buffer_value.h"
#include "xla/service/heap_simulator/allocation_block.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_alias_analysis.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_value.h"
#include "xla/service/memory_space_assignment/allocation.h"
#include "xla/service/memory_space_assignment/cost_analysis.h"
#include "xla/service/memory_space_assignment/memory_space_assignment.pb.h"
#include "xla/service/memory_space_assignment/options.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace xla {
namespace memory_space_assignment {

// Pair of chunks for even and odd loop iterations.
using EvenOddChunkPair = std::pair<std::optional<HeapSimulator::Chunk>,
                                   std::optional<HeapSimulator::Chunk>>;

// LoopOptimizerBestFitHeap extends GlobalDecreasingSizeBestFitHeap to track
// allocated buffers and their live intervals for the MemoryBoundLoopOptimizer.
// * We model 4 loop iterations.
// * The 0th and 2nd iterations are even. The 1st and 3rd are odd.
// * Allocations in even iterations are required to have the same offsets.
//   Likewise, allocations in odd iterations are required to have the same
//   offset.
// * Allocations may have different offsets between odd and even iterations.
// * Buffers can span up to 2 iterations.
// * The algorithm uses the 0th and 1st iterations to account for buffers that
//   start in those iterations but are still alive in the 2nd and 3rd
//   iterations. The 2nd and 3rd iterations are used to give the complete loop
//   buffer picture.
class LoopOptimizerBestFitHeap
    : public GlobalDecreasingSizeBestFitHeap<AllocationBlock> {
 public:
  explicit LoopOptimizerBestFitHeap(uint64_t size_limit_per_heap,
                                    int64_t loop_size,
                                    int64_t alignment_in_bytes)
      : GlobalDecreasingSizeBestFitHeap<AllocationBlock>(alignment_in_bytes),
        size_limit_per_heap_(size_limit_per_heap),
        loop_size_(loop_size) {}
  ~LoopOptimizerBestFitHeap() override = default;

  // Frees the memory space denoted by chunk from [begin_idx_in_loop,
  // end_idx_in_loop] from all iterations.
  void RemoveEvenOddChunkPair(int64_t begin_idx_in_loop,
                              int64_t end_idx_in_loop,
                              EvenOddChunkPair& chunks);

  // Displays the current memory usage vs time for 6 loop iterations by default.
  // Note: The 0th and the 1st iterations are just to account for loop around
  // for buffers that go across one or two loop boundaries. The 2nd and the 3rd
  // iterations present the actual memory view of the allocation. The 4th and
  // 5th iterations show buffers from previous two iterations that go across one
  // or two loop boundaries. begin_iteration_idx and end_iteration_idx are both
  // inclusive, 0 indexed.
  std::string MemoryUsageToAsciiArt(int64_t begin_iteration = 0,
                                    int64_t end_iteration = 5) const;

  // Returns a vector of size loop_size, where the i'th element denotes the
  // available(unfragmented) alternate memory in bytes at loop_idx i.
  std::vector<int64_t> RemainingMemoryByTime() const;

  // Returns an integer denoting the largest occupied memory location in the
  // alternate memory.
  int64_t LastMemoryOffsetOccupied() const;

  // Finds free memory chunks of size "size" between [begin_idx_in_loop,
  // end_idx_in_loop] in the even and odd loop iterations, only if free chunks
  // are found in both iterations. The even and odd iteration offsets may be
  // different.
  EvenOddChunkPair FindEvenAndOddAllocationBetween(
      int64_t begin_idx_in_loop, int64_t end_idx_in_loop, int64_t size,
      std::pair<int64_t, int64_t> preferred_offsets = {-1, -1});

  // Finds and reserves free memory chunks of size "size" between
  // [begin_idx_in_loop, end_idx_in_loop] in the even and odd loop iterations,
  // only if free chunks are found in both iterations. The even and odd
  // iteration offsets may be different.
  EvenOddChunkPair AllocateEvenAndOddBetween(
      int64_t begin_idx_in_loop, int64_t end_idx_in_loop, int64_t size,
      std::pair<int64_t, int64_t> preferred_offsets = {-1, -1});

  // Finds free memory chunks of size "size" between [begin_idx_in_loop,
  // end_idx_in_loop] in the even and odd loop iterations, only if free chunks
  // are found in both iterations. The even and odd iteration offsets are same.
  EvenOddChunkPair FindSameEvenAndOddAllocationBetween(
      int64_t begin_idx_in_loop, int64_t end_idx_in_loop, int64_t size,
      int64_t preferred_offset = -1);

  // Finds and reserves free memory chunks of size "size" between
  // [begin_idx_in_loop, end_idx_in_loop] in the even and odd loop iterations,
  // only if free chunks are found in both iterations. The even and odd
  // iteration offsets are same.
  EvenOddChunkPair AllocateSameEvenAndOddBetween(int64_t begin_idx_in_loop,
                                                 int64_t end_idx_in_loop,
                                                 int64_t size,
                                                 int64_t preferred_offset = -1);

 private:
  // REQUIRES:
  // - begin_idx_in_loop <= end_idx_in_loop
  // - begin_idx_in_loop is within [-loop_size loop_size)
  // - end_idx_in_loop is within [0, 2 * loop_size)
  // - end_idx_in_loop - begin_idx_in_loop + 1 <= 2 * loop_size (allocation
  //   colocated in even (or odd) iterations cannot span more than 2 loop
  //   iterations)
  void CheckAllocationIntervalValid(int64_t begin_idx_in_loop,
                                    int64_t end_idx_in_loop) const;

  // Shifts allocation interval at [begin_idx_in_loop, end_idx_in_loop] to
  // [begin_idx_in_loop + loop_size, end_idx_in_loop + loop_size], if
  // begin_idx_in_loop is negative.
  void ShiftAllocationIntervalIfRequired(int64_t& begin_idx_in_loop,
                                         int64_t& end_idx_in_loop) const;

  // Returns pointer to a newly created allocation block that is added to
  // allocation_blocks_.
  const AllocationBlock& GetAllocationBlock(int64_t start_time,
                                            int64_t end_time, int64_t size);

  // Creates a BufferInterval corresponding to the AllocationBlock and adds it
  // to buffer_intervals_.
  void CreateBufferInterval(const AllocationBlock& allocation_block,
                            const AllocationBlock* colocated_with = nullptr);

  std::optional<Chunk> MaybeFindChunkCandidate(
      const AllocationBlock& allocation_block, int64_t preferred_offset = -1);

  std::optional<Chunk> FindAndCommitChunkCandidate(
      const AllocationBlock& allocation_block, int64_t preferred_offset = -1);

  void RemoveChunk(int64_t start_time, int64_t end_time, Chunk chunk);

  void RemoveEvenChunks(int64_t begin_idx_in_loop, int64_t end_idx_in_loop,
                        std::optional<HeapSimulator::Chunk>& chunk);

  void RemoveOddChunks(int64_t begin_idx_in_loop, int64_t end_idx_in_loop,
                       std::optional<HeapSimulator::Chunk>& chunk);

  // Creates colocated allocation blocks for loop interval [begin_idx_in_loop,
  // end_idx_in_loop] in 0th, 1st, 2nd and 3rd loop iterations and returns one
  // AllocationBlock.
  const AllocationBlock& CreateSameEvenAndOddAllocationBlock(
      int64_t begin_idx_in_loop, int64_t end_idx_in_loop, int64_t size);

  // Creates colocated allocation blocks for loop interval [begin_idx_in_loop,
  // end_idx_in_loop] in 0th and 2nd loop iterations and returns one
  // AllocationBlock.
  const AllocationBlock& CreateEvenAllocationBlock(int64_t begin_idx_in_loop,
                                                   int64_t end_idx_in_loop,
                                                   int64_t size);
  // Creates colocated allocation blocks for loop interval [begin_idx_in_loop,
  // end_idx_in_loop] in 1st and 3rd loop iterations and returns one
  // AllocationBlock.
  const AllocationBlock& CreateOddAllocationBlock(int64_t begin_idx_in_loop,
                                                  int64_t end_idx_in_loop,
                                                  int64_t size);

  uint64_t size_limit_per_heap_;
  int64_t loop_size_;
  std::list<AllocationBlock> allocation_blocks_;
};

// An optimizer for unrolled memory-bound loops. It keeps track of alternate
// memory capacity and default memory bandwidth to decide the allocations of
// each tensor within a loop iteration. The assumption is that all of the
// unrolled loop iterations will use the same allocation decisions, so we can
// spend more time to optimize this one iteration as optimally as possible.
//
// To represent instructions, we keep track of three iterations (previous,
// current, and next), as well as the header and footer regions that are before
// and after the loop, respectively.
//
// We classify each tensor used in the current iteration as one of the following
// allocations based on its positions and uses:
//
// Temporary Allocations: These are produced by a producer in the current
// iteration and consumed either in this or the next iteration. For these, we
// try to give them alternate memory allocations for their entire live range.
//
// Case 1: producer and consumer all in the current iteration.
//                                     p-----c--c
// Case 2: producer is in the current iter, consumer is in the next iter.
//                                           p-----c
//  idx:       |...| 0  1  2  3  4| 0  1  2  3  4| 0  1  2  3  4|...|
// iter: head  |...|      prev    |    current   |     next     |...| foot
//
// Loop Carried Dependences: This is where the last use is at a larger index
// than the producer. This would require 2X peak buffer consumption because both
// this and next iteration's buffer is alive at the same time. This case is
// currently not supported.
//
// Case 3: producer is in the current iter, consumer is in the next iter
//         (consumer idx >= producer idx).
//                                           p-----------------c
//  idx:       |...| 0  1  2  3  4| 0  1  2  3  4| 0  1  2  3  4|...|
// iter: head  |...|      prev    |    current   |     next     |...| foot
//
// Pinned Allocations: These are values produced at the header and are used in
// every iteration at the same indices. For these, we just allocate the buffer
// for the duration of the loop:
//
// Case 4: producer: kHead, consumer: kCurrent
//         p---------------c--------------c--------------c--------
//  idx:       |...| 0  1  2  3  4| 0  1  2  3  4| 0  1  2  3  4|...|
// iter: head  |...|      prev    |    current   |     next     |...| foot
//
// Prefetch Allocations: These are values produced at the header and are used in
// the current (and possibly next) iteration. We will try to prefetch these
// values into the alternate memory:
//
// Case 5: producer: kHead, consumer: kCurrent
//         p---------------------------------c--------c
//  idx:       |...| 0  1  2  3  4| 0  1  2  3  4| 0  1  2  3  4|...|
// iter: head  |...|      prev    |    current   |     next     |...| foot
class MemoryBoundLoopOptimizer {
 public:
  // We represent each tensor used in the current iteration as a LoopValue,
  // wrapping the relevant information such as its HLO value, indices and
  // pointers to its use and position sites in different iterations.
  struct LoopValue {
    // An enum that encodes the allocation type that is suitable for this
    // LoopValue. See the comment above on what each of these mean.
    enum class AllocationType {
      kTemporary,
      kLoopCarriedDependence,
      kPinned,
      kPrefetch,
      kUnsupported
    };

    // ToString methods for logging/debugging.
    static std::string AllocationTypeToString(AllocationType allocation_type);
    std::string ToString() const;

    // Returns true if memory-bound loop optimizer supports allocating this type
    // of a loop value.
    bool IsAllocationTypeSupported() const;

    // The HloValues that correspond to this LoopValue.
    std::vector<const HloValue*> hlo_values;
    // The position in the header, if any.
    std::optional<HloPosition> header_position;
    // The loop index and position in the previous and current iterations.
    std::vector<std::pair<int64_t, HloPosition>> prev_iteration_positions;
    std::vector<std::pair<int64_t, HloPosition>> loop_positions;
    // The loop index and use in the current and next iterations.
    std::vector<std::pair<int64_t, HloUse>> loop_uses;
    std::vector<std::pair<int64_t, HloUse>> next_iteration_uses;
    // The allocation type.
    AllocationType allocation_type;
    // Size of this tensor.
    int64_t size;
    // The default memory bandwidth savings were we to successfully put this in
    // the alternate memory using the allocation type, in bytes.
    float savings;
    // The savings divided by the size. This is typically 2 for temporary
    // allocations (skip a write and a read to the default memory). More complex
    // production/consumption patterns may result in higher or lower values. We
    // use this value to sort LoopValues so that the algorithm can prioritize
    // allocating the buffers with the highest savings per byte to the alternate
    // memory.
    float savings_per_byte;
    // The optimized AllocationSequence.
    AllocationSequence allocations;
  };

  // Factory method to create and initialize a MemoryBoundLoopOptimizer.
  static absl::StatusOr<std::unique_ptr<MemoryBoundLoopOptimizer>> Create(
      int loop_start, int loop_end, uint64_t alternate_memory_size,
      const MemoryBoundLoopOptimizerOptions& options,
      const HloLiveRange& hlo_live_range,
      const HloAliasAnalysis& alias_analysis_,
      const CostAnalysis& cost_analysis,
      const BufferValue::SizeFunction& size_function,
      const ReservedScopedMemoryFunction& reserved_scoped_memory_fn);

  // Optimize the loop. Initialize must be called first.
  void Optimize();

  // Calculate the steady-state execution time of one loop iteration using the
  // allocation decisions so far.
  float CalculateExecutionTime() const;

  // Return the LoopValues.
  const std::vector<LoopValue>& loop_values() const { return loop_values_; }
  std::vector<LoopValue>& loop_values() { return loop_values_; }

  // Return the remaining memory vector for each point in time in the loop using
  // the allocation decisions so far.
  const std::vector<int64_t>& remaining_memory() const {
    return remaining_memory_;
  }

  int64_t MaxAlternateMemoryUsed() const {
    return alternate_memory_size_ - *std::min_element(remaining_memory_.begin(),
                                                      remaining_memory_.end());
  }

  // The loop start, end, and size accessors.
  int loop_start() const { return loop_start_; }
  int loop_end() const { return loop_end_; }
  int loop_size() const { return loop_size_; }

 private:
  // Temporary data structures used by the AllocatePrefetch function.
  struct AllocatePrefetchesContext {
    // The values that are requested to be prefetched.
    absl::Span<LoopValue*> values;

    // A list of indices into values array, sorted by the start time of the
    // first use.
    std::vector<int> value_indices;

    // Default memory remaining bandwidths assuming all prefetches succeeded.
    std::vector<float> bandwidth_idle_times;

    // Additional memory used while performing prefetching.
    std::vector<int64_t> additional_memory_used;
  };

  MemoryBoundLoopOptimizer(
      int loop_start, int loop_end, uint64_t alternate_memory_size,
      const MemoryBoundLoopOptimizerOptions& options,
      const HloLiveRange& hlo_live_range,
      const HloAliasAnalysis& alias_analysis_,
      const CostAnalysis& cost_analysis,
      const BufferValue::SizeFunction& size_function,
      const ReservedScopedMemoryFunction& reserved_scoped_memory_fn);

  // Initializes the data structures used by the optimizer.
  absl::Status Initialize();

  // Given an HloBuffer object, determines if this buffer represents a LoopValue
  // that can be optimized by the optimizer, and if so it adds a LoopValue to
  // the back of loop_values_ that represents the HloBuffer. Otherwise, no new
  // LoopValue is added to loop_values_.
  void MaybeCreateLoopValue(const HloBuffer& buffer,
                            const HloComputation* loop_computation);

  // Sort LoopValues by savings_per_byte.
  void SortLoopValues();

  // After allocation finishes, we fix up by creating Allocation objects to any
  // LoopValues that didn't get alternate memory allocations.
  void PostProcess();

  // Allocate LoopValues by dispatching to the correct Allocate method.
  void AllocateLoopValues();

  // Allocate and reserve memory between the given indices.
  bool AllocateBetween(int64_t begin_idx, int64_t end_idx, int64_t size);

  // Perform allocation type kTemporary. Return true if successful.
  bool AllocateTemporary(LoopValue& value);

  // Perform allocation type kPinned. Return true if successful.
  bool AllocatePinned(LoopValue& value);

  // Perform allocation type kPrefetch. Unlike the other Allocate methods, this
  // performs allocation of multiple LoopValues in order to consider the effect
  // of remaining bandwidth assuming the other prefetches were successful.
  // Return true if successful.
  bool AllocatePrefetches(absl::Span<LoopValue*> values);

  // Allocate one prefetch for the loop value index that corresponds to
  // context.context.values. Returns true if successful.
  bool AllocatePrefetch(int value_index, AllocatePrefetchesContext& context);

  // Keeps track of successful allocation of all uses and positions of this
  // LoopValue.
  void AddAllLoopPositionsAndUses(LoopValue& value,
                                  bool allocate_next_iteration_uses);

  // Returns the default memory bandwidth idle time at the index.
  float GetBandwidthIdleTime(int idx) const;

  // Returns the default memory bandwidth idle time at the index assuming the
  // given uses and positions got alternate memory allocations.
  float GetBandwidthIdleTime(
      int idx,
      const absl::flat_hash_map<const HloInstruction*,
                                std::vector<std::pair<int64_t, ShapeIndex>>>&
          additional_uses_in_alternate_mem,
      const absl::flat_hash_map<const HloInstruction*, std::vector<ShapeIndex>>&
          additional_positions_in_alternate_mem) const;

  // Returns the instruction elapsed at the index.
  float GetInstructionElapsed(int idx) const;

  int loop_start_;
  int loop_end_;
  int loop_size_;
  uint64_t alternate_memory_size_;
  MemoryBoundLoopOptimizerOptions options_;
  const HloLiveRange& hlo_live_range_;
  const HloAliasAnalysis& alias_analysis_;
  const CostAnalysis& cost_analysis_;
  BufferValue::SizeFunction size_function_;

  absl::flat_hash_map<const HloInstruction*, int64_t> instructions_in_loop_;
  absl::flat_hash_map<const HloInstruction*, int64_t>
      instructions_in_prev_iteration_;
  absl::flat_hash_map<const HloInstruction*, int64_t>
      instructions_in_next_iteration_;
  std::vector<LoopValue> loop_values_;
  std::vector<int64_t> remaining_memory_;
  absl::flat_hash_map<const HloInstruction*,
                      std::vector<std::pair<int64_t, ShapeIndex>>>
      uses_in_alternate_mem_;
  absl::flat_hash_map<const HloInstruction*, std::vector<ShapeIndex>>
      positions_in_alternate_mem_;
  const ReservedScopedMemoryFunction& reserved_scoped_memory_fn_;
};

}  // namespace memory_space_assignment
}  // namespace xla

#endif  // XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_MEMORY_BOUND_LOOP_OPTIMIZER_H_

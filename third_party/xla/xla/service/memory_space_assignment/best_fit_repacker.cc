/* Copyright 2020 The OpenXLA Authors.

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

/*
BestFitRepacker Algorithm

Preliminary terminology/ideas
  * Consider a sliced AllocationBlock B with S slices.
    - Let s_{B,i} be the size of slice at the ith smallest offset.
    - Let s_B be the sum(s_{B,i}, for all i).
    - t_{B,i} be the ith earliest start time for a slice in B. (Note, an
      AllocationBlock has not determined which slice offset maps to which slice
      start time.)
    - Let e_B be the allocation end time for all slices.

    We defined the following:
    - B's full buffer interval
      * Defined by the buffer interval spanning time [s_{B,0}, e_B], with size
        sum(s_{B, i}, for all i).
      * In the illustration below, B's full buffer interval occupies W+X+Y+Z
      * In more colloquial language, the smallest rectangle that can bound
        a sliced allocation.
    - B's minimum buffer interval
      * Defined by the buffer interval spanning the time [t_{B,S-1}, e_B],
        with size s_B.
      * In the illustration below, B's minimum buffer interval occupies W.
      * In more colloquial language, the sliced allocation once all slices are
        allocated.
    - Potential placement of B
      * A potential placement of B occupies W+X. Note other placements are
        possible, e.g., starting the slice at offset o at time t_{B,0} or
        t_{B,2}.

    Illustration of B with S=3:
                      space
                        ^
                  o+s_B |  +-------------------------+------------+
                        |  |            Y            |            |
      o+s_{B,0}+s_{B,1} |  +-------------------------+            |
                        |  |                         |      W     |
              o+s_{B,0} |  +------------+     X      |            |
                        |  |      Z     |            |            |
                      o |  +------------+------------+------------+
                        |
                        +--|------------|------------|------------|-> time
                         t_{B,0}      t_{B,1}      t_{B,2}       e_B

  * For a non-sliced AllocationBlock the full buffer interval == the miniumum
    buffer interval == potential placement (given starting offset o).

Step 1: Data structure construction (ImportAllocationBlocks())
  * allocation_blocks_: Contains the input AllocationBlocks to repack. If we
        find a valid repacking, we update these AllocationBlocks with repacking
        placement data.
  * buffer_intervals_: Data structure owned by GlobalDecreasingSizeBestFitHeap.
        We use it to map each each AllocationBlock to its minimum buffer
        interval.
  * full_buffer_interval_map_: Maps each AllocationBlock to its minimum buffer
        interval.
  * sliced_buffer_interval_map_: Maps each AllocationBlock to a
        SlicedBufferInterval that wraps the corresponding full buffer interval
        stored in full_buffer_interval_map_.

Step 2: Sort buffers (GetSortedBufferIntervals())
  * This step prioritizes the order in which we will try place each buffer in
    the repacking.
  * GetSortedBufferIntervals() sorts the buffer intervals in buffer_intervals_.
    The DefaultBufferIntervalCompare() maps each buffer in buffer_intervals to
    its full buffer interval, and uses the full buffer interval's
    properties/dimensions for sorting.

Step 3: Find and commit buffer chunks (FindAndCommitChunks())
  * We iterate through AllocationBlocks (in the sorted order from Step 2),
    finding a location for them.
  * When we try to find a placement for an AllocationBlock B, we also find
    locations for its colocations. Colocations are done in tandem with B because
    B cannot be committed to a location if the same offset does not work for its
    colocations.
  * Colocations are handled as follows:
    - If a colocation is not sliced, MakeFreeChunks() will make sure we do not
      consider any placement for B that does not also accomodate the colocation.
    - If a colocation is sliced, MakeFreeChunks() will be inadequate for
      enforcing a placement for the colocation. The problem is that during
      placement, we have not yet determined the mapping between offsets and
      slice times for a sliced colocation. So, we have 2 options.
      Option 1) Overestimate the locations of free space available for B,
                knowing that some may not work when we try to place sliced
                colocations.
      Option 2) Underestimate the locations of free space available for B,
                knowing this will cause us not to choose some valid placement
                options for B.
      We have chosen option 1. To compensate for the free space over estimation,
      every time we find a place for B, we must explicitly check that each of
      its sliced colocations will also fit.

Step 4: Check if the repacking fits in the heap size
  * If the repacking does not fit in the heap size, the repacking fails.

Step 5: Update AllocationBlocks with the repacking placements
  * We update the offset and repacked_slicing fields of an AllocationBlock.
    Callers extract that data to get the repacking locations.
*/

#include "xla/service/memory_space_assignment/best_fit_repacker.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/comparison_util.h"
#include "xla/service/heap_simulator/allocation_block.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"

namespace xla {
namespace {

bool IsSliced(const AllocationBlock* block) {
  return block->original_slice_data.has_value();
}

template <typename T>
std::vector<const AllocationBlock*> SortAllocationBlocks(const T& container) {
  std::vector<const AllocationBlock*> result;
  result.insert(result.end(), container.begin(), container.end());
  absl::c_sort(
      result, [](const AllocationBlock* lhs, const AllocationBlock* rhs) {
        return std::make_tuple(lhs->inclusive_start_time, lhs->end_time,
                               lhs->initial_offset, lhs->size) <
               std::make_tuple(rhs->inclusive_start_time, rhs->end_time,
                               rhs->initial_offset, rhs->size);
      });

  return result;
}

const SlicedAllocationData* GetSlicedAllocationDataPointer(
    const std::optional<SlicedAllocationData>& sliced_allocation_data) {
  if (!sliced_allocation_data.has_value()) {
    return nullptr;
  }
  return &(*sliced_allocation_data);
}

// A slice-aware best-fit repacker.
class BestFitRepacker
    : public GlobalDecreasingSizeBestFitHeap<AllocationBlock> {
 public:
  BestFitRepacker(
      const memory_space_assignment::MemorySpaceAssignmentBestFitRepacker::
          BestFitRepackOptions& options,
      SliceTimePermutationIterator::Ty slice_time_permutation_iterator_type,
      int64_t max_size, int64_t alignment)
      : GlobalDecreasingSizeBestFitHeap<AllocationBlock>(
            alignment, kCustom,
            (options.buffer_interval_compare ? options.buffer_interval_compare
                                             : DefaultBufferIntervalCompare()),
            slice_time_permutation_iterator_type),
        validate_(options.validate),
        max_size_(max_size) {}

  // Initialize our basic data structures: allocation_blocks_,
  // buffer_intervals_, full_buffer_interval_map_, and
  // sliced_buffer_interval_map_.
  void ImportAllocationBlocks(absl::Span<AllocationBlock*> allocations) {
    allocation_blocks_ = allocations;
    // We loop through allocation_blocks_ once to build
    // full_buffer_interval_map_, with colocations fully specified.
    for (AllocationBlock* allocation_block : allocation_blocks_) {
      // Check if any of the colocations are already added to buffer_intervals_.
      bool need_allocation = true;
      CHECK_NE(allocation_block->next_colocated, nullptr);
      for (AllocationBlock* colocated = allocation_block->next_colocated;
           colocated != allocation_block;
           colocated = colocated->next_colocated) {
        auto aliased_it = full_buffer_interval_map_.find(colocated);
        if (aliased_it != full_buffer_interval_map_.end() &&
            aliased_it->second.need_allocation) {
          aliased_it->second.colocations.push_back(allocation_block);
          need_allocation = false;
          break;
        }
      }
      full_buffer_interval_map_.insert(
          std::make_pair(allocation_block,
                         BufferInterval{allocation_block,
                                        allocation_block->size,
                                        allocation_block->inclusive_start_time,
                                        allocation_block->end_time,
                                        {},
                                        need_allocation}));
    }

    // Now that full_buffer_interval_map_ has full colocation specifications,
    // we loop through allocation_blocks_ again to build
    // sliced_buffer_interval_map_ and buffer_intervals_. Also note, at this
    // point we will not longer add or remove items from
    // full_buffer_interval_map_. This is important because
    // sliced_buffer_interval_map_ will retain pointers to BufferIntervals in
    // full_buffer_interval_map_.
    for (AllocationBlock* allocation_block : allocation_blocks_) {
      BufferInterval& full_buffer_interval =
          full_buffer_interval_map_[allocation_block];
      SlicedBufferInterval& sliced_buffer_interval =
          sliced_buffer_interval_map_
              .insert(std::make_pair(
                  allocation_block, SlicedBufferInterval::CreateMutableInterval(
                                        full_buffer_interval)))
              .first->second;
      if (IsSliced(allocation_block)) {
        const SlicedAllocationData& original_slice_data =
            allocation_block->original_slice_data.value();
        CHECK(!original_slice_data.slices_sorted_by_offset.empty());

        sliced_buffer_interval.Slice(original_slice_data.SizesSortedByOffset());
        sliced_buffer_interval.UpdateInclusiveSliceStartTimes(
            original_slice_data.SortedInclusiveStartTimes());
      }

      // We use buffer_intervals_ to store the minimum buffer interval for
      // allocation_block. See the algorithm description (at the head of this
      // file) for more details.
      buffer_intervals_[allocation_block] =
          sliced_buffer_interval.IntervalForMakeFreeChunks(
              sliced_buffer_interval.num_slices() - 1);
    }

    CHECK_EQ(allocation_blocks_.size(), buffer_intervals_.size());
    CHECK_EQ(allocation_blocks_.size(), full_buffer_interval_map_.size());
    CHECK_EQ(allocation_blocks_.size(), sliced_buffer_interval_map_.size());

    VLOG(2) << [&]() -> std::string {
      int sliced_blocks = 0;
      int colocation_sets = 0;
      int colocation_sets_with_multiple_sliced_blocks = 0;
      absl::flat_hash_set<const AllocationBlock*> seen_blocks;

      for (const auto& allocation_and_buffer_interval : buffer_intervals_) {
        const AllocationBlock* block = allocation_and_buffer_interval.first;
        const BufferInterval& min_buffer_interval =
            allocation_and_buffer_interval.second;

        if (IsSliced(block)) {
          ++sliced_blocks;
        }
        if (seen_blocks.contains(block)) {
          continue;
        }
        seen_blocks.insert(block);
        // We process a block and all its colocations at once below. So every
        // time a block is unseen, it indicates a new colocation set.
        ++colocation_sets;

        int num_sliced_colocations = (IsSliced(block) ? 1 : 0);
        // GetTranstivieColocations must be called on BufferIntervals from
        // buffer_intervals_.
        for (const AllocationBlock* colocation :
             GetTransitiveColocations(min_buffer_interval)) {
          seen_blocks.insert(colocation);
          if (IsSliced(colocation)) {
            ++num_sliced_colocations;
          }
        }
        if (num_sliced_colocations > 1) {
          ++colocation_sets_with_multiple_sliced_blocks;
        }
      }

      return absl::StrCat(
          "Imported repacking stats: num_blocks=", allocation_blocks_.size(),
          "; num_sliced_blocks=", sliced_blocks,
          "; num_colocation_sets=", colocation_sets,
          "; num_colocation_sets_with_multiple_sliced_blocks=",
          colocation_sets_with_multiple_sliced_blocks);
    }();
  }

  BufferIntervalCompare DefaultBufferIntervalCompare() const {
    return LessThanByKey([this](const BufferInterval& x) {
      const BufferInterval& full_buffer_interval =
          full_buffer_interval_map_.at(x.buffer);
      int64_t full_buffer_interval_end = full_buffer_interval.end;

      // GetTranstivieColocations must be called on BufferIntervals from
      // buffer_intervals_.
      for (auto colocation : GetTransitiveColocations(x)) {
        full_buffer_interval_end =
            std::max(full_buffer_interval_end,
                     full_buffer_interval_map_.at(colocation).end);
      }

      // Sort by duration (descending), size (descending), buffer (ascending).
      return std::make_tuple(
          full_buffer_interval.start - full_buffer_interval_end,
          -full_buffer_interval.size, std::cref(*full_buffer_interval.buffer));
    });
  }

  // CommitChunks() does the following:
  // 1) Commits chunks to interval_tree_.
  // 2) Updates the entries in new_offsets_ and new_repacked_slicing_ for
  //    allocation_block, with the information from chunks.
  // 3) Updates result._heap_size with the information from chunks.
  //
  // REQUIRED:
  // - chunks is sorted in slice time order
  void CommitChunks(const AllocationBlock* allocation_block,
                    const std::vector<Chunk>& chunks) {
    VLOG(3) << "Committing repack chunks for " << allocation_block->ToString();

    int64_t new_offset = -1;
    std::optional<SlicedAllocationData> repacked_slice_data = std::nullopt;

    if (IsSliced(allocation_block)) {
      const SlicedAllocationData& original_slice_data =
          allocation_block->original_slice_data.value();

      // We should have a chunk per slice time.
      CHECK_EQ(chunks.size(),
               original_slice_data.slices_sorted_by_offset.size());
      repacked_slice_data = SlicedAllocationData();
      repacked_slice_data->slices_sorted_by_offset.reserve(chunks.size());

      // Chunks and start times are sorted in start time order.
      std::vector<int64_t> sorted_inclusive_start_times =
          original_slice_data.SortedInclusiveStartTimes();
      for (int i = 0; i < chunks.size(); ++i) {
        const Chunk& chunk = chunks[i];
        int64_t start_time = sorted_inclusive_start_times[i];
        result_.heap_size = result_.UpdatedHeapSize(chunk);
        VLOG(3) << "Adding sliced chunk " << chunk.ToString() << " at ["
                << start_time << ", " << allocation_block->end_time << "]";
        interval_tree_.Add(start_time, allocation_block->end_time, chunk);
        new_offset = (new_offset == -1 ? chunk.offset
                                       : std::min(new_offset, chunk.offset));
        repacked_slice_data->slices_sorted_by_offset.push_back(
            AllocatedSlice({chunk.size, chunk.offset, start_time}));
      }
      absl::c_sort(repacked_slice_data->slices_sorted_by_offset,
                   [](const AllocatedSlice& lhs, const AllocatedSlice& rhs) {
                     return lhs.offset < rhs.offset;
                   });
    } else {
      CHECK_EQ(chunks.size(), 1);
      new_offset = chunks.front().offset;
      result_.heap_size = result_.UpdatedHeapSize(chunks.front());
      VLOG(3) << "Adding unsliced chunk " << chunks.front().ToString()
              << " at [" << allocation_block->inclusive_start_time << ", "
              << allocation_block->end_time << ")";
      interval_tree_.Add(allocation_block->inclusive_start_time,
                         allocation_block->end_time, chunks.front());
    }

    CHECK_NE(new_offset, -1);
    CHECK(!new_offsets_.contains(allocation_block));
    new_offsets_[allocation_block] = new_offset;
    if (repacked_slice_data.has_value()) {
      CHECK(IsSliced(allocation_block));
      CHECK(!new_repacked_slicing_.contains(allocation_block));
      new_repacked_slicing_[allocation_block] = *repacked_slice_data;
    }
  }

  // A helper structure of information we keep in FindAndCommitChunks() for
  // each sliced colocation.
  struct SlicedColocationData {
    SlicedBufferInterval* sliced_buffer_interval;
    SlicedAllocationFinder sliced_allocation_finder;
    std::vector<Chunk> chunks;
  };

  // Finds and commits chunks for the AllocationBlock associated with
  // min_buffer_interval, and all of that block's colocations.
  void FindAndCommitChunks(BufferInterval* min_buffer_interval) {
    const AllocationBlock* allocation_block = min_buffer_interval->buffer;
    SlicedBufferInterval& sliced_buffer_interval =
        sliced_buffer_interval_map_.at(allocation_block);

    int64_t max_colocation_size = GetMaxColocationSize(*min_buffer_interval);

    // Additional data we track for sliced colocations. In particular, we create
    // a SlicedAllocationFinder for each sliced colocation, so we can check if
    // a sliced colocation can fit at a particular offset, to accommodate the
    // overestimation of free space in MakeFreeChunks(), as described at the top
    // of this file.
    absl::flat_hash_map<const AllocationBlock*, SlicedColocationData>
        sliced_buffer_map;
    for (auto colocation :
         SortAllocationBlocks(GetTransitiveColocations(*min_buffer_interval))) {
      if (IsSliced(colocation)) {
        SlicedBufferInterval& colocation_sliced_buffer_interval =
            sliced_buffer_interval_map_.at(colocation);
        SlicedAllocationFinder sliced_colocation_finder =
            CreateSlicedAllocationFinder(
                colocation_sliced_buffer_interval, max_colocation_size,
                /*preferred_offset=*/-1,
                SliceTimePermutationIterator::CreateForRepack(
                    slice_time_permutation_iterator_type(),
                    GetSlicedAllocationDataPointer(
                        colocation->original_slice_data)),
                &SlicedAllocationFinder::AllOffsetsAllowed);
        sliced_buffer_map.insert(std::make_pair(
            colocation,
            SlicedColocationData{&colocation_sliced_buffer_interval,
                                 std::move(sliced_colocation_finder),
                                 /*chunks=*/{}}));
      }
    }

    // Using the SlicedAllocationFinders for each sliced colocation, this
    // function returns true if we can place all sliced colocations at a given
    // offset.
    auto is_offset_allowed = [this, &sliced_buffer_map](int64_t offset) {
      for (auto& block_and_colocation_data : sliced_buffer_map) {
        SlicedColocationData& sliced_colocation_data =
            block_and_colocation_data.second;
        auto colocation_chunks =
            sliced_colocation_data.sliced_allocation_finder.FindForOffset(
                offset);
        colocation_chunks = PostProcessFindChunkCandidatesResult(
            *sliced_colocation_data.sliced_buffer_interval,
            std::move(colocation_chunks));
        if (colocation_chunks.empty()) {
          return false;
        }
        sliced_colocation_data.chunks = std::move(colocation_chunks);
      }

      return true;
    };

    // Find chunks for allocation_block and its colocations.
    SlicedAllocationFinder finder = CreateSlicedAllocationFinder(
        sliced_buffer_interval, max_colocation_size, /*preferred_offset=*/-1,
        SliceTimePermutationIterator::CreateForRepack(
            slice_time_permutation_iterator_type(),
            GetSlicedAllocationDataPointer(
                allocation_block->original_slice_data)),
        is_offset_allowed);
    std::vector<Chunk> chunks = PostProcessFindChunkCandidatesResult(
        sliced_buffer_interval, finder.Find());
    int64_t min_offset =
        absl::c_min_element(chunks, [](const Chunk& lhs, const Chunk& rhs) {
          return lhs.offset < rhs.offset;
        })->offset;

    // Commit chunks for allocation_block.
    CommitChunks(allocation_block, chunks);

    // Commit chunks for colocations.
    for (auto colocation : GetTransitiveColocations(*min_buffer_interval)) {
      if (IsSliced(colocation)) {
        CommitChunks(colocation, sliced_buffer_map.at(colocation).chunks);
      } else {
        const BufferInterval& colocation_full_buffer_interval =
            full_buffer_interval_map_[colocation];
        CommitChunks(colocation,
                     {Chunk::FromOffsetSize(
                         min_offset, colocation_full_buffer_interval.size)});
      }
    }
  }

  // We do not use result_.chunk_map, and we have our own method that combines
  // finding chunks with committing them; thus, we expect this method to never
  // be called.
  void AddToChunkMap(const AllocationBlock* buffer, Chunk chunk) override {
    LOG(FATAL) << "We should never get here.";
  }

  absl::StatusOr<Result> Finish() override {
    std::vector<BufferInterval> sorted_buffer_intervals =
        GetSortedBufferIntervals();

    for (auto& buffer_interval : sorted_buffer_intervals) {
      if (!buffer_interval.need_allocation) {
        continue;
      }

      FindAndCommitChunks(&buffer_interval);
    }

    Result result;
    result.heap_size = result_.heap_size;
    result.heap_results.push_back(result_);
    return result;
  }

  // A data structure for storing a chunk and its live time for use in
  // debugging.
  struct TimedChunk {
    std::string id;
    const AllocationBlock* block;
    int64_t start_inclusive;
    int64_t end_inclusive;
    Chunk chunk;

    bool Overlaps(const TimedChunk& timed_chunk) {
      if (timed_chunk.start_inclusive > end_inclusive ||
          timed_chunk.end_inclusive < start_inclusive) {
        return false;
      }
      return chunk.OverlapsWith(timed_chunk.chunk);
    }
  };

  void DebuggingValidate() {
    std::vector<TimedChunk> timed_chunks;
    for (const AllocationBlock* block : allocation_blocks_) {
      if (IsSliced(block)) {
        for (int i = 0;
             i < block->repacked_slice_data->slices_sorted_by_offset.size();
             ++i) {
          const AllocatedSlice& slice =
              block->repacked_slice_data->slices_sorted_by_offset[i];
          timed_chunks.push_back(
              TimedChunk{absl::StrCat(((int64_t)block), "_slice_", i), block,
                         slice.inclusive_start_time, block->end_time,
                         Chunk::FromOffsetSize(slice.offset, slice.size)});
        }
      } else {
        timed_chunks.push_back(
            TimedChunk{absl::StrCat(((int64_t)block)), block,
                       block->inclusive_start_time, block->end_time,
                       Chunk::FromOffsetSize(block->offset, block->size)});
      }
    }

    bool overlap_found = false;
    for (int i = 0; i < timed_chunks.size(); ++i) {
      for (int j = i + 1; j < timed_chunks.size(); ++j) {
        if (timed_chunks[i].Overlaps(timed_chunks[j])) {
          overlap_found = true;
          LOG(ERROR) << "Allocation block overlap\n"
                     << "     " << timed_chunks[i].block->ToString()
                     << "\n     " << timed_chunks[j].block->ToString();
        }
      }
    }

    if (overlap_found) {
      LOG(FATAL) << "Allocation overlap found";
    }
  }

  bool Repack() {
    TF_CHECK_OK(Finish().status());
    bool success = result_.heap_size <= max_size_;
    if (!success) {
      VLOG(1) << "Repacking unsuccessful with heap size " << result_.heap_size;
      return false;
    }

    // Update AllocationBlocks.
    for (AllocationBlock* block : allocation_blocks_) {
      CHECK(new_offsets_.contains(block));
      block->offset = new_offsets_[block];
      if (!IsSliced(block)) {
        continue;
      }

      CHECK(new_repacked_slicing_.contains(block));
      block->repacked_slice_data = std::move(new_repacked_slicing_[block]);
    }

    if (validate_) {
      DebuggingValidate();
    }

    if (VLOG_IS_ON(2)) {
      for (AllocationBlock* block : allocation_blocks_) {
        VLOG(2) << "AllocationBlock after repacking: " << block->ToString();
      }
    }

    VLOG(1) << "Repacking successful with heap size " << result_.heap_size;

    return true;
  }

 private:
  // If true, we run a potentially expensive validation to make sure there are
  // no overlaps in the repacked chunks. Note, there should never be an overlap.
  bool validate_ = false;

  // Maximum heap size.
  int64_t max_size_;

  // Input AllocationBlocks to repack.
  absl::Span<AllocationBlock*> allocation_blocks_;

  absl::flat_hash_map<const AllocationBlock*, BufferInterval>
      full_buffer_interval_map_;
  absl::flat_hash_map<const AllocationBlock*, SlicedBufferInterval>
      sliced_buffer_interval_map_;

  // Data structures updated with repacking placement information as we compute
  // it.
  absl::flat_hash_map<const AllocationBlock*, int64_t> new_offsets_;
  absl::flat_hash_map<const AllocationBlock*, SlicedAllocationData>
      new_repacked_slicing_;
};

}  // namespace

namespace memory_space_assignment {

absl::StatusOr<bool> MemorySpaceAssignmentBestFitRepacker::Repack(
    absl::Span<AllocationBlock*> allocations) {
  BestFitRepacker best_fit_repacker = BestFitRepacker(
      options_, slice_time_permutation_iterator_type_, max_size_, alignment_);
  best_fit_repacker.ImportAllocationBlocks(allocations);
  return best_fit_repacker.Repack();
}

}  // namespace memory_space_assignment
}  // namespace xla

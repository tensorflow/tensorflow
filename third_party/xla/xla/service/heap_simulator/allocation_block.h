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

// TODO(b/319135034): create a heap_simulator sub directory and move
// allocation_block.h/cc to it.

// This file contains a number of data structures to describe how blocks of
// data are allocated. It is used by Memory Space Assignment repacking to
// understand how data was allocated before the repacking.

#ifndef XLA_SERVICE_HEAP_SIMULATOR_ALLOCATION_BLOCK_H_
#define XLA_SERVICE_HEAP_SIMULATOR_ALLOCATION_BLOCK_H_

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace xla {

// Data about a slice in a sliced allocation.
struct AllocatedSlice {
  int64_t size;
  int64_t offset;
  int64_t inclusive_start_time;

  std::string ToString() const;

  std::tuple<int64_t, int64_t, int64_t> ToTuple() const;

  bool operator==(const AllocatedSlice& rhs) const;
};

// Slice data about a sliced allocation.
struct SlicedAllocationData {
  std::vector<AllocatedSlice> slices_sorted_by_offset;

  std::vector<int64_t> SizesSortedByOffset() const;

  std::vector<int64_t> SortedInclusiveStartTimes() const;

  int64_t num_slices() const { return slices_sorted_by_offset.size(); }

  std::string ToString() const;

  bool operator==(const SlicedAllocationData& rhs) const;
};

// A contiguous block of allocation consisting of start and end (logical)
// times, size, and the initial offset. After repacking, if the repacking was
// successful and the allocations were modified, the offset field holds the
// new offset. To support aliased allocations, AllocationBlock also includes a
// pointer to the next colocated AllocationBlock called next_colocated. The
// colocations form a circular singly-linked list. Therefore, next_colocated
// should never be a nullptr (it should point to itself for AllocationBlocks
// without any other colocations). All AllocationBlock objects within the
// colocations must get the same offset. The id should be unique and is used
// to ensure determinism for comparison tie-breaker.
//
// Each AllocationBlock can be treated as an allocation that requires size
// space from start_time to end_time. However, some allocations are really
// composed of slices. In such cases, the repacker can utilize
// the information in the original_slice_data field to achieve an even more
// efficient repacking.
struct AllocationBlock {
  int64_t inclusive_start_time;
  int64_t end_time;
  int64_t size;
  int64_t offset;
  int64_t initial_offset;
  int64_t id;
  AllocationBlock* next_colocated;

  // Optional data structures that are used to improve repacking, when an
  // allocation is sliced, e.g., from a sliced prefetch.
  std::optional<SlicedAllocationData> original_slice_data;
  std::optional<SlicedAllocationData> repacked_slice_data;

  std::string ToString() const;

  // Returns the number of AllocationBlocks colocated with this (including
  // this AllocationBlock).
  int GetColocationsCount() const;

  // Returns the AllocationBlocks colocated with this (including this
  // AllocationBlock).
  std::vector<AllocationBlock*> GetColocations();

  // This is required by BufferIntervalCompare as a tie breaker. Use a unique
  // and deterministic id.
  bool operator<(const AllocationBlock& other) const;
};

}  // namespace xla

#endif  // XLA_SERVICE_HEAP_SIMULATOR_ALLOCATION_BLOCK_H_

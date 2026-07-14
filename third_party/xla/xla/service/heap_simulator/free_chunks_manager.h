/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_HEAP_SIMULATOR_FREE_CHUNKS_MANAGER_H_
#define XLA_SERVICE_HEAP_SIMULATOR_FREE_CHUNKS_MANAGER_H_

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"

namespace xla {

// Manages the free space created by the given free chunks.
// Maintains the chunks:
// - sorted by offset: to allow for efficient insertion of a chunk.
//   This may split a free chunk into two chunks (see Allocate()).
// - sorted by size: to allow for efficient querying of free chunks of size
//   at least the given size.
class FreeChunksManager {
 public:
  // Represents a chunk of memory, which can be in different states: allocated,
  // free, or a candidate for allocation.
  class MemoryChunk {
   public:
    MemoryChunk(int64_t offset, int64_t end, int64_t aligned_chunk_offset,
                int64_t id);
    MemoryChunk(int64_t offset, int64_t end) : offset_(offset), end_(end) {}

    // Returns the usable size of this chunk for aligned allocations.
    int64_t aligned_size() const { return end_ - aligned_chunk_offset_; }

    static bool OffsetLessThan(const MemoryChunk& a, const MemoryChunk& b);

    static bool AlignedSizeLessThan(const MemoryChunk& a, const MemoryChunk& b);

    bool operator<(const MemoryChunk& other) const;

    bool operator==(const MemoryChunk& other) const;

    int64_t offset() const { return offset_; }
    int64_t end() const { return end_; }
    int64_t aligned_chunk_offset() const { return aligned_chunk_offset_; }
    int64_t id() const { return id_; }

   private:
    int64_t offset_ = -1;  // Inclusive offset of this memory chunk.
    int64_t end_ = -1;     // Exclusive end of this memory chunk.
    // The smallest address >= offset_ that satisfies alignment requirements.
    // When an allocation is placed in a free chunk, it must begin at an aligned
    // offset.
    // The smallest aligned offset in [offset_, end_) is
    // aligned_chunk_offset_, therefore a free chunk [offset_, end_) can contain
    // an allocation of size `S` only if `S <= end_ - aligned_chunk_offset_`.
    int64_t aligned_chunk_offset_ = -1;
    // A unique ID assigned to free chunks, used internally by FreeChunksManager
    // to track removed chunks for lazy removal.
    int64_t id_ = -1;
  };

  explicit FreeChunksManager(
      absl::AnyInvocable<int64_t(int64_t)> chunk_alignment);

  // Allocates the given interval [offset, end), removing it from free chunks.
  // If [offset, end) is carved out of a larger free chunk, the remainder
  // portion(s) remain free.
  //
  // REQUIRES: The allocated interval must be contained in a single free chunk.
  void Allocate(int64_t offset, int64_t end);

  // Deallocates the given interval [offset, end), adding it to free chunks.
  void Deallocate(int64_t offset, int64_t end);

  // Returns a free chunk that is large enough to fit the given size, or
  // std::nullopt if no such chunk exists.
  std::optional<MemoryChunk> FindJustLargeEnough(int64_t size);

  // (Slow -- for testing/debugging) returns all the free chunks in a vector,
  // sorted by chunk offset.
  std::vector<MemoryChunk> GetFreeChunks();

 private:
  // Creates and adds a new free chunk representing interval [offset, end) to
  // free_chunks_by_offset_ and free_chunks_by_size_.
  // REQUIRES: No part of [offset-1, end+1) is contained in an existing free
  // chunk.
  void AddNewFreeChunk(int64_t offset, int64_t end);

  using FreeChunksByOffset =
      absl::btree_set<MemoryChunk, decltype(&MemoryChunk::OffsetLessThan)>;
  using FreeChunksByAlignedSize =
      absl::btree_set<MemoryChunk, decltype(&MemoryChunk::AlignedSizeLessThan)>;

  // Removes free chunk pointed by 'it' from free_chunks_by_offset_, and marks
  // it as removed in free_chunks_to_be_removed_ for lazy removal from
  // free_chunks_by_size_.
  void RemoveFreeChunk(FreeChunksByOffset::iterator it);

  // Removes any existing free chunks that overlap with or are adjacent to
  // [offset, end), merges them into a single chunk, and adds that chunk.
  void InvalidateAndMerge(int64_t offset, int64_t end);

  absl::AnyInvocable<int64_t(int64_t)> chunk_alignment_;

  // Free chunks sorted by offset. The endpoint is used as a tie breaker
  // and is only used when querying (using lower/upper bound).
  FreeChunksByOffset free_chunks_by_offset_;

  // These 2 data structures tell us which chunks are free.
  // free_chunks_to_be_removed_ contains the unique ids of recently removed free
  // chunks that have not yet been reflected in free_chunks_by_size. This is an
  // optimization to avoid expensive removals from free_chunks_by_size_ when a
  // free chunk is removed. Removing a chunk from free_chunks_by_size_ takes
  // O(logN). Instead of paying that cost on every removal, we mark chunks as
  // removed here in O(1) and lazily remove them from free_chunks_by_size_ in
  // FindJustLargeEnough, when we iterate past them.
  FreeChunksByAlignedSize free_chunks_by_size_;
  absl::flat_hash_set<int64_t> free_chunks_to_be_removed_;

  // The ID of the next inserted free chunk.
  int64_t next_free_chunk_id_ = 1;
};

}  // namespace xla

#endif  // XLA_SERVICE_HEAP_SIMULATOR_FREE_CHUNKS_MANAGER_H_

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

#include "xla/service/heap_simulator/free_chunks_manager.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"

namespace xla {
namespace {

// kMinQueryId is used as the ID in query chunks for lower_bound
// searches in free_chunks_by_size_. It ensures that among chunks of equal
// size and offset, any chunk with a valid ID (>=0) is considered.
constexpr int64_t kMinQueryId = -1;
const int64_t kAlignedOffsetThatIsNotUsed = -1;

}  // namespace

FreeChunksManager::MemoryChunk::MemoryChunk(int64_t offset, int64_t end,
                                            int64_t aligned_chunk_offset,
                                            int64_t id)
    : offset_(offset),
      end_(end),
      aligned_chunk_offset_(aligned_chunk_offset),
      id_(id) {}

bool FreeChunksManager::MemoryChunk::OffsetLessThan(const MemoryChunk& a,
                                                    const MemoryChunk& b) {
  return std::forward_as_tuple(a.offset(), a.end()) <
         std::forward_as_tuple(b.offset(), b.end());
}

bool FreeChunksManager::MemoryChunk::AlignedSizeLessThan(const MemoryChunk& a,
                                                         const MemoryChunk& b) {
  return std::tuple(a.aligned_size(), a.offset(), a.id()) <
         std::tuple(b.aligned_size(), b.offset(), b.id());
}

bool FreeChunksManager::MemoryChunk::operator<(const MemoryChunk& other) const {
  return std::forward_as_tuple(offset_, end_, aligned_chunk_offset_, id_) <
         std::forward_as_tuple(other.offset_, other.end_,
                               other.aligned_chunk_offset_, other.id_);
}

bool FreeChunksManager::MemoryChunk::operator==(
    const MemoryChunk& other) const {
  return std::forward_as_tuple(offset_, end_, aligned_chunk_offset_, id_) ==
         std::forward_as_tuple(other.offset_, other.end_,
                               other.aligned_chunk_offset_, other.id_);
}

FreeChunksManager::FreeChunksManager(
    absl::AnyInvocable<int64_t(int64_t)> chunk_alignment)
    : chunk_alignment_(std::move(chunk_alignment)),
      free_chunks_by_offset_(&MemoryChunk::OffsetLessThan),
      free_chunks_by_size_(&MemoryChunk::AlignedSizeLessThan) {
  // Add [-2, -1) and [INT64_MAX - 2, INT64_MAX - 1) as two dummy chunks to
  // ensure there is always a disconnected chunk before and after the actual
  // chunks.
  // The [-2, -1) dummy chunk is chosen because it won't be united with the
  // real free chunks when deallocating chunks of the form [0, n). The
  // `aligned_chunk_offset` for these dummy chunks is not used because they are
  // not inserted into `free_chunks_by_size_`, so their `aligned_size()` is
  // never calculated or used for sorting. We pass -1 as a sentinel value for
  // this unused parameter.
  FreeChunksManager::MemoryChunk very_beginning_chunk(
      -2, -1, kAlignedOffsetThatIsNotUsed, next_free_chunk_id_++);
  FreeChunksManager::MemoryChunk real_free_chunk(
      0, INT64_MAX - 3, chunk_alignment_(0), next_free_chunk_id_++);
  FreeChunksManager::MemoryChunk very_end_chunk(INT64_MAX - 2, INT64_MAX - 1,
                                                kAlignedOffsetThatIsNotUsed,
                                                next_free_chunk_id_++);

  free_chunks_by_offset_.insert(very_beginning_chunk);
  free_chunks_by_offset_.insert(real_free_chunk);
  free_chunks_by_offset_.insert(very_end_chunk);
  free_chunks_by_size_.insert(real_free_chunk);
}

void FreeChunksManager::Allocate(int64_t offset, int64_t end) {
  auto it = free_chunks_by_offset_.lower_bound(
      FreeChunksManager::MemoryChunk(offset, INT64_MAX));
  // All allocations are for non-negative offsets. Because we insert a dummy
  // chunk [-2, -1) in the constructor which is ordered before any chunk with
  // non-negative offset, `lower_bound` will return an iterator pointing after
  // `begin()`, so `--it` is safe.
  --it;
  FreeChunksManager::MemoryChunk free_chunk = *it;
  CHECK(it->offset() <= offset && end <= it->end())
      << "Cannot allocate [" << offset << ", " << end
      << ") because it is not fully contained in free chunk [" << it->offset()
      << ", " << it->end() << ")";
  RemoveFreeChunk(it);
  if (free_chunk.end() > end) {
    AddNewFreeChunk(end, free_chunk.end());
  }
  if (free_chunk.offset() < offset) {
    AddNewFreeChunk(free_chunk.offset(), offset);
  }
}

void FreeChunksManager::AddNewFreeChunk(int64_t offset, int64_t end) {
  int64_t id = next_free_chunk_id_++;
  FreeChunksManager::MemoryChunk chunk(offset, end, chunk_alignment_(offset),
                                       id);
  VLOG(1) << "AddNewFreeChunk: offset=" << offset << ", end=" << end
          << ", aligned_size=" << chunk.aligned_size() << ", id=" << id;
  free_chunks_by_offset_.insert(chunk);
  free_chunks_by_size_.insert(chunk);
}

void FreeChunksManager::RemoveFreeChunk(
    FreeChunksManager::FreeChunksByOffset::iterator it) {
  free_chunks_to_be_removed_.insert(it->id());
  free_chunks_by_offset_.erase(it);
}

void FreeChunksManager::Deallocate(int64_t offset, int64_t end) {
  VLOG(1) << "Deallocate: offset=" << offset << ", end=" << end;
  auto free_chunk_above = free_chunks_by_offset_.lower_bound(
      FreeChunksManager::MemoryChunk(end, end + 1));
  int64_t right = end;
  if (free_chunk_above != free_chunks_by_offset_.end() &&
      free_chunk_above->offset() == end) {
    right = free_chunk_above->end();
    VLOG(1) << "Deallocate removing chunk: offset="
            << free_chunk_above->offset() << ", end=" << free_chunk_above->end()
            << ", aligned_size=" << free_chunk_above->aligned_size()
            << ", id=" << free_chunk_above->id();
    RemoveFreeChunk(free_chunk_above);
  }

  auto free_chunk_below = free_chunks_by_offset_.lower_bound(
      FreeChunksManager::MemoryChunk(offset, offset + 1));
  --free_chunk_below;
  // Find free chunk to the left and remove if its right next to [offset,
  // end].
  int64_t left = offset;
  if (free_chunk_below->end() >= offset) {
    left = free_chunk_below->offset();
    right = std::max(right, free_chunk_below->end());
    VLOG(1) << "Deallocate removing chunk: offset="
            << free_chunk_below->offset() << ", end=" << free_chunk_below->end()
            << ", aligned_size=" << free_chunk_below->aligned_size()
            << ", id=" << free_chunk_below->id();
    RemoveFreeChunk(free_chunk_below);
  }
  AddNewFreeChunk(left, right);
}

std::optional<FreeChunksManager::MemoryChunk>
FreeChunksManager::FindJustLargeEnough(int64_t size) {
  // We are looking for a chunk with size at least 'size'. We construct a query
  // chunk with size 'size', and lowest possible offset and id, to use in
  // lower_bound. This will return the first chunk that is not smaller than
  // the query chunk, i.e. chunk with size >= 'size'.
  FreeChunksManager::MemoryChunk query_chunk(0, 0, -size, kMinQueryId);
  auto it = free_chunks_by_size_.lower_bound(query_chunk);
  while (it != free_chunks_by_size_.end() &&
         free_chunks_to_be_removed_.contains(it->id())) {
    // Remove free chunks that have been invalidated.
    it = free_chunks_by_size_.erase(it);
  }

  if (it == free_chunks_by_size_.end()) {
    return std::nullopt;
  }

  return *it;
}

std::vector<FreeChunksManager::MemoryChunk> FreeChunksManager::GetFreeChunks() {
  std::vector<FreeChunksManager::MemoryChunk> chunks;
  auto it = free_chunks_by_offset_.begin();
  ++it;  // Skip the dummy chunk at the beginning.
  for (; it != free_chunks_by_offset_.end(); ++it) {
    chunks.emplace_back(it->offset(), it->end());
  }
  chunks.pop_back();  // Remove the dummy chunk at the end.
  return chunks;
}

}  // namespace xla

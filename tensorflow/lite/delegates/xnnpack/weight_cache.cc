/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/xnnpack/weight_cache.h"

#include <fcntl.h>

#if defined(_MSC_VER)
#include <io.h>
#define F_OK 0
#else
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>  // IWYU pragma: keep
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "experimental.h"  // from @XNNPACK
#include "xnnpack.h"  // from @XNNPACK
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/verifier.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/xnnpack/file_util.h"
#include "tensorflow/lite/delegates/xnnpack/macros.h"
#include "tensorflow/lite/delegates/xnnpack/mmap_handle.h"
#include "tensorflow/lite/delegates/xnnpack/weight_cache_schema_generated.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite::xnnpack {

namespace {
constexpr size_t kMinAlignment = 128;
#if defined(XNNPACK_CACHE_NO_FILE_MAPPING_FOR_DEBUG)
// Debug heap allocations (`new[]`) aren't page-aligned, so new chunks reserve
// extra space to align the first buffer to `kMinAlignment`.
constexpr size_t kChunkAlignmentPadding = kMinAlignment;
#else
// OS mappings (`mmap` and `MapViewOfFile` at offset 0) are page-aligned
// (`>= 4 KiB`), which is already a multiple of `kMinAlignment`.
constexpr size_t kChunkAlignmentPadding = 0;
#endif
// Name of the anonymous mapping holding cached packed weights, which shows up
// as `[anon:xnnpack-weights-cache]` in `/proc/<pid>/maps`.
constexpr char kWeightCacheVmaName[] = "xnnpack-weights-cache";
// Caps single-step chunk growth at 1 MiB so unused tail capacity stays bounded
// while amortizing mapping calls.
constexpr size_t kMaxChunkGrowth = 1024 * 1024;

const char* Sanitize(const char* path) { return path ? path : ""; }

// Checks if the given path is a special value to use an in-memory cache.
bool IsInMemoryCachePath(const char* path) {
  // Use strncmp to check for the prefix.
  return path &&
         !strncmp(path, kInMemoryCachePath, sizeof(kInMemoryCachePath) - 1);
}

// Checks if the given path is a special value to use an in-memory cache.
bool IsInMemoryCachePath(const std::string& path) {
  // Use strncmp to check for the prefix.
  return IsInMemoryCachePath(path.c_str());
}

// Returns the next offset value that is aligned to `alignement`.
size_t Align(size_t offset, const size_t alignment) {
  const size_t misalign = offset % alignment;
  return offset + (misalign ? alignment - misalign : 0);
}

template <class T>
T* Align(T* offset, const size_t alignment) {
  return reinterpret_cast<T*>(
      Align(reinterpret_cast<uintptr_t>(offset), alignment));
}

// Returns true if the given path exists.
[[nodiscard]]
bool FileExists(const char* path) {
  return access(path, F_OK) != -1;
}

bool CheckFingerprints(const cache::schema::BufferList* buffer_list) {
  if (buffer_list->fingerprints()) {
    for (uint64_t cache_fingerprint : *buffer_list->fingerprints()) {
      xnn_fingerprint fingerprint;
      static_assert(sizeof(fingerprint) == sizeof(cache_fingerprint));
      std::memcpy(&fingerprint, &cache_fingerprint, sizeof(fingerprint));
      XNNPACK_RETURN_CHECK(
          xnn_check_fingerprint(fingerprint) == xnn_status_success,
          "fingerprint %s could not be matched",
          xnn_fingerprint_id_to_string_u32(fingerprint.id));
    }
  }
  return true;
}

bool IsValidHeader(const XNNPackCacheHeader& header) {
  XNNPACK_RETURN_CHECK(
      header.magic_value == XNNPackCacheHeader::kMagicValue,
      "Cache header magic value is incorrect. Expected 0x%016" PRIx64
      ", got 0x%016" PRIx64 ".",
      XNNPackCacheHeader::kMagicValue, header.magic_value);
  XNNPACK_RETURN_CHECK(header.version == XNNPackCacheHeader::kVersion,
                       "Cache header version is incompatible. Expected %" PRIu64
                       ", got %" PRIu64 ".",
                       XNNPackCacheHeader::kVersion, header.version);
  XNNPACK_RETURN_CHECK(!header.stale, "Cache file is stale.");
  return true;
}

}  // namespace

#define XNN_MOVE_CONSTRUCT_MEMBER(x) x(std::move(other.x))
WeightCacheBuilder::WeightCacheBuilder(WeightCacheBuilder&& other)
    : XNN_MOVE_CONSTRUCT_MEMBER(data_),
      XNN_MOVE_CONSTRUCT_MEMBER(schema_),
      XNN_MOVE_CONSTRUCT_MEMBER(capacity_),
      XNN_MOVE_CONSTRUCT_MEMBER(build_segment_size_),
      XNN_MOVE_CONSTRUCT_MEMBER(build_segment_start_),
      XNN_MOVE_CONSTRUCT_MEMBER(first_write_done_),
      XNN_MOVE_CONSTRUCT_MEMBER(buffer_list_loaded_),
      XNN_MOVE_CONSTRUCT_MEMBER(fd_),
      XNN_MOVE_CONSTRUCT_MEMBER(file_path_) {}
#undef XNN_MOVE_CONSTRUCT_MEMBER

WeightCacheBuilder& WeightCacheBuilder::operator=(WeightCacheBuilder&& other) {
#define XNN_MOVE_MEMBER(x) x = std::move(other.x)
  XNN_MOVE_MEMBER(data_);
  XNN_MOVE_MEMBER(schema_);
  XNN_MOVE_MEMBER(capacity_);
  XNN_MOVE_MEMBER(build_segment_size_);
  XNN_MOVE_MEMBER(build_segment_start_);
  XNN_MOVE_MEMBER(first_write_done_);
  XNN_MOVE_MEMBER(buffer_list_loaded_);
  XNN_MOVE_MEMBER(fd_);
  XNN_MOVE_MEMBER(file_path_);
#undef XNN_MOVE_MEMBER
  return *this;
}

bool WeightCacheBuilder::Start(const char* path, const FileDescriptor& fd) {
  XNNPACK_RETURN_CHECK(!IsStarted());
  file_path_ = Sanitize(path);

  XNNPACK_RETURN_CHECK(fd.IsValid(), "File descriptor isn't valid ('%s'): %s.",
                       file_path_.c_str(), strerror(errno));
  fd_ = fd;

  // Write data in the header, this will be overwritten in the `Finalize` call.
  // We explicitly set the header as invalid. If any error happens during
  // the build, reloading the cache file will fail.
  XNNPackCacheHeader header{XNNPackCacheHeader::kInvalidHeader};
  header.buffer_list_offset = sizeof(header);

  XNNPACK_RETURN_CHECK(fd_.Truncate(0), "could not truncate weight cache.");
  XNNPACK_RETURN_CHECK(fd_.SetPos(0) == 0, "couldn't move to file start.");
  XNNPACK_RETURN_CHECK(fd_.Write(&header, sizeof(header)),
                       "could not write initial cache header in %s: %s.",
                       file_path_.c_str(), strerror(errno));

  schema_ = cache::schema::BufferListT{};
  schema_.base_offset = Align(sizeof(header), kMinAlignment);
  build_segment_start_ = header.buffer_list_offset;
  buffer_list_loaded_ = true;

  XNNPACK_RETURN_CHECK(StartBuildStep(), "failed to start initial write step.");
  XNNPACK_RETURN_CHECK(StopBuildStep(), "failed to write initial step.");
  // The initial step writes an empty buffer list at `schema_.base_offset`.
  build_segment_start_ = schema_.base_offset;

  return true;
}

bool WeightCacheBuilder::StartBuildStep() {
  XNNPACK_RETURN_CHECK(IsStarted(),
                       "Trying to start a build step in an invalid builder.");
  XNNPACK_RETURN_CHECK(!is_build_step_.exchange(true),
                       "Failed to start build step: already started. This may "
                       "be a concurrency issue.");
  // Don't leave the builder stuck in a build step if the setup fails.
  ScopeGuard reset_build_step_on_fail([this] { is_build_step_ = false; });

  if (!buffer_list_loaded_) {
    // Reload flatbuffer data.
    XNNPackCacheHeader header;
    fd_.SetPos(0);
    XNNPACK_RETURN_CHECK(fd_.Read(&header, sizeof(header)),
                         "could not read cache file header.");
    if (header.buffer_list_size) {
      MMapHandle buffer_list_data;
      XNNPACK_RETURN_CHECK(buffer_list_data.Map(fd_, header.buffer_list_offset,
                                                file_path_.c_str()),
                           "could not map buffer list mapping");
      flatbuffers::Verifier verifier(
          reinterpret_cast<const uint8_t*>(buffer_list_data.data()),
          header.buffer_list_size);
      XNNPACK_RETURN_CHECK(cache::schema::VerifyBufferListBuffer(verifier),
                           "could not verify buffer list mapping");
      cache::schema::GetBufferList(buffer_list_data.data())->UnPackTo(&schema_);
    } else {
      const uint64_t base_offset = schema_.base_offset;
      schema_ = cache::schema::BufferListT{};
      schema_.base_offset = base_offset;
    }
    build_segment_start_ = header.buffer_list_offset;
  }

  // Move cursor to end of existing data.
  build_segment_size_ = 0;
  XNNPACK_RETURN_CHECK(fd_.SetPos(build_segment_start_) != -1);
  buffer_list_loaded_ = true;

  reset_build_step_on_fail.Deactivate();
  return true;
}

void WeightCacheBuilder::Reset() { *this = WeightCacheBuilder(); }

void WeightCacheBuilder::ReleaseStagingBuffer() {
  if (!is_build_step_.load(std::memory_order_acquire)) {
    data_.reset();
    capacity_ = 0;
  }
}

void* WeightCacheBuilder::Reserve(size_t size) {
  if (size > capacity_) {
    // We don't care about the data when we are reserving space. We save memory
    // by deleting the existing buffer first.
    data_.reset(nullptr);
    data_ = std::make_unique<uint8_t[]>(size + kMinAlignment);
    capacity_ = size;
  }
  return Align(data_.get(), kMinAlignment);
}

BufferLocation WeightCacheBuilder::Append(PackIdentifier pack_id,
                                          const void* data, uint64_t size,
                                          int32_t fingerprint_id) {
  if (!is_build_step_) {
    TFLITE_LOG_PROD(
        tflite::TFLITE_LOG_ERROR,
        "XNNPack weight cache: cannot append data to an unstarted builder.");
    return BufferLocation::Invalid();
  }
  // Not passing a fingerprint id is a logic error on XNNPack's side. If we
  // don't have a fingerprint for an operation, we have no way of ensuring that
  // the generation of the cached data hasn't changed when reloading the cache.
  //
  // If we just log this and continue on with the work. This run will build a
  // cache with cached data that can't be checked in the future. This will lead,
  // in future runs that reuse the cache, to crashes that are impossible to
  // debug or outputs that are nonsensical without any chance of linking this
  // back to this error.
  if (fingerprint_id == 0) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                    "XNNPack weight cache: no fingerprint identifier was set "
                    "when appending a buffer to the cache file.");
    return BufferLocation::Invalid();
  }
  const xnn_fingerprint* fingerprint = xnn_get_fingerprint(fingerprint_id);
  if (!fingerprint) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                    "XNNPack weight cache: could not find a fingerprint with "
                    "id %s when appending a buffer to the cache file.",
                    xnn_fingerprint_id_to_string_u32(fingerprint_id));
    return BufferLocation::Invalid();
  }

  // Add some padding so that the cache file can be mmaped and the buffer
  // stays aligned correctly.
  const size_t offset = Align(fd_.GetPos(), kMinAlignment);
  if (fd_.SetPos(offset) == -1) {
    return BufferLocation::Invalid();
  }
  if (!fd_.Write(data, size)) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                    "XNNPack weight cache: cannot append buffer to cache file");
    return BufferLocation::Invalid();
  }
  buffer_list_loaded_ = false;

  uint64_t fingerprint_value;
  static_assert(sizeof(fingerprint_value) == sizeof(*fingerprint));
  std::memcpy(&fingerprint_value, fingerprint, sizeof(*fingerprint));
  if (std::find(schema_.fingerprints.begin(), schema_.fingerprints.end(),
                fingerprint_value) == schema_.fingerprints.end()) {
    schema_.fingerprints.push_back(fingerprint_value);
  }

  BufferLocation loc{/*offset=*/offset - schema_.base_offset, /*size=*/size};
  cache::schema::BufferT buffer;
  buffer.packing_algorithm_id = pack_id.pack_algorithm_id;
  buffer.weights_id = pack_id.weights_id;
  buffer.bias_id = pack_id.bias_id;
  buffer.offset = loc.offset;
  buffer.size = loc.size;
  schema_.buffers.push_back(std::make_unique<cache::schema::BufferT>(buffer));
  return loc;
}

bool WeightCacheBuilder::StopBuildStep(bool release_staging_buffer) {
  XNNPACK_RETURN_CHECK(is_build_step_.load(std::memory_order_acquire),
                       "Attempting to stop a non existing build step. This may "
                       "be a concurrency issue.");
  // End the build step on every exit path. Free the staging buffer before
  // clearing `is_build_step_` so a concurrent `StartBuildStep` cannot race on
  // `data_` (`ReleaseStagingBuffer()` is a no-op while building).
  ScopeGuard end_build_step([this, release_staging_buffer] {
    if (release_staging_buffer) {
      data_.reset();
      capacity_ = 0;
    }
    is_build_step_.store(false, std::memory_order_release);
  });
  XNNPACK_RETURN_CHECK(fd_.IsValid(), "cache file ('%s') is not open.",
                       file_path_.c_str());

  if (fd_.GetPos() == build_segment_start_ && first_write_done_) {
    // Nothing was written to the file, we can exit early.
    return true;
  }

  flatbuffers::FlatBufferBuilder builder;
  // Add a fake size and the base offset to mutate them afterwards. Otherwise
  // space for it won't be added to the flatbuffer.
  cache::schema::FinishBufferListBuffer(
      builder, cache::schema::BufferList::Pack(builder, &schema_));

  // Add some padding so that the cache file can be mmaped and the buffer
  // stays aligned correctly.
  const size_t layout_offset = Align(fd_.GetPos(), kMinAlignment);
  XNNPACK_RETURN_CHECK(fd_.SetPos(layout_offset) != -1,
                       "could not move in the file: %s", strerror(errno));

  XNNPackCacheHeader header{XNNPackCacheHeader::kVersion,
                            XNNPackCacheHeader::kMagicValue};
  header.buffer_list_offset = fd_.GetPos();
  header.buffer_list_size = builder.GetSize();

  // Write the flatbuffer which serves as a header to index the buffer data.
  XNNPACK_RETURN_CHECK(fd_.Write(builder.GetBufferPointer(), builder.GetSize()),
                       "cannot write buffer list to '%s'.", file_path_.c_str());

  // Save the segment size for that it can be individually mapped.
  build_segment_size_ = fd_.GetPos() - build_segment_start_;

  // Write the header at the beginning of the file.
  XNNPACK_RETURN_CHECK(fd_.SetPos(0) != -1,
                       "could not move in the file to write header to %s",
                       strerror(errno));
  XNNPACK_RETURN_CHECK(fd_.Write(&header, sizeof(header)),
                       "cannot write cache header to %s.", file_path_.c_str());

  TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE,
                  "XNNPack weight cache: written to '%s'.", file_path_.c_str());
  first_write_done_ = true;
  return true;
}

MMapArena::MMapArena(MMapArena&& other) noexcept
    : chunks_(std::move(other.chunks_)),
      pending_ptr_(std::exchange(other.pending_ptr_, nullptr)),
      pending_size_(std::exchange(other.pending_size_, 0)),
      pending_chunk_offset_before_(
          std::exchange(other.pending_chunk_offset_before_, 0)) {}

MMapArena& MMapArena::operator=(MMapArena&& other) noexcept {
  if (this != &other) {
    chunks_ = std::move(other.chunks_);
    pending_ptr_ = std::exchange(other.pending_ptr_, nullptr);
    pending_size_ = std::exchange(other.pending_size_, 0);
    pending_chunk_offset_before_ =
        std::exchange(other.pending_chunk_offset_before_, 0);
  }
  return *this;
}

void* MMapArena::Reserve(size_t size, MMapHandle* evicted_mapping) {
  const auto evict_empty_tail = [&]() {
    if (evicted_mapping != nullptr) {
      *evicted_mapping = std::move(chunks_.back().mapping);
    }
    chunks_.pop_back();
  };
  const auto evict_oversized_empty_tail = [&]() {
    if (!chunks_.empty() && chunks_.back().used == 0 &&
        chunks_.back().mapping.size() > kMaxChunkGrowth) {
      evict_empty_tail();
    }
  };

  // Calls to Reserve / Commit do not interleave. Rewind an uncommitted
  // reservation so unused `Reserve()` calls don't accumulate memory.
  if (pending_ptr_ != nullptr) {
    chunks_.back().used = pending_chunk_offset_before_;
    pending_ptr_ = nullptr;
    pending_size_ = 0;
  }
  if (size > SIZE_MAX - kMinAlignment) {
    evict_oversized_empty_tail();
    return nullptr;
  }
  const size_t alloc_size = std::max<size_t>(size, 1);
  const size_t min_map_size = alloc_size + kChunkAlignmentPadding;
  if (!chunks_.empty() && chunks_.back().used == 0 &&
      chunks_.back().mapping.size() > kMaxChunkGrowth &&
      (min_map_size <= kMaxChunkGrowth ||
       chunks_.back().mapping.size() - kMaxChunkGrowth >= min_map_size)) {
    evict_empty_tail();
  }

  const auto aligned_offset_in = [](const Chunk& chunk) -> size_t {
    const uintptr_t base = reinterpret_cast<uintptr_t>(chunk.mapping.data());
    return Align(base + chunk.used, kMinAlignment) - base;
  };
  const auto record_reservation = [&](Chunk& chunk,
                                      size_t aligned_offset) -> void* {
    pending_chunk_offset_before_ = chunk.used;
    pending_ptr_ = chunk.mapping.data() + aligned_offset;
    pending_size_ = size;
    chunk.used = aligned_offset + alloc_size;
    return pending_ptr_;
  };

  if (!chunks_.empty()) {
    Chunk& chunk = chunks_.back();
    const size_t aligned_offset = aligned_offset_in(chunk);
    if (aligned_offset <= chunk.mapping.size() &&
        alloc_size <= chunk.mapping.size() - aligned_offset) {
      return record_reservation(chunk, aligned_offset);
    }

    // Try expanding the active mapping in place so earlier addresses stay
    // valid.
    if (alloc_size <= SIZE_MAX - aligned_offset) {
      const size_t required = aligned_offset + alloc_size;
      const size_t old_size = chunk.mapping.size();
      const size_t grow_step = std::min(old_size, kMaxChunkGrowth);
      const size_t grown_size =
          old_size <= SIZE_MAX - grow_step ? old_size + grow_step : required;
      const size_t target_size =
          RoundUpToPageSize(std::max(required, grown_size));
      if (target_size != 0 && chunk.mapping.Resize(target_size)) {
        return record_reservation(chunk, aligned_offset);
      }
    }
  }

  const size_t prev_chunk_size =
      chunks_.empty() ? 0 : chunks_.back().mapping.size();
  const size_t next_chunk_size =
      prev_chunk_size == 0
          ? 0
          : std::min(prev_chunk_size <= SIZE_MAX / 2 ? prev_chunk_size * 2
                                                     : kMaxChunkGrowth,
                     kMaxChunkGrowth);
  const size_t map_size = std::max(min_map_size, next_chunk_size);
  MMapHandle new_mapping;
  if (!new_mapping.Map(MMapHandle::kAnonymous, map_size, kWeightCacheVmaName)) {
    evict_oversized_empty_tail();
    return nullptr;
  }
  if (!chunks_.empty() && chunks_.back().used == 0) {
    if (evicted_mapping != nullptr) {
      *evicted_mapping = std::move(chunks_.back().mapping);
    }
    chunks_.back().mapping = std::move(new_mapping);
  } else {
    chunks_.push_back(Chunk{std::move(new_mapping), /*used=*/0});
  }

  Chunk& chunk = chunks_.back();
  return record_reservation(chunk, aligned_offset_in(chunk));
}

bool MMapArena::Commit(const void* ptr, size_t size) {
  if (pending_ptr_ == nullptr || ptr != pending_ptr_ || size > pending_size_) {
    return false;
  }
  Chunk& chunk = chunks_.back();
  const size_t aligned_offset =
      static_cast<const uint8_t*>(pending_ptr_) - chunk.mapping.data();
  chunk.used = aligned_offset + std::max<size_t>(size, 1);
  pending_ptr_ = nullptr;
  pending_size_ = 0;
  return true;
}

bool MMapArena::OverrunsActiveChunk(const void* data,
                                    size_t size) const noexcept {
  if (size == 0 || data == nullptr || chunks_.empty()) {
    return false;
  }
  const Chunk& chunk = chunks_.back();
  const uintptr_t start = reinterpret_cast<uintptr_t>(chunk.mapping.data());
  const uintptr_t end = start + chunk.mapping.size();
  const uintptr_t addr = reinterpret_cast<uintptr_t>(data);
  return start <= addr && addr < end && size > end - addr;
}

CacheMissHandler::CacheMissHandler(CacheMissHandler&& other) noexcept
    : arena_(std::move(other.arena_)),
      min_offset_(std::exchange(other.min_offset_, 0)),
      buffer_count_(std::exchange(other.buffer_count_, 0)),
      append_count_(std::exchange(other.append_count_, 0)),
      reserve_count_(std::exchange(other.reserve_count_, 0)) {}

CacheMissHandler& CacheMissHandler::operator=(
    CacheMissHandler&& other) noexcept {
  if (this != &other) {
    arena_ = std::move(other.arena_);
    min_offset_ = std::exchange(other.min_offset_, 0);
    buffer_count_ = std::exchange(other.buffer_count_, 0);
    append_count_ = std::exchange(other.append_count_, 0);
    reserve_count_ = std::exchange(other.reserve_count_, 0);
  }
  return *this;
}

void* CacheMissHandler::Reserve(size_t size) {
  void* const ptr = arena_.Reserve(size);
  if (ptr != nullptr) {
    ++reserve_count_;
  }
  return ptr;
}

BufferLocation CacheMissHandler::Append(
    PackIdentifier pack_id, const void* data, uint64_t size, int fingerprint_id,
    std::map<size_t, void*>& offset_to_addr) {
  (void)pack_id;
  (void)fingerprint_id;
  if (size > SIZE_MAX) {
    return BufferLocation::Invalid();
  }
  void* ptr = nullptr;
  if (arena_.Commit(data, size)) {
    ptr = const_cast<void*>(data);
  } else {
    if (arena_.OverrunsActiveChunk(data, size)) {
      return BufferLocation::Invalid();
    }
    MMapHandle evicted_mapping;
    ptr = arena_.Reserve(size, &evicted_mapping);
    if (ptr == nullptr) {
      return BufferLocation::Invalid();
    }
    ++reserve_count_;
    if (size > 0 && ptr != data) {
      std::memmove(ptr, data, size);
    }
    if (!arena_.Commit(ptr, size)) {
      return BufferLocation::Invalid();
    }
  }
  const BufferLocation loc{/*offset=*/min_offset_ + buffer_count_,
                           /*size=*/size};
  ++buffer_count_;
  ++append_count_;
  offset_to_addr.insert({loc.offset, ptr});
  return loc;
}

#define XNN_MOVE_CONSTRUCT_MEMBER(member) member(std::move(other.member))
MMapWeightCacheProvider::MMapWeightCacheProvider(
    MMapWeightCacheProvider&& other)
    : XNN_MOVE_CONSTRUCT_MEMBER(cache_provider_),
      XNN_MOVE_CONSTRUCT_MEMBER(file_path_),
      XNN_MOVE_CONSTRUCT_MEMBER(buffer_address_to_identifier_),
      XNN_MOVE_CONSTRUCT_MEMBER(buffer_remaps_),
      XNN_MOVE_CONSTRUCT_MEMBER(cache_key_to_offset_),
      XNN_MOVE_CONSTRUCT_MEMBER(mmap_handles_),
      XNN_MOVE_CONSTRUCT_MEMBER(mmap_buffer_base_offset_),
      XNN_MOVE_CONSTRUCT_MEMBER(file_descriptor_),
      XNN_MOVE_CONSTRUCT_MEMBER(builder_),
      XNN_MOVE_CONSTRUCT_MEMBER(cache_miss_handler_),
      XNN_MOVE_CONSTRUCT_MEMBER(offset_to_addr_),
      in_memory_(std::exchange(other.in_memory_, false)) {
  // The contexts need to keep pointing to their owning object.
  cache_provider_.context = this;
  other.cache_provider_.context = &other;
}
#undef XNN_MOVE_CONSTRUCT_MEMBER

MMapWeightCacheProvider& MMapWeightCacheProvider::operator=(
    MMapWeightCacheProvider&& other) {
  this->WriteCacheMissFlag();
#define XNN_MOVE_MEMBER(member) member = std::move(other.member)
  XNN_MOVE_MEMBER(cache_provider_);
  // The contexts need to keep pointing to their owning object.
  cache_provider_.context = this;
  other.cache_provider_.context = &other;
  XNN_MOVE_MEMBER(file_path_);
  XNN_MOVE_MEMBER(buffer_address_to_identifier_);
  XNN_MOVE_MEMBER(buffer_remaps_);
  XNN_MOVE_MEMBER(cache_key_to_offset_);
  XNN_MOVE_MEMBER(mmap_handles_);
  XNN_MOVE_MEMBER(mmap_buffer_base_offset_);
  XNN_MOVE_MEMBER(file_descriptor_);
  XNN_MOVE_MEMBER(builder_);
  XNN_MOVE_MEMBER(cache_miss_handler_);
  XNN_MOVE_MEMBER(offset_to_addr_);
  in_memory_ = std::exchange(other.in_memory_, false);
#undef XNN_MOVE_MEMBER
  return *this;
}

bool MMapWeightCacheProvider::WriteCacheMissFlag() {
  // In-memory caches don't have a file to mark as stale: all their buffers go
  // through the cache miss handler.
  if (cache_miss_handler_.HasCacheMisses() && !in_memory_) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                    "Cache file is stale. Setting stale flag.");
    return WriteStaleFlag();
  }
  return true;
}

bool MMapWeightCacheProvider::WriteStaleFlag() {
  if (file_descriptor_.IsValid()) {
    constexpr decltype(XNNPackCacheHeader::stale) stale = 1;
    constexpr size_t stale_offset = offsetof(XNNPackCacheHeader, stale);
    XNNPACK_RETURN_CHECK(file_descriptor_.SetPos(stale_offset) == stale_offset,
                         "Could not move file cursor to write stale flag.");
    XNNPACK_RETURN_CHECK(
        file_descriptor_.Write(&stale, sizeof(XNNPackCacheHeader::stale)),
        "Cannot write stale flag to cache file.");
  }
  return true;
}

MMapWeightCacheProvider::~MMapWeightCacheProvider() { WriteCacheMissFlag(); }

void MMapWeightCacheProvider::SetFilePath(const char* path) {
  XNNPACK_ABORT_CHECK(
      !IsBuilding(),
      "Cannot change the path of a cache that has already been loaded.");
  const char* const safe_path = Sanitize(path);
  if (file_path_ != safe_path) {
    // We try to keep file_path_'s data as stable as possible. Don't overwrite
    // if the path hasn't changed.
    file_path_ = safe_path;
  }
}

bool MMapWeightCacheProvider::LoadOrStartBuild(const char* path,
                                               FileDescriptor fd) {
  if (!path && !fd.IsValid()) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                    "Cannot load or build XNNPack cache without specifying a "
                    "path or a file descriptor.");
    return false;
  }
  const char* const safe_path = Sanitize(path);
  FileDescriptor build_fd = fd.Duplicate();
  if (!IsInMemoryCachePath(safe_path) && !IsFileEmpty(safe_path, fd) &&
      Load(safe_path, std::move(fd))) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE,
                    "XNNPack weight cache loaded from '%s'.", safe_path);
    return true;
  } else if (StartBuild(safe_path, std::move(build_fd))) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE,
                    "XNNPack weight cache build for '%s' started.", safe_path);
    return true;
  }
  return false;
}

bool MMapWeightCacheProvider::StartBuild(const char* path, FileDescriptor fd) {
  const char* const safe_path = Sanitize(path);
  SetFilePath(safe_path);
  ResetCache();

  if (!fd.IsValid() && IsInMemoryCachePath(file_path_)) {
    // In-memory caches are never written to a file. Packed buffers are stored
    // by the cache miss handler as they are inserted, without build steps.
    in_memory_ = true;
    return true;
  }

  if (!fd.IsValid()) {
    fd = FileDescriptor::Open(file_path_.c_str(), O_CREAT | O_TRUNC | O_RDWR,
                              0644);
  }
  XNNPACK_RETURN_CHECK(fd.IsValid(), "could not open file ('%s'): %s.",
                       file_path_.c_str(), strerror(errno));
  file_descriptor_ = std::move(fd);
  return builder_.Start(safe_path, file_descriptor_);
}

bool MMapWeightCacheProvider::Load(const std::string& path, FileDescriptor fd) {
  SetFilePath(path.c_str());
  ResetCache();
  file_descriptor_ = std::move(fd);
  return Load();
}

bool MMapWeightCacheProvider::Load() {
  mmap_buffer_base_offset_ = 0;
  cache_key_to_offset_.clear();
  mmap_handles_.resize(1);
  MMapHandle& mmap_handle = mmap_handles_.front();
  ScopeGuard unmap_on_fail([this] { mmap_handles_.clear(); });

  if (file_descriptor_.IsValid()) {
    XNNPACK_RETURN_CHECK(
        mmap_handle.Map(file_descriptor_, /*offset=*/0, file_path_.c_str()));
  } else {
    XNNPACK_RETURN_CHECK(!file_path_.empty(),
                         "Path wasn't provided to weight cache provider.");
    if (!FileExists(file_path_.c_str())) {
      TFLITE_LOG(tflite::TFLITE_LOG_WARNING,
                 "XNNPack weight cache: could not load '%s': %s.",
                 file_path_.c_str(), strerror(errno));
      return false;
    }
    XNNPACK_RETURN_CHECK(mmap_handle.Map(file_path_.c_str()));
  }

  XNNPACK_RETURN_CHECK(mmap_handle.size() >= sizeof(XNNPackCacheHeader),
                       "invalid cache file size: %zu, expected at least %zu.",
                       mmap_handle.size(), sizeof(XNNPackCacheHeader));

  const XNNPackCacheHeader header = [&mmap_handle] {
    XNNPackCacheHeader header;
    memcpy(&header, mmap_handle.data(), sizeof(header));
    return header;
  }();

  XNNPACK_RETURN_CHECK(IsValidHeader(header), "Cache needs to be built again.");

  XNNPACK_RETURN_CHECK(header.buffer_list_offset < mmap_handle.size(),
                       "invalid offset for buffer list descriptor.");

  XNNPACK_RETURN_CHECK(
      header.buffer_list_size == mmap_handle.size() - header.buffer_list_offset,
      "invalid size for buffer list descriptor.");

  // Verify the flatbuffer part of the file.
  flatbuffers::Verifier verifier(mmap_handle.data() + header.buffer_list_offset,
                                 header.buffer_list_size);
  XNNPACK_RETURN_CHECK(cache::schema::VerifyBufferListBuffer(verifier),
                       "buffer list validation failed.");

  // Load flatbuffer.
  const cache::schema::BufferList* buffer_list = cache::schema::GetBufferList(
      mmap_handle.data() + header.buffer_list_offset);
  XNNPACK_RETURN_CHECK(buffer_list,
                       "could not get packed weights from flatbuffer.");

  XNNPACK_RETURN_CHECK(CheckFingerprints(buffer_list));

  size_t max_buffer_offset = 0;
  mmap_buffer_base_offset_ = buffer_list->base_offset();
  XNNPACK_RETURN_CHECK(mmap_handle.size() >= mmap_buffer_base_offset_,
                       "buffer base offset overflows the mapping size.");
  if (const auto buffers = buffer_list->buffers(); buffers) {
    for (auto* buffer : *buffers) {
      XNNPACK_RETURN_CHECK(buffer, "invalid buffer address in buffer list.");
      XNNPACK_RETURN_CHECK(
          mmap_handle.size() - mmap_buffer_base_offset_ >= buffer->offset(),
          "buffer start is after the end of the mapping.");
      XNNPACK_RETURN_CHECK(
          mmap_handle.size() - mmap_buffer_base_offset_ - buffer->offset() >
              buffer->size(),
          "buffer end is after the end of the mapping.");
      cache_key_to_offset_.emplace(
          PackIdentifier{/*pack_algorithm_id=*/buffer->packing_algorithm_id(),
                         /*weights_id=*/buffer->weights_id(),
                         /*bias_id=*/buffer->bias_id()},
          BufferLocation{/*offset=*/buffer->offset(), /*size=*/buffer->size()});
      offset_to_addr_.insert(
          {buffer->offset(),
           mmap_handle.data() + mmap_buffer_base_offset_ + buffer->offset()});
      max_buffer_offset = std::max<size_t>(max_buffer_offset, buffer->offset());
    }
  }
  cache_miss_handler_.SetMinOffset(max_buffer_offset + 1);
  unmap_on_fail.Deactivate();
  return true;
}

bool MMapWeightCacheProvider::LockMemory() {
  for (auto& mmap_handle : mmap_handles_) {
    XNNPACK_RETURN_CHECK(mmap_handle.LockMemory(),
                         "could not lock cache in memory.");
  }
  return true;
}

bool MMapWeightCacheProvider::UnlockMemory() {
  for (auto& mmap_handle : mmap_handles_) {
    XNNPACK_RETURN_CHECK(mmap_handle.UnlockMemory(),
                         "could not unlock cache in memory.");
  }
  return true;
}

bool MMapWeightCacheProvider::LoadLastBuildStep() {
  if (mmap_handles_.empty()) {
    return Load();
  }

  if (builder_.LastBuildStepSize() == 0) {
    return true;
  }

  const XNNPackCacheHeader header = [this] {
    XNNPackCacheHeader header;
    memcpy(&header, mmap_handles_.front().data(), sizeof(header));
    return header;
  }();

  // Map last data segment:
  // - either resize the last mmap handle;
  // - or add a new mapping handle.
  {
    MMapHandle& last_mmap_handle = mmap_handles_.back();
    const size_t last_mmap_size = last_mmap_handle.size();
    if (!last_mmap_handle.Resize(last_mmap_size +
                                 builder_.LastBuildStepSize())) {
      mmap_handles_.emplace_back();
      if (file_descriptor_.IsValid()) {
        XNNPACK_RETURN_CHECK(
            mmap_handles_.back().Map(file_descriptor_,
                                     /*offset=*/builder_.LastBuildStepStart(),
                                     file_path_.c_str()),
            "could not map last build step");
      } else {
        XNNPACK_RETURN_CHECK(
            mmap_handles_.back().Map(file_path_.c_str(),
                                     /*offset=*/builder_.LastBuildStepStart()),
            "could not map last build step");
      }
    }
  }
  // Read the updated buffer list.
  MMapHandle& segment_mmap_handle = mmap_handles_.back();
  const size_t buffer_list_offset =
      header.buffer_list_offset - segment_mmap_handle.offset();

  flatbuffers::Verifier verifier(
      segment_mmap_handle.data() + buffer_list_offset, header.buffer_list_size);
  XNNPACK_RETURN_CHECK(cache::schema::VerifyBufferListBuffer(verifier),
                       "buffer list validation failed.");

  const cache::schema::BufferList* buffer_list = cache::schema::GetBufferList(
      segment_mmap_handle.data() + buffer_list_offset);
  XNNPACK_RETURN_CHECK(buffer_list,
                       "could not get packed weights from flatbuffer.");

  // Update offset_to_addr_ with new offsets
  const ptrdiff_t offset_modifier =
      buffer_list->base_offset() - segment_mmap_handle.offset();
  for (const auto* buffer : *(buffer_list->buffers())) {
    const size_t offset = buffer->offset();
    if (!offset_to_addr_.count(offset)) {
      offset_to_addr_.insert(
          {offset, segment_mmap_handle.data() + offset + offset_modifier});
    }
  }
  return true;
}

bool MMapWeightCacheProvider::StartBuildStep() {
  if (in_memory_) {
    // In-memory caches store buffers as they are inserted. Build steps are
    // no-ops.
    return true;
  }
  XNNPACK_RETURN_CHECK(CanStartBuildStep(),
                       "cannot append data to an existing cache file.");
  if (IsBuilding()) {
    return true;
  }
  return builder_.StartBuildStep();
}

bool MMapWeightCacheProvider::StopBuildStep(bool release_staging_buffer) {
  if (in_memory_) {
    return true;
  }
  XNNPACK_RETURN_CHECK(builder_.StopBuildStep(release_staging_buffer));
#if defined(XNNPACK_CACHE_NO_FILE_MAPPING_FOR_DEBUG)
  if (!mmap_handles_.empty()) {
    // Sync mmap_handles_.data() with the content updated by
    // builder_.StopBuildStep().
    XNNPACK_RETURN_CHECK(file_descriptor_.IsValid());
    XNNPACK_RETURN_CHECK(mmap_handles_.front().Map(
        file_descriptor_, /*offset=*/0, file_path_.c_str()));
  }
#endif
  return LoadLastBuildStep();
}

bool MMapWeightCacheProvider::MapTensorIdentifiers(
    const TfLiteTensor* tensors, const size_t size,
    const std::unordered_map<size_t, size_t>& tensor_index_to_identifier) {
  for (const auto [index, identifier] : tensor_index_to_identifier) {
    XNNPACK_RETURN_CHECK(index < size,
                         "Tensor index corresponds to a non existing tensor.");
    const TfLiteTensor& t = tensors[index];
    XNNPACK_RETURN_CHECK(MapBufferIdentifier(t.data.data, t.bytes, identifier));
  }
  return true;
}

bool MMapWeightCacheProvider::MapBufferIdentifier(const void* buffer,
                                                  size_t size,
                                                  uint64_t identifier) {
  if (!buffer) {
    return false;
  }
  buffer_address_to_identifier_.emplace(
      buffer, OriginalBufferMetadata{identifier, size});
  return true;
}

void MMapWeightCacheProvider::RemapDataBuffer(const void* const buffer,
                                              const void* const new_buffer) {
  buffer_remaps_[new_buffer] = buffer;
}

size_t MMapWeightCacheProvider::LookUp(
    const xnn_weights_cache_look_up_key* cache_key) {
  if (!cache_key) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                    "XNNPack weight cache: a null cache key was provided.");
    return SIZE_MAX;
  }
  const PackIdentifier pack_id = BuildPackIdentifier(*cache_key);
  // Note: we don't check if pack_id is valid here, the lookup in the map will
  // fail in any case.
  if (auto offset_it = cache_key_to_offset_.find(pack_id);
      offset_it != cache_key_to_offset_.end()) {
    return offset_it->second.offset;
  }
  return SIZE_MAX;
}

void* MMapWeightCacheProvider::ReserveSpace(size_t size) {
  return builder_.IsBuilding() ? builder_.Reserve(size)
                               : cache_miss_handler_.Reserve(size);
}

size_t MMapWeightCacheProvider::LookUpOrInsert(
    const xnn_weights_cache_look_up_key* cache_key, void* ptr, size_t size) {
  XNNPACK_ABORT_CHECK(cache_key, "A null cache key was provided.");

  const PackIdentifier pack_id = BuildPackIdentifier(*cache_key);
  XNNPACK_ABORT_CHECK(pack_id.HasValidBufferIds(),
                      "An invalid pack identifier was created.");
  if (auto offset_it = cache_key_to_offset_.find(pack_id);
      offset_it != cache_key_to_offset_.end()) {
    return offset_it->second.offset;
  }

  const BufferLocation location =
      builder_.IsBuilding()
          ? builder_.Append(pack_id, ptr, size, cache_key->fingerprint_id)
          : cache_miss_handler_.Append(
                pack_id, ptr, size, cache_key->fingerprint_id, offset_to_addr_);
  XNNPACK_ABORT_CHECK(!location.IsInvalid(),
                      "Inserting data in the cache failed.");
  cache_key_to_offset_.emplace(pack_id, location);

  // Signals the system that the original weights are not needed anymore.
  auto SignalDataNotNeeded = [&](const void* data) {
    if (auto it = buffer_address_to_identifier_.find(data);
        it != buffer_address_to_identifier_.end()) {
      MarkMemoryNotNeeded(const_cast<void*>(data), it->second.size);
    }
  };

  SignalDataNotNeeded(cache_key->kernel);
  SignalDataNotNeeded(cache_key->bias);

  return location.offset;
}

void* MMapWeightCacheProvider::OffsetToAddr(const size_t offset) {
  auto it = offset_to_addr_.find(offset);
  if (it == offset_to_addr_.end()) {
    WriteStaleFlag();
    XNNPACK_ABORT_CHECK(
        false,  // The check for finding the offset if done above.
        "No address is mapped to the given offset (%zu). This is a critical "
        "error that should not happen. Check that the model hasn't changed "
        "since the cache was created.",
        offset);
  }
  return it->second;
}

void MMapWeightCacheProvider::ResetCache() {
  // Flag the cache file before forgetting about the cache misses.
  WriteCacheMissFlag();
  cache_key_to_offset_.clear();
  mmap_handles_.clear();
  offset_to_addr_.clear();
  mmap_buffer_base_offset_ = 0;
  // The builder only has a view of `file_descriptor_`: reset it first.
  builder_ = WeightCacheBuilder();
  cache_miss_handler_ = CacheMissHandler();
  file_descriptor_.Close();
  in_memory_ = false;
}

void MMapWeightCacheProvider::Release() {
  ResetCache();
  buffer_address_to_identifier_.clear();
  buffer_remaps_.clear();
  file_path_.clear();
}

size_t MMapWeightCacheProvider::look_up(
    void* context, const xnn_weights_cache_look_up_key* cache_key) {
  return reinterpret_cast<MMapWeightCacheProvider*>(context)->LookUp(cache_key);
}

void* MMapWeightCacheProvider::reserve_space(void* context, size_t n) {
  return reinterpret_cast<MMapWeightCacheProvider*>(context)->ReserveSpace(n);
}

size_t MMapWeightCacheProvider::look_up_or_insert(
    void* context, const xnn_weights_cache_look_up_key* cache_key, void* ptr,
    size_t size) {
  return reinterpret_cast<MMapWeightCacheProvider*>(context)->LookUpOrInsert(
      cache_key, ptr, size);
}

bool MMapWeightCacheProvider::is_finalized(void* context) {
  return reinterpret_cast<MMapWeightCacheProvider*>(context)->IsActive();
}

void* MMapWeightCacheProvider::offset_to_addr(void* context, size_t offset) {
  return reinterpret_cast<MMapWeightCacheProvider*>(context)->OffsetToAddr(
      offset);
}

enum xnn_status MMapWeightCacheProvider::delete_cache(void* context) {
  reinterpret_cast<MMapWeightCacheProvider*>(context)->Release();
  return xnn_status_success;
}

enum xnn_status MMapWeightCacheProvider::alias_data(void* context, void* alias,
                                                    void* original) {
  reinterpret_cast<MMapWeightCacheProvider*>(context)->RemapDataBuffer(original,
                                                                       alias);
  return xnn_status_success;
}

PackIdentifier MMapWeightCacheProvider::BuildPackIdentifier(
    const xnn_weights_cache_look_up_key& key) {
  const auto get_buffer_id = [&](const void* buffer) -> size_t {
    if (!buffer) {
      return PackIdentifier::kNoId;
    }
    // Look the buffer up in the primary map.
    const auto identifier_it = buffer_address_to_identifier_.find(buffer);
    if (identifier_it != buffer_address_to_identifier_.end()) {
      return identifier_it->second.identifier;
    }
    // We could have several layers of remapping. We look through
    // buffer_remaps_ until we find a valid identifier or nothing is mapped to
    // the current buffer pointer.
    for (auto remapped_it = buffer_remaps_.find(buffer);
         remapped_it != buffer_remaps_.end();
         remapped_it = buffer_remaps_.find(remapped_it->second)) {
      const auto remapped_identifier_it =
          buffer_address_to_identifier_.find(remapped_it->second);
      if (remapped_identifier_it != buffer_address_to_identifier_.end()) {
        return remapped_identifier_it->second.identifier;
      }
    }
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                    "Unknown constant buffer passed to BuildPackIdentifier.");
    return PackIdentifier::kInvalidId;
  };
  return PackIdentifier{/*pack_algorithm_id=*/key.seed,
                        /*weights_id=*/get_buffer_id(key.kernel),
                        /*bias_id=*/get_buffer_id(key.bias)};
}

bool IsCompatibleCacheFile(const char* path) {
  FileDescriptor fd = FileDescriptor::Open(path, O_RDONLY);
  XNNPACK_RETURN_CHECK(fd.IsValid(), "Could not open file: %s: %s.", path,
                       strerror(errno));
  return IsCompatibleCacheFile(std::move(fd));
}

bool IsCompatibleCacheFile(FileDescriptorView fd) {
  XNNPACK_RETURN_CHECK(fd.IsValid(), "Invalid file descriptor: %d.",
                       fd.Value());
  const size_t current_pos = fd.GetPos();
  ScopeGuard reset_pos_on_return(
      [current_pos, &fd] { fd.SetPos(current_pos); });
  XNNPACK_RETURN_CHECK(fd.SetPos(0) != -1,
                       "Couldn't move to the start of the file.");

  XNNPackCacheHeader header;
  XNNPACK_RETURN_CHECK(fd.Read(&header, sizeof(header)),
                       "Couldn't read file header.");
  XNNPACK_RETURN_CHECK(IsValidHeader(header));

  fd.SetPos(header.buffer_list_offset);
  auto buffer = std::make_unique<uint8_t[]>(header.buffer_list_size);
  XNNPACK_RETURN_CHECK(fd.Read(buffer.get(), header.buffer_list_size));

  flatbuffers::Verifier verifier(buffer.get(), header.buffer_list_size);
  XNNPACK_RETURN_CHECK(cache::schema::VerifyBufferListBuffer(verifier),
                       "buffer list validation failed.");

  const cache::schema::BufferList* buffer_list =
      cache::schema::GetBufferList(buffer.get());
  XNNPACK_RETURN_CHECK(buffer_list,
                       "could not get packed weights from flatbuffer.");

  XNNPACK_RETURN_CHECK(
      xnn_initialize(/*allocator=*/nullptr) == xnn_status_success,
      "xnn_initialize failed.");
  XNNPACK_RETURN_CHECK(CheckFingerprints(buffer_list));
  return true;
}

}  // namespace tflite::xnnpack

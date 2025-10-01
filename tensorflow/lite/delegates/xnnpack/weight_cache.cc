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

#include <cerrno>  // IWYU pragma: keep
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "xnnpack.h"  // from @XNNPACK
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/verifier.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/xnnpack/file_util.h"
#include "tensorflow/lite/delegates/xnnpack/mmap_handle.h"
#include "tensorflow/lite/delegates/xnnpack/weight_cache_schema_generated.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"

#define XNNPACK_ABORT_CHECK(TEST, ...)                      \
  if (!(TEST)) {                                            \
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR, __VA_ARGS__); \
    std::abort();                                           \
  }

#define XNNPACK_VAR_ARG_HEAD(FIRST, ...) FIRST

#define XNNPACK_RETURN_CHECK(TEST, ...)                              \
  if (!(TEST)) {                                                     \
    if (sizeof(XNNPACK_VAR_ARG_HEAD("" __VA_ARGS__)) > sizeof("")) { \
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,                      \
                      "XNNPack weight cache: " __VA_ARGS__);         \
    }                                                                \
    return false;                                                    \
  }

namespace tflite::xnnpack {

namespace {
constexpr size_t kMinAlignment = 128;

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

// Returns true if the given path exists.
[[nodiscard]]
bool FileExists(const char* path) {
  return access(path, F_OK) != -1;
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

  XNNPACK_RETURN_CHECK(fd_.Truncate(0), "could not truncate weight cache");
  XNNPACK_RETURN_CHECK(fd_.Write(&header, sizeof(header)),
                       "could not write initial cache header in %s: %s.",
                       file_path_.c_str(), strerror(errno));

  schema_.base_offset = Align(sizeof(header), kMinAlignment);
  return true;
}

bool WeightCacheBuilder::StartBuildStep() {
  XNNPACK_RETURN_CHECK(IsStarted());

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
  }

  // Move cursor to end of existing data.
  build_segment_size_ = 0;
  build_segment_start_ = fd_.SetPos(header.buffer_list_offset);
  XNNPACK_RETURN_CHECK(build_segment_start_ != -1);

  is_build_step_ = true;
  return true;
}

void WeightCacheBuilder::Reset() { *this = WeightCacheBuilder(); }

void* WeightCacheBuilder::Reserve(size_t size) {
  if (size > capacity_) {
    // We don't care about the data when we are reserving space. We save memory
    // by deleting the existing buffer first.
    data_.reset(nullptr);
    data_ = std::make_unique<uint8_t[]>(size + kMinAlignment);
    capacity_ = size;
  }
  return reinterpret_cast<void*>(
      Align(reinterpret_cast<size_t>(data_.get()), kMinAlignment));
}

BufferLocation WeightCacheBuilder::Append(PackIdentifier pack_id,
                                          const void* data, uint64_t size) {
  XNNPACK_ABORT_CHECK(is_build_step_,
                      "cannot append data to an unstarted builder.");
  // Add some padding so that the cache file can be mmaped and the buffer
  // stays aligned correctly.
  const size_t offset = Align(fd_.GetPos(), kMinAlignment);
  if (fd_.SetPos(offset) == -1) {
    return BufferLocation::Invalid();
  }

  BufferLocation loc{/*offset=*/offset - schema_.base_offset, /*size=*/size};
  cache::schema::BufferT buffer;
  buffer.packing_algorithm_id = pack_id.pack_algorithm_id;
  buffer.weights_id = pack_id.weights_id;
  buffer.bias_id = pack_id.bias_id;
  buffer.offset = loc.offset;
  buffer.size = loc.size;
  schema_.buffers.push_back(std::make_unique<cache::schema::BufferT>(buffer));

  if (!fd_.Write(data, size)) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                    "XNNPack weight cache: cannot append buffer to cache file");
    return BufferLocation::Invalid();
  }
  return loc;
}

bool WeightCacheBuilder::StopBuildStep() {
  if (!is_build_step_) {
    return true;
  }
  XNNPACK_RETURN_CHECK(fd_.IsValid(),
                       "cache file ('%s') is not open for writing: %s.",
                       file_path_.c_str(), strerror(errno));

  is_build_step_ = false;
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

  XNNPACK_RETURN_CHECK(
      sizeof(XNNPackCacheHeader::xnnpack_build_identifier) ==
          xnn_experimental_get_build_identifier_size(),
      "cache file ('%s') header cannot hold XNNPack's build identifier: %s.",
      file_path_.c_str(), strerror(errno));

  XNNPackCacheHeader header{XNNPackCacheHeader::kVersion};
  memcpy(header.xnnpack_build_identifier,
         xnn_experimental_get_build_identifier_data(),
         xnn_experimental_get_build_identifier_size());
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
      XNN_MOVE_CONSTRUCT_MEMBER(building_run_),
      XNN_MOVE_CONSTRUCT_MEMBER(offset_to_addr_) {
  // The contexts need to keep pointing to their owning object.
  cache_provider_.context = this;
  other.cache_provider_.context = &other;
}
#undef XNN_MOVE_CONSTRUCT_MEMBER

MMapWeightCacheProvider& MMapWeightCacheProvider::operator=(
    MMapWeightCacheProvider&& other) {
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
  XNN_MOVE_MEMBER(building_run_);
  XNN_MOVE_MEMBER(offset_to_addr_);
#undef XNN_MOVE_MEMBER
  return *this;
}

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
  if (!IsInMemoryCachePath(safe_path) && Load(safe_path, std::move(fd))) {
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

  if (!fd.IsValid()) {
    if (IsInMemoryCachePath(file_path_)) {
      fd = CreateInMemoryFileDescriptor("XNNPack in-memory weight cache");
    } else {
      fd = FileDescriptor::Open(file_path_.c_str(), O_CREAT | O_TRUNC | O_RDWR,
                                0644);
    }
  }
  XNNPACK_RETURN_CHECK(fd.IsValid(), "could not open file ('%s'): %s.",
                       file_path_.c_str(), strerror(errno));
  file_descriptor_ = std::move(fd);
  building_run_ = builder_.Start(safe_path, file_descriptor_);
  return building_run_;
}

bool MMapWeightCacheProvider::Load(const std::string& path, FileDescriptor fd) {
  SetFilePath(path.c_str());
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
    XNNPACK_ABORT_CHECK(!file_path_.empty(),
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

  XNNPACK_RETURN_CHECK(header.version == XNNPackCacheHeader::kVersion,
                       "incompatible header version. Got %zd, expected %zd. "
                       "Cache needs to be built again.",
                       header.version, XNNPackCacheHeader::kVersion);

  XNNPACK_RETURN_CHECK(xnn_experimental_check_build_identifier(
                           header.xnnpack_build_identifier,
                           sizeof(header.xnnpack_build_identifier)),
                       "XNNPack weight cache: incompatible XNNPack version. "
                       "Cache needs to be built again.");

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

  mmap_buffer_base_offset_ = buffer_list->base_offset();
  if (const auto buffers = buffer_list->buffers(); buffers) {
    for (auto* buffer : *buffers) {
      XNNPACK_RETURN_CHECK(buffer, "invalid buffer address in buffer list.");
      cache_key_to_offset_.emplace(
          PackIdentifier{/*pack_algorithm_id=*/buffer->packing_algorithm_id(),
                         /*weights_id=*/buffer->weights_id(),
                         /*bias_id=*/buffer->bias_id()},
          BufferLocation{/*offset=*/buffer->offset(), /*size=*/buffer->size()});
      offset_to_addr_.insert(
          {buffer->offset(),
           mmap_handle.data() + mmap_buffer_base_offset_ + buffer->offset()});
    }
  }

  unmap_on_fail.Deactivate();
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
    const int last_mmap_size = last_mmap_handle.size();
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
  XNNPACK_RETURN_CHECK(CanStartBuildStep(),
                       "cannot append data to an existing cache file.");
  if (IsBuilding()) {
    return true;
  }
  return builder_.StartBuildStep();
}

bool MMapWeightCacheProvider::StopBuildStep() {
  XNNPACK_RETURN_CHECK(builder_.StopBuildStep());
#if defined(XNNPACK_CACHE_NO_MMAP_FOR_TEST)
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

void MMapWeightCacheProvider::MapTensorIdentifiers(
    const TfLiteTensor* tensors, const size_t size,
    const std::unordered_map<size_t, size_t>& tensor_index_to_identifier) {
  for (const auto [index, identifier] : tensor_index_to_identifier) {
    XNNPACK_ABORT_CHECK(index < size,
                        "Tensor index corresponds to a non existing tensor.");
    buffer_address_to_identifier_[tensors[index].data.data] = identifier;
  }
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
  if (auto offset_it = cache_key_to_offset_.find(pack_id);
      offset_it != cache_key_to_offset_.end()) {
    return offset_it->second.offset;
  }
  return SIZE_MAX;
}

void* MMapWeightCacheProvider::ReserveSpace(size_t size) {
  XNNPACK_ABORT_CHECK(IsBuilding(),
                      "Cannot reserve space in a cache that isn't building.");
  return builder_.Reserve(size);
}

size_t MMapWeightCacheProvider::LookUpOrInsert(
    const xnn_weights_cache_look_up_key* cache_key, void* ptr, size_t size) {
  XNNPACK_ABORT_CHECK(cache_key, "A null cache key was provided.");

  const PackIdentifier pack_id = BuildPackIdentifier(*cache_key);
  if (auto offset_it = cache_key_to_offset_.find(pack_id);
      offset_it != cache_key_to_offset_.end()) {
    return offset_it->second.offset;
  }

  const BufferLocation location = builder_.Append(pack_id, ptr, size);
  XNNPACK_ABORT_CHECK(!location.IsInvalid(),
                      "Inserting data in the cache failed.");
  cache_key_to_offset_.emplace(pack_id, location);
  return location.offset;
}

void* MMapWeightCacheProvider::OffsetToAddr(const size_t offset) {
  // While the cache is being built, the buffer could grow and need to be
  // reallocated so we cannot ensure pointer stability.
  XNNPACK_ABORT_CHECK(
      !IsBuilding(),
      "Cannot get the address of a buffer in a cache during a building step.");
  return offset_to_addr_[offset];
}

void MMapWeightCacheProvider::Release() {
  buffer_address_to_identifier_.clear();
  cache_key_to_offset_.clear();
  mmap_handles_.clear();
  mmap_buffer_base_offset_ = 0;
  builder_ = WeightCacheBuilder();
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

PackIdentifier MMapWeightCacheProvider::BuildPackIdentifier(
    const xnn_weights_cache_look_up_key& key) {
  const auto get_buffer_id = [&](const void* buffer) -> size_t {
    if (buffer) {
      const auto identifier_it = buffer_address_to_identifier_.find(buffer);
      if (identifier_it != buffer_address_to_identifier_.end()) {
        return identifier_it->second;
      }
      // We could have several layers of remapping. We look through
      // buffer_remaps_ until we find a valid identifier or nothing is mapped to
      // the current buffer pointer.
      auto remapped_it = buffer_remaps_.find(buffer);
      while (remapped_it != buffer_remaps_.end()) {
        const auto remapped_identifier_it =
            buffer_address_to_identifier_.find(remapped_it->second);
        if (remapped_identifier_it != buffer_address_to_identifier_.end()) {
          return remapped_identifier_it->second;
        }
        remapped_it = buffer_remaps_.find(remapped_it->second);
      }
      XNNPACK_ABORT_CHECK(
          remapped_it != buffer_remaps_.end(),
          "Unknown constant buffer passed to BuildPackIdentifier.");
    }
    return PackIdentifier::kNoId;
  };
  return PackIdentifier{/*pack_algorithm_id=*/key.seed,
                        /*weights_id=*/get_buffer_id(key.kernel),
                        /*bias_id=*/get_buffer_id(key.bias)};
}

bool IsCompatibleCacheFile(const char* path) {
  FileDescriptor fd = FileDescriptor::Open(path, O_RDONLY);
  XNNPACK_RETURN_CHECK(fd.IsValid(), "Could not open file: %s: %s.", path,
                       strerror(errno));
  XNNPackCacheHeader header;
  XNNPACK_RETURN_CHECK(fd.Read(&header, sizeof(header)),
                       "Couldn't read file header.");
  XNNPACK_RETURN_CHECK(
      header.version == XNNPackCacheHeader::kVersion,
      "Cache header version is incompatible. Expected %d, got %d.",
      XNNPackCacheHeader::kVersion, header.version);
  XNNPACK_RETURN_CHECK(xnn_experimental_check_build_identifier(
                           header.xnnpack_build_identifier,
                           sizeof(header.xnnpack_build_identifier)),
                       "Cache header build identifier is different.");
  return true;
}

}  // namespace tflite::xnnpack

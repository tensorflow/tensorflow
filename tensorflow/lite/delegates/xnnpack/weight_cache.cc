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
#include <sys/stat.h>

#if defined(_MSC_VER)
#include <io.h>
#define F_OK 0
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <algorithm>
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
#include "tensorflow/lite/delegates/xnnpack/weight_cache_schema_generated.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"

#define XNNPACK_ABORT_CHECK(TEST, ...)                      \
  if (!(TEST)) {                                            \
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR, __VA_ARGS__); \
    std::abort();                                           \
  }

namespace tflite::xnnpack {

namespace {
constexpr size_t kMinAlignment = 64;

// Returns the next offset value that is aligned to `alignement`.
size_t Align(size_t offset, const size_t alignment) {
  const size_t misalign = offset % alignment;
  return offset + (misalign ? alignment - misalign : 0);
}

// Class the provided callback at then end of the scope this was created into.
template <class F>
class ScopeGuard {
 public:
  explicit ScopeGuard(F&& callback) : callback_(std::forward<F>(callback)) {}
  ScopeGuard(const ScopeGuard&) = delete;
  ScopeGuard& operator=(const ScopeGuard&) = delete;
  ScopeGuard(ScopeGuard&& other)
      : active_(other.active_), callback_(std::move(other.callback_)) {
    other.Deactivate();
  }
  ScopeGuard& operator=(ScopeGuard&& other) {
    if (this != &other) {
      active_ = std::move(other.active_);
      callback_ = std::move(other.callback_);
      other.Deactivate();
    }
  }

  ~ScopeGuard() {
    if (active_) {
      callback_();
    }
  }

  void Deactivate() { active_ = false; }

 private:
  F callback_;
  bool active_ = true;
};

template <class F>
ScopeGuard(F&&) -> ScopeGuard<F>;

// Returns true if the given path exists.
[[nodiscard]]
bool FileExists(const char* path) {
  return access(path, F_OK) != -1;
}

}  // namespace

void swap(MMapHandle& a, MMapHandle& b) {
  using std::swap;
  swap(a.size_, b.size_);
  swap(a.data_, b.data_);
}

MMapHandle::~MMapHandle() { UnMap(); }

MMapHandle::MMapHandle(MMapHandle&& other) { swap(*this, other); }

MMapHandle& MMapHandle::operator=(MMapHandle&& other) {
  swap(*this, other);
  return *this;
}

bool MMapHandle::Map(const char* path) {
  this->UnMap();

  const int fd = open(path, O_RDONLY);
  if (fd == -1) {
    TFLITE_LOG_PROD(
        tflite::TFLITE_LOG_ERROR,
        "XNNPack weight cache: could not open file to mmap ('%s'): %s.", path,
        strerror(errno))
    return false;
  }

  const ScopeGuard close_fd_on_return([&fd] {
    if (fd >= 0) {
      close(fd);
    }
  });

  struct stat file_stats;
  if (fstat(fd, &file_stats)) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                    "XNNPack weight cache: could not access file stats to get "
                    "size ('%s'): %s.",
                    path, strerror(errno))
    return false;
  }

  size_ = file_stats.st_size;
#if defined(_MSC_VER)
  data_ = new uint8_t[size_];
  {
    uint8_t* data_reader = data_;
    size_t remaining_bytes = size_;
    while (remaining_bytes > 0) {
      const auto bytes = read(fd, data_reader, remaining_bytes);
      if (bytes == -1) {
        TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                        "XNNPack weight cache: could not read file ('%s'): %s.",
                        path, strerror(errno))
        UnMap();
        return false;
      }
      remaining_bytes -= bytes;
      data_reader += bytes;
    }
  }
#else
  data_ = static_cast<uint8_t*>(
      mmap(/*addr=*/nullptr, size_, PROT_READ, MAP_SHARED, fd, /*offset=*/0));
  if (data_ == MAP_FAILED) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                    "XNNPack weight cache: could not mmap file (%s): %s.", path,
                    strerror(errno));
    data_ = nullptr;
    size_ = 0;
    return false;
  }
#endif

  return true;
}

void MMapHandle::UnMap() {
  if (data_) {
#if defined(_MSC_VER)
    delete[] data_;
#else
    munmap(data_, size_);
#endif
    data_ = nullptr;
    size_ = 0;
  }
}

void swap(WeightCacheBuilder& a, WeightCacheBuilder& b) {
  using std::swap;
  swap(a.schema_, b.schema_);
  swap(a.data_, b.data_);
  swap(a.capacity_, b.capacity_);
  swap(a.fd_, b.fd_);
  swap(a.file_path_, b.file_path_);
}

WeightCacheBuilder::WeightCacheBuilder(WeightCacheBuilder&& other) {
  swap(*this, other);
}

WeightCacheBuilder& WeightCacheBuilder::operator=(WeightCacheBuilder&& other) {
  Reset();
  swap(*this, other);
  return *this;
}

WeightCacheBuilder::~WeightCacheBuilder() { Reset(); }

namespace {

bool WriteData(const int fd, const uint8_t* data, size_t size,
               const char* const file_path, const char* step_description) {
  for (size_t bytes = 0; bytes < size;) {
    const auto written_bytes = write(fd, data + bytes, size - bytes);
    if (written_bytes == -1) {
      TFLITE_LOG_PROD(
          tflite::TFLITE_LOG_ERROR,
          "XNNPack weight cache: file write incomplete (%s). %s: %s.",
          file_path, step_description, strerror(errno))
    }
    bytes += written_bytes;
  }
  return true;
}

}  // namespace

bool WeightCacheBuilder::Start(const char* path) {
  Reset();
  ScopeGuard reset_on_error([this] { Reset(); });

  file_path_ = path;
  fd_ = open(file_path_.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
  if (fd_ == -1) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR, "Could not open file ('%s'): %s.",
                    file_path_.c_str(), strerror(errno));
    return false;
  }

  // Write data in the header, this will be overwritten in the `Finalize` call.
  const XNNPackCacheHeader header{
      // We explicitly set the header as invalid. If any error happens during
      // the build, reloading the cache file will fail.
      .version = XNNPackCacheHeader::kInvalidHeader,
      .buffer_list_offset = 0,
      .buffer_list_size = 0,
  };

  if (!WriteData(fd_, (const uint8_t*)&header, sizeof(header),
                 file_path_.c_str(), "padding for flatbuffer offset")) {
    return false;
  }

  schema_.base_offset = Align(sizeof(header), kMinAlignment);

  reset_on_error.Deactivate();
  return true;
}

void WeightCacheBuilder::Reset() {
  if (fd_ != -1) {
    close(fd_);
    fd_ = -1;
    file_path_.clear();
  }
  data_.reset(nullptr);
  capacity_ = 0;
}

void* WeightCacheBuilder::Reserve(size_t size) {
  if (size > capacity_) {
    // We don't care about the data when we are reserving space. We save memory
    // by deleting the existing buffer first.
    data_.reset(nullptr);
    data_ = std::make_unique<uint8_t[]>(size);
    capacity_ = size;
  }
  return data_.get();
}

BufferLocation WeightCacheBuilder::Append(PackIdentifier pack_id,
                                          const void* data, uint64_t size) {
  XNNPACK_ABORT_CHECK(IsStarted(),
                      "Cannot append data to an unstarted builder.");
  // Add some padding so that the cache file can be mmaped and the buffer
  // stays aligned correctly.
  const size_t offset = Align(lseek(fd_, 0, SEEK_CUR), kMinAlignment);
  if (lseek(fd_, offset, SEEK_SET) != offset) {
    return BufferLocation{};
  }

  BufferLocation loc{.offset = offset - schema_.base_offset, .size = size};
  schema_.buffers.push_back(std::make_unique<cache::schema::BufferT>(
      cache::schema::BufferT{.packing_algorithm_id = pack_id.pack_algorithm_id,
                             .weights_id = pack_id.weights_id,
                             .bias_id = pack_id.bias_id,
                             .offset = loc.offset,
                             .size = loc.size}));
  WriteData(fd_, reinterpret_cast<const uint8_t*>(data), size,
            file_path_.c_str(), "Append buffer to cache file");
  return loc;
}

bool WeightCacheBuilder::ShouldFinalize() const { return fd_ != -1; }

bool WeightCacheBuilder::Finalize() {
  if (fd_ == -1) {
    TFLITE_LOG_PROD(
        tflite::TFLITE_LOG_ERROR,
        "XNNPack weight cache: cache file ('%s') is not open for writing: %s.",
        file_path_.c_str(), strerror(errno))
    return false;
  }

  flatbuffers::FlatBufferBuilder builder;
  // Add a fake size and the base offset to mutate them afterwards. Otherwise
  // space for it won't be added to the flatbuffer.
  cache::schema::FinishBufferListBuffer(
      builder, cache::schema::BufferList::Pack(builder, &schema_));

  // Add some padding so that the cache file can be mmaped and the buffer
  // stays aligned correctly.
  const size_t layout_offset = Align(lseek(fd_, 0, SEEK_CUR), kMinAlignment);
  if (lseek(fd_, layout_offset, SEEK_SET) != layout_offset) {
    return false;
  }

  if (sizeof(XNNPackCacheHeader::xnnpack_build_identifier) !=
      xnn_experimental_get_build_identifier_size()) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                    "XNNPack weight cache: cache file ('%s') header cannot "
                    "hold XNNPack's build identifier: %s.",
                    file_path_.c_str(), strerror(errno))
    return false;
  }

  XNNPackCacheHeader header{XNNPackCacheHeader::kVersion};
  memcpy(header.xnnpack_build_identifier,
         xnn_experimental_get_build_identifier_data(),
         xnn_experimental_get_build_identifier_size());
  header.buffer_list_offset = lseek(fd_, 0, SEEK_CUR);
  header.buffer_list_size = builder.GetSize();

  // Write the flatbuffer which serves as a header to index the buffer data.
  if (!WriteData(fd_, builder.GetBufferPointer(), builder.GetSize(),
                 file_path_.c_str(), "Buffer list")) {
    return false;
  }

  // Write the header at the beginning of the file.
  lseek(fd_, 0, SEEK_SET);
  WriteData(fd_, (const uint8_t*)&header, sizeof(header), file_path_.c_str(),
            "Writing header");

  TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE,
                  "XNNPack weight cache: written to '%s'.", file_path_.c_str());
  Reset();
  return true;
}

MMapWeightCacheProvider::MMapWeightCacheProvider(
    MMapWeightCacheProvider&& other) {
  *this = std::move(other);
}

MMapWeightCacheProvider& MMapWeightCacheProvider::operator=(
    MMapWeightCacheProvider&& other) {
  using std::swap;
  swap(cache_provider_, other.cache_provider_);
  // The contexts need to keep pointing to their owning object.
  cache_provider_.context = this;
  other.cache_provider_.context = &other;
  swap(file_path_, other.file_path_);
  swap(buffer_address_to_identifier_, other.buffer_address_to_identifier_);
  swap(cache_key_to_offset_, other.cache_key_to_offset_);
  swap(mmap_handle_, other.mmap_handle_);
  swap(mmap_buffer_base_offset_, other.mmap_buffer_base_offset_);
  swap(builder_, other.builder_);
  return *this;
}

void MMapWeightCacheProvider::SetFilePath(const char* path) {
  XNNPACK_ABORT_CHECK(
      !IsFinalized(),
      "Cannot change the path of a cache that has already been loaded.");
  // We try to keep file_path_'s data as stable as possible. Don't overwrite
  // if the path hasn't changed.
  if (file_path_ != path) {
    file_path_ = path;
  }
}

bool MMapWeightCacheProvider::LoadOrStartBuild(const char* path) {
  if (Load(path)) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE,
                    "XNNPack weight cache loaded from '%s'.", path);
    return true;
  } else if (StartBuild(path)) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE,
                    "XNNPack weight cache build for '%s' started.", path);

    return true;
  }
  return false;
}

bool MMapWeightCacheProvider::StartBuild(const char* path) {
  SetFilePath(path);
  return builder_.Start(path);
}

bool MMapWeightCacheProvider::Load(const std::string& path) {
  SetFilePath(path.c_str());
  return Load();
}

bool MMapWeightCacheProvider::Load() {
  XNNPACK_ABORT_CHECK(!file_path_.empty(),
                      "Path wasn't provided to weight cache provider.");
  mmap_buffer_base_offset_ = 0;
  cache_key_to_offset_.clear();

  if (!FileExists(file_path_.c_str())) {
    TFLITE_LOG(tflite::TFLITE_LOG_WARNING,
               "XNNPack weight cache: could not load '%s': %s.",
               file_path_.c_str(), strerror(errno));
    return false;
  }

  if (!mmap_handle_.Map(file_path_.c_str())) {
    return false;
  }

  ScopeGuard unmap_on_fail([this] { mmap_handle_.UnMap(); });

  if (mmap_handle_.size() < sizeof(XNNPackCacheHeader)) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                    "XNNPack weight cache: invalid cache file size.");
    return false;
  }

  const XNNPackCacheHeader header = [this] {
    XNNPackCacheHeader header;
    memcpy(&header, mmap_handle_.data(), sizeof(header));
    return header;
  }();

  if (header.version != XNNPackCacheHeader::kVersion) {
    TFLITE_LOG(tflite::TFLITE_LOG_VERBOSE,
               "XNNPack weight cache: incompatible header version. Cache "
               "needs to be built again.");
    return false;
  }

  if (!xnn_experimental_check_build_identifier(
          header.xnnpack_build_identifier,
          sizeof(header.xnnpack_build_identifier))) {
    TFLITE_LOG(tflite::TFLITE_LOG_VERBOSE,
               "XNNPack weight cache: incompatible XNNPack version. Cache "
               "needs to be built again.");
    return false;
  }

  if (header.buffer_list_offset >= mmap_handle_.size()) {
    TFLITE_LOG_PROD(
        tflite::TFLITE_LOG_ERROR,
        "XNNPack weight cache: invalid offset for buffer list descriptor.");
    return false;
  }

  if (header.buffer_list_size !=
      mmap_handle_.size() - header.buffer_list_offset) {
    TFLITE_LOG_PROD(
        tflite::TFLITE_LOG_ERROR,
        "XNNPack weight cache: invalid size for buffer list descriptor.");
    return false;
  }

  // Verifiy the flabuffer part of the file.
  flatbuffers::Verifier verifier(
      mmap_handle_.data() + header.buffer_list_offset, header.buffer_list_size);
  if (!cache::schema::VerifyBufferListBuffer(verifier)) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                    "XNNPack weight cache: buffer list validation failed.");
    return false;
  }

  // Load flatbuffer.
  const cache::schema::BufferList* buffer_list = cache::schema::GetBufferList(
      mmap_handle_.data() + header.buffer_list_offset);
  if (!buffer_list) {
    TFLITE_LOG_PROD(
        tflite::TFLITE_LOG_ERROR,
        "XNNPack weight cache: could not get packed weights from flatbuffer.");
    return false;
  }
  mmap_buffer_base_offset_ = buffer_list->base_offset();
  if (const auto buffers = buffer_list->buffers(); buffers) {
    for (auto* buffer : *buffers) {
      if (!buffer) {
        TFLITE_LOG_PROD(
            tflite::TFLITE_LOG_ERROR,
            "XNNPack weight cache: Invalid buffer address in buffer list.");
        return false;
      }
      cache_key_to_offset_.emplace(
          PackIdentifier{.pack_algorithm_id = buffer->packing_algorithm_id(),
                         .weights_id = buffer->weights_id(),
                         .bias_id = buffer->bias_id()},
          BufferLocation{.offset = buffer->offset(), .size = buffer->size()});
    }
  }

  unmap_on_fail.Deactivate();
  return true;
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
  XNNPACK_ABORT_CHECK(!IsFinalized(),
                      "Cannot reserve space in a finalized cache.");
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

  XNNPACK_ABORT_CHECK(!IsFinalized(),
                      "Cannot insert a buffer in a finalized cache.");

  const BufferLocation location = builder_.Append(pack_id, ptr, size);
  cache_key_to_offset_.emplace(pack_id, location);
  return location.offset;
}

void* MMapWeightCacheProvider::OffsetToAddr(const size_t offset) {
  // While the cache is being built, the buffer could grow and need to be
  // reallocated so we cannot ensure pointer stability.
  XNNPACK_ABORT_CHECK(
      IsFinalized(),
      "Cannot get the address of a buffer in a non finalized cache.");
  return mmap_handle_.data() + mmap_buffer_base_offset_ + offset;
}

void MMapWeightCacheProvider::Release() {
  buffer_address_to_identifier_.clear();
  cache_key_to_offset_.clear();
  mmap_handle_ = MMapHandle();
  mmap_buffer_base_offset_ = 0;
  builder_ = WeightCacheBuilder();
}

bool MMapWeightCacheProvider::Finalize() {
  if (IsFinalized()) {
    return true;
  }
  if (file_path_.empty()) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                    "XNNPack weight cache: file path wasn't set. Cannot "
                    "finalize the cache.");
    return false;
  }
  if (!builder_.Finalize()) {
    return false;
  }
  builder_ = WeightCacheBuilder();

  return Load();
}

bool MMapWeightCacheProvider::IsFinalized() const {
  return mmap_handle_.IsMapped();
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
  return reinterpret_cast<MMapWeightCacheProvider*>(context)->IsFinalized();
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
      XNNPACK_ABORT_CHECK(identifier_it != buffer_address_to_identifier_.end(),
                          "Unknown constant buffer passed to HashCacheKey.");
      return identifier_it->second;
    }
    return PackIdentifier::kNoId;
  };
  return PackIdentifier{.pack_algorithm_id = key.seed,
                        .weights_id = get_buffer_id(key.kernel),
                        .bias_id = get_buffer_id(key.bias)};
}

}  // namespace tflite::xnnpack

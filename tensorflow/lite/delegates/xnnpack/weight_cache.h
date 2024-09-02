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
#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_WEIGHT_CACHE_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_WEIGHT_CACHE_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "xnnpack.h"  // from @XNNPACK
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/xnnpack/weight_cache_schema_generated.h"

#if defined(__linux__) || defined(__ANDROID__)
#ifndef TFLITE_XNNPACK_ENABLE_IN_MEMORY_WEIGHT_CACHE
#include <sys/syscall.h>
#ifdef SYS_memfd_create
#define TFLITE_XNNPACK_ENABLE_IN_MEMORY_WEIGHT_CACHE 1
#endif
#endif
#endif

// WARNING: the interface in this file is still under experimentation and WILL
// CHANGE. Do not rely on it.

// TFLite doesn't use absl hashing utilities.

namespace tflite {
namespace xnnpack {

// Reserved value to request the delegate to use an in-memory cache instead of
// saving it to disk.
//
// This is useful when disk space is not available or when having to manage the
// cache file freshness is too complicated and still provides the deduplication
// mechanism for constant buffers that are reused accross graph signatures.
inline constexpr char kInMemoryCachePath[] = ":memory";

// This structure is written at the start of every cache file.
//
// When changing this structure or anything in the cache file layout,
// `kVersion` should be incremented by one.
//
// When creating a new cache file, `version` should be set to `kVersion`.
//
// When reading a cache file, the cache should be rejected if `version`
// doesn't match `kVersion`.
struct XNNPackCacheHeader {
  enum : uint64_t { kInvalidHeader = 0, kVersion = 1 };
  uint64_t version;
  uint8_t xnnpack_build_identifier[32];
  uint64_t buffer_list_offset;
  uint64_t buffer_list_size;
};

struct PackIdentifier {
  enum { kNoId = SIZE_MAX };
  uint64_t pack_algorithm_id = kNoId;
  uint64_t weights_id = kNoId;
  uint64_t bias_id = kNoId;

  friend bool operator==(const PackIdentifier& a, const PackIdentifier& b) {
    return a.pack_algorithm_id == b.pack_algorithm_id &&
           a.weights_id == b.weights_id && a.bias_id == b.bias_id;
  }

  struct Hash {
    size_t operator()(const PackIdentifier& p) const {
      std::hash<uint64_t> hasher;
      return hasher(p.pack_algorithm_id) ^ hasher(p.weights_id) ^
             hasher(p.bias_id);
    }
  };
};

struct BufferLocation {
  uint64_t offset;
  uint64_t size;

  static constexpr BufferLocation Invalid() { return {SIZE_MAX, SIZE_MAX}; }

  constexpr bool IsInvalid() const {
    constexpr BufferLocation invalid = Invalid();
    return offset == invalid.offset && size == invalid.size;
  }
};

class FileDescriptor {
 public:
  explicit FileDescriptor(int fd) : fd_(fd) {}

  FileDescriptor() = default;

  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;

  FileDescriptor(FileDescriptor&& other) : fd_(other.fd_) { other.fd_ = -1; }

  FileDescriptor& operator=(FileDescriptor&& other) {
    Close();
    fd_ = other.fd_;
    other.fd_ = -1;
    return *this;
  }

  ~FileDescriptor() { Close(); }

  // Checks that the file descriptor has a valid value.
  //
  // WARNING: this does not check that the descriptor points to an open file.
  bool IsValid() const { return fd_ >= 0; }

  // Returns the file descriptor value.
  int Value() const { return fd_; }

  // Closes the current file descriptor if needed and assigns the given value.
  void Reset(int new_fd);

  // Returns the cursor position in the current file.
  //
  // WARNING: the file descriptor must be valid and the file must be opened.
  off_t GetPos() const;

  // Sets the absolute cursor position in the current file.
  //
  // WARNING: the file descriptor must be valid and the file must be opened.
  off_t SetPos(off_t position);

  // Moves the cursor position by the given offset in the current file.
  //
  // WARNING: the file descriptor must be valid and the file must be opened.
  off_t MovePos(off_t offset);

  // Duplicates the current file descriptor and returns the new file descriptor.
  FileDescriptor Duplicate() const;

  // Closes the current file descriptor and set it to -1.
  void Close();

  // Returns the current file descriptor value and stops managing it.
  int Release() {
    const int fd = fd_;
    fd_ = -1;
    return fd;
  }

  friend void swap(FileDescriptor& f1, FileDescriptor& f2) {
    using std::swap;
    swap(f1.fd_, f2.fd_);
  }

 private:
  int fd_ = -1;
};

// Handles MMap allocations lifetime.
//
// When mapped, provides a view over the allocation for convenience.
//
// WARNING: the interface in this file is still under experimentation and WILL
// CHANGE. Do not rely on it.
class MMapHandle {
 public:
  using value_type = uint8_t;

  MMapHandle() = default;
  ~MMapHandle();
  MMapHandle(const MMapHandle&) = delete;
  MMapHandle& operator=(const MMapHandle&) = delete;
  MMapHandle(MMapHandle&&);
  MMapHandle& operator=(MMapHandle&&);

  // Maps the file at the given path.
  [[nodiscard /*Mapping a file can fail.*/]]
  bool Map(const char* path);

  [[nodiscard /*Mapping a file can fail.*/]]
  bool Map(int fd, const char* debug_path = "unspecified");

  // Unmaps an existing mapping.
  void UnMap();

  // Returns true if a mapping exists.
  bool IsMapped() const { return data_ != nullptr; }

  // Returns the mapping buffer.
  uint8_t* data() { return data_; }

  // Returns the mapping buffer.
  const uint8_t* data() const { return data_; }

  // Returns the mapping size in bytes.
  size_t size() const { return size_; }

  uint8_t* begin() { return data(); }

  const uint8_t* begin() const { return data(); }

  uint8_t* end() { return data() + size(); }

  const uint8_t* end() const { return data() + size(); }

  friend void swap(MMapHandle& a, MMapHandle& b);

 private:
  size_t size_ = 0;
  uint8_t* data_ = nullptr;
};

// Provides storage to write the packed buffers to and saves those to disk.
//
// WARNING: the interface in this file is still under experimentation and WILL
// CHANGE. Do not rely on it.
class WeightCacheBuilder {
 public:
  WeightCacheBuilder() = default;
  ~WeightCacheBuilder();

  // Non-copyable.
  WeightCacheBuilder(const WeightCacheBuilder&) = delete;
  WeightCacheBuilder& operator=(const WeightCacheBuilder&) = delete;

  // Moveable.
  WeightCacheBuilder(WeightCacheBuilder&&);
  WeightCacheBuilder& operator=(WeightCacheBuilder&&);

  [[nodiscard /*Starting the builder may fail.*/]]
  bool Start(const char* path);

  [[nodiscard]]
  bool IsStarted() const {
    return fd_.IsValid();
  }

  // Resets the builder, discarding any data that hasn't been written.
  void Reset();

  // Reserves space in the data buffer for the required size in bytes and
  // returns the address of that space.
  //
  // Sets `last_reserve` to the offset from `buffer_data_`'s start and `n`.
  //
  // A call to `Reserve` should alway be followed by a call to `Append`.
  [[nodiscard /*The pointer to reserved space should be used.*/]]
  void* Reserve(size_t size);

  // Adds a buffer to the cache.
  //
  // The buffer space must have been reserved before using `Reserve`. If not, a
  // new call to `Reserve` will be done and the data will be copied over.
  [[nodiscard /*The location to the appended data should be saved.*/]]
  BufferLocation Append(PackIdentifier pack_id, const void* data,
                        uint64_t size);

  // Checks whether this builder has data that needs to be written to disk.
  bool ShouldFinalize() const;

  // Writes the flatbuffer to disk.
  [[nodiscard /*Writing the weight cache can fail.*/]]
  bool Finalize();

  // Returns the file descriptor.
  const FileDescriptor& GetFileDescriptor() const { return fd_; }

  // Returns the capacity of the underlying reserved buffer.
  //
  // WARNING: this exposes class implementation details for testing purposes and
  // may be removed at any time.
  size_t capacity() const { return capacity_; }

  // Returns the address of the underlying reserved buffer.
  //
  // YOU SHOULD BE GETTING THAT ADDRESS FROM THE `Reserve` FUNCTION.
  //
  // WARNING: this exposes class implementation details for testing purposes and
  // may be removed at any time.
  uint8_t* data() const { return data_.get(); }

  friend void swap(WeightCacheBuilder& a, WeightCacheBuilder& b);

 private:
  std::unique_ptr<uint8_t[]> data_ = nullptr;
  cache::schema::BufferListT schema_;
  size_t capacity_ = 0;
  // Temporary file descriptor to write the weights to disk immediately.
  FileDescriptor fd_;
  std::string file_path_;
};

// Allows XNNPack to directly load packed weights from disk instead of having to
// repack them every time.
//
// XNNPack kernels do not have knowledge of the TFLite context. The only thing
// they can access is the buffers address. We rely on the fact that the address
// provided by TFLite is unique in order to find out the buffer identifier.
//
// To use the cache you need to:
//
//  - Map the buffer addresses to their identifier with `MapTensorIdentifiers`
//  - Load the cache file.
//  - Finalize the cache before calling the run functions of XNNPack (setup and
//    reshape are ok).
class MMapWeightCacheProvider {
 public:
  MMapWeightCacheProvider() = default;
  MMapWeightCacheProvider(const MMapWeightCacheProvider&) = delete;
  MMapWeightCacheProvider& operator=(const MMapWeightCacheProvider&) = delete;
  MMapWeightCacheProvider(MMapWeightCacheProvider&&);
  MMapWeightCacheProvider& operator=(MMapWeightCacheProvider&&);

  // Changes the file path to save the cache to.
  //
  // WARNING: Can only be called if the cache isn't finalized.
  void SetFilePath(const char* file_path);

  const std::string& GetFilePath() const { return file_path_; }

  // Tries to load the given file. If the file doesn't exist starts building the
  // cache for it.
  [[nodiscard /*Loading a cache file may fail.*/]]
  bool LoadOrStartBuild(const char* file_path);

  [[nodiscard /*Starting to build a cache file may fail.*/]]
  bool StartBuild(const char* file_path);

  // Set the weight file path and loads it.
  [[nodiscard /*Loading a cache file may fail.*/]]
  bool Load(const std::string& path);

  // Loads the weight cache previouslt set with `SetFilePath`.
  [[nodiscard /*Loading cache data may fail.*/]]
  bool Load();

  // Creates the tensor map.
  void MapTensorIdentifiers(
      const TfLiteTensor* tensors, size_t size,
      const std::unordered_map<size_t, size_t>& tensor_index_to_identifier);

  // In case a constant buffer data needs to be moved for some reason, this will
  // map the new buffer data to its identifier.
  void RemapDataBuffer(const void* buffer, const void* new_buffer);

  // Returns the offset of the buffer identified by `cache_key`.
  //
  // If the buffer isn't found, return SIZE_MAX.
  [[nodiscard]]
  size_t LookUp(const xnn_weights_cache_look_up_key* cache_key);

  // Reserves space for a buffer of given size and returns a pointer to it.
  //
  // The buffer data should be filled and `LookUpOrInsert` should be immediately
  // called.
  [[nodiscard]]
  void* ReserveSpace(size_t size);

  // Returns the offset of the buffer identified by `cache_key`. If the lookup
  // fails, inserts the span `[ptr, ptr+size)`.
  //
  // This should be called after ReserveSpace and `ptr` should be the result of
  // that call with the given `size`.
  //
  // WARNING: The cache key cannot be null.
  [[nodiscard]]
  size_t LookUpOrInsert(const xnn_weights_cache_look_up_key* cache_key,
                        void* ptr, size_t size);

  // Gets the pointer to the buffer at the given offset.
  //
  // WARNING: This requires the buffer to be finalized.
  // WARNING: This does not check the validity of the passed offset.
  void* OffsetToAddr(size_t offset);

  // Releases the weight cache's memory.
  void Release();

  // Ensures that the cache is ready.
  //
  // If the cache file already exists, this is a no-op. Otherwise, this writes
  // the file to disk and reloads it.
  [[nodiscard /*Writing the cache file may fail.*/]]
  bool Finalize();

  // Checks whether the cache is ready to be used.
  bool IsFinalized() const;

  // Returns true if any weights have been added to the underlying builder.
  bool IsBuilding() const { return !IsFinalized() && !file_path_.empty(); };

  // Returns true if a file is mapped or a file path is set.
  bool IsActive() const { return IsFinalized() || !file_path_.empty(); };

  // Returns the cache provider expected by XNNPack.
  xnn_weights_cache_provider& GetCacheProvider() { return cache_provider_; }

  // C interface: `xnn_weights_cache_provider` callback.
  static size_t look_up(void* context,
                        const xnn_weights_cache_look_up_key* cache_key);

  // C interface: `xnn_weights_cache_provider` callback.
  static void* reserve_space(void* context, size_t n);

  // C interface: `xnn_weights_cache_provider` callback.
  static size_t look_up_or_insert(
      void* context, const xnn_weights_cache_look_up_key* cache_key, void* ptr,
      size_t size);

  // C interface: `xnn_weights_cache_provider` callback.
  static bool is_finalized(void* context);

  // C interface: `xnn_weights_cache_provider` callback.
  static void* offset_to_addr(void* context, size_t offset);

  // C interface: `xnn_weights_cache_provider` callback.
  static enum xnn_status delete_cache(void* context);

 private:
  // Hashes a cache key to lookup in `cache_key_to_identifier_`.
  PackIdentifier BuildPackIdentifier(const xnn_weights_cache_look_up_key& key);

  // Cache provider implementation for XNNPack.
  xnn_weights_cache_provider cache_provider_{
      /*context=*/this,
      /*look_up=*/MMapWeightCacheProvider::look_up,
      /*reserve_space=*/MMapWeightCacheProvider::reserve_space,
      /*look_up_or_insert=*/MMapWeightCacheProvider::look_up_or_insert,
      /*is_finalized=*/MMapWeightCacheProvider::is_finalized,
      /*offset_to_addr=*/MMapWeightCacheProvider::offset_to_addr,
      /*delete_cache=*/MMapWeightCacheProvider::delete_cache};

  // Path to the cache file.
  std::string file_path_;

  // Maps buffer addresses to buffer identifiers.
  std::unordered_map<const void*, uint64_t> buffer_address_to_identifier_;

  std::unordered_map<const void*, const void*> buffer_remaps_;

  // Maps cache request hashes to the buffer identifier.
  std::unordered_multimap<PackIdentifier, BufferLocation, PackIdentifier::Hash>
      cache_key_to_offset_;

  // MMap allocation handler.
  MMapHandle mmap_handle_;

  // The offset to the first buffer data in the MMap allocation.
  size_t mmap_buffer_base_offset_;

  // Can hold a file descriptor when building a temporary cache to prevent it
  // from being deleted.
  FileDescriptor temporary_file_descriptor_;

  // Used to build the cache.
  WeightCacheBuilder builder_;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_WEIGHT_CACHE_H_

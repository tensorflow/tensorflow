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
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/xnnpack/file_util.h"
#include "tensorflow/lite/delegates/xnnpack/weight_cache_schema_generated.h"

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
  bool Map(const char* path, size_t offset = 0);

  // Maps the fd associated to the file descriptor.
  //
  // The debug_path is printed along the error messages.
  [[nodiscard /*Mapping a file can fail.*/]]
  bool Map(const FileDescriptor& fd, size_t offset = 0,
           const char* debug_path = "unspecified");

  // Tries to resize the current mapping.
  //
  // Only succeeds if the mapping could be resized without being moved.
  //
  // WARNING: expects `IsMapped()` to be true.
  [[nodiscard /*Resizing a file can fail.*/]]
  bool Resize(size_t new_size);

  // Unmaps an existing mapping.
  void UnMap();

  // Returns true if a mapping exists.
  bool IsMapped() const { return data_ != nullptr; }

  // Returns the mapping buffer.
  uint8_t* data() { return data_ + offset_page_adjustment_; }

  // Returns the mapping buffer.
  const uint8_t* data() const { return data_ + offset_page_adjustment_; }

  // Returns the mapping size in bytes.
  size_t size() const { return size_; }

  size_t offset() const { return offset_; }

  uint8_t* begin() { return data(); }

  const uint8_t* begin() const { return data(); }

  uint8_t* end() { return data() + size(); }

  const uint8_t* end() const { return data() + size(); }

  friend void swap(MMapHandle& a, MMapHandle& b);

 private:
  size_t size_ = 0;
  size_t offset_ = 0;
  size_t offset_page_adjustment_ = 0;
  uint8_t* data_ = nullptr;
};

// Provides storage to write the packed buffers to and saves those to disk.
//
// WARNING: the interface in this file is still under experimentation and WILL
// CHANGE. Do not rely on it.
class WeightCacheBuilder {
 public:
  WeightCacheBuilder() = default;
  ~WeightCacheBuilder() = default;

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

  // Reopens the given file to add data to it.
  //
  // This should be only called from the weight cache provider.
  [[nodiscard /*Starting a build step may fail.*/]]
  bool StartBuildStep();

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

  // Writes the flatbuffer to disk.
  [[nodiscard /*Writing the weight cache can fail.*/]]
  bool StopBuildStep();

  // Get the offset in the cache file of the data written during the last step.
  //
  // This includes the buffers that were appended and the whole buffer mapping.
  [[nodiscard]]
  size_t LastBuildStepStart() const {
    return build_segment_start_;
  }

  // Get the size of the data written during the last step.
  //
  // This includes the buffers that were appended and the whole buffer mapping.
  [[nodiscard]]
  size_t LastBuildStepSize() const {
    return build_segment_size_;
  }

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

 private:
  std::unique_ptr<uint8_t[]> data_ = nullptr;
  cache::schema::BufferListT schema_;
  size_t capacity_ = 0;
  // Size of the data written between StartBuildStep and StopBuildStep.
  size_t build_segment_size_ = 0;
  // Offset in the cache file when StartBuildStep was called.
  size_t build_segment_start_ = 0;
  // The call to StopBuildStep may short circuit when nothing was written to the
  // cache. To ensure a smooth reloading, we need to ensure that the file header
  // is correct. This flag lets us know if that has happened.
  bool first_write_done_ = false;
  // Temporary file descriptor to write the weights to disk immediately.
  FileDescriptor fd_;
  std::string file_path_;

  bool is_build_step_ = false;
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

  // Loads the weight cache previously set with `SetFilePath`.
  [[nodiscard /*Loading cache data may fail.*/]]
  bool Load();

  // Checks if the cache is currently being built or if it was loaded from a
  // file.
  [[nodiscard]]
  bool CanStartBuildStep() const {
    return building_run_;
  };

  // Prepares to add new data to the cache.
  [[nodiscard /*Updating cache data may fail.*/]]
  bool StartBuildStep();

  // Prepares to use data that was added to the cache during a build step.
  [[nodiscard /*Updating cache data may fail.*/]]
  bool StopBuildStep();

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

  // Returns true if any weights have been added to the underlying builder.
  [[nodiscard]]
  bool IsBuilding() const {
    return is_build_step_;
  };

  // Returns true if a file is mapped or a file path is set.
  [[nodiscard]]
  bool IsActive() const {
    return !mmap_handles_.empty() || builder_.IsStarted();
  };

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

  // Loads the data written by the last call to `builder_.BuildStepStop()`.
  [[nodiscard /*Loading cache data may fail.*/]]
  bool LoadLastBuildStep();

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
  std::vector<MMapHandle> mmap_handles_;

  // The offset to the first buffer data in the MMap allocation.
  size_t mmap_buffer_base_offset_;

  // Can hold a file descriptor when building a temporary cache to prevent it
  // from being deleted.
  FileDescriptor temporary_file_descriptor_;

  // Used to build the cache.
  WeightCacheBuilder builder_;

  // True if the current run is the one building the cache file.
  //
  // We cannot distinguish between a wrong/outdated cache and one that is not
  // fully done. To detect misuse, we still want to raise an error when XNNPack
  // tries to append data to an existing file (i.e. when this is `false`).
  bool building_run_ = false;

  // True between StartBuildStep and StopBuildStep.
  //
  // This is used to check whether the builder is active, which means that some
  // of the buffers are not available/can't be retrieved.
  bool is_build_step_ = false;

  // Stores the loaded buffer addresses corresponding to the given offset in the
  // cache file.
  std::map<size_t, void*> offset_to_addr_;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_WEIGHT_CACHE_H_

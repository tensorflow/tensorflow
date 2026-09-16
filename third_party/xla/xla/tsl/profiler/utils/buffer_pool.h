/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_TSL_PROFILER_UTILS_BUFFER_POOL_H_
#define XLA_TSL_PROFILER_UTILS_BUFFER_POOL_H_

#include <vector>

#include "absl/synchronization/mutex.h"
#include "tsl/platform/thread_annotations.h"

namespace tsl {
namespace profiler {

// A lightweight buffer management class for tracking fixed sized buffers that
// can be reused. ReusableBuffers only manages buffers that have been
// reclaimed (i.e. relinquished by client).
// This class is thread-safe.
class BufferPool {
 public:
  // Allocated buffers will be of a fixed size specified during initialization.
  explicit BufferPool(size_t buffer_size_in_bytes);

  ~BufferPool();

  // Returns a previously reclaimed buffer for use. If there are no buffers
  // being managed, this allocates and returns 8B aligned buffers of size
  // `buffer_size_in_bytes_`. The content of returned buffers is undefined.
  // If `is_allocated` is not null, it is set to true if a new buffer was
  // allocated, or false if a reclaimed buffer was reused.
  uint8_t* GetOrCreateBuffer(bool* is_allocated = nullptr);

  // Reclaims exclusive ownership of a buffer. Clients must pass in a buffer
  // that was obtained from `GetOrCreateBuffer()`.
  void ReclaimBuffer(uint8_t* buffer);

  // Frees all relinquished buffers from memory.
  void DestroyAllBuffers();

  // Gets size of a single buffer in bytes.
  size_t GetBufferSizeInBytes() const;

  // Returns the number of reclaimed buffers currently available in the pool.
  size_t GetFreeBuffersCount() const;

  // Reclaims the buffer only if the current number of free buffers is less than
  // `max_buffers`. Returns true if reclaimed, or false if not reclaimed (the
  // caller should drop/free the buffer).
  bool ReclaimBufferIfLessThan(uint8_t* buffer, size_t max_buffers);

 protected:
  mutable absl::Mutex buffers_mutex_;
  std::vector<uint8_t*> buffers_ TF_GUARDED_BY(buffers_mutex_);
  size_t buffer_size_in_bytes_;
};

// A wrapper for BufferPool that manages pre-allocation, capacity, and zeroing.
// Pre-allocates a configurable number of buffers (default 8) and zeroes them
// out. When a buffer is returned/reclaimed:
// - If the internal queue has equal or greater than pre-allocation count, the
//   buffer is dropped (freed from memory).
// - Otherwise, the buffer is zeroed out and returned to the internal free list.
// Any dynamically allocated buffers (when pool is exhausted) are also zeroed.
class BufferPoolWrapper {
 public:
  static constexpr size_t kDefaultPreallocationCount = 8;

  explicit BufferPoolWrapper(
      size_t buffer_size_in_bytes,
      size_t preallocation_count = kDefaultPreallocationCount);

  ~BufferPoolWrapper() = default;

  // Returns a zeroed buffer for use.
  uint8_t* GetOrCreateBuffer();

  // Reclaims ownership of a buffer.
  // If the internal queue has >= preallocation_count, drops (frees) the buffer.
  // Otherwise, zeroes the buffer and returns it to the internal free list.
  void ReclaimBuffer(uint8_t* buffer);

  // Frees all relinquished buffers from memory.
  void DestroyAllBuffers();

  // Gets size of a single buffer in bytes.
  size_t GetBufferSizeInBytes() const;

  // Gets the number of pre-allocated buffers configured.
  size_t GetPreallocationCount() const;

  // Access to underlying BufferPool.
  BufferPool& GetBufferPool() { return buffer_pool_; }
  const BufferPool& GetBufferPool() const { return buffer_pool_; }

 private:
  BufferPool buffer_pool_;
  size_t preallocation_count_;
};

}  // namespace profiler
}  // namespace tsl

#endif  // XLA_TSL_PROFILER_UTILS_BUFFER_POOL_H_

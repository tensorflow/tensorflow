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

#include "xla/tsl/profiler/utils/buffer_pool.h"

#include <cstdint>
#include <cstring>
#include <ios>

#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/util/safe_reinterpret_cast.h"
#include "tsl/platform/mem.h"

namespace tsl {
namespace profiler {

BufferPool::BufferPool(size_t buffer_size_in_bytes)
    : buffer_size_in_bytes_(buffer_size_in_bytes) {}

BufferPool::~BufferPool() { DestroyAllBuffers(); }

uint8_t* BufferPool::GetOrCreateBuffer(bool* is_allocated) {
  if (is_allocated != nullptr) {
    *is_allocated = false;
  }
  // Get a relinquished buffer if it exists.
  {
    absl::MutexLock lock(buffers_mutex_);
    if (!buffers_.empty()) {
      uint8_t* buffer = buffers_.back();
      buffers_.pop_back();
      if (!buffer) {
        LOG(ERROR) << "A reused buffer must not be null!";
        return nullptr;
      }
      VLOG(3) << "Reused Buffer, buffer=" << std::hex
              << safe_reinterpret_cast<std::uintptr_t>(buffer) << std::dec;
      return buffer;
    }
  }

  // Allocate and return a new buffer.
  constexpr size_t kBufferAlignSize = 8;
  uint8_t* buffer = reinterpret_cast<uint8_t*>(port::AlignedMalloc(
      buffer_size_in_bytes_, static_cast<std::align_val_t>(kBufferAlignSize)));
  if (buffer == nullptr) {
    LOG(WARNING) << "Buffer not allocated.";
    return nullptr;
  }
  if (is_allocated != nullptr) {
    *is_allocated = true;
  }
  VLOG(3) << "Allocated Buffer, buffer=" << std::hex
          << safe_reinterpret_cast<std::uintptr_t>(buffer) << std::dec
          << " size=" << buffer_size_in_bytes_;
  return buffer;
}

void BufferPool::ReclaimBuffer(uint8_t* buffer) {
  absl::MutexLock lock(buffers_mutex_);

  buffers_.push_back(buffer);
  VLOG(3) << "Reclaimed Buffer, buffer=" << std::hex
          << safe_reinterpret_cast<std::uintptr_t>(buffer) << std::dec;
}

void BufferPool::DestroyAllBuffers() {
  absl::MutexLock lock(buffers_mutex_);
  for (uint8_t* buffer : buffers_) {
    VLOG(3) << "Freeing Buffer, buffer:" << std::hex
            << safe_reinterpret_cast<std::uintptr_t>(buffer) << std::dec;
    port::AlignedFree(buffer);
  }
  buffers_.clear();
}

size_t BufferPool::GetBufferSizeInBytes() const {
  return buffer_size_in_bytes_;
}

size_t BufferPool::GetFreeBuffersCount() const {
  absl::MutexLock lock(buffers_mutex_);
  return buffers_.size();
}

bool BufferPool::ReclaimBufferIfLessThan(uint8_t* buffer, size_t max_buffers) {
  absl::MutexLock lock(buffers_mutex_);
  if (buffers_.size() >= max_buffers) {
    return false;
  }
  buffers_.push_back(buffer);
  VLOG(3) << "Reclaimed Buffer, buffer=" << std::hex
          << safe_reinterpret_cast<std::uintptr_t>(buffer) << std::dec;
  return true;
}

BufferPoolWrapper::BufferPoolWrapper(size_t buffer_size_in_bytes,
                                     size_t preallocation_count)
    : buffer_pool_(buffer_size_in_bytes),
      preallocation_count_(preallocation_count) {
  std::vector<uint8_t*> preallocated;
  preallocated.reserve(preallocation_count_);
  for (size_t i = 0; i < preallocation_count_; ++i) {
    uint8_t* buffer = buffer_pool_.GetOrCreateBuffer();
    if (buffer != nullptr) {
      std::memset(buffer, 0, buffer_size_in_bytes);
      preallocated.push_back(buffer);
    }
  }
  for (uint8_t* buffer : preallocated) {
    buffer_pool_.ReclaimBuffer(buffer);
  }
}

uint8_t* BufferPoolWrapper::GetOrCreateBuffer() {
  bool is_allocated = false;
  uint8_t* buffer = buffer_pool_.GetOrCreateBuffer(&is_allocated);
  if (is_allocated && buffer != nullptr) {
    std::memset(buffer, 0, buffer_pool_.GetBufferSizeInBytes());
  }
  return buffer;
}

void BufferPoolWrapper::ReclaimBuffer(uint8_t* buffer) {
  if (buffer == nullptr) return;
  // If the internal queue already has >= preallocation_count buffers, drop
  // (free) the buffer.
  if (buffer_pool_.GetFreeBuffersCount() >= preallocation_count_) {
    port::AlignedFree(buffer);
    return;
  }
  // Otherwise, zerofy and put back into the internal free list.
  std::memset(buffer, 0, buffer_pool_.GetBufferSizeInBytes());
  if (!buffer_pool_.ReclaimBufferIfLessThan(buffer, preallocation_count_)) {
    port::AlignedFree(buffer);
  }
}

void BufferPoolWrapper::DestroyAllBuffers() {
  buffer_pool_.DestroyAllBuffers();
}

size_t BufferPoolWrapper::GetBufferSizeInBytes() const {
  return buffer_pool_.GetBufferSizeInBytes();
}

size_t BufferPoolWrapper::GetPreallocationCount() const {
  return preallocation_count_;
}

}  // namespace profiler
}  // namespace tsl

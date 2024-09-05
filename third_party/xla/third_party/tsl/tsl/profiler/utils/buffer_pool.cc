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

#include "tsl/profiler/utils/buffer_pool.h"

#include <ios>

#include "tsl/platform/logging.h"
#include "tsl/platform/mem.h"
#include "tsl/platform/mutex.h"

namespace tsl {
namespace profiler {

BufferPool::BufferPool(size_t buffer_size_in_bytes)
    : buffer_size_in_bytes_(buffer_size_in_bytes) {}

BufferPool::~BufferPool() { DestroyAllBuffers(); }

uint8_t* BufferPool::GetOrCreateBuffer() {
  // Get a relinquished buffer if it exists.
  {
    mutex_lock lock(buffers_mutex_);
    if (!buffers_.empty()) {
      uint8_t* buffer = buffers_.back();
      buffers_.pop_back();
      if (!buffer) {
        LOG(ERROR) << "A reused buffer must not be null!";
        return nullptr;
      }
      VLOG(3) << "Reused Buffer, buffer=" << std::hex
              << reinterpret_cast<uintptr_t>(buffer) << std::dec;
      return buffer;
    }
  }

  // Allocate and return a new buffer.
  constexpr size_t kBufferAlignSize = 8;
  uint8_t* buffer = reinterpret_cast<uint8_t*>(
      port::AlignedMalloc(buffer_size_in_bytes_, kBufferAlignSize));
  if (buffer == nullptr) {
    LOG(WARNING) << "Buffer not allocated.";
    return nullptr;
  }
  VLOG(3) << "Allocated Buffer, buffer=" << std::hex
          << reinterpret_cast<uintptr_t>(buffer) << std::dec
          << " size=" << buffer_size_in_bytes_;
  return buffer;
}

void BufferPool::ReclaimBuffer(uint8_t* buffer) {
  mutex_lock lock(buffers_mutex_);

  buffers_.push_back(buffer);
  VLOG(3) << "Reclaimed Buffer, buffer=" << std::hex
          << reinterpret_cast<uintptr_t>(buffer) << std::dec;
}

void BufferPool::DestroyAllBuffers() {
  mutex_lock lock(buffers_mutex_);
  for (uint8_t* buffer : buffers_) {
    VLOG(3) << "Freeing Buffer, buffer:" << std::hex
            << reinterpret_cast<uintptr_t>(buffer) << std::dec;
    port::AlignedFree(buffer);
  }
  buffers_.clear();
}

size_t BufferPool::GetBufferSizeInBytes() const {
  return buffer_size_in_bytes_;
}

}  // namespace profiler
}  // namespace tsl

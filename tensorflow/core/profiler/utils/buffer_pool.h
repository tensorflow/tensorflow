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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_BUFFER_POOL_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_BUFFER_POOL_H_

#include <vector>

#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
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
  uint8_t* GetOrCreateBuffer();

  // Reclaims exclusive ownership of a buffer. Clients must pass in a buffer
  // that was obtained from `GetOrCreateBuffer()`.
  void ReclaimBuffer(uint8_t* buffer);

  // Frees all relinquished buffers from memory.
  void DestroyAllBuffers();

  // Gets size of a single buffer in bytes.
  size_t GetBufferSizeInBytes() const;

 protected:
  mutex buffers_mutex_;
  std::vector<uint8_t*> buffers_ TF_GUARDED_BY(buffers_mutex_);
  size_t buffer_size_in_bytes_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_BUFFER_POOL_H_

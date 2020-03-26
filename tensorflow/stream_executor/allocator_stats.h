/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_STREAM_EXECUTOR_ALLOCATOR_STATS_H_
#define TENSORFLOW_STREAM_EXECUTOR_ALLOCATOR_STATS_H_

#include <string>

#include "absl/types/optional.h"
#include "tensorflow/stream_executor/platform/port.h"

namespace stream_executor {

// Runtime statistics collected by an allocator. Exactly the same as
// tensorflow::AllocatorStats, but independently defined to preserve the mutual
// independence of StreamExecutor and TensorFlow.
struct AllocatorStats {
  int64 num_allocs;          // Number of allocations.
  int64 bytes_in_use;        // Number of bytes in use.
  int64 peak_bytes_in_use;   // The peak bytes in use.
  int64 largest_alloc_size;  // The largest single allocation seen.

  // The upper limit of bytes of user allocatable device memory, if such a limit
  // is known.
  absl::optional<int64> bytes_limit;

  // Stack related memory usage.
  int64 bytes_reserved;       // Number of bytes reserved on the stack.
  int64 peak_bytes_reserved;  // The peak number of bytes reserved on the stack.
  // The upper limit on the number bytes of reservable memory on the stack,
  // if such a limit is known.
  absl::optional<int64> bytes_reservable_limit;

  AllocatorStats()
      : num_allocs(0),
        bytes_in_use(0),
        peak_bytes_in_use(0),
        largest_alloc_size(0),
        bytes_reserved(0),
        peak_bytes_reserved(0) {}

  std::string DebugString() const;
};

}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_ALLOCATOR_STATS_H_

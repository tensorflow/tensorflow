/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_MEMORY_ALLOCATOR_H_
#define XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_MEMORY_ALLOCATOR_H_

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor_interface.h"

namespace stream_executor {

// Default memory allocator for a platform which uses
// StreamExecutor::Allocate/Deallocate.
class StreamExecutorMemoryAllocator : public DeviceMemoryAllocator {
 public:
  // Create an allocator supporting a single device, corresponding to the
  // passed executor.
  explicit StreamExecutorMemoryAllocator(StreamExecutorInterface *executor);

  // Create an allocator supporting multiple stream executors.
  //
  // Precondition: all stream_executors have different device ordinals.
  StreamExecutorMemoryAllocator(
      const Platform *platform,
      absl::Span<StreamExecutorInterface *const> stream_executors);

  absl::StatusOr<OwningDeviceMemory> Allocate(int device_ordinal, uint64_t size,
                                              bool retry_on_failure,
                                              int64_t memory_space) override;

  // Pull in two-arg overload that sets retry_on_failure to true.
  using DeviceMemoryAllocator::Allocate;

  absl::Status Deallocate(int device_ordinal, DeviceMemoryBase mem) override;

  bool AllowsAsynchronousDeallocation() const override;

  // Gets-or-creates a stream for a given `device_ordinal` from an appropriate
  // stream executor.
  absl::StatusOr<Stream *> GetStream(int device_ordinal) override;

  // Gets the stream executor for given device ordinal.
  absl::StatusOr<StreamExecutorInterface *> GetStreamExecutor(
      int device_ordinal) const;

 private:
  // Available stream executors. Each stream executor has a different device
  // ordinal.
  std::vector<StreamExecutorInterface *> stream_executors_;

  absl::Mutex mutex_;

  // Cache of streams for GetStream.
  std::map<int, std::unique_ptr<Stream>> streams_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_MEMORY_ALLOCATOR_H_

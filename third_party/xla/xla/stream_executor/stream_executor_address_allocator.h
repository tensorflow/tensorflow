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

#ifndef XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_ADDRESS_ALLOCATOR_H_
#define XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_ADDRESS_ALLOCATOR_H_

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {

// Default address allocator for a platform which uses StreamExecutor
// Allocate/Deallocate APIs.
class StreamExecutorAddressAllocator : public DeviceAddressAllocator {
 public:
  // Create an allocator supporting a single device, corresponding to the
  // passed executor.
  explicit StreamExecutorAddressAllocator(StreamExecutor* executor);

  // Create an allocator supporting multiple stream executors.
  //
  // Precondition: all stream_executors have different device ordinals.
  StreamExecutorAddressAllocator(
      const Platform* platform,
      absl::Span<StreamExecutor* const> stream_executors);

  absl::StatusOr<ScopedDeviceAddress<uint8_t>> Allocate(
      int device_ordinal, uint64_t size, bool retry_on_failure,
      int64_t memory_space) override;

  // Pull in two-arg overload that sets retry_on_failure to true.
  using DeviceAddressAllocator::Allocate;

  absl::Status Deallocate(int device_ordinal, DeviceAddressBase mem) override;

  bool AllowsAsynchronousDeallocation() const override;

  // Gets-or-creates a stream for a given `device_ordinal` from an appropriate
  // stream executor.
  absl::StatusOr<Stream*> GetStream(int device_ordinal) override;

  // Gets the stream executor for given device ordinal.
  absl::StatusOr<StreamExecutor*> GetStreamExecutor(int device_ordinal) const;

 private:
  // Available stream executors. Each stream executor has a different device
  // ordinal.
  std::vector<StreamExecutor*> stream_executors_;

  absl::Mutex mutex_;

  // Cache of streams for GetStream.
  std::map<int, std::unique_ptr<Stream>> streams_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_STREAM_EXECUTOR_ADDRESS_ALLOCATOR_H_

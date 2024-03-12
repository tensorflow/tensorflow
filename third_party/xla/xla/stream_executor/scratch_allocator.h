/* Copyright 2015 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_SCRATCH_ALLOCATOR_H_
#define XLA_STREAM_EXECUTOR_SCRATCH_ALLOCATOR_H_

#include <cstddef>
#include <cstdint>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "tsl/platform/statusor.h"

namespace stream_executor {

class Stream;

// Interface for "scratch" allocator for device memory, which deallocates all
// buffers it has allocated at destruction. Returned memory pointers are not
// owning.
//
// Used by stream operations (e.g. Stream::ThenConvolveWithScratch) to
// optionally request scratch space to speed up the operation.
class ScratchAllocator {
 public:
  virtual ~ScratchAllocator() {}

  // Returns a limit of memory this scratch allocator wants to produce, in
  // bytes. This information may be used to help select an algorithm.
  //
  // Returns values < 0 to indicate that there is no recommended limit.
  virtual int64_t GetMemoryLimitInBytes() = 0;

  // Returns an allocation on byte_size bytes for use in an operation on stream.
  //
  // This is a temporary allocation, and the caller is responsible for
  // deallocating at some known-safe point. See the class comment above.
  virtual absl::StatusOr<DeviceMemory<uint8_t>> AllocateBytes(
      int64_t byte_size) = 0;
};

// Can allocate several times -- this memory is deallocated when the scratch
// allocator is destroyed.
//
// Thread-compatible, but not thread-safe (use in scenarios where only one
// thread will request the scratch allocation).
template <size_t N = 1>
class OwningScratchAllocator : public ScratchAllocator {
 public:
  OwningScratchAllocator(int device_ordinal, DeviceMemoryAllocator* allocator)
      : device_ordinal_(device_ordinal), allocator_(allocator) {}

  int64_t GetMemoryLimitInBytes() override { return -1; }

  absl::StatusOr<DeviceMemory<uint8_t>> AllocateBytes(
      int64_t byte_size) override {
    TF_ASSIGN_OR_RETURN(OwningDeviceMemory buffer,
                        allocator_->Allocate(device_ordinal_, byte_size,
                                             /*retry_on_failure=*/false));
    buffers_.push_back(std::move(buffer));
    return *buffers_.back();
  }

 private:
  int device_ordinal_;
  DeviceMemoryAllocator* allocator_;
  absl::InlinedVector<OwningDeviceMemory, N> buffers_;

  OwningScratchAllocator(const OwningScratchAllocator&) = delete;
  void operator=(const OwningScratchAllocator&) = delete;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_SCRATCH_ALLOCATOR_H_

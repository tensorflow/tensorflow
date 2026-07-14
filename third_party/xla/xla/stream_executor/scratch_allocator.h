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
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor {

// Interface for "scratch" allocator for device memory, which deallocates all
// buffers it has allocated at destruction. Returned memory pointers are not
// owning.
//
// Used by stream operations (e.g. Stream::ThenConvolveWithScratch) to
// optionally request scratch space to speed up the operation.
class ScratchAllocator {
 public:
  virtual ~ScratchAllocator() = default;

  // Returns a limit of memory this scratch allocator wants to produce, in
  // bytes. This information may be used to help select an algorithm.
  //
  // Returns values < 0 to indicate that there is no recommended limit.
  virtual int64_t GetMemoryLimitInBytes() = 0;

  // Returns an allocation on byte_size bytes for use in an operation on stream.
  //
  // This is a temporary allocation, and the caller is responsible for
  // deallocating at some known-safe point. See the class comment above.
  virtual absl::StatusOr<DeviceAddress<uint8_t>> AllocateBytes(
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
  OwningScratchAllocator(int device_ordinal, DeviceAddressAllocator* allocator)
      : device_ordinal_(device_ordinal), allocator_(allocator) {
    CHECK(allocator_ != nullptr);
  }

  OwningScratchAllocator(OwningScratchAllocator&&) = default;
  // The default move assignment operator will immediately destroy any existing
  // `buffers_` in the target object. This can cause a use-after-free bug if
  // those buffers are still in use by the GPU. Deleting the move assignment
  // operator to prevent this issue. If move assignment is needed, a custom
  // implementation must be provided that defers the cleanup of the old buffers
  // similar to the destructor.
  OwningScratchAllocator& operator=(OwningScratchAllocator&&) = delete;

  ~OwningScratchAllocator() override {
    if (buffers_.empty()) {
      return;
    }
    // If the allocator supports asynchronous deallocation, we can rely on the
    // default destruction of `buffers_` (which calls `Deallocate` via
    // `ScopedDeviceAddress`) because the allocator will handle the
    // synchronization.
    if (allocator_->AllowsAsynchronousDeallocation()) {
      return;
    }
    absl::StatusOr<Stream*> stream = allocator_->GetStream(device_ordinal_);
    if (!stream.ok()) {
      LOG(ERROR) << "Failed to get stream for asynchronous deallocation: "
                 << stream.status();
    } else if (*stream == nullptr) {
      LOG(ERROR) << "Allocator returned a null stream for asynchronous "
                    "deallocation.";
    } else {
      if (absl::Status s = (*stream)->DoHostCallback(
              [buffers = std::move(buffers_)]() mutable { buffers.clear(); });
          !s.ok()) {
        LOG(ERROR) << "Failed to schedule scratch allocator cleanup: " << s;
      }
    }
  }

  int64_t GetMemoryLimitInBytes() override { return -1; }

  absl::StatusOr<DeviceAddress<uint8_t>> AllocateBytes(
      int64_t byte_size) override {
    if (byte_size < 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("byte_size must be non-negative, but got ", byte_size));
    }
    ASSIGN_OR_RETURN(ScopedDeviceAddress<uint8_t> buffer,
                     allocator_->Allocate(device_ordinal_, byte_size,
                                          /*retry_on_failure=*/false));
    buffers_.push_back(std::move(buffer));
    return *buffers_.back();
  }

 private:
  int device_ordinal_;
  DeviceAddressAllocator* allocator_;
  absl::InlinedVector<ScopedDeviceAddress<uint8_t>, N> buffers_;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_SCRATCH_ALLOCATOR_H_

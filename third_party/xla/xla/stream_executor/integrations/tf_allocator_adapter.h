/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_INTEGRATIONS_TF_ALLOCATOR_ADAPTER_H_
#define XLA_STREAM_EXECUTOR_INTEGRATIONS_TF_ALLOCATOR_ADAPTER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/framework/allocator.h"

namespace stream_executor {

// Adapter class that wraps a Tensorflow allocator.
//
// Assumes that the Tensorflow allocator permits asynchronous deallocation:
// see comment on `AllowsAsynchronousDeallocation()`.
class TfAllocatorAdapter : public DeviceAddressAllocator {
 public:
  // Creates a device address allocator by wrapping a tsl::Allocator.
  //
  // wrapped:       underlying memory allocator, which in practice is almost
  //                always a BFC allocator wrapping a physical memory allocator.
  //
  // stream:        stream on which this allocator can only be used. If
  //                non-null, the allocator cannot be used on any other stream.
  //
  // min_alignment: minimum alignment passed to tsl::Allocator::AllocateRaw.
  //                Different memory spaces may require different alignment
  //                (e.g. symmetric memory requires higher alignment than
  //                default memory used for on-device compute).
  TfAllocatorAdapter(
      tsl::Allocator* wrapped, Stream* stream,
      size_t min_alignment = tsl::Allocator::kAllocatorAlignment);

  // Constructor for cases where `stream` is not available.
  TfAllocatorAdapter(
      tsl::Allocator* wrapped, const Platform* platform,
      size_t min_alignment = tsl::Allocator::kAllocatorAlignment);

  ~TfAllocatorAdapter() override;

  absl::StatusOr<ScopedDeviceAddress<uint8_t>> Allocate(
      int device_ordinal, uint64_t size, bool retry_on_failure,
      int64_t memory_space) override;

  absl::Status Deallocate(int device_ordinal, DeviceAddressBase mem) override;

  // The Tensorflow BFC allocator used on GPU allows host-side deallocation
  // before GPU execution takes place. Tensorflow uses the ordering of the main
  // compute stream to enforce a happens-before relationship between a memory
  // allocation and code that reuses the same memory. If Tensorflow adds
  // support for multiple GPU streams or allocators with different ordering
  // requirements, this code may need to change.
  // (This attribute has no effect on CPU.)
  bool AllowsAsynchronousDeallocation() const override { return true; }

  absl::StatusOr<Stream*> GetStream(int device_ordinal) override;

  absl::StatusOr<tsl::Allocator*> GetAllocator(int device_ordinal);

 private:
  tsl::Allocator* wrapped_;
  Stream* stream_;
  size_t min_alignment_;
};

// Adapter class that wraps per-device TF allocators with corresponding streams
// as a TfAllocatorAdapter. Assumes that the Tensorflow allocator permits
// asynchronous deallocation; see comment on `AllowsAsynchronousDeallocation()`.
class MultiDeviceAdapter : public DeviceAddressAllocator {
 public:
  // Describes a per-device allocator for a specific memory space. Multiple
  // AllocatorInfo entries can share the same underlying tsl::Allocator (e.g.
  // kDefault and kCollective memory spaces backed by the same BFC allocator
  // with different alignment requirements).
  //
  // allocator:      underlying allocator (e.g. BFC); shared_ptr allows the
  //                 same allocator to be referenced by multiple memory spaces.
  //
  // stream:         compute stream for this device. If null, `platform` must
  //                 be set instead.
  //
  // memory_space:   identifies which memory space this entry serves
  //                 (e.g. kDefault=0, kCollective=1).
  //
  // device_ordinal: explicit device ordinal. When nullopt, inferred from
  //                 `stream`.
  //
  // platform:       platform pointer, used when `stream` is null.
  //
  // min_alignment:  minimum alignment passed to tsl::Allocator::AllocateRaw.
  //                 Symmetric/collective memory typically needs higher
  //                 alignment than default compute buffers.
  struct AllocatorInfo {
    std::shared_ptr<tsl::Allocator> allocator;
    Stream* stream;
    int64_t memory_space;
    std::optional<int32_t> device_ordinal = std::nullopt;
    const Platform* platform = nullptr;
    size_t min_alignment = tsl::Allocator::kAllocatorAlignment;
  };

  MultiDeviceAdapter(const Platform* platform,
                     std::vector<AllocatorInfo> allocators);

  absl::StatusOr<ScopedDeviceAddress<uint8_t>> Allocate(
      int device_ordinal, uint64_t size, bool retry_on_failure,
      int64_t memory_space) override;

  absl::Status Deallocate(int device_ordinal, DeviceAddressBase mem) override;

  // The Tensorflow BFC allocator used on GPU allows host-side deallocation
  // before GPU execution takes place. Tensorflow uses the ordering of the main
  // compute stream to enforce a happens-before relationship between a memory
  // allocation and code that reuses the same memory. If Tensorflow adds
  // support for multiple GPU streams or allocators with different ordering
  // requirements, this code may need to change.
  // (This attribute has no effect on CPU.)
  bool AllowsAsynchronousDeallocation() const override { return true; }

  absl::StatusOr<Stream*> GetStream(int device_ordinal) override;

  absl::StatusOr<tsl::Allocator*> GetAllocator(int device_ordinal);

 private:
  absl::flat_hash_map<int64_t, std::vector<std::shared_ptr<TfAllocatorAdapter>>>
      memory_space_to_per_device_allocators_;
  // Map of device ordinal, buffer to which memory space it resides in.
  absl::Mutex mu_;
  absl::flat_hash_map<std::pair<int, void*>, int64_t> buffer_memory_spaces_
      ABSL_GUARDED_BY(mu_);
  std::vector<std::shared_ptr<tsl::Allocator>> allocators_;
};

// Creates a status with a payload indicating an error while allocating `size`
// bytes of memory.
absl::Status MemoryAllocationError(uint64_t size, bool is_host_mem);

// Checks whether the status is a memory allocation error.
bool IsMemoryAllocationError(absl::Status status);

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_INTEGRATIONS_TF_ALLOCATOR_ADAPTER_H_

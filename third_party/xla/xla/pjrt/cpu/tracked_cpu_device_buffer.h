/* Copyright 2021 The OpenXLA Authors.

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

#ifndef XLA_PJRT_CPU_TRACKED_CPU_DEVICE_BUFFER_H_
#define XLA_PJRT_CPU_TRACKED_CPU_DEVICE_BUFFER_H_

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/pjrt/abstract_tracked_device_buffer.h"
#include "xla/pjrt/cpu/cpu_event.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {

// A region of device memory that can be used to construct PjRt buffers. Device
// memory can be either owned or non-owned.
//
// CpuDeviceMemory has an asynchronous memory allocation semantics, as the size
// of the allocation might depend on a result of another computation (pending
// async value), and must be delayed until the async value becomes available.
//
// Synchronous allocations of the raw memory (same semantics as `aligned_malloc`
// and `free`) is handled via the `CpuDeviceMemory::Allocator` interface.
//
// Types of CpuDeviceMemory:
//
//   OWNED:    raw memory was allocated for the CpuDeviceMemory and will be
//             freed when when CpuDeviceMemory is destroyed.
//
//   FOREIGN:  raw memory was allocated by another entity (i.e. it can be a view
//             into a buffer owned by a different runtime) and the owner will be
//             notified via the on_delete_callback when CpuDeviceMemory is
//             destroyed.
//
//   CONSTANT: raw memory has a lifetime that is not bound to the
//             CpuDeviceMemory (i.e. a global static).
//
class CpuDeviceMemory {
 public:
  class Allocator;

  virtual ~CpuDeviceMemory() = default;

  CpuDeviceMemory(const CpuDeviceMemory&) = delete;
  CpuDeviceMemory& operator=(const CpuDeviceMemory&) = delete;

  virtual void* untyped_data() const = 0;
  virtual size_t size_bytes() const = 0;

  // Creates an unavailable AsyncValueRef placeholder for a delayed
  // memory allocation (see `AllocateInto` below).
  static tsl::AsyncValueRef<CpuDeviceMemory> CreateDelayedMemory();

  // Creates an available AsyncValueRef to a CpuDeviceMemory that wraps foreign
  // memory. Will call on_delete_callback on the last-ref.
  static tsl::AsyncValueRef<CpuDeviceMemory> CreateForeignMemory(
      void* base, size_t size,
      absl::AnyInvocable<void() &&> on_delete_callback);

  // Creates an available AsyncValueRef to a CpuDeviceMemory that wraps a
  // constant memory with infinite lifetime. No action will be taken on decref.
  static tsl::AsyncValueRef<CpuDeviceMemory> CreateConstantMemory(void* base,
                                                                  size_t size);

  // Allocates owning memory wrapped in an available `AsyncValueRef`.
  static absl::StatusOr<tsl::AsyncValueRef<CpuDeviceMemory>> Allocate(
      size_t size_bytes, const Allocator& allocator = DefaultAllocator());

  // Allocates owning memory into the previously created delayed memory
  // placeholder (see `CreateDelayedMemory` above).
  static absl::Status AllocateInto(
      size_t size_bytes, tsl::AsyncValuePtr<CpuDeviceMemory> delayed_memory,
      const Allocator& allocator = DefaultAllocator());

  //===--------------------------------------------------------------------===//
  // Custom raw memory allocation APIs.
  //===--------------------------------------------------------------------===//

  // Default allocator uses aligned allocation and free APIs from tsl.
  static Allocator& DefaultAllocator();

  // Returns a new instance of the default allocator.
  static std::unique_ptr<Allocator> MakeDefaultAllocator();

  // A raw memory allocation that can be used to construct a CpuDeviceMemory.
  class RawMemory {
   public:
    virtual ~RawMemory() = default;
    virtual void* base() const = 0;
    virtual size_t size_bytes() const = 0;
  };

  // A raw memory allocator that allocates memory buffers for constructing
  // CpuDeviceMemory.
  //
  // This is a virtual interface to allow for different memory allocation
  // strategies, e.g. aligned_alloc, pre-mapped DMA buffers, etc. For example,
  // when XLA:CPU is running as a part of host-offloading computation, we want
  // all buffers used by XLA:CPU to be pre-mapped with the accelerator device,
  // so that we can issue zero-copy DMA transfers operations if needed.
  class Allocator {
   public:
    virtual ~Allocator() = default;
    virtual absl::StatusOr<std::unique_ptr<RawMemory>> Allocate(
        size_t size_bytes, size_t alignment) const = 0;
  };

 protected:
  CpuDeviceMemory() = default;
};

// A class that represents a CPU device buffer: it can be a single memory region
// or multiple memory regions for a tuple buffers. It also tracks the definition
// and usage of the memory to allow for synchronized usage and deletion of CPU
// memory. This class is thread-compatible.
class TrackedCpuDeviceBuffer : public AbstractTrackedDeviceBuffer {
 public:
  // Variant with single definition event.
  TrackedCpuDeviceBuffer(tsl::RCReference<CommonPjRtRawBuffer> raw_buffer,
                         tsl::AsyncValueRef<CpuEvent> definition_event);

  TrackedCpuDeviceBuffer(TrackedCpuDeviceBuffer&&) noexcept = default;
  TrackedCpuDeviceBuffer& operator=(TrackedCpuDeviceBuffer&&) noexcept =
      default;

  ~TrackedCpuDeviceBuffer();

  const tsl::AsyncValueRef<CpuDeviceMemory>& buffer();

  size_t BufferSize();

  const tsl::AsyncValueRef<CpuEvent>& definition_event() const {
    return definition_event_;
  }

  void AddUsageEvents(absl::Span<tsl::AsyncValueRef<CpuEvent>> events);

  // Return the usage events for the buffers. After
  // LockUseAndTransferUsageEvents is called, it is illegal to AddUsageEvent.
  absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4>
  LockUseAndTransferUsageEvents();

  std::vector<tsl::RCReference<tsl::AsyncValue>> GetAsyncValueDefinitionEvents()
      override;

  absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>> GetDefinitionEvent(
      PjRtMemorySpace* memory_space) override;

  void AddUsageEvent(tsl::RCReference<PjRtDeviceEvent> event) override;

  void Delete(PjRtMemorySpace* memory_space) override;

  Future<> GetReadyFuture(PjRtMemorySpace* memory_space) override;

  absl::Status BlockForOperationsToComplete(
      PjRtMemorySpace* memory_space) override;

  bool AddDefinitionEventsToSet(PjRtDeviceEventSet& events) override;

  void AddUsageEventsToSet(PjRtDeviceEventSet& events) override;

 private:
  void ConfirmDonation() override;

  // The definition event are associated with CPU operations that write to the
  // buffers.
  tsl::AsyncValueRef<CpuEvent> definition_event_;
  // Usage events are associated with CPU operations that read from the buffers.
  absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4> usage_events_;
};
}  // namespace xla

#endif  // XLA_PJRT_CPU_TRACKED_CPU_DEVICE_BUFFER_H_

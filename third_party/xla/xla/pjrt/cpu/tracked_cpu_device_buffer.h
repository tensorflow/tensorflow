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

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/pjrt/abstract_tracked_device_buffer.h"
#include "xla/pjrt/cpu/cpu_event.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla {

// A region of device memory that can be used to construct PjRt buffers. Device
// memory can be either owned or non-owned.
class CpuDeviceMemory {
 public:
  virtual ~CpuDeviceMemory() = default;

  CpuDeviceMemory(const CpuDeviceMemory&) = delete;
  CpuDeviceMemory& operator=(const CpuDeviceMemory&) = delete;

  void* untyped_data() const { return base_; }
  size_t size_bytes() const { return size_bytes_; }

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
      size_t size_bytes);

  // Allocates owning memory into the previously created delayed memory
  // placeholder (see `CreateDelayedMemory` above).
  static absl::Status AllocateInto(
      size_t size_bytes, tsl::AsyncValuePtr<CpuDeviceMemory> delayed_memory);

 protected:
  CpuDeviceMemory(void* base, size_t size) : base_(base), size_bytes_(size) {}

  void* base_;
  size_t size_bytes_;
};

// A class that represents a CPU device buffer: it can be a single memory region
// or multiple memory regions for a tuple buffers. It also tracks the definition
// and usage of the memory to allow for synchronized usage and deletion of CPU
// memory. This class is thread-compatible.
class TrackedCpuDeviceBuffer : public AbstractTrackedDeviceBuffer {
 public:
  // For non-tuple, takes a single buffer.
  // For tuple, takes the leaf buffers. Tuple index table created internally.
  // Nested tuple is not supported.

  // Constructor for allocated cpu memory, i.e., `buffer` should have concrete
  // states. Definition event is after the list of `definition_events`.
  TrackedCpuDeviceBuffer(
      bool owns_buffers, tsl::AsyncValueRef<CpuDeviceMemory> buffer,
      absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4> definition_events);

  // Variant with single definition event.
  TrackedCpuDeviceBuffer(bool owns_buffers,
                         tsl::AsyncValueRef<CpuDeviceMemory> buffer,
                         tsl::AsyncValueRef<CpuEvent> definition_event);

  // Constructor for unallocated cpu memory, i.e., `buffer` will have
  // unconstructed states, and we also need to provide `buffer_size` which will
  // be the size of the `buffer` after allocation. Definition event is after the
  // list of `definition_events`. Callers need to ensure cpu memory is allocated
  // before the definition event is ready.
  TrackedCpuDeviceBuffer(
      bool owns_buffers, tsl::AsyncValueRef<CpuDeviceMemory> buffer,
      size_t buffer_size,
      absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4> definition_events);

  // Variant with single definition event.
  TrackedCpuDeviceBuffer(bool owns_buffers,
                         tsl::AsyncValueRef<CpuDeviceMemory> buffer,
                         size_t buffer_size,
                         tsl::AsyncValueRef<CpuEvent> definition_event);

  TrackedCpuDeviceBuffer(TrackedCpuDeviceBuffer&&) noexcept = default;
  TrackedCpuDeviceBuffer& operator=(TrackedCpuDeviceBuffer&&) noexcept =
      default;

  ~TrackedCpuDeviceBuffer();

  const tsl::AsyncValueRef<CpuDeviceMemory>& buffer() { return buffer_; }

  size_t BufferSize();

  const tsl::AsyncValueRef<CpuEvent>& definition_event() const {
    return definition_event_;
  }

  absl::Span<const tsl::AsyncValueRef<CpuEvent>> UsageEvents() const {
    return usage_events_;
  }

  void AddUsageEvents(absl::Span<tsl::AsyncValueRef<CpuEvent>> events);

  // Return the usage events for the buffers. After
  // LockUseAndTransferUsageEvents is called, it is illegal to AddUsageEvent.
  absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4>
  LockUseAndTransferUsageEvents();

  bool owns_buffers() const { return owns_buffers_; }

  std::vector<tsl::RCReference<tsl::AsyncValue>> GetAsyncValueDefinitionEvents()
      override;

  tsl::RCReference<CommonPjRtRawBuffer> GetRawBuffer(
      PjRtMemorySpace* memory_space) override;

  void AddUsageEvent(tsl::RCReference<PjRtDeviceEvent> event) override;

 private:
  // Relinquishes ownership of the buffer's device memory, e.g., after the
  // buffer is passed to a computation that aliases its inputs to outputs.
  void ReleaseDeviceMemory();

  void ConfirmDonation() override { ReleaseDeviceMemory(); }

  bool owns_buffers_;

  // If non-tuple, `buffers_` contains 1 buffer; otherwise all leaf buffers.
  tsl::AsyncValueRef<CpuDeviceMemory> buffer_;
  // Should correspond to size of each buffer in `buffers_` when `buffers_` is
  // available.
  size_t buffer_size_;
  // The definition event are associated with CPU operations that write to the
  // buffers.
  tsl::AsyncValueRef<CpuEvent> definition_event_;
  // Usage events are associated with CPU operations that read from the buffers.
  absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4> usage_events_;
};
}  // namespace xla

#endif  // XLA_PJRT_CPU_TRACKED_CPU_DEVICE_BUFFER_H_

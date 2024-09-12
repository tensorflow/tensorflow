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

#ifndef XLA_PJRT_CPU_TRACKED_TFRT_CPU_DEVICE_BUFFER_H_
#define XLA_PJRT_CPU_TRACKED_TFRT_CPU_DEVICE_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/cpu_function_runtime.h"
#include "xla/service/cpu/cpu_event.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/util.h"
#include "tsl/platform/mem.h"
#include "tsl/platform/statusor.h"

namespace xla {

class MaybeOwningCpuMemory {
 public:
  using OwnedDataPtr = std::unique_ptr<uint8_t[], void (*)(void*)>;

  MaybeOwningCpuMemory() = default;

  // Non-owning.
  MaybeOwningCpuMemory(void* buf, size_t size) : buf_(buf), size_(size) {}

  // Owning.
  MaybeOwningCpuMemory(OwnedDataPtr data, size_t size)
      : buf_(data.get()), data_(std::move(data)), size_(size) {}

  // Move-only.
  MaybeOwningCpuMemory(MaybeOwningCpuMemory&&) = default;
  MaybeOwningCpuMemory& operator=(MaybeOwningCpuMemory&&) = default;
  MaybeOwningCpuMemory(const MaybeOwningCpuMemory&) = delete;
  MaybeOwningCpuMemory& operator=(const MaybeOwningCpuMemory&) = delete;

  // Allocates owning memory wrapped in an available `AsyncValueRef`.
  static absl::StatusOr<tsl::AsyncValueRef<MaybeOwningCpuMemory>>
  AllocateAvailableAvr(size_t size) {
    TF_ASSIGN_OR_RETURN(auto memory, Allocate(size));
    return tsl::MakeAvailableAsyncValueRef<MaybeOwningCpuMemory>(
        std::move(memory));
  }

  // Allocates raw owning memory. The typical usage is for delayed allocation.
  static absl::StatusOr<MaybeOwningCpuMemory> Allocate(size_t size) {
    uint8_t* data = static_cast<uint8_t*>(
        tsl::port::AlignedMalloc(size, cpu_function_runtime::MinAlign()));
    if (!data) {
      return ResourceExhausted("Out of memory allocating %d bytes.", size);
    }
    return MaybeOwningCpuMemory(OwnedDataPtr{data, tsl::port::AlignedFree},
                                size);
  }

  void* data() const { return buf_; }
  size_t size() const { return size_; }

 private:
  void* buf_ = nullptr;                  // Non-owning data pointer.
  OwnedDataPtr data_ = {nullptr, free};  // Owning data pointer;
  size_t size_ = 0;                      // Size in number of bytes.
};

// Class that represents CPU buffers. It optionally owns the buffers. It also
// tracks the definition and usage of the memory to allow for synchronized usage
// and deletion of CPU memory. This class is thread-compatible.
class TrackedTfrtCpuDeviceBuffer {
 public:
  // For non-tuple, takes a single buffer.
  // For tuple, takes the leaf buffers. Tuple index table created internally.
  // Nested tuple is not supported.

  // Constructor for allocated cpu memory, i.e., `buffers` should have concrete
  // states. Definition event is after the list of `definition_events`.
  TrackedTfrtCpuDeviceBuffer(
      bool is_tuple, bool owns_buffers,
      absl::InlinedVector<tsl::AsyncValueRef<MaybeOwningCpuMemory>, 4> buffers,
      absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4> definition_events,
      absl::AnyInvocable<void() &&> on_delete_callback = nullptr);

  // Variant with single definition event.
  TrackedTfrtCpuDeviceBuffer(
      bool is_tuple, bool owns_buffers,
      absl::InlinedVector<tsl::AsyncValueRef<MaybeOwningCpuMemory>, 4> buffers,
      tsl::AsyncValueRef<CpuEvent> definition_event,
      absl::AnyInvocable<void() &&> on_delete_callback = nullptr);

  // Constructor for unallocated cpu memory, i.e., `buffers` have unconstructed
  // states, also needs to provide `buffer_sizes` which will be the sizes of
  // the `buffers` after allocation. Definition event is after the list of
  // `definition_events`. Callers need to ensure cpu memory is allocated before
  // the definition event is ready.
  TrackedTfrtCpuDeviceBuffer(
      bool is_tuple, bool owns_buffers,
      absl::InlinedVector<tsl::AsyncValueRef<MaybeOwningCpuMemory>, 4> buffers,
      absl::InlinedVector<size_t, 4> buffer_sizes,
      absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4> definition_events,
      absl::AnyInvocable<void() &&> on_delete_callback = nullptr);

  // Variant with single definition event.
  TrackedTfrtCpuDeviceBuffer(
      bool is_tuple, bool owns_buffers,
      absl::InlinedVector<tsl::AsyncValueRef<MaybeOwningCpuMemory>, 4> buffers,
      absl::InlinedVector<size_t, 4> buffer_sizes,
      tsl::AsyncValueRef<CpuEvent> definition_event,
      absl::AnyInvocable<void() &&> on_delete_callback = nullptr);

  // Move-only.
  TrackedTfrtCpuDeviceBuffer(TrackedTfrtCpuDeviceBuffer&&) noexcept = default;
  TrackedTfrtCpuDeviceBuffer& operator=(TrackedTfrtCpuDeviceBuffer&&) noexcept =
      default;
  TrackedTfrtCpuDeviceBuffer(const TrackedTfrtCpuDeviceBuffer&) = delete;
  TrackedTfrtCpuDeviceBuffer& operator=(const TrackedTfrtCpuDeviceBuffer&) =
      delete;

  ~TrackedTfrtCpuDeviceBuffer();

  absl::Span<const tsl::AsyncValueRef<MaybeOwningCpuMemory>> Buffers() {
    return buffers_;
  }

  absl::Span<const size_t> BufferSizes() { return buffer_sizes_; }

  tsl::AsyncValueRef<MaybeOwningCpuMemory> Buffer(
      const ShapeIndex& shape_index);

  size_t BufferSize(const ShapeIndex& shape_index);

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

  // Relinquishes ownership of the buffer's device memory, e.g., after the
  // buffer is passed to a computation that aliases its inputs to outputs.
  void ReleaseDeviceMemory();

  bool owns_buffers() const { return owns_buffers_; }

 private:
  bool is_tuple_;
  bool owns_buffers_;
  // If tuple, tuple index table is created and stored.
  tsl::AsyncValueRef<MaybeOwningCpuMemory> tuple_index_table_;
  // If non-tuple, `buffers_` contains 1 buffer; otherwise all leaf buffers.
  absl::InlinedVector<tsl::AsyncValueRef<MaybeOwningCpuMemory>, 4> buffers_;
  // Should correspond to size of each buffer in `buffers_` when `buffers_` is
  // available.
  absl::InlinedVector<size_t, 4> buffer_sizes_;
  // The definition event are associated with CPU operations that write to the
  // buffers.
  tsl::AsyncValueRef<CpuEvent> definition_event_;
  // Usage events are associated with CPU operations that read from the buffers.
  absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4> usage_events_;
  // A callback to call when the TrackedTfrtCpuDeviceBuffer is about to be
  // destroyed.
  absl::AnyInvocable<void() &&> on_delete_callback_;
};
}  // namespace xla

#endif  // XLA_PJRT_CPU_TRACKED_TFRT_CPU_DEVICE_BUFFER_H_

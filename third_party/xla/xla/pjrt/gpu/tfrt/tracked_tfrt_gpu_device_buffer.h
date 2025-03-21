/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_PJRT_GPU_TFRT_TRACKED_TFRT_GPU_DEVICE_BUFFER_H_
#define XLA_PJRT_GPU_TFRT_TRACKED_TFRT_GPU_DEVICE_BUFFER_H_

#include <cstddef>
#include <functional>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "xla/pjrt/gpu/tfrt/gpu_event.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/framework/allocator.h"

namespace xla {
// TODO(b/400541410): Refactor and Merge this with MaybeOwningDeviceMemory.

// MaybeOwningGpuMemory represents either an owned or unowned GPU memory. It
// owns GPU memory if an allocator is provided. When the object goes output of
// scope, it will free the underlying memory if it owns it.
class MaybeOwningGpuMemory {
 public:
  MaybeOwningGpuMemory() = default;

  // Non-owning underlying GPU memory `buffer`.
  explicit MaybeOwningGpuMemory(stream_executor::DeviceMemoryBase buffer)
      : allocator_(nullptr), buffer_(buffer) {}

  // Owning underlying GPU memory `buffer`. When the object goes out of scope,
  // it will free the underlying memory.
  explicit MaybeOwningGpuMemory(tsl::Allocator* allocator,
                                stream_executor::DeviceMemoryBase buffer)
      : allocator_(allocator), buffer_(buffer) {
    DCHECK(allocator != nullptr);
  }

  MaybeOwningGpuMemory(const MaybeOwningGpuMemory&) = delete;
  MaybeOwningGpuMemory& operator=(const MaybeOwningGpuMemory&) = delete;

  // Move-only.
  MaybeOwningGpuMemory(MaybeOwningGpuMemory&& other) {
    allocator_ = other.allocator_;
    buffer_ = other.buffer_;
    other.allocator_ = nullptr;
    other.buffer_ = se::DeviceMemoryBase();
  }

  MaybeOwningGpuMemory& operator=(MaybeOwningGpuMemory&& other) {
    allocator_ = other.allocator_;
    buffer_ = other.buffer_;
    other.allocator_ = nullptr;
    other.buffer_ = se::DeviceMemoryBase();
    return *this;
  }

  ~MaybeOwningGpuMemory() {
    if (owns_data()) {
      allocator_->DeallocateRaw(buffer_.opaque());
    }
  }

  ShapedBuffer AsShapedBuffer(const Shape& on_device_shape,
                              const PjRtDevice* device) const;

  // Change ownership from owning to non-owning. Used for buffer donation.
  void SetUnOwned();

  // Owning.
  static absl::StatusOr<MaybeOwningGpuMemory> AllocateShared(
      tsl::Allocator* allocator, size_t size);

  tsl::Allocator* allocator() const { return allocator_; }
  stream_executor::DeviceMemoryBase buffer() const { return buffer_; }
  size_t size() const { return buffer_.size(); }
  bool owns_data() const { return allocator_ != nullptr; }

 private:
  tsl::Allocator* allocator_;  // nullptr if unowned.
  se::DeviceMemoryBase buffer_;
};

// Class that represents a GPU buffer. It optionally owns the buffer. It also
// tracks the definition and usage of the memory to allow for synchronized usage
// and deletion of GPU memory. This class is thread-compatible.
class TrackedTfrtGpuDeviceBuffer {
 public:
  TrackedTfrtGpuDeviceBuffer(
      tsl::AsyncValueRef<MaybeOwningGpuMemory> buffer,
      absl::InlinedVector<tsl::AsyncValueRef<GpuEvent>, 4> definition_events,
      std::function<void()> on_delete_callback = nullptr);

  TrackedTfrtGpuDeviceBuffer(
      tsl::AsyncValueRef<MaybeOwningGpuMemory> buffer,
      tsl::AsyncValueRef<GpuEvent> definition_event,
      std::function<void()> on_delete_callback = nullptr);

  // Move-only.
  TrackedTfrtGpuDeviceBuffer(TrackedTfrtGpuDeviceBuffer&&) = default;
  TrackedTfrtGpuDeviceBuffer& operator=(TrackedTfrtGpuDeviceBuffer&&) = default;
  TrackedTfrtGpuDeviceBuffer(const TrackedTfrtGpuDeviceBuffer&) = delete;
  TrackedTfrtGpuDeviceBuffer& operator=(const TrackedTfrtGpuDeviceBuffer&) =
      delete;

  ~TrackedTfrtGpuDeviceBuffer();

  const tsl::AsyncValueRef<MaybeOwningGpuMemory>& buffer() const {
    return buffer_;
  }

  const tsl::AsyncValueRef<GpuEvent>& definition_event() const {
    return definition_event_;
  }

  const tsl::AsyncValueRef<GpuEvent>& deallocation_event() const {
    return deallocation_event_;
  }

  // Adds usage events to the buffer. This usage events could be any device
  // buffer related events, e.g. D2H/D2D
  void AddUsageEvents(absl::Span<tsl::AsyncValueRef<GpuEvent>> events);

  // Returns an AsyncValueRef<GpuEvent> that will be ready after all the async
  // values in usage events are ready. If errors occurs, one of the errors will
  // be propagated through the returned async value.
  tsl::AsyncValueRef<GpuEvent> AfterAllUsageEvents();

  // Return the usage events for the buffers. After
  // LockUseAndTransferUsageEvents is called, it is illegal to AddUsageEvent.
  tsl::AsyncValueRef<GpuEvent> LockUseAndTransferUsageEvents();

  // Relinquishes ownership of the buffer's device memory, e.g., after the
  // buffer is passed to a computation that aliases its inputs to outputs.
  void ReleaseDeviceMemory();

  // Change ownership of underlying MaybeOwningGpuMemory from owning to
  // non-owning. Used for buffer donation.
  void SetUnOwned();

 private:
  tsl::AsyncValueRef<MaybeOwningGpuMemory> buffer_;

  // The definition event are associated with GPU operations that write to the
  // buffers.
  tsl::AsyncValueRef<GpuEvent> definition_event_;

  // Usage events are associated with GPU operations that read from the buffers.
  TfrtEventSet usage_events_;

  // An event triggered after this buffer is freed or donated. This event is
  // used to make sure that allocations are sequenced with respect to
  // deallocations in program order.
  tsl::AsyncValueRef<GpuEvent> deallocation_event_;

  // A callback to call when the TrackedTfrtGpuDeviceBuffer is about to be
  // destroyed.
  std::function<void()> on_delete_callback_;
};
}  // namespace xla

#endif  // XLA_PJRT_GPU_TFRT_TRACKED_TFRT_GPU_DEVICE_BUFFER_H_

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

#ifndef XLA_PJRT_GPU_TRACKED_TFRT_GPU_DEVICE_BUFFER_H_
#define XLA_PJRT_GPU_TRACKED_TFRT_GPU_DEVICE_BUFFER_H_

#include <functional>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "xla/pjrt/gpu/gpu_event.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/shaped_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/util.h"

namespace xla {

class MaybeOwningGpuMemory {
 public:
  MaybeOwningGpuMemory() = default;

  // Non-owning.
  explicit MaybeOwningGpuMemory(se::DeviceMemoryBase buffer)
      : allocator_(nullptr), buffer_(buffer) {}

  // Owning.
  explicit MaybeOwningGpuMemory(tsl::Allocator* allocator,
                                se::DeviceMemoryBase buffer)
      : allocator_(allocator), buffer_(buffer) {
    DCHECK(allocator != nullptr);
  }

  ~MaybeOwningGpuMemory() {
    if (owns_data()) {
      allocator_->DeallocateRaw(buffer_.opaque());
    }
  }

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

  ShapedBuffer AsShapedBuffer(const Shape& on_device_shape,
                              const PjRtDevice* device) const {
    ShapedBuffer shaped_buffer(on_device_shape,
                               device->local_device_id().value(),
                               device->local_hardware_id().value());
    ShapeTree<se::DeviceMemoryBase>::iterator iterator =
        shaped_buffer.buffers().begin();
    CHECK(iterator != shaped_buffer.buffers().end());
    iterator->second = buffer_;
    ++iterator;
    CHECK(iterator == shaped_buffer.buffers().end());
    return shaped_buffer;
  }

  // Change ownership from owning to non-owning. Used for buffer donation.
  void SetUnOwned() {
    CHECK(owns_data())
        << "SetUnOwned can only be called on an owning MaybeOwningGpuMemory.";
    allocator_ = nullptr;
  }

  MaybeOwningGpuMemory(const MaybeOwningGpuMemory&) = delete;
  MaybeOwningGpuMemory& operator=(const MaybeOwningGpuMemory&) = delete;

  // Owning.
  static absl::StatusOr<MaybeOwningGpuMemory> AllocateShared(
      tsl::Allocator* allocator, size_t size) {
    if (size == 0) return MaybeOwningGpuMemory(se::DeviceMemoryBase());
    void* data =
        allocator->AllocateRaw(tsl::Allocator::kAllocatorAlignment, size);
    if (!data) {
      return ResourceExhausted("Out of memory allocating %d bytes.", size);
    }
    return MaybeOwningGpuMemory(allocator, se::DeviceMemoryBase(data, size));
  }

  tsl::Allocator* allocator() const { return allocator_; }
  se::DeviceMemoryBase buffer() const { return buffer_; }
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

  tsl::AsyncValueRef<GpuEvent> AfterAllUsageEvents() {
    return usage_events_.AfterAll();
  }

  const tsl::AsyncValueRef<GpuEvent>& deallocation_event() const {
    return deallocation_event_;
  }

  void AddUsageEvents(absl::Span<tsl::AsyncValueRef<GpuEvent>> events);

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

#endif  // XLA_PJRT_GPU_TRACKED_TFRT_GPU_DEVICE_BUFFER_H_

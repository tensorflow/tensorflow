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
#include "xla/pjrt/gpu/tfrt/tracked_tfrt_gpu_device_buffer.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "xla/pjrt/gpu/tfrt/gpu_event.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/stacktrace.h"

namespace xla {

ShapedBuffer MaybeOwningGpuMemory::AsShapedBuffer(
    const Shape& on_device_shape, const PjRtDevice* device) const {
  ShapedBuffer shaped_buffer(on_device_shape, device->local_device_id().value(),
                             device->local_hardware_id().value());
  ShapeTree<se::DeviceMemoryBase>::iterator iterator =
      shaped_buffer.buffers().begin();
  CHECK(iterator != shaped_buffer.buffers().end());
  iterator->second = buffer_;
  ++iterator;
  CHECK(iterator == shaped_buffer.buffers().end());
  return shaped_buffer;
}

void MaybeOwningGpuMemory::SetUnOwned() {
  CHECK(owns_data())
      << "SetUnOwned can only be called on an owning MaybeOwningGpuMemory.";
  owning_buffer_.Release();
}

absl::StatusOr<MaybeOwningGpuMemory> MaybeOwningGpuMemory::AllocateShared(
    se::DeviceMemoryAllocator* allocator, int device_ordinal, size_t size) {
  return AllocateShared(allocator, device_ordinal, size,
                        static_cast<int>(se::MemoryType::kDevice));
}

absl::StatusOr<MaybeOwningGpuMemory> MaybeOwningGpuMemory::AllocateShared(
    se::DeviceMemoryAllocator* allocator, int device_ordinal, size_t size,
    int64_t memory_space) {
  if (size == 0) {
    return MaybeOwningGpuMemory(se::DeviceMemoryBase());
  }
  TF_ASSIGN_OR_RETURN(
      stream_executor::OwningDeviceMemory memory,
      allocator->Allocate(device_ordinal, size, /*retry_on_failure=*/true,
                          memory_space));
  return MaybeOwningGpuMemory(std::move(memory));
}

TrackedTfrtGpuDeviceBuffer::TrackedTfrtGpuDeviceBuffer(
    tsl::AsyncValueRef<MaybeOwningGpuMemory> buffer,
    absl::InlinedVector<tsl::AsyncValueRef<GpuEvent>, 4> definition_events,
    std::function<void()> on_delete_callback)
    : TrackedTfrtGpuDeviceBuffer(std::move(buffer), AfterAll(definition_events),
                                 std::move(on_delete_callback)) {
  if (VLOG_IS_ON(4)) {
    LOG(INFO) << "TrackedTfrtGpuDeviceBuffer::TrackedTfrtGpuDeviceBuffer: "
              << this << "\n " << tsl::CurrentStackTrace();
  }
}

TrackedTfrtGpuDeviceBuffer::TrackedTfrtGpuDeviceBuffer(
    tsl::AsyncValueRef<MaybeOwningGpuMemory> buffer,
    tsl::AsyncValueRef<GpuEvent> definition_event,
    std::function<void()> on_delete_callback)
    : buffer_(std::move(buffer)),
      definition_event_(std::move(definition_event)),
      deallocation_event_(tsl::MakeConstructedAsyncValueRef<GpuEvent>()),
      on_delete_callback_(std::move(on_delete_callback)) {
  if (VLOG_IS_ON(4)) {
    LOG(INFO) << "TrackedTfrtGpuDeviceBuffer::TrackedTfrtGpuDeviceBuffer: "
              << this << "\n " << tsl::CurrentStackTrace();
  }
  DCHECK(definition_event_);
}

TrackedTfrtGpuDeviceBuffer::~TrackedTfrtGpuDeviceBuffer() {
  if (VLOG_IS_ON(4)) {
    LOG(INFO) << "TrackedTfrtGpuDeviceBuffer::~TrackedTfrtGpuDeviceBuffer: "
              << this << " opaque: " << buffer_->buffer().opaque() << "\n "
              << tsl::CurrentStackTrace();
  }

  ReleaseDeviceMemory();
  if (on_delete_callback_) {
    on_delete_callback_();
  }
}

void TrackedTfrtGpuDeviceBuffer::AddUsageEvents(
    absl::Span<tsl::AsyncValueRef<GpuEvent>> events) {
  for (auto& ev : events) {
    usage_events_.Add(std::move(ev));
  }
}

tsl::AsyncValueRef<GpuEvent> TrackedTfrtGpuDeviceBuffer::AfterAllUsageEvents() {
  return usage_events_.AfterAll();
}

// Schedule tasks to wait for all usage events to be ready. Clear all the usage
// events that are scheduled and return the ready event. Since all usage events
// are AsyncValueRef, even TrackedTfrtGpuDeviceBuffer no longer holds the usage
// events, the usage events must be still alive and held by someone who is
// responsible to set event ready.
tsl::AsyncValueRef<GpuEvent>
TrackedTfrtGpuDeviceBuffer::LockUseAndTransferUsageEvents() {
  auto after_all = usage_events_.AfterAll();
  usage_events_.Clear();
  return after_all;
}

void TrackedTfrtGpuDeviceBuffer::ReleaseDeviceMemory() {
  buffer_.reset();
  definition_event_.reset();
  usage_events_.Clear();
  deallocation_event_.SetStateConcrete();
}

void TrackedTfrtGpuDeviceBuffer::SetUnOwned() {
  if (buffer_.IsAvailable()) {
    if (buffer_.IsError()) {
      VLOG(3) << "Setting buffer to unowned: buffer has error state.";
      return;
    }
    buffer_->SetUnOwned();
  } else {
    buffer_.AndThen([buffer = buffer_]() {
      if (buffer.IsError()) {
        VLOG(3) << "Setting buffer to unowned: buffer has error state.";
        return;
      }
      buffer->SetUnOwned();
    });
  }
}

}  // namespace xla

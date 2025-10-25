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

#include "xla/pjrt/gpu/tracked_tfrt_gpu_device_buffer.h"

#include <functional>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "xla/pjrt/gpu/gpu_event.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/platform/stacktrace.h"

namespace xla {

TrackedTfrtGpuDeviceBuffer::TrackedTfrtGpuDeviceBuffer(
    tsl::AsyncValueRef<MaybeOwningGpuMemory> buffer,
    absl::InlinedVector<tsl::AsyncValueRef<GpuEvent>, 4> definition_events,
    std::function<void()> on_delete_callback)
    : TrackedTfrtGpuDeviceBuffer(std::move(buffer), AfterAll(definition_events),
                                 std::move(on_delete_callback)) {
  VLOG(4) << "TrackedTfrtGpuDeviceBuffer::TrackedTfrtGpuDeviceBuffer: " << this
          << "\n " << tsl::CurrentStackTrace();
}

TrackedTfrtGpuDeviceBuffer::TrackedTfrtGpuDeviceBuffer(
    tsl::AsyncValueRef<MaybeOwningGpuMemory> buffer,
    tsl::AsyncValueRef<GpuEvent> definition_event,
    std::function<void()> on_delete_callback)
    : buffer_(std::move(buffer)),
      definition_event_(std::move(definition_event)),
      deallocation_event_(tsl::MakeConstructedAsyncValueRef<GpuEvent>()),
      on_delete_callback_(std::move(on_delete_callback)) {
  VLOG(4) << "TrackedTfrtGpuDeviceBuffer::TrackedTfrtGpuDeviceBuffer: " << this
          << "\n " << tsl::CurrentStackTrace();
  DCHECK(definition_event_);
}

TrackedTfrtGpuDeviceBuffer::~TrackedTfrtGpuDeviceBuffer() {
  VLOG(4) << "TrackedTfrtGpuDeviceBuffer::~TrackedTfrtGpuDeviceBuffer: " << this
          << " opaque: " << buffer_->buffer().opaque() << "\n "
          << tsl::CurrentStackTrace();
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
    buffer_->SetUnOwned();
  } else {
    buffer_.AndThen([buffer = buffer_]() { buffer->SetUnOwned(); });
  }
}

}  // namespace xla

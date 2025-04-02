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

#include "xla/pjrt/cpu/tracked_cpu_device_buffer.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/pjrt/cpu/cpu_event.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/mem.h"

namespace xla {
namespace {

// Returns an AsyncValueRef<CpuEvent> that will be ready after all the async
// values in `events` are ready. If errors occurs, one of the errors will be
// propagated through the returned async value.
tsl::AsyncValueRef<CpuEvent> AfterAll(
    absl::Span<const tsl::AsyncValueRef<CpuEvent>> events) {
  if (events.empty()) return tsl::MakeAvailableAsyncValueRef<CpuEvent>();

  struct State {
    State(int count, tsl::AsyncValueRef<CpuEvent> after_all)
        : count(count), after_all(std::move(after_all)) {}
    std::atomic<int> count;
    tsl::AsyncValueRef<CpuEvent> after_all;

    absl::Mutex mutex;
    absl::Status error;
  };

  auto after_all = tsl::MakeConstructedAsyncValueRef<CpuEvent>();
  auto* state = new State(events.size(), after_all);

  for (auto& event : events) {
    event.AndThen([state, event = event.AsPtr()]() {
      if (event.IsError()) {
        absl::MutexLock lock(&state->mutex);
        state->error = event.GetError();
      }

      if (state->count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        if (!state->error.ok()) {
          state->after_all.SetError(state->error);
        } else {
          state->after_all.SetStateConcrete();
        }
        delete state;
      }
    });
  }

  return after_all;
}

}  // namespace

// Creates non-owning CPU device memory from a raw data pointer.
CpuDeviceMemory::CpuDeviceMemory(void* data, size_t size_bytes)
    : data_(data), size_bytes_(size_bytes) {}

// Creates owning CPU device memory from an owned data pointer.
CpuDeviceMemory::CpuDeviceMemory(OwnedData data, size_t size_bytes)
    : data_(data.get()),
      owned_data_(std::move(data)),
      size_bytes_(size_bytes) {}

// Allocates owning memory wrapped in an available `AsyncValueRef`.
absl::StatusOr<tsl::AsyncValueRef<CpuDeviceMemory>>
CpuDeviceMemory::AllocateAvailable(size_t size_bytes) {
  TF_ASSIGN_OR_RETURN(CpuDeviceMemory memory, Allocate(size_bytes));
  return tsl::MakeAvailableAsyncValueRef<CpuDeviceMemory>(std::move(memory));
}

// Allocates raw owning memory. The typical usage is for delayed allocation.
absl::StatusOr<CpuDeviceMemory> CpuDeviceMemory::Allocate(size_t size_bytes) {
  if (void* data = tsl::port::AlignedMalloc(size_bytes, cpu::MinAlign())) {
    return CpuDeviceMemory(
        OwnedData{static_cast<uint8_t*>(data), tsl::port::AlignedFree},
        size_bytes);
  }
  return ResourceExhausted("Out of memory allocating %d bytes.", size_bytes);
}

TrackedCpuDeviceBuffer::TrackedCpuDeviceBuffer(
    bool owns_buffers, tsl::AsyncValueRef<CpuDeviceMemory> buffer,
    absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4> definition_events,
    absl::AnyInvocable<void() &&> on_delete_callback)
    : TrackedCpuDeviceBuffer(owns_buffers, std::move(buffer),
                             AfterAll(definition_events),
                             std::move(on_delete_callback)) {}

TrackedCpuDeviceBuffer::TrackedCpuDeviceBuffer(
    bool owns_buffers, tsl::AsyncValueRef<CpuDeviceMemory> buffer,
    size_t buffer_size,
    absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4> definition_events,
    absl::AnyInvocable<void() &&> on_delete_callback)
    : TrackedCpuDeviceBuffer(owns_buffers, std::move(buffer), buffer_size,
                             AfterAll(definition_events),
                             std::move(on_delete_callback)) {}

TrackedCpuDeviceBuffer::TrackedCpuDeviceBuffer(
    bool owns_buffers, tsl::AsyncValueRef<CpuDeviceMemory> buffer,
    tsl::AsyncValueRef<CpuEvent> definition_event,
    absl::AnyInvocable<void() &&> on_delete_callback)
    : owns_buffers_(owns_buffers),
      buffers_({std::move(buffer)}),
      definition_event_(std::move(definition_event)),
      on_delete_callback_(std::move(on_delete_callback)) {
  DCHECK(definition_event_);
  for (const auto& buffer : buffers_) {
    CHECK(buffer.IsConcrete());
    buffer_sizes_.push_back(buffer->size_bytes());
  }
  CHECK_EQ(buffers_.size(), 1);
}

TrackedCpuDeviceBuffer::TrackedCpuDeviceBuffer(
    bool owns_buffers, tsl::AsyncValueRef<CpuDeviceMemory> buffer,
    size_t buffer_size, tsl::AsyncValueRef<CpuEvent> definition_event,
    absl::AnyInvocable<void() &&> on_delete_callback)
    : owns_buffers_(owns_buffers),
      buffers_({std::move(buffer)}),
      buffer_sizes_({buffer_size}),
      definition_event_(std::move(definition_event)),
      on_delete_callback_(std::move(on_delete_callback)) {
  DCHECK(definition_event_);
  CHECK_EQ(buffers_.size(), 1);
}

TrackedCpuDeviceBuffer::~TrackedCpuDeviceBuffer() {
  ReleaseDeviceMemory();
  if (on_delete_callback_) {
    std::move(on_delete_callback_)();
  }
}

tsl::AsyncValuePtr<CpuDeviceMemory> TrackedCpuDeviceBuffer::Buffer(
    const ShapeIndex& shape_index) {
  CHECK(shape_index.empty());
  return buffers_[0].AsPtr();
}

size_t TrackedCpuDeviceBuffer::BufferSize() { return buffer_sizes_[0]; }

void TrackedCpuDeviceBuffer::AddUsageEvents(
    absl::Span<tsl::AsyncValueRef<CpuEvent>> events) {
  // Periodically remove available usage events to prevent memory blowup.
  if (usage_events_.size() >= 1024) {
    int i = 0;
    while (i < usage_events_.size()) {
      auto& event = usage_events_[i];
      if (event.IsAvailable()) {
        using std::swap;
        swap(event, usage_events_.back());
        usage_events_.pop_back();
        continue;
      }
      ++i;
    }
  }
  for (auto& ev : events) {
    usage_events_.push_back(std::move(ev));
  }
}

absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4>
TrackedCpuDeviceBuffer::LockUseAndTransferUsageEvents() {
  return std::move(usage_events_);
}

void TrackedCpuDeviceBuffer::ReleaseDeviceMemory() {
  buffers_.clear();
  definition_event_.reset();
  usage_events_.clear();
}

}  // namespace xla

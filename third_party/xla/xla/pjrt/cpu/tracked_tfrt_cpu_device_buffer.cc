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

#include "xla/pjrt/cpu/tracked_tfrt_cpu_device_buffer.h"

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
#include "xla/service/cpu/cpu_event.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"

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

TrackedTfrtCpuDeviceBuffer::TrackedTfrtCpuDeviceBuffer(
    bool is_tuple, bool owns_buffers,
    absl::InlinedVector<tsl::AsyncValueRef<MaybeOwningCpuMemory>, 4> buffers,
    absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4> definition_events,
    absl::AnyInvocable<void() &&> on_delete_callback)
    : TrackedTfrtCpuDeviceBuffer(is_tuple, owns_buffers, std::move(buffers),
                                 AfterAll(definition_events),
                                 std::move(on_delete_callback)) {}

TrackedTfrtCpuDeviceBuffer::TrackedTfrtCpuDeviceBuffer(
    bool is_tuple, bool owns_buffers,
    absl::InlinedVector<tsl::AsyncValueRef<MaybeOwningCpuMemory>, 4> buffers,
    absl::InlinedVector<size_t, 4> buffer_sizes,
    absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4> definition_events,
    absl::AnyInvocable<void() &&> on_delete_callback)
    : TrackedTfrtCpuDeviceBuffer(
          is_tuple, owns_buffers, std::move(buffers), std::move(buffer_sizes),
          AfterAll(definition_events), std::move(on_delete_callback)) {}

TrackedTfrtCpuDeviceBuffer::TrackedTfrtCpuDeviceBuffer(
    bool is_tuple, bool owns_buffers,
    absl::InlinedVector<tsl::AsyncValueRef<MaybeOwningCpuMemory>, 4> buffers,
    tsl::AsyncValueRef<CpuEvent> definition_event,
    absl::AnyInvocable<void() &&> on_delete_callback)
    : is_tuple_(is_tuple),
      owns_buffers_(owns_buffers),
      buffers_(std::move(buffers)),
      definition_event_(std::move(definition_event)),
      on_delete_callback_(std::move(on_delete_callback)) {
  DCHECK(definition_event_);
  for (const auto& buffer : buffers_) {
    CHECK(buffer.IsConcrete());
    buffer_sizes_.push_back(buffer->size());
  }
  if (is_tuple) {
    size_t index_table_byte_size = buffers_.size() * sizeof(void*);
    // We assume tuple table allocations will not fail.
    tuple_index_table_ =
        MaybeOwningCpuMemory::AllocateAvailableAvr(index_table_byte_size)
            .value();
    uintptr_t* index_table =
        reinterpret_cast<uintptr_t*>(tuple_index_table_->data());
    for (int i = 0; i < buffers_.size(); ++i) {
      index_table[i] = absl::bit_cast<uintptr_t>(buffers_[i]->data());
    }
  }
}

TrackedTfrtCpuDeviceBuffer::TrackedTfrtCpuDeviceBuffer(
    bool is_tuple, bool owns_buffers,
    absl::InlinedVector<tsl::AsyncValueRef<MaybeOwningCpuMemory>, 4> buffers,
    absl::InlinedVector<size_t, 4> buffer_sizes,
    tsl::AsyncValueRef<CpuEvent> definition_event,
    absl::AnyInvocable<void() &&> on_delete_callback)
    : is_tuple_(is_tuple),
      owns_buffers_(owns_buffers),
      buffers_(std::move(buffers)),
      buffer_sizes_(std::move(buffer_sizes)),
      definition_event_(std::move(definition_event)),
      on_delete_callback_(std::move(on_delete_callback)) {
  DCHECK(definition_event_);
  if (is_tuple) {
    tuple_index_table_ =
        tsl::MakeUnconstructedAsyncValueRef<MaybeOwningCpuMemory>();
    tsl::RunWhenReady(
        absl::MakeConstSpan(buffers_),
        [buffers = buffers_, tuple_index_table = tuple_index_table_] {
          size_t index_table_byte_size = buffers.size() * sizeof(void*);
          // We assume tuple table allocations will not fail.
          tuple_index_table.emplace(
              MaybeOwningCpuMemory::Allocate(index_table_byte_size).value());
          uintptr_t* index_table =
              reinterpret_cast<uintptr_t*>(tuple_index_table->data());
          for (int i = 0; i < buffers.size(); ++i) {
            index_table[i] = absl::bit_cast<uintptr_t>(buffers[i]->data());
          }
        });
  }
}

TrackedTfrtCpuDeviceBuffer::~TrackedTfrtCpuDeviceBuffer() {
  ReleaseDeviceMemory();
  if (on_delete_callback_) {
    std::move(on_delete_callback_)();
  }
}

tsl::AsyncValueRef<MaybeOwningCpuMemory> TrackedTfrtCpuDeviceBuffer::Buffer(
    const ShapeIndex& shape_index) {
  if (shape_index.empty()) {
    // shape_index={}
    if (is_tuple_) return tuple_index_table_;
    return buffers_[0];
  }
  // shape_index={i}
  CHECK(is_tuple_);
  CHECK_EQ(shape_index.size(), 1) << "nested tuple not supported";
  return buffers_[shape_index[0]];
}

size_t TrackedTfrtCpuDeviceBuffer::BufferSize(const ShapeIndex& shape_index) {
  if (shape_index.empty()) {
    // shape_index={}
    if (is_tuple_) return buffers_.size() * sizeof(void*);
    return buffer_sizes_[0];
  }
  // shape_index={i}
  CHECK(is_tuple_);
  CHECK_EQ(shape_index.size(), 1) << "nested tuple not supported";
  return buffer_sizes_[shape_index[0]];
}

void TrackedTfrtCpuDeviceBuffer::AddUsageEvents(
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
TrackedTfrtCpuDeviceBuffer::LockUseAndTransferUsageEvents() {
  return std::move(usage_events_);
}

void TrackedTfrtCpuDeviceBuffer::ReleaseDeviceMemory() {
  tuple_index_table_.reset();
  buffers_.clear();
  definition_event_.reset();
  usage_events_.clear();
}

}  // namespace xla

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
#include "xla/pjrt/gpu/tfrt/gpu_event.h"

#include <atomic>
#include <utility>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla {

// See TfrtEventSet::AfterAll for the documentation.
tsl::AsyncValueRef<GpuEvent> AfterAll(
    absl::Span<const tsl::AsyncValueRef<GpuEvent>> events) {
  if (events.empty()) return tsl::MakeAvailableAsyncValueRef<GpuEvent>();
  if (events.size() == 1) return events.front().CopyRef();

  struct State {
    State(int count, tsl::AsyncValueRef<GpuEvent> after_all)
        : count(count), after_all(std::move(after_all)) {}
    std::atomic<int> count;
    tsl::AsyncValueRef<GpuEvent> after_all;

    absl::Mutex mutex;
    absl::Status error_status;
  };

  auto after_all = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  auto* state = new State(events.size(), after_all);

  for (auto& event : events) {
    event.AndThen([state, event = event.AsPtr()]() {
      if (event.IsError()) {
        absl::MutexLock lock(&state->mutex);
        state->error_status = event.GetError();
      }

      if (state->count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        if (!state->error_status.ok()) {
          state->after_all.SetError(state->error_status);
        } else {
          state->after_all.SetStateConcrete();
        }
        delete state;
      }
    });
  }

  return after_all;
}

void TfrtEventSet::Add(tsl::AsyncValueRef<GpuEvent> event) {
  events_.push_back(std::move(event));
}

tsl::AsyncValueRef<GpuEvent> TfrtEventSet::AfterAll() {
  auto after_all = xla::AfterAll(events_);
  Clear();
  events_.push_back(after_all.CopyRef());
  return after_all;
}

void TfrtEventSet::Clear() { events_.clear(); }

}  // namespace xla

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

#include "xla/pjrt/gpu/gpu_event.h"

#include <atomic>
#include <string>
#include <utility>

#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "third_party/tf_runtime/include/tfrt/host_context/async_value_ref.h"

namespace xla {

// See TfrtEventSet::AfterAll for the documentation.
tfrt::AsyncValueRef<GpuEvent> AfterAll(
    absl::Span<const tfrt::AsyncValueRef<GpuEvent>> events) {
  if (events.empty()) return tfrt::MakeAvailableAsyncValueRef<GpuEvent>();
  if (events.size() == 1) return events.front().CopyRef();

  struct State {
    State(int count, tfrt::AsyncValueRef<GpuEvent> after_all)
        : count(count), after_all(std::move(after_all)) {}
    std::atomic<int> count;
    tfrt::AsyncValueRef<GpuEvent> after_all;

    absl::Mutex mutex;
    std::string error_message;
  };

  auto after_all = tfrt::MakeConstructedAsyncValueRef<GpuEvent>();
  auto* state = new State(events.size(), after_all);

  for (auto& event : events) {
    event.AndThen([state, event = event.AsPtr()]() {
      if (event.IsError()) {
        absl::MutexLock lock(&state->mutex);
        state->error_message = event.GetError().message();
      }

      if (state->count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        if (!state->error_message.empty()) {
          state->after_all.SetError(event.GetError());
        } else {
          state->after_all.SetStateConcrete();
        }
        delete state;
      }
    });
  }

  return after_all;
}

TfrtEventSet::TfrtEventSet() = default;

void TfrtEventSet::Add(tfrt::AsyncValueRef<GpuEvent> event) {
  events_.push_back(std::move(event));
}

tfrt::AsyncValueRef<GpuEvent> TfrtEventSet::AfterAll() {
  auto after_all = xla::AfterAll(events_);
  events_.clear();
  events_.push_back(after_all.CopyRef());
  return after_all;
}

void TfrtEventSet::Clear() { events_.clear(); }

}  // namespace xla

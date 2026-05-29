/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PJRT_DEVICE_EVENT_UTILS_H_
#define XLA_PJRT_DEVICE_EVENT_UTILS_H_

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/pjrt/device_event.h"
#include "xla/tsl/concurrency/executor.h"

namespace xla {

// Runs a task in an optional executor after a sequence of dependency events.
class ScopedLauncher {
 public:
  explicit ScopedLauncher(tsl::Executor::Task task,
                          tsl::Executor* executor = nullptr);
  ~ScopedLauncher();
  ScopedLauncher(const ScopedLauncher&) = delete;
  ScopedLauncher(ScopedLauncher&&) = delete;

  void AddDependency(PjRtDeviceEventPtr dependency);

  void AddDependency(const PjRtDeviceEventRef& dependency) {
    AddDependency(dependency.ptr());
  }

  void AddDependency(tsl::AsyncValue* dependency);

  void AddDependency(absl::Span<const PjRtDeviceEventRef> dependencies);

 private:
  class Callee;
  tsl::RCReference<Callee> state_;
  tsl::Executor::Task task_;
  tsl::Executor* executor_ = nullptr;
};

// Executes `task` on `executor` when all `events` are ready.
inline void ExecuteWhenReady(absl::Span<const PjRtDeviceEventRef> events,
                             tsl::Executor* executor,
                             tsl::Executor::Task task) {
  ScopedLauncher launcher_(std::move(task), executor);
  for (auto& dep : events) {
    launcher_.AddDependency(dep);
  }
}

// Runs `callback` when all `events` are ready.
inline void RunWhenReady(absl::Span<const PjRtDeviceEventRef> events,
                         tsl::Executor::Task task) {
  ExecuteWhenReady(events, nullptr, std::move(task));
}

// Returns an error if any of the `events` have an error.
// Does not wait for events to become ready.
absl::Status GetErrors(absl::Span<const PjRtDeviceEventRef> events);

void BlockUntilReady(PjRtDeviceEventPtr event);

inline void BlockUntilReady(const PjRtDeviceEventRef& event) {
  BlockUntilReady(event.ptr());
}

}  // namespace xla

#endif  // XLA_PJRT_DEVICE_EVENT_UTILS_H_

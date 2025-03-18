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

#ifndef XLA_PJRT_GPU_TFRT_GPU_EVENT_H_
#define XLA_PJRT_GPU_TFRT_GPU_EVENT_H_

#include <cstddef>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla {

// tsl::AsyncValueRef<GpuEvent> is used to indicate the completion of a GPU
// operation, e.g., data transfer or running a program.
// TODO(b/400996007) :make GpuEvent contain CudaEvent.
struct GpuEvent {
  GpuEvent() = default;
};

// Returns an AsyncValueRef<GpuEvent> that will be ready after all the async
// values in `events` are ready. If errors occurs, one of the errors will be
// propagated through the returned async value.
tsl::AsyncValueRef<GpuEvent> AfterAll(
    absl::Span<const tsl::AsyncValueRef<GpuEvent>> events);

// Represents a set of TFRT events. Not thread-safe.
class TfrtEventSet {
 public:
  TfrtEventSet() = default;
  TfrtEventSet(const TfrtEventSet&) = delete;
  TfrtEventSet(TfrtEventSet&&) = delete;
  TfrtEventSet& operator=(const TfrtEventSet&) = delete;
  TfrtEventSet& operator=(TfrtEventSet&&) = delete;

  // Adds an event to the set. Periodically, events that have already been
  // triggered will be removed from the set.
  void Add(tsl::AsyncValueRef<GpuEvent> event);

  // Returns an AsyncValueRef<GpuEvent> that will be ready after all the async
  // values in `events` are ready. If errors occurs, one of the errors will be
  // propagated through the returned async value.
  tsl::AsyncValueRef<GpuEvent> AfterAll();

  size_t size() const { return events_.size(); }

  void Clear();

 private:
  absl::InlinedVector<tsl::AsyncValueRef<GpuEvent>, 4> events_;
};

// A RAII helper class used to set an AsyncValueRef<GpuEvent> to a ready state
// upon destruction. In many cases in PjRt implementation, there will be
// multiple return statements in the function, all of which require setting
// some AsyncValueRef<GpuEvent> to be ready. This class could make such code
// more robust by using setting the AsyncValue in the destructor.
class MarkEventReadyOnExit {
 public:
  explicit MarkEventReadyOnExit(tsl::AsyncValueRef<GpuEvent> event)
      : event_(std::move(event)) {}

  MarkEventReadyOnExit(const MarkEventReadyOnExit&) = delete;
  MarkEventReadyOnExit& operator=(const MarkEventReadyOnExit&) = delete;
  MarkEventReadyOnExit(MarkEventReadyOnExit&&) = default;
  MarkEventReadyOnExit& operator=(MarkEventReadyOnExit&&) = default;

  ~MarkEventReadyOnExit() {
    if (event_) event_.SetStateConcrete();
  }

  tsl::AsyncValueRef<GpuEvent> Release() && { return std::move(event_); }

 private:
  tsl::AsyncValueRef<GpuEvent> event_;
};

}  // namespace xla

#endif  // XLA_PJRT_GPU_TFRT_GPU_EVENT_H_

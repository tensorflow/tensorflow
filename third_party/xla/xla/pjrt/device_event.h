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

#ifndef XLA_PJRT_DEVICE_EVENT_H_
#define XLA_PJRT_DEVICE_EVENT_H_

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {

// A common base class between events and promises that allow adding extra
// metadata.
class PjRtDeviceEventOrPromise
    : public tsl::ReferenceCounted<PjRtDeviceEventOrPromise> {
 public:
  virtual ~PjRtDeviceEventOrPromise() = default;

  // If this event is based on async-value, return it.
  virtual tsl::AsyncValue* async_value() { return nullptr; }

  // If this event type supports tracking, add tracking information.
  virtual void AppendDescriptionToEvent(
      absl::string_view description,
      absl::Span<PjRtDeviceEventOrPromise* const> waiters) {}

  // If this event type supports tracking, add dependency async values.
  virtual void AddEventDependencies(
      absl::Span<const tsl::RCReference<tsl::AsyncValue>> dependencies) {}

  // If this event type supports tracking, report that a thread is waiting.
  virtual void RegisterClientThreadWait(absl::string_view description) {}
};

// A device event occurs (potentially) on a device. It can be waited on
// directly or passed between APIs which may be able to handle these events
// directly.
class PjRtDeviceEvent : public PjRtDeviceEventOrPromise {
 public:
  ~PjRtDeviceEvent() override = default;

  enum class State {
    kPending,
    kReady,
    kError,
  };

  // Runs a callback when an event becomes ready.
  virtual void AndThen(absl::AnyInvocable<void() &&> cb) = 0;

  // Polls current event state.
  virtual State state() const = 0;

  // Check if ready.
  bool ok() const { return state() != State::kError; }

  // Fetches the error if this event is in state kError.
  virtual const absl::Status& status() const = 0;

  // Converts a device-event into a future.
  virtual PjRtFuture<> GetReadyFuture() = 0;
};

// Instead of taking a device event as an argument, apis may instead decide to
// return a promise which is fulfilled later.
class PjRtDeviceEventPromise : public PjRtDeviceEventOrPromise {
 public:
  ~PjRtDeviceEventPromise() override = default;

  // Fulfill the promise.
  virtual void Set(tsl::RCReference<PjRtDeviceEvent> event) = 0;

  // Mark the promise as an error.
  virtual void SetError(absl::Status s) = 0;

  // Mark the event as ready.
  virtual void SetReady() = 0;
};

}  // namespace xla

#endif  // XLA_PJRT_DEVICE_EVENT_H_

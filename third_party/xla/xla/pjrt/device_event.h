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
#include "xla/future.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {

// A device event occurs (potentially) on a device. It can be waited on
// directly or passed between APIs which may be able to handle these events
// directly.
class PjRtDeviceEvent : public tsl::ReferenceCounted<PjRtDeviceEvent> {
 public:
  virtual ~PjRtDeviceEvent() = default;

  // Runs a callback when an event becomes ready.
  template <typename Waiter>
  void AndThen(Waiter&& cb) {
    async_value()->AndThen(std::forward<Waiter>(cb));
  }

  // The underlying AsyncValue.
  virtual tsl::AsyncValue* async_value() const = 0;
};

// Instead of taking a device event as an argument, apis may instead decide to
// return a promise which is fulfilled later.
class PjRtDeviceEventPromise
    : public tsl::ReferenceCounted<PjRtDeviceEventPromise> {
 public:
  virtual ~PjRtDeviceEventPromise() = default;

  // The underlying AsyncValue.
  virtual tsl::AsyncValue* async_value() const = 0;

  // Fulfill the promise.
  virtual void Set(tsl::RCReference<PjRtDeviceEvent> event) = 0;

  // Mark the promise as an error.
  virtual void SetError(absl::Status s) = 0;

  // Mark the event as ready.
  virtual void SetReady() = 0;
};

// A collection of events. This is not an event itself because we may want to
// add events in the future.
class PjRtDeviceEventSet {
 public:
  virtual ~PjRtDeviceEventSet() = default;
};

}  // namespace xla

#endif  // XLA_PJRT_DEVICE_EVENT_H_

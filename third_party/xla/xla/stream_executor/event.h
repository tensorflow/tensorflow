/* Copyright 2015 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_EVENT_H_
#define XLA_STREAM_EXECUTOR_EVENT_H_

#include <cstdint>

#include "absl/status/status.h"

namespace stream_executor {

// The Event class, when supported by a platform, enables low-overhead status
// reporting for a Stream. An Event is inserted at a location in a stream via
// the Stream::RecordEvent() API. From then on, the Event's status can be
// monitored via the nonblocking Event::PollForStatus() call.
class Event {
 public:
  // Potential states for an Event. If PollForStatus() returns anything aside
  // from kPending or kComplete, an error has occurred; kUnknown is a bad state.
  // Not all implementations are able to return all enumeration values. Refer to
  // the platform-specific implementation for details.
  enum class Status {
    kUnknown,
    kError,
    kPending,
    kComplete,
  };

  // Releases any resources held by the Event object.
  virtual ~Event() = default;

  // Returns the current Status for the event.
  virtual Status PollForStatus() { return Status::kError; }

  // Blocks `stream` on this event. `stream` is a raw platform-specific
  // stream (e.g. GpuStreamHandle).
  virtual absl::Status WaitForEventOnExternalStream(std::intptr_t stream) {
    return absl::UnimplementedError("Not supported for this Event.");
  }
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_EVENT_H_

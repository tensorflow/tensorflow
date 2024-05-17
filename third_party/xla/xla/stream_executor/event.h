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
#include <memory>

#include "absl/status/status.h"

namespace stream_executor {

class EventInterface;
class StreamExecutorInterface;

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

  explicit Event(StreamExecutorInterface* stream_exec);  // NOLINT

  // Releases any resources held by the Event object.
  ~Event();

  // Performs any platform-specific or potentially error-generating
  // initialization.
  bool Init();

  // Returns the current Status for the event.
  Status PollForStatus();

  // Blocks `stream` on this event. `stream` is a raw platform-specific
  // stream (e.g. GpuStreamHandle).
  absl::Status WaitForEventOnExternalStream(std::intptr_t stream);

  // Returns a pointer to the underlying platform-specific implementation.
  EventInterface* implementation() { return implementation_.get(); }

  Event(Event&&);
  Event& operator=(Event&&);

 private:
  // Pointer to the StreamExecutorInterface interface used to create this
  // object. Not owned.
  StreamExecutorInterface* stream_exec_;

  // Pointer to the platform-specific EventInterface implementation underlying
  // the object. Owned.
  std::unique_ptr<EventInterface> implementation_;

  Event(const Event&) = delete;
  void operator=(const Event&) = delete;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_EVENT_H_

/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_STREAM_EXECUTOR_EVENT_H_
#define TENSORFLOW_STREAM_EXECUTOR_EVENT_H_

#include <memory>

#include "tensorflow/stream_executor/platform/port.h"

namespace perftools {
namespace gputools {

namespace internal {
class EventInterface;
}

class Stream;
class StreamExecutor;

// The Event class, when supported by a platform, enables low-overhead status
// reporting for a Stream. An Event is inserted at a location in a stream via
// the Stream::ThenRecordEvent() API. From then on, the Event's status can be
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

  explicit Event(StreamExecutor* stream_exec);  // NOLINT

  // Releases any resources held by the Event object.
  ~Event();

  // Performs any platform-specific or potentially error-generating
  // initialization.
  bool Init();

  // Returns the current Status for the event.
  Status PollForStatus();

  // Returns a pointer to the underlying platform-specific implementation.
  internal::EventInterface* implementation() { return implementation_.get(); }

 private:
  friend class Stream;

  // Pointer to the StreamExecutor interface used to create this object.
  // Not owned.
  StreamExecutor* stream_exec_;

  // Pointer to the platform-specific EventInterface implementation underlying
  // the object. Owned.
  std::unique_ptr<internal::EventInterface> implementation_;

  SE_DISALLOW_COPY_AND_ASSIGN(Event);
};

}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_EVENT_H_

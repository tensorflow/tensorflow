/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_EVENT_BASED_TIMER_H_
#define XLA_STREAM_EXECUTOR_EVENT_BASED_TIMER_H_

#include "absl/status/statusor.h"
#include "absl/time/time.h"

namespace stream_executor {

// This class defines an interface for an Event-based timer.  It allows the
// timing via Events from the creation of an EventBasedTimer to some arbitrary
// later point when the GetElapsedDuration method is called.
class EventBasedTimer {
 public:
  virtual ~EventBasedTimer() = default;

  // Stops the timer on the first call and returns the elapsed duration.
  // Subsequent calls error out.
  virtual absl::StatusOr<absl::Duration> GetElapsedDuration() = 0;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_EVENT_BASED_TIMER_H_

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_EVENT_SPAN_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_EVENT_SPAN_H_

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/utils/timespan.h"

namespace tensorflow {
namespace profiler {

// The various event types. Enumerations are numbered such that a bigger number
// has a higher priority than a smaller number when used in execution-time
// breakdown.
enum EventType {
  // Invalid type.
  INVALID_BREAKDOWN_TYPE = 0,
  // Both host and device are idle.
  BOTH_IDLE = 1,
  // Host is compiling.
  HOST_COMPILE = 2,
  // Host-to-host communication.
  HOST_TO_HOST = 3,
  // Host-to-device communication.
  HOST_TO_DEVICE = 4,
  // Host is preparing to launch a computation on device.
  HOST_PREPARE = 5,
  // Host is computing.
  HOST_COMPUTE = 6,
  // Host is waiting for input.
  HOST_WAIT_INPUT = 7,
  // Device-to-device communication.
  DEVICE_TO_DEVICE = 8,
  // Device-to-host communication.
  DEVICE_TO_HOST = 9,
  // Device is computing.
  DEVICE_COMPUTE = 10,
  // Device is waiting for another device.
  DEVICE_WAIT_DEVICE = 11,
  // Device is waiting for host.
  DEVICE_WAIT_HOST = 12
};

struct EventTypeSpan {
  EventType type;  // type of this event.
  Timespan span;   // timespan of this event.
  EventTypeSpan(EventType t, Timespan s) : type(t), span(s) {}
};

// Map from step-number to the events happened in that step.
using StepEvents =
    absl::flat_hash_map<int64 /*step number*/, std::vector<EventTypeSpan>>;

// Returns the event type of the given CPU event.
EventType ClassifyCpuEvent(absl::string_view event_name, int64 correlation_id);

// Returns the event type of the given GPU event.
EventType ClassifyGpuEvent(absl::string_view event_name);

// Combines the src StepEvents into dst.
void CombineStepEvents(const StepEvents& src, StepEvents* dst);

// Converts from overlapped step-events to non-overlapped step-events.
StepEvents ToNonOverlappedStepEvents(const StepEvents& overlapped_step_events);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_EVENT_SPAN_H_

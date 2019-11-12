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

#include "tensorflow/core/profiler/utils/event_span.h"

#include "absl/strings/match.h"

namespace tensorflow {
namespace profiler {

EventType ClassifyGpuEvent(absl::string_view event_name) {
  if (absl::StartsWithIgnoreCase(event_name, "MEMCPYHtoD"))
    return HOST_TO_DEVICE;
  if (absl::StartsWithIgnoreCase(event_name, "MEMCPYDtoH"))
    return DEVICE_TO_HOST;
  if (absl::StartsWithIgnoreCase(event_name, "MEMCPYDtoD"))
    return DEVICE_TO_DEVICE;
  return DEVICE_COMPUTE;
}

EventType ClassifyCpuEvent(absl::string_view event_name, int64 correlation_id) {
  if (absl::StartsWithIgnoreCase(event_name, "MEMCPYHtoD"))
    return HOST_TO_DEVICE;
  if (absl::StartsWithIgnoreCase(event_name, "MEMCPYHtoH")) return HOST_TO_HOST;
  if (correlation_id >= 0 ||
      absl::StartsWithIgnoreCase(event_name, "ExecutorState::Process")) {
    return HOST_PREPARE;
  } else {
    if (absl::StartsWithIgnoreCase(event_name, "IteratorGetNext"))
      return HOST_WAIT_INPUT;
    return HOST_COMPUTE;
  }
}

void CombineStepEvents(const StepEvents& src, StepEvents* dst) {
  for (const auto& step_events : src) {
    int64 step_number = step_events.first;
    const std::vector<EventTypeSpan>& src_events = step_events.second;
    std::vector<EventTypeSpan>& dst_events = (*dst)[step_number];
    // Simply appends src_events to dst_events.
    dst_events.insert(dst_events.end(), src_events.begin(), src_events.end());
  }
}

// Converts from overlapped step-events to non-overlapped step-events.
StepEvents ToNonOverlappedStepEvents(const StepEvents& overlapped_step_events) {
  // TODO(ckluk): Implement this function.
  return overlapped_step_events;
}

}  // namespace profiler
}  // namespace tensorflow

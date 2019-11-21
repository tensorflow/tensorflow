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

#include <thread>  // NOLINT
#include <vector>

#include "absl/strings/match.h"

namespace tensorflow {
namespace profiler {

namespace {

// Converts from overlapped events to non-overlapped events.
std::vector<EventTypeSpan> ToNonOverlappedEvents(
    const std::vector<EventTypeSpan>& overlapped_events) {
  // TODO(ckluk): Implement this function.
  return overlapped_events;
}

}  // namespace.

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
  size_t num_steps = overlapped_step_events.size();
  std::vector<std::thread> workers;
  workers.resize(num_steps);
  std::vector<int64> step_numbers;
  step_numbers.resize(num_steps);
  std::vector<std::vector<EventTypeSpan>> vec;
  vec.resize(num_steps);
  int64 i = 0;
  // Sets up 1 worker per step to convert overlapped events to non-overlapped
  // events.
  for (const auto& step_events : overlapped_step_events) {
    step_numbers[i] = step_events.first;
    const std::vector<EventTypeSpan>* overlapped_events = &step_events.second;
    std::vector<EventTypeSpan>* non_overlapped_events = &vec[i];
    workers[i] = std::thread([overlapped_events, non_overlapped_events]() {
      *non_overlapped_events = ToNonOverlappedEvents(*overlapped_events);
    });
    i += 1;
  }
  // Runs the workers in parallel.
  std::for_each(workers.begin(), workers.end(),
                [](std::thread& t) { t.join(); });
  StepEvents non_overlapped_step_events;
  // Moves non-overlapped events to the corresponding step in the map.
  for (i = 0; i < step_numbers.size(); i++) {
    non_overlapped_step_events[step_numbers[i]] = std::move(vec[i]);
  }
  return non_overlapped_step_events;
}

}  // namespace profiler
}  // namespace tensorflow

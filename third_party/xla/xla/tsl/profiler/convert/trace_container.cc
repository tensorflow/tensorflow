/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "xla/tsl/profiler/convert/trace_container.h"

#include <algorithm>
#include <string>
#include <vector>

#include "tsl/platform/protobuf.h"

namespace tsl {
namespace profiler {

bool TraceContainer::ParseMetadataFromString(const std::string& description) {
  return protobuf::TextFormat::ParseFromString(description, &metadata_);
}

void TraceContainer::CapEvents(const uint32_t max_count) {
  const size_t total_count = events_.size();

  if (total_count <= max_count) {
    // Nothing to do. Events are not known sorted after return.
    return;
  }

  // Partially sort the events according to start time.
  const std::vector<TraceEvent*>::iterator end = events_.begin() + max_count;
  std::partial_sort(
      events_.begin(), end, events_.end(),
      [](const TraceEvent* const lhs, const TraceEvent* const rhs) -> bool {
        return lhs->timestamp_ps() < rhs->timestamp_ps();
      });
  for (std::vector<TraceEvent*>::iterator i = end; i != events_.end(); ++i) {
    delete *i;
  }
  events_.erase(end, events_.end());
  // All events are known sorted here.
}

void TraceContainer::FlushAndSerializeEvents(std::string* const output) {
  Trace trace = metadata_;
  for (TraceEvent* const event : events_) {
    trace.mutable_trace_events()->AddAllocated(event);
  }
  // Ownership was transferred to the `trace` message. since
  // the container assumes it owns all the events in its storage
  // buffer, we must clear the buffer, to prevent overreleasing.
  events_.clear();
  trace.SerializeToString(output);
}

}  // namespace profiler
}  // namespace tsl

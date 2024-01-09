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
#include "tensorflow/core/profiler/convert/trace_viewer/trace_events_util.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {

// Functor that compares flow events for sorting.
struct FlowEventsComparator {
  bool operator()(const TraceEvent* a, const TraceEvent* b) const {
    if (a->timestamp_ps() < b->timestamp_ps()) return true;
    if (a->timestamp_ps() > b->timestamp_ps()) return false;
    return (a->flow_entry_type() < b->flow_entry_type());
  }
};

std::vector<TraceEventFlow> SplitEventFlow(TraceEventFlow&& flow) {
  std::vector<TraceEventFlow> flows;
  absl::c_sort(flow, FlowEventsComparator());
  TraceEventFlow* current = nullptr;
  for (TraceEvent* event : flow) {
    if (current == nullptr ||
        event->flow_entry_type() == TraceEvent::FLOW_START) {
      current = &flows.emplace_back();
    }
    current->push_back(event);
    if (event->flow_entry_type() == TraceEvent::FLOW_END) {
      current = nullptr;
    }
  }
  return flows;
}

void ExpandTraceSpan(const tsl::profiler::Timespan& span, Trace* trace) {
  if (!trace->has_min_timestamp_ps() ||
      span.begin_ps() < trace->min_timestamp_ps()) {
    trace->set_min_timestamp_ps(span.begin_ps());
  }
  if (!trace->has_max_timestamp_ps() ||
      span.end_ps() > trace->max_timestamp_ps()) {
    trace->set_max_timestamp_ps(span.end_ps());
  }
}

class DefaultResourceGrouper : public ResourceGrouperInterface {
 public:
  explicit DefaultResourceGrouper(uint32_t device_id, absl::string_view name)
      : device_id_(device_id), name_(name) {}

  std::vector<std::pair<uint32_t, absl::string_view>> Devices() const override {
    return {{device_id_, name_}};
  }

  uint32_t GetDeviceId(uint32_t resource_id) const override {
    return device_id_;
  }

 private:
  uint32_t device_id_;
  absl::string_view name_;
};

std::unique_ptr<ResourceGrouperInterface> CreateDefaultResourceGrouper(
    uint32_t device_id, absl::string_view name) {
  return std::make_unique<DefaultResourceGrouper>(device_id, name);
}

}  // namespace profiler
}  // namespace tensorflow

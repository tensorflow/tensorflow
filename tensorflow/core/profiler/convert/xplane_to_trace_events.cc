/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/convert/xplane_to_trace_events.h"

#include "tensorflow/core/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"

namespace tensorflow {
namespace profiler {

namespace {

Device BuildDeviceAndResource(const XPlaneVisitor& plane) {
  Device device;
  device.set_name(std::string(plane.Name()));
  device.set_device_id(plane.Id());
  plane.ForEachLine([&](const XLineVisitor& line) {
    Resource resource;
    resource.set_resource_id(line.Id());
    resource.set_name(std::string(line.Name()));
    (*device.mutable_resources())[line.Id()] = resource;
  });
  return device;
}

}  // namespace

void MaybeDropEventsForTraceViewer(Trace* trace, uint32 limit) {
  auto* trace_events = trace->mutable_trace_events();
  size_t trace_event_size = trace_events->size();
  if (trace_event_size <= limit) return;  // Nothing to do.
  // Sort the events according to start time.
  std::vector<uint64> timestamps;
  timestamps.reserve(trace_event_size);
  for (const auto& event : *trace_events) {
    timestamps.push_back(event.timestamp_ps());
  }
  std::partial_sort(timestamps.begin(), timestamps.begin() + limit,
                    timestamps.end(), std::less<uint64>());
  uint64 cutoff_timestamp = timestamps[limit - 1];
  trace_events->erase(std::remove_if(trace_events->begin(), trace_events->end(),
                                     [&](const TraceEvent& event) {
                                       return event.timestamp_ps() >
                                              cutoff_timestamp;
                                     }),
                      trace_events->end());
}

void ConvertXSpaceToTraceEvents(const XSpace& xspace, Trace* trace) {
  auto* trace_devices = trace->mutable_devices();

  for (const auto& raw_plane : xspace.planes()) {
    XPlaneVisitor xplane = CreateTfXPlaneVisitor(&raw_plane);
    // Convert devices and resources.
    int64 device_id = xplane.Id();
    (*trace_devices)[device_id] = BuildDeviceAndResource(xplane);

    // Convert events.
    xplane.ForEachLine([&](const XLineVisitor& xline) {
      int64 resource_id = xline.Id();  // Either thread id or CUDA stream id.
      xline.ForEachEvent([&](const XEventVisitor& xevent) {
        auto* event = trace->add_trace_events();
        auto& args = *event->mutable_args();
        event->set_device_id(device_id);
        event->set_resource_id(resource_id);
        if (xevent.HasDisplayName()) {
          event->set_name(string(xevent.DisplayName()));
          args["long_name"] = string(xevent.Name());
        } else {
          event->set_name(string(xevent.Name()));
        }
        event->set_timestamp_ps(xevent.TimestampPs());
        event->set_duration_ps(xevent.DurationPs());

        xevent.ForEachStat([&](const XStatVisitor& stat) {
          if (stat.ValueCase() == XStat::VALUE_NOT_SET) return;
          if (IsInternalStat(stat.Type())) return;
          args[string(stat.Name())] = stat.ToString();
        });
      });
    });
  }

  // Trace viewer (non-streaming) has scalability issues, we need to drop
  // events to avoid loading failure for trace viewer.
  constexpr uint64 kMaxEvents = 1000000;
  MaybeDropEventsForTraceViewer(trace, kMaxEvents);
}

}  // namespace profiler
}  // namespace tensorflow

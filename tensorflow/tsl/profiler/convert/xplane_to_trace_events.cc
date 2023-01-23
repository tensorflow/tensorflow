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

#include "tensorflow/tsl/profiler/convert/xplane_to_trace_events.h"

#include <stddef.h>

#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/tsl/platform/types.h"
#include "tensorflow/tsl/profiler/protobuf/trace_events.pb.h"
#include "tensorflow/tsl/profiler/protobuf/xplane.pb.h"
#include "tensorflow/tsl/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/tsl/profiler/utils/trace_utils.h"
#include "tensorflow/tsl/profiler/utils/xplane_schema.h"
#include "tensorflow/tsl/profiler/utils/xplane_utils.h"
#include "tensorflow/tsl/profiler/utils/xplane_visitor.h"

namespace tsl {
namespace profiler {

namespace {

using tensorflow::profiler::Device;
using tensorflow::profiler::Resource;
using tensorflow::profiler::Trace;
using tensorflow::profiler::TraceEvent;
using tensorflow::profiler::XSpace;

void BuildDeviceAndResources(uint32 device_id, const XPlaneVisitor& plane,
                             Device* device) {
  device->set_name(std::string(plane.Name()));
  device->set_device_id(device_id);

  bool sort_by_ordinal = (device_id == kHostThreadsDeviceId);
  int ordinal = 0;
  plane.ForEachLine([&](const XLineVisitor& line) {
    uint32 resource_id = line.DisplayId();
    Resource& resource = (*device->mutable_resources())[resource_id];
    resource.set_resource_id(resource_id);
    resource.set_name(std::string(line.DisplayName()));
    if (sort_by_ordinal) {
      // When sort_index is absent (i.e. 0), resource id will be used.
      // Therefore sort_index starts with 1.
      resource.set_sort_index(++ordinal);
    }
  });
}

void ConvertXPlaneToTraceEvents(uint32 device_id, const XPlaneVisitor& xplane,
                                Trace* trace) {
  // Convert devices and resources.
  BuildDeviceAndResources(device_id, xplane,
                          &(*trace->mutable_devices())[device_id]);

  // Convert events.
  xplane.ForEachLine([device_id, trace](const XLineVisitor& xline) {
    uint32 resource_id = xline.DisplayId();
    if (xline.DisplayName() == tsl::profiler::kXlaAsyncOpLineName) {
      return;
    }
    xline.ForEachEvent(
        [device_id, resource_id, trace](const XEventVisitor& xevent) {
          int64_t event_type =
              xevent.Type().value_or(HostEventType::kUnknownHostEventType);
          if (IsInternalEvent(event_type)) return;
          auto* event = trace->add_trace_events();
          auto& args = *event->mutable_args();
          event->set_device_id(device_id);
          event->set_resource_id(resource_id);
          if (xevent.HasDisplayName()) {
            event->set_name(std::string(xevent.DisplayName()));
            args["long_name"] = std::string(xevent.Name());
          } else {
            event->set_name(std::string(xevent.Name()));
          }
          event->set_timestamp_ps(xevent.TimestampPs());
          event->set_duration_ps(xevent.DurationPs());

          auto for_each_stat = [&](const XStatVisitor& stat) {
            if (stat.ValueCase() == XStat::VALUE_NOT_SET) return;
            if (IsInternalStat(stat.Type())) return;
            if (stat.Type() == StatType::kStepName) {
              event->set_name(stat.ToString());
            }
            args[std::string(stat.Name())] = stat.ToString();
          };
          // The metadata stats should appear before the per-occurrence stats.
          xevent.Metadata().ForEachStat(for_each_stat);
          xevent.ForEachStat(for_each_stat);
        });
  });
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

uint64 GetTraceViewerMaxEvents() {
  constexpr uint64 kMaxEvents = 1000000;
  // Testing only env variable, not recommended for use
  char* max_events = getenv("TF_PROFILER_TRACE_VIEWER_MAX_EVENTS");
  if (max_events != nullptr) {
    return std::stoull(max_events, nullptr, 10);
  } else {
    return kMaxEvents;
  }
}

void ConvertXSpaceToTraceEvents(const XSpace& xspace, Trace* trace) {
  const XPlane* host_plane = FindPlaneWithName(xspace, kHostThreadsPlaneName);
  if (host_plane != nullptr) {
    XPlaneVisitor xplane = CreateTfXPlaneVisitor(host_plane);
    ConvertXPlaneToTraceEvents(kHostThreadsDeviceId, xplane, trace);
  }
  std::vector<const XPlane*> device_planes =
      FindPlanesWithPrefix(xspace, kGpuPlanePrefix);
  // We don't expect GPU and TPU planes and custom devices to be present in the
  // same XSpace.
  if (device_planes.empty()) {
    device_planes = FindPlanesWithPrefix(xspace, kTpuPlanePrefix);
  }
  if (device_planes.empty()) {
    device_planes = FindPlanesWithPrefix(xspace, kCustomPlanePrefix);
  }
  for (const XPlane* device_plane : device_planes) {
    XPlaneVisitor xplane = CreateTfXPlaneVisitor(device_plane);
    uint32 device_id = kFirstDeviceId + xplane.Id();
    ConvertXPlaneToTraceEvents(device_id, xplane, trace);
  }

  // Trace viewer (non-streaming) has scalability issues, we need to drop
  // events to avoid loading failure for trace viewer.
  uint64 viewer_max_events = GetTraceViewerMaxEvents();
  MaybeDropEventsForTraceViewer(trace, viewer_max_events);
}

void ConvertXSpaceToTraceEventsString(const XSpace& xspace,
                                      std::string* content) {
  Trace trace;
  ConvertXSpaceToTraceEvents(xspace, &trace);
  trace.SerializeToString(content);
}

}  // namespace profiler
}  // namespace tsl

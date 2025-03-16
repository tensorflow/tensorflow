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

#include "xla/tsl/profiler/convert/xplane_to_trace_events.h"

#include <stddef.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/tsl/platform/types.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/trace_utils.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/trace_events.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {

namespace {

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
                                TraceContainer& container) {
  // Convert devices and resources.
  BuildDeviceAndResources(device_id, xplane,
                          container.MutableDevice(device_id));

  // Convert events.
  xplane.ForEachLine([device_id, &container](const XLineVisitor& xline) {
    uint32 resource_id = xline.DisplayId();
    if (xline.DisplayName() == tsl::profiler::kXlaAsyncOpLineName) {
      return;
    }
    xline.ForEachEvent(
        [device_id, resource_id, &container](const XEventVisitor& xevent) {
          int64_t event_type =
              xevent.Type().value_or(HostEventType::kUnknownHostEventType);
          if (IsInternalEvent(event_type)) return;
          TraceEvent* event = container.CreateEvent();
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

TraceContainer ConvertXSpaceToTraceContainer(const XSpace& xspace) {
  TraceContainer container;
  const XPlane* host_plane = FindPlaneWithName(xspace, kHostThreadsPlaneName);
  if (host_plane != nullptr) {
    XPlaneVisitor xplane = CreateTfXPlaneVisitor(host_plane);
    ConvertXPlaneToTraceEvents(kHostThreadsDeviceId, xplane, container);
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
    ConvertXPlaneToTraceEvents(device_id, xplane, container);
  }
  // Trace viewer (non-streaming) has scalability issues, we need to drop
  // events to avoid loading failure for trace viewer.
  uint64 viewer_max_events = GetTraceViewerMaxEvents();
  container.CapEvents(viewer_max_events);
  return container;
}

void ConvertXSpaceToTraceEventsString(const XSpace& xspace,
                                      std::string* content) {
  ConvertXSpaceToTraceContainer(xspace).FlushAndSerializeEvents(content);
}

}  // namespace profiler
}  // namespace tsl

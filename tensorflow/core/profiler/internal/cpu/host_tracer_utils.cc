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
#include "tensorflow/core/profiler/internal/cpu/host_tracer_utils.h"

#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/profiler/internal/parse_annotation.h"
#include "tensorflow/core/profiler/internal/traceme_recorder.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"

namespace tensorflow {
namespace profiler {

void MakeCompleteEvents(TraceMeRecorder::Events* events) {
  // Track events created by ActivityStart and copy their data to events created
  // by ActivityEnd. TraceMe records events in its destructor, so this results
  // in complete events sorted by their end_time in the thread they ended.
  // Within the same thread, the record created by ActivityStart must appear
  // before the record created by ActivityEnd. Cross-thread events must be
  // processed in a separate pass. A single map can be used because the
  // activity_id is globally unique.
  absl::flat_hash_map<uint64, TraceMeRecorder::Event*> start_events;
  std::vector<TraceMeRecorder::Event*> end_events;
  for (auto& thread : *events) {
    for (auto& event : thread.events) {
      if (IsStartEvent(event)) {
        start_events.emplace(event.activity_id, &event);
      } else if (IsEndEvent(event)) {
        auto iter = start_events.find(event.activity_id);
        if (iter != start_events.end()) {  // same thread
          auto* start_event = iter->second;
          event.name = std::move(start_event->name);
          event.start_time = start_event->start_time;
          start_events.erase(iter);
        } else {  // cross-thread
          end_events.push_back(&event);
        }
      }
    }
  }
  for (auto* event : end_events) {  // cross-thread
    auto iter = start_events.find(event->activity_id);
    if (iter != start_events.end()) {
      auto* start_event = iter->second;
      event->name = std::move(start_event->name);
      event->start_time = start_event->start_time;
      start_events.erase(iter);
    }
  }
}

void ConvertCompleteEventsToXPlane(uint64 start_timestamp_ns,
                                   const TraceMeRecorder::Events& events,
                                   XPlane* raw_plane) {
  XPlaneBuilder xplane(raw_plane);
  absl::flat_hash_map<string, XEventMetadata*> xevent_metadata_by_name;
  absl::flat_hash_map<string, XStatMetadata*> xstat_metadata_by_name;
  for (const auto& thread : events) {
    XLineBuilder xline = xplane.AddLine();
    xline.SetId(thread.thread.tid);
    xline.SetName(thread.thread.name);
    xline.SetTimestampNs(start_timestamp_ns);
    xline.ReserveEvents(thread.events.size());
    for (const auto& event : thread.events) {
      if (!IsCompleteEvent(event)) continue;
      Annotation annotation = ParseAnnotation(event.name);
      XEventMetadata*& xevent_metadata =
          xevent_metadata_by_name[annotation.name];
      if (xevent_metadata == nullptr) {
        xevent_metadata =
            xplane.GetOrCreateEventMetadata(xevent_metadata_by_name.size());
        xevent_metadata->set_name(string(annotation.name));
      }
      XEventBuilder xevent = xline.AddEvent(*xevent_metadata);
      xevent.SetTimestampNs(event.start_time);
      xevent.SetEndTimestampNs(event.end_time);
      xevent.ReserveStats(annotation.metadata.size());
      for (const auto& metadata : annotation.metadata) {
        XStatMetadata*& xstat_metadata = xstat_metadata_by_name[metadata.key];
        if (xstat_metadata == nullptr) {
          xstat_metadata =
              xplane.GetOrCreateStatMetadata(xstat_metadata_by_name.size());
          xstat_metadata->set_name(string(metadata.key));
        }
        xevent.ParseAndAddStatValue(*xstat_metadata, metadata.value);
      }
    }
  }
}

}  // namespace profiler
}  // namespace tensorflow

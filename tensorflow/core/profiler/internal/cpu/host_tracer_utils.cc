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

#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/internal/cpu/traceme_recorder.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/parse_annotation.h"
#include "tensorflow/core/profiler/utils/tf_op_utils.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

void MayAddDisplayName(XEventMetadata* xevent_metadata) {
  if (!xevent_metadata->display_name().empty()) return;
  std::string tf_op_event_name = TfOpEventName(xevent_metadata->name());
  if (tf_op_event_name != xevent_metadata->name()) {
    xevent_metadata->set_display_name(std::move(tf_op_event_name));
  }
}

}  // namespace

void ConvertCompleteEventsToXPlane(uint64 start_timestamp_ns,
                                   TraceMeRecorder::Events&& events,
                                   XPlane* raw_plane) {
  XPlaneBuilder xplane(raw_plane);
  for (auto& thread : events) {
    XLineBuilder xline = xplane.GetOrCreateLine(thread.thread.tid);
    xline.SetName(thread.thread.name);
    xline.SetTimestampNs(start_timestamp_ns);
    xline.ReserveEvents(thread.events.size());
    while (!thread.events.empty()) {
      auto event = std::move(thread.events.front());
      thread.events.pop_front();
      if (!event.IsComplete()) continue;
      if (event.start_time < start_timestamp_ns) continue;
      if (!HasMetadata(event.name)) {
        XEventMetadata* xevent_metadata =
            xplane.GetOrCreateEventMetadata(std::move(event.name));
        MayAddDisplayName(xevent_metadata);
        XEventBuilder xevent = xline.AddEvent(*xevent_metadata);
        xevent.SetTimestampNs(event.start_time);
        xevent.SetEndTimestampNs(event.end_time);
        continue;
      }
      Annotation annotation = ParseAnnotation(event.name);
      XEventMetadata* xevent_metadata =
          xplane.GetOrCreateEventMetadata(annotation.name);
      MayAddDisplayName(xevent_metadata);
      XEventBuilder xevent = xline.AddEvent(*xevent_metadata);
      xevent.SetTimestampNs(event.start_time);
      xevent.SetEndTimestampNs(event.end_time);
      xevent.ReserveStats(annotation.metadata.size());
      for (const auto& metadata : annotation.metadata) {
        XStatMetadata* xstat_metadata =
            xplane.GetOrCreateStatMetadata(metadata.key);
        xevent.ParseAndAddStatValue(*xstat_metadata, metadata.value);
      }
    }
  }
  SortXLinesBy(raw_plane, XLinesComparatorByName());
}

}  // namespace profiler
}  // namespace tensorflow

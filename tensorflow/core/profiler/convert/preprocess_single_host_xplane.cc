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
#include "tensorflow/core/profiler/convert/preprocess_single_host_xplane.h"

#include <vector>

#include "tensorflow/core/profiler/utils/derived_timeline.h"
#include "tensorflow/core/profiler/utils/group_events.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/tsl/profiler/utils/preprocess_xplane.h"
#include "tensorflow/tsl/profiler/utils/xplane_utils.h"

namespace tensorflow {
namespace profiler {

void PreprocessSingleHostXSpace(XSpace* space, bool step_grouping,
                                bool derived_timeline) {
  if (step_grouping && !tsl::profiler::IsXSpaceGrouped(*space)) {
    // Grouping (i.e. marking step number) events in the XSpace.
    std::vector<XPlane*> device_traces;
    bool isTpu = false;
    for (XPlane& plane : *space->mutable_planes()) {
      if (tsl::profiler::IsDevicePlane(plane)) {
        device_traces.push_back(&plane);
      }
      // Preprocess XPlane to convert stats to Traceme2 semantics
      tsl::profiler::PreprocessXPlane(&plane);

      if (!isTpu && absl::StartsWith(plane.name(), kTpuPlanePrefix)) {
        isTpu = true;
      }
    }

    EventForest event_forest;
    if (isTpu) {
      // group TPU events
      GroupTpuEventsOSS(space, device_traces, &event_forest);
    } else {
      // group GPU events
      GroupTfEvents(space, &event_forest);
    }

    if (derived_timeline) {
      // Generated miscellaneous derived time lines for device planes.
      GenerateDerivedTimeLines(event_forest.GetGroupMetadataMap(), space);
    }
  }
}

}  // namespace profiler
}  // namespace tensorflow

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
#include "tensorflow/core/profiler/convert/post_process_single_host_xplane.h"

#include "tensorflow/core/profiler/utils/derived_timeline.h"
#include "tensorflow/core/profiler/utils/group_events.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"

namespace tensorflow {
namespace profiler {

void MergeHostPlanes(XSpace* space) {
  const XPlane* cupti_driver_api_plane =
      FindPlaneWithName(*space, kCuptiDriverApiPlaneName);
  const XPlane* python_tracer_plane =
      FindPlaneWithName(*space, kPythonTracerPlaneName);
  if (cupti_driver_api_plane || python_tracer_plane) {
    XPlane* host_plane =
        FindOrAddMutablePlaneWithName(space, kHostThreadsPlaneName);
    if (cupti_driver_api_plane) {
      MergePlanes(*cupti_driver_api_plane, host_plane);
    }
    if (python_tracer_plane) {
      MergePlanes(*python_tracer_plane, host_plane);
    }
    SortXLinesBy(host_plane, XLinesComparatorByName());
    if (cupti_driver_api_plane) {
      RemovePlane(space, cupti_driver_api_plane);
    }
    if (python_tracer_plane) {
      RemovePlane(space, python_tracer_plane);
    }
  }
}

void PostProcessSingleHostXSpace(XSpace* space, uint64 start_time_ns) {
  VLOG(3) << "Post processing local profiler XSpace.";
  // Post processing the collected XSpace without hold profiler lock.
  // 1. Merge plane of host events with plane of CUPTI driver api.
  MergeHostPlanes(space);

  // 2. Normalize all timestamps by shifting timeline to profiling start time.
  // NOTE: this have to be done before sorting XSpace due to timestamp overflow.
  NormalizeTimestamps(space, start_time_ns);
  // 3. Sort each plane of the XSpace
  SortXSpace(space);
  // 4. Grouping (i.e. marking step number) events in the XSpace.
  EventForest event_forest;
  GroupTfEvents(space, &event_forest);
  // 5. Generated miscellaneous derived time lines for device planes.
  GenerateDerivedTimeLines(event_forest.GetGroupMetadataMap(), space);
}

}  // namespace profiler
}  // namespace tensorflow

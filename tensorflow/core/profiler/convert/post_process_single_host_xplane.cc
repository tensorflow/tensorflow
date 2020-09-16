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

#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/profiler/internal/profiler_factory.h"
#include "tensorflow/core/profiler/utils/derived_timeline.h"
#include "tensorflow/core/profiler/utils/group_events.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#endif

namespace tensorflow {
namespace profiler {

void PostProcessSingleHostXSpace(XSpace* space, uint64 start_time_ns) {
  VLOG(3) << "Post processing local profiler XSpace.";
  // Post processing the collected XSpace without hold profiler lock.
  // 1. Merge plane of host events with plane of CUPTI driver api.
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
    // NOTE: RemovePlaneWithName might invalidate plane pointers. so do these
    // at the last step.
    if (cupti_driver_api_plane) {
      RemovePlaneWithName(space, kCuptiDriverApiPlaneName);
    }
    if (python_tracer_plane) {
      RemovePlaneWithName(space, kPythonTracerPlaneName);
    }
  }

  // 2. Normalize all timestamps by shifting timeline to profiling start time.
  // NOTE: this have to be done before sorting XSpace due to timestamp overflow.
  NormalizeTimestamps(space, start_time_ns);
  // 3. Sort each plane of the XSpace
  SortXSpace(space);
  // 4. Grouping (i.e. marking step number) events in the XSpace.
  GroupMetadataMap group_metadata_map;
  GroupTfEvents(space, &group_metadata_map);
  // 5. Generated miscellaneous derived time lines for device planes.
  GenerateDerivedTimeLines(group_metadata_map, space);
}

}  // namespace profiler
}  // namespace tensorflow

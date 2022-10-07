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

#include "tensorflow/core/profiler/utils/derived_timeline.h"
#include "tensorflow/core/profiler/utils/group_events.h"
#include "tensorflow/tsl/profiler/utils/xplane_utils.h"

namespace tensorflow {
namespace profiler {

void PreprocessSingleHostXSpace(XSpace* space, bool step_grouping,
                                bool derived_timeline) {
  if (step_grouping && !tsl::profiler::IsXSpaceGrouped(*space)) {
    // Grouping (i.e. marking step number) events in the XSpace.
    EventForest event_forest;
    GroupTfEvents(space, &event_forest);
    if (derived_timeline) {
      // Generated miscellaneous derived time lines for device planes.
      GenerateDerivedTimeLines(event_forest.GetGroupMetadataMap(), space);
    }
  }
}

}  // namespace profiler
}  // namespace tensorflow

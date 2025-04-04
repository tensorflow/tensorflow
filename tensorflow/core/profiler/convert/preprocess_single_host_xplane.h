/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_PREPROCESS_SINGLE_HOST_XPLANE_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_PREPROCESS_SINGLE_HOST_XPLANE_H_

#include "xla/tsl/profiler/utils/group_events.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/utils/hlo_module_map.h"  // from @org_xprof

namespace tensorflow {
namespace profiler {

// Preprocess XSpaces before tools conversion.
// If step_grouping, perform events grouping for step tracking.
// If derived_timeline, generate derived timeline (XLines).
// HloModuleMap is used to cache the results of parsing the HloModuleMap from
// Xspace for use in later processing.
// If group_metadata_map is not nullptr, populate the group metadata map.
void PreprocessSingleHostXSpace(
    XSpace* space, bool step_grouping, bool derived_timeline,
    HloModuleMap& hlo_module_map,
    tsl::profiler::GroupMetadataMap* group_metadata_map = nullptr);

// Preprocess XSpaces before tools conversion.
// If step_grouping, perform events grouping for step tracking.
// If derived_timeline, generate derived timeline (XLines).
// If group_metadata_map is not nullptr, populate the group metadata map.
inline void PreprocessSingleHostXSpace(
    XSpace* space, bool step_grouping, bool derived_timeline,
    tsl::profiler::GroupMetadataMap* group_metadata_map = nullptr) {
  HloModuleMap hlo_module_map;
  PreprocessSingleHostXSpace(space, step_grouping, derived_timeline,
                             hlo_module_map, group_metadata_map);
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_PREPROCESS_SINGLE_HOST_XPLANE_H_

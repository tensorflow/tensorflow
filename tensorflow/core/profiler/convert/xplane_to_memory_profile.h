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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_MEMORY_PROFILE_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_MEMORY_PROFILE_H_

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/memory_profile.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {

// Process the host threads XPlane and generate MemoryProfile result; at most
// max_num_snapshots will be displayed on the UI.
// REQUIRED: host_plane should have been grouped by calling GroupTfEvents().
MemoryProfile ConvertXPlaneToMemoryProfile(const XPlane& host_plane,
                                           int64 max_num_snapshots = 1000);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_MEMORY_PROFILE_H_

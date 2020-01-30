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
#ifndef TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_UTILS_H_

#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {

// Returns the plane with the given name or nullptr if not found.
const XPlane* FindPlaneWithName(const XSpace& space, absl::string_view name);

// Returns all the planes with a given prefix.
std::vector<const XPlane*> FindPlanesWithPrefix(const XSpace& space,
                                                absl::string_view prefix);

// Returns the plane with the given name, create it if necessary.
XPlane* GetOrCreatePlane(XSpace* space, absl::string_view name);

// Returns true if event is nested by parent.
bool IsNested(const tensorflow::profiler::XEvent& event,
              const tensorflow::profiler::XEvent& parent);

void AddOrUpdateIntStat(int64 metadata_id, int64 value,
                        tensorflow::profiler::XEvent* event);

void AddOrUpdateStrStat(int64 metadata_id, absl::string_view value,
                        tensorflow::profiler::XEvent* event);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_UTILS_H_

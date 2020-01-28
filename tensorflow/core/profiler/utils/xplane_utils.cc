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
#include "tensorflow/core/profiler/utils/xplane_utils.h"

#include "absl/strings/match.h"

namespace tensorflow {
namespace profiler {

const XPlane* FindPlaneWithName(const XSpace& space, absl::string_view name) {
  for (const XPlane& plane : space.planes()) {
    if (plane.name() == name) return &plane;
  }
  return nullptr;
}

std::vector<const XPlane*> FindPlanesWithPrefix(const XSpace& space,
                                                absl::string_view prefix) {
  std::vector<const XPlane*> result;
  for (const XPlane& plane : space.planes()) {
    if (absl::StartsWith(plane.name(), prefix)) result.push_back(&plane);
  }
  return result;
}

XPlane* GetOrCreatePlane(XSpace* space, absl::string_view name) {
  for (XPlane& plane : *space->mutable_planes()) {
    if (plane.name() == name) return &plane;
  }
  XPlane* plane = space->add_planes();
  plane->set_name(std::string(name));
  return plane;
}

}  // namespace profiler
}  // namespace tensorflow

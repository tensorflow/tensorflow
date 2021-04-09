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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_TOOLS_DATA_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_TOOLS_DATA_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"

namespace tensorflow {
namespace profiler {

// Convert multiple XSpaces in <xspace_paths> to tool specific data.
// Return the serialized string of tool specific data and whether the conversion
// is successful.
std::pair<std::string, bool> ConvertMultiXSpacesToToolData(
    const std::vector<std::string>& xspace_paths,
    const absl::string_view tool_name);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_TOOLS_DATA_H_

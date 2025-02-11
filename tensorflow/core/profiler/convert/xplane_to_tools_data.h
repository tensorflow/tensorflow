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

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/profiler/convert/repository.h"
#include "tensorflow/core/profiler/convert/tool_options.h"

namespace tensorflow {
namespace profiler {

// Convert XSpace protos to a tool specific data.
// Return the serialized string of tool specific data when the conversion is
// successful, else return error status.
absl::StatusOr<std::string> ConvertMultiXSpacesToToolData(
    const SessionSnapshot& session_snapshot, absl::string_view tool_name,
    const ToolOptions& options);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_TOOLS_DATA_H_

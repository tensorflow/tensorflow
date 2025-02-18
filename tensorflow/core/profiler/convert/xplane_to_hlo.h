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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_HLO_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_HLO_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/service/hlo.pb.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/profiler/convert/repository.h"

namespace tensorflow {
namespace profiler {

// Get HLO proto by module name.
absl::StatusOr<xla::HloProto> GetHloProtoByModuleName(
    const SessionSnapshot& session_snapshot, absl::string_view module_name);

// Converts multiple XSpaces to HLO protos.
// Stores the HLO protos as files in the same directory as the xspace files.
// Returns whether there are HLO protos in this profile.
absl::StatusOr<bool> ConvertMultiXSpaceToHloProto(
    const SessionSnapshot& session_snapshot);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_HLO_H_

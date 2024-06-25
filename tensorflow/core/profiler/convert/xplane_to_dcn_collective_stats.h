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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_DCN_COLLECTIVE_STATS_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_DCN_COLLECTIVE_STATS_H_

#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/profiler/convert/repository.h"
#include "tensorflow/core/profiler/protobuf/dcn_slack_analysis.pb.h"

namespace tensorflow {
namespace profiler {

// Converts multiple XSpaces to dcn collective stats.
// Stores the dcn collective stats as files in the same directory
// as the xspace files.
absl::StatusOr<bool> ConvertMultiXSpaceToDcnCollectiveStats(
    const SessionSnapshot& session_snapshot);

// Returns whether there are dcn collective stats in the profile.
absl::StatusOr<bool> HasDcnCollectiveStatsInMultiXSpace(
    const SessionSnapshot& session_snapshot);

// Gets DcnSlackAnalysis proto for a host.
absl::StatusOr<DcnSlackAnalysis> GetDcnSlackAnalysisByHostName(
    const SessionSnapshot& session_snapshot, std::string hostname);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_DCN_COLLECTIVE_STATS_H_

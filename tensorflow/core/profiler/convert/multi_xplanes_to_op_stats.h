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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_MULTI_XPLANES_TO_OP_STATS_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_MULTI_XPLANES_TO_OP_STATS_H_

#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/profiler/convert/repository.h"
#include "tensorflow/core/profiler/convert/xplane_to_op_stats.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"

namespace tensorflow {
namespace profiler {

// Converts and combines multiple XSpace protos into a single OpStats
// <combined_op_stats>.
// Return the first error status during conversion, or return OkStatus() if
// there is no error.
absl::Status ConvertMultiXSpacesToCombinedOpStats(
    const SessionSnapshot& session_snapshot, const OpStatsOptions& options,
    OpStats* combined_op_stats);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_MULTI_XPLANES_TO_OP_STATS_H_

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

#include "tensorflow/core/profiler/convert/multi_xplanes_to_op_stats.h"

#include <vector>

#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/profiler/convert/op_stats_combiner.h"
#include "tensorflow/core/profiler/convert/preprocess_single_host_xplane.h"
#include "tensorflow/core/profiler/convert/repository.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/utils/hardware_type_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {

absl::Status ConvertMultiXSpacesToCombinedOpStats(
    const SessionSnapshot& session_snapshot, const OpStatsOptions& options,
    OpStats* combined_op_stats) {
  // Read multiple XSpaces and convert to multiple OpStats.
  // TODO(profiler): Change the combiner to convert and combine one OpStats at a
  // time, to reduce peak memory usage.
  std::vector<OpStats> all_op_stats;
  all_op_stats.reserve(session_snapshot.XSpaceSize());
  for (int i = 0; i < session_snapshot.XSpaceSize(); i++) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<XSpace> xspace,
                        session_snapshot.GetXSpace(i));
    PreprocessSingleHostXSpace(xspace.get(), /*step_grouping=*/true,
                               /*derived_timeline=*/true);
    all_op_stats.push_back(ConvertXSpaceToOpStats(*xspace, options));
  }

  // Combine OpStats.
  std::vector<OpStatsInfo> all_op_stats_info;
  all_op_stats_info.reserve(all_op_stats.size());
  for (int i = 0; i < all_op_stats.size(); i++) {
    all_op_stats_info.emplace_back(
        &all_op_stats[i],
        ParseHardwareType(all_op_stats[i].run_environment().device_type()), i);
  }

  // Do not limit the maximum number of steps during the merge of OpStats.
  StepIntersection step_intersection =
      ComputeStepIntersectionToMergeOpStats(all_op_stats_info, kuint32max);
  CombineAllOpStats(all_op_stats_info, step_intersection, combined_op_stats);

  return absl::OkStatus();
}

}  // namespace profiler
}  // namespace tensorflow

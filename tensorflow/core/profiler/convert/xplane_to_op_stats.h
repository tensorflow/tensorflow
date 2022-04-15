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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_OP_STATS_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_OP_STATS_H_

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/profiler/protobuf/hardware_types.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {

struct OpStatsOptions {
  bool maybe_drop_incomplete_steps = false;
  bool generate_op_metrics_db = false;
  bool generate_step_db = false;
  bool generate_kernel_stats_db = false;
};

// NOTE: call GroupTfEvents before if OpStats.step_db needs to be generated.
OpStats ConvertXSpaceToOpStats(const XSpace& space,
                               const OpStatsOptions& options);

// Propagate and dedup the diagnostics in XSpace and add to OpStats.
void PropagateXSpaceDiagnosticsToOpStats(const XSpace& space,
                                         OpStats* op_stats);

// Populates PerfEnv.
PerfEnv MakePerfEnv(double peak_tera_flops_per_second,
                    double peak_hbm_bw_giga_bytes_per_second);

// Extracts PerfEnv from XPlane stats.
PerfEnv GetPerfEnvFromXPlane(const XPlane& device_plane);

// Takes an XSpace proto message, converts to OpStats, and
// combine them to a single OpStats in <combined_op_stats>.
// Return the first error status during conversion, or return Status::OK() if
// there is no error.
Status ConvertMultiXSpacesToCombinedOpStats(const std::vector<XSpace>& xspaces,
                                            const OpStatsOptions& options,
                                            OpStats* combined_op_stats);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_OP_STATS_H_

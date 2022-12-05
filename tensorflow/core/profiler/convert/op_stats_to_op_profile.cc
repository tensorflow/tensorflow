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

#include "tensorflow/core/profiler/convert/op_stats_to_op_profile.h"

#include <string>

#include "absl/strings/match.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/profiler/convert/op_profile_builder.h"
#include "tensorflow/core/profiler/protobuf/hardware_types.pb.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/op_profile.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/utils/math_utils.h"
#include "tensorflow/core/profiler/utils/op_metrics_db_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tensorflow::profiler::GigaToGibi;
using ::tensorflow::profiler::IsIdleOp;
using ::tensorflow::profiler::OpMetrics;
using ::tensorflow::profiler::OpProfileBuilder;
using ::tensorflow::profiler::OpProfileOptions;
using ::tensorflow::profiler::OpStats;
using ::tensorflow::profiler::TeraToGiga;
using ::tensorflow::profiler::TotalTimePs;
using ::tensorflow::profiler::op_profile::Node;

void BuildOpProfileNodeTree(const OpStats& op_stats, bool group_by_program,
                            bool exclude_idle_ops, Node* root) {
  const auto& metrics_db = op_stats.device_op_metrics_db();
  if (metrics_db.metrics_db().empty()) return;

  OpProfileOptions options = {group_by_program,
                              /*group_by_deduplicated_name=*/true,
                              /*children_per_node=*/100};
  OpProfileBuilder builder(options, root, &op_stats.program_id_to_name_map());

  for (const OpMetrics& op_metrics : metrics_db.metrics_db()) {
    DCHECK(!op_metrics.name().empty());
    // Don't add ops that cannot be symbolized.
    if (absl::StartsWith(op_metrics.name(), "region")) continue;
    if (exclude_idle_ops && IsIdleOp(op_metrics)) continue;
    builder.AddOp(op_metrics);
  }

  const auto& perf_env = op_stats.perf_env();
  double max_gigaflops_per_second_per_core =
      TeraToGiga(perf_env.peak_tera_flops_per_second());
  double max_gibibytes_per_second_per_core =
      GigaToGibi(perf_env.peak_bw_giga_bytes_per_second());
  double max_hbm_gibibytes_per_second_per_core =
      GigaToGibi(perf_env.peak_hbm_bw_giga_bytes_per_second());
  builder.Finalize(max_gigaflops_per_second_per_core,
                   max_gibibytes_per_second_per_core,
                   max_hbm_gibibytes_per_second_per_core,
                   TotalTimePs(metrics_db, exclude_idle_ops));
}

}  // namespace

void ConvertOpStatsToOpProfile(
    const OpStats& op_stats, tensorflow::profiler::HardwareType hardware_type,
    tensorflow::profiler::op_profile::Profile& profile) {
  profile.set_device_type(HardwareType_Name(hardware_type));
  BuildOpProfileNodeTree(op_stats,
                         /*group_by_program=*/false,
                         /*exclude_idle_ops=*/false,
                         profile.mutable_by_category());

  BuildOpProfileNodeTree(op_stats,
                         /*group_by_program=*/false,
                         /*exclude_idle_ops=*/true,
                         profile.mutable_by_category_exclude_idle());

  // Don't generate per program profile if there's only a single program.
  if (op_stats.program_id_to_name_map_size() > 1) {
    BuildOpProfileNodeTree(op_stats,
                           /*group_by_program=*/true,
                           /*exclude_idle_ops=*/false,
                           profile.mutable_by_program());

    BuildOpProfileNodeTree(op_stats,
                           /*group_by_program=*/true,
                           /*exclude_idle_ops=*/true,
                           profile.mutable_by_program_exclude_idle());
  }
}

}  // namespace profiler
}  // namespace tensorflow

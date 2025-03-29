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
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/match.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "tensorflow/core/profiler/convert/op_profile_builder.h"
#include "tensorflow/core/profiler/protobuf/hardware_types.pb.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/op_profile.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "plugin/tensorboard_plugin_profile/protobuf/hardware_types.pb.h"  // from @org_xprof
#include "plugin/tensorboard_plugin_profile/protobuf/op_stats.pb.h"  // from @org_xprof
#include "xprof/utils/op_metrics_db_utils.h"  // from @org_xprof

namespace tensorflow {
namespace profiler {
namespace {

using ::tensorflow::profiler::IsIdleOp;
using ::tensorflow::profiler::OpMetrics;
using ::tensorflow::profiler::OpProfileBuilder;
using ::tensorflow::profiler::OpProfileOptions;
using ::tensorflow::profiler::OpStats;
using ::tensorflow::profiler::TotalTimePs;
using ::tensorflow::profiler::op_profile::Node;

void BuildOpProfileNodeTree(const OpStats& op_stats, bool group_by_program,
                            bool exclude_idle_ops, int op_profile_limit,
                            Node* root) {
  const auto& metrics_db = op_stats.device_op_metrics_db();
  if (metrics_db.metrics_db().empty()) return;

  OpProfileOptions options = {group_by_program,
                              /*group_by_deduplicated_name=*/true,
                              /*children_per_node=*/op_profile_limit};
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
      tsl::profiler::TeraToGiga(perf_env.peak_tera_flops_per_second());
  std::vector<double> peak_bws;
  for (auto bw : perf_env.peak_bws_giga_bytes_per_second()) {
    peak_bws.push_back(tsl::profiler::GigaToGibi(bw));
  }
  builder.Finalize(max_gigaflops_per_second_per_core, peak_bws,
                   TotalTimePs(metrics_db, exclude_idle_ops));
}

}  // namespace

void ConvertOpStatsToOpProfile(
    const OpStats& op_stats, tensorflow::profiler::HardwareType hardware_type,
    tensorflow::profiler::op_profile::Profile& profile, int op_profile_limit) {
  profile.set_device_type(HardwareType_Name(hardware_type));
  BuildOpProfileNodeTree(op_stats,
                         /*group_by_program=*/false,
                         /*exclude_idle_ops=*/false, op_profile_limit,
                         profile.mutable_by_category());

  BuildOpProfileNodeTree(op_stats,
                         /*group_by_program=*/false,
                         /*exclude_idle_ops=*/true, op_profile_limit,
                         profile.mutable_by_category_exclude_idle());

  BuildOpProfileNodeTree(op_stats,
                         /*group_by_program=*/true,
                         /*exclude_idle_ops=*/false, op_profile_limit,
                         profile.mutable_by_program());

  BuildOpProfileNodeTree(op_stats,
                         /*group_by_program=*/true,
                         /*exclude_idle_ops=*/true, op_profile_limit,
                         profile.mutable_by_program_exclude_idle());
}

}  // namespace profiler
}  // namespace tensorflow

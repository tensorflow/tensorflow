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

#include "tensorflow/compiler/jit/report_clustering_info_pass.h"

#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/xla_activity_listener.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"

namespace tensorflow {
Status ReportClusteringInfoPass::Run(
    const GraphOptimizationPassOptions& options) {
  XlaAutoClusteringActivity activity;
  *activity.mutable_summary() = GetXlaAutoClusteringSummary(**options.graph);
  activity.set_global_jit_level(GetGlobalJitLevelForGraph(options));
  activity.set_cpu_global_jit_enabled(
      GetMarkForCompilationPassFlags()->tf_xla_cpu_global_jit);
  return BroadcastXlaActivity(std::move(activity));
}
}  // namespace tensorflow
